import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
#Download before importing -- NLTK doesn't download data by default.
# nltk.download('stopwords')
from nltk.corpus import stopwords
import ssl
import ast
# This module extract text from bodies of HTML.
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import xgboost as xgb
# from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
import gc
import warnings
import concurrent.futures
from tqdm import tqdm
from pocket import Pocket, PocketException
from sqlite_utils import SQLiteConn, clean_text, get_html_text

warnings.simplefilter('ignore')
pd.set_option('display.max_colwidth', 300)
db = '../data/pocket.db'

def generate_new_features():
	"""
	Uses a predefined SQLite View that uses the Python user-defined SQLite functions
	to clean and prep the text fields.

	NOTE:
		This works because SQLite does no checks on view definitions.
		So, you can create a SQLite view using functions (Python runtime functions)
		that don't exist at the time of view creation.
	"""
	print('Generating new text features..')
	with SQLiteConn(db) as conn:
		conn.execute("insert into text_features select * from v_get_feature_text").fetchall()
		conn.commit()

def add_new_tags(concurrency_bool=False):
	"""
	Collects the outstanding delta of tag predictions whose
	article have not yet been updated with the Pocket API.

	Submits tag predictions (additional tags) to the Pocket API.

	(Optional: Concurrency specification that will execute many requests concurrently.
	Useful to process a large amount of predictions quickly).
	"""
	with SQLiteConn(db) as conn:
		new_tags = conn.execute("""
			SELECT item_id,
				   group_concat(tag, ',') tags,
				   strftime('%s', 'now')  ts
			from v_new_tags
			group by 1, 3;
		""").fetchall()

	pauth = json.loads(open('auth.json').read())
	p = Pocket(
		consumer_key=os.environ.get('pocket_add_modify_consumer_key'),
		access_token=os.environ.get('pocket_add_modify_access_token')
	)

	total_record_cnt = len(new_tags)
	print(f'Total records to process: {total_record_cnt}')

	def add_tag(t):
		#         print(*t)
		return p.tags_add(*t).commit()

	responses = []
	if concurrency_bool:
		with concurrent.futures.ThreadPoolExecutor() as tpe:
			results = {tpe.submit(add_tag, t): t[0] for t in new_tags}
			with tqdm(total=total_record_cnt) as pbar:
				for res in concurrent.futures.as_completed(results):
					responses.append(res.result())
					pbar.update(1)
	else:
		with tqdm(total=total_record_cnt) as pbar:
			for t in new_tags:
				responses.append(p.tags_add(*t))
				pbar.update(1)
	return responses

def check_ntlk_dependencies():
	"""Ensures the ability to download NLTK data"""
	try:
		_create_unverified_https_context = ssl._create_unverified_context
	except AttributeError:
		pass
	else:
		ssl._create_default_https_context = _create_unverified_https_context

	nltk.download()

def get_articles_df(fetch_all=False):
	"""Gets new batch of article for running the ML model against"""
	# TODO: This date is hard coded.
	#  Should be able to grab the max(last_modified) from items table.
	if not fetch_all:
		return pd.read_sql("""select t.*, i.tags, i.resolved_title, i.resolved_url
			from text_features t
					 join items i on t.item_id = i.item_id
			where date(date_added) = date('2020-06-26')
		""", conn)
	elif fetch_all:
		return pd.read_sql("""select t.*, i.tags, i.resolved_title, i.resolved_url
			from text_features t
					 join items i on t.item_id = i.item_id
		""", conn)


def plot_tag_counts(number_of_tags=15):
	"""Plots counts of articles by tags(labels)"""
	with SQLiteConn(db) as conn:
		tags = pd.read_sql(f"""select * from v_all_labels limit {number_of_tags}""", conn)
		plt.figure(figsize=(12, 15))
		ax = sns.barplot(data=tags, x="tag_count", y="tags", palette='bright')
		ax.set(ylabel='Count')
		plt.savefig('tag_counts.png')
		plt.show()
		conn.close()


def plot_word_frequencies(number_of_terms=15):
	"""Plots frequency of words
	TODO: Replace raw counts w/ TFIDF keywords (that should automatically down-weight any common filler words).
	"""
	stop_words = set(stopwords.words('english'))
	clean_non_ascii = lambda wrd: re.sub(r"[^{}]".format(string.ascii_letters), " ", wrd.lower())
	exclusions = list(stop_words) + ['using', 'use', 'one', 'used', 'new', 'time', 'many']
	remove_stop_words = lambda text: ' '.join([w for w in text.split() if not w in exclusions])
	df_col = pd.read_sql('select excerpt from text_features', conn).excerpt
	df_col = df_col.apply(clean_non_ascii).apply(remove_stop_words)
	all_words = ' '.join([text for text in df_col])
	all_words = all_words.split()
	fdist = nltk.FreqDist(all_words)
	words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

	words_df = words_df.nlargest(columns="count", n=number_of_terms)

	plt.figure(figsize=(12, 15))
	ax = sns.barplot(data=words_df, x="count", y="word", palette='bright')
	ax.set(ylabel='Word')
	plt.savefig('word_frequencies.png')
	plt.show()

def get_categories():
	"""Get unique list of tags(labels)"""
	return set(pd.read_sql('select tags from v_all_labels', conn).tags)

def save_models(models: dict):
	"""Derived from example at: 
	https://gist.github.com/JonathanRaiman/aa0bdfd8e3511c59f3af

	Saves model as pickled BLOBs in the SQLite DB.
	"""
	with SQLiteConn(db) as conn:
		conn.execute('drop table if exists models')
		conn.execute("""create table if not exists models (
						label text, 
						model pickle, 
						attributes json, 
						pyobj_type text
							);""")

		for mname, model in models.items():
			conn.execute("""insert into models 
							(label, model, attributes, pyobj_type) 
							 values (?, ?, ?, ?)""",
						 (mname,
						  model,
						  json.dumps(model.attributes()),
						  str(type(model)))
						 )
			conn.commit()

def save_model_output(model_outputs_df):
	with SQLiteConn(db) as conn:
		"""Saves model probability output to the 'model_output' tables"""
		for i in model_outputs_df.to_dict(orient='records'):
			item_id = i['item_id']
			del i['item_id']
			conn.execute("""insert into model_output 
							(item_id, model_probabilities) 
							values (?, ?)""", (item_id, json.dumps(i)))
		conn.commit()

def prep_dataframe(df, text_col):
	tdf = df[['item_id', 'tags', text_col]]
	tdf.tags = tdf.tags.fillna('{}')
	extract_tags = lambda x: list(ast.literal_eval(x).keys())
	tdf['tag_list'] = tdf.tags.apply(extract_tags)
	all_tags = get_categories()
	del tdf['tags']

	for t in all_tags:
		tdf[t] = np.NaN
		tdf[t] = tdf[t].fillna(0)
		tdf[t] = tdf[t].astype('int64')

	for row in tdf.itertuples():
		for tag in all_tags:
			if tag in row.tag_list:
				tdf.at[row.Index, tag] = int(1)
			else:
				tdf.at[row.Index, tag] = int(0)

	return tdf, all_tags

def train_models(tdf, tags):
	"""
	Trains an individual XGBoost classification model for each tag.

	This is a brute force method of getting around the Imbalanced Class
	problem inherent in this dataset
	(e.g. hundreds of articles for some tags and next to 0 for others).
	"""
	class_names = tags

	text_col = 'combined_text'
	train, test = train_test_split(tdf, random_state=42, test_size=0.33, shuffle=True)

	train_text = train[text_col]
	test_text = test[text_col]
	all_text = pd.concat([train, test])

	train = train.loc[:, class_names]

	print("TFIDF")
	word_vectorizer = TfidfVectorizer(
		sublinear_tf=True,
		strip_accents='unicode',
		analyzer='word',
		token_pattern=r'\w{1,}',
		stop_words='english',
		ngram_range=(1, 1),
		norm='l2',
		min_df=0,
		smooth_idf=False,
		max_features=15000)
	word_vectorizer.fit(all_text)
	train_word_features = word_vectorizer.transform(train_text)
	test_word_features = word_vectorizer.transform(test_text)

	char_vectorizer = TfidfVectorizer(
		sublinear_tf=True,
		strip_accents='unicode',
		analyzer='char',
		stop_words='english',
		ngram_range=(2, 6),
		norm='l2',
		min_df=0,
		smooth_idf=False,
		max_features=50000)
	char_vectorizer.fit(all_text)
	train_char_features = char_vectorizer.transform(train_text)
	test_char_features = char_vectorizer.transform(test_text)

	train_features = hstack([train_char_features, train_word_features])
	del train_char_features, train_word_features
	test_features = hstack([test_char_features, test_word_features])
	del test_char_features, test_word_features

	print(train_features.shape)
	print(test_features.shape)
	d_test = xgb.DMatrix(test_features)
	del test_features
	gc.collect()

	print("Modeling")
	cv_scores = []
	xgb_preds = []
	output = pd.DataFrame.from_dict({'item_id': test['item_id']})
	errors = dict()
	models = dict()
	for class_name in class_names:
		try:
			train_target = train[class_name]
			# Split out a validation set
			X_train, X_valid, y_train, y_valid = train_test_split(
				train_features, train_target, test_size=0.25, random_state=23)

			xgb_params = {
				'eta': 0.3,
				'max_depth': 5,
				'subsample': 0.8,
				'colsample_bytree': 0.8,
				'objective': 'binary:logistic',
				'eval_metric': 'auc',
				'seed': 23
			}

			d_train = xgb.DMatrix(X_train, y_train)
			d_valid = xgb.DMatrix(X_valid, y_valid)

			watchlist = [(d_valid, 'valid')]
			model = xgb.train(xgb_params, d_train, 200, watchlist, verbose_eval=False, early_stopping_rounds=30)
			print("class Name: {}".format(class_name))
			print(model.attributes()['best_msg'])
			cv_scores.append(float(model.attributes()['best_score']))
			output[class_name] = model.predict(d_test)
			models[class_name.replace(' ', '_')] = model
			print('Total CV score is {} for the label: "{}"'.format(np.mean(cv_scores), class_name))
			del X_train, X_valid, y_train, y_valid
			gc.collect()
		except Exception as e:
			errors[class_name] = e
	#             print(e)

	output.to_csv('output.csv', index=False)
	save_models(models)
	save_model_output(output)
	return output
	# TODO: Save errors as well
	# TODO: Add xgb_params to save_models()
	# TODO: Research Error -- https://github.com/dmlc/xgboost/issues/505


def predict_new(df):
	"""Runs ML classification predictions for new records"""
	#     df = df[['item_id', 'combined_text']]
	class_names = list(set(pd.read_sql('select tags from v_all_labels', conn).tags))
	text_col = 'combined_text'

	all_text = df[text_col]

	print("TFIDF")
	word_vectorizer = TfidfVectorizer(
		sublinear_tf=True,
		strip_accents='unicode',
		analyzer='word',
		token_pattern=r'\w{1,}',
		stop_words='english',
		ngram_range=(1, 1),
		norm='l2',
		min_df=0,
		smooth_idf=False,
		max_features=15000)

	word_vectorizer.fit(all_text)
	word_features = word_vectorizer.transform(all_text)

	char_vectorizer = TfidfVectorizer(
		sublinear_tf=True,
		strip_accents='unicode',
		analyzer='char',
		stop_words='english',
		ngram_range=(2, 6),
		norm='l2',
		min_df=0,
		smooth_idf=False,
		max_features=50000)
	char_vectorizer.fit(all_text)
	char_features = char_vectorizer.transform(all_text)

	features = hstack([char_features, word_features])
	del char_features, word_features

	print(features.shape)
	d_features = xgb.DMatrix(features)
	del features
	gc.collect()

	print("Modeling")
	cv_scores = []
	xgb_preds = []
	output = pd.DataFrame.from_dict({'item_id': df['item_id']})
	errors = dict()
	with SQLiteConn(db) as conn:
		models = {l: m for l, m in conn.execute("""select label, model from models""").fetchall()}
	for class_name, model in models.items():
		try:
			print('Predicting for the label: "{}"'.format(class_name))
			output[class_name] = model.predict(d_features)
		#             models[class_name.replace(' ', '_')] = model
		except Exception as e:
			errors[class_name] = e
			print(e)
	#     print(errors)
	return output

