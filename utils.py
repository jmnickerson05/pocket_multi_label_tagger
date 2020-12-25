import pandas as pd
import numpy as np
import json
import nltk
import re
import matplotlib.pyplot as plt 
import seaborn as sns
import keyring
import requests
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import time
import ssl
import sqlite3 
import ast
import itertools
from collections import Counter
import pickle
import xgboost
import html2text
from datetime import datetime
import pathlib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import xgboost as xgb
from xgboost.sklearn import XGBClassifier # <3
from sklearn.model_selection import train_test_split
import gc
import warnings; warnings.simplefilter('ignore')

pd.set_option('display.max_colwidth', 300)

sqlite3.register_converter("pickle", pickle.loads)
sqlite3.register_adapter(xgboost.core.Booster, pickle.dumps)
conn = sqlite3.connect('data/pocket.db', detect_types=sqlite3.PARSE_DECLTYPES)

stop_words = set(stopwords.words('english'))
clean_non_ascii = lambda wrd: re.sub(r"[^{}]".format(string.ascii_letters), " ", wrd.lower())
remove_stop_words = lambda text: ' '.join([w for w in text.split() if not w in stop_words])
extract_tags = lambda x: list(ast.literal_eval(x).keys())

def get_html_text(url):
    try:
        return html2text.html2text(requests.get(url, timeout=3).text)
    except Exception as e:
        pass
#             print(e)
#             errors[row.item_id] = e 
        return None

def clean_text(text):
    clean_non_ascii = lambda wrd: re.sub(r"[^{}]".format(string.ascii_letters), " ", wrd.lower())
    remove_stop_words = lambda text: ' '.join([w for w in text.split() if not w in stop_words])
    return remove_stop_words(clean_non_ascii(text))

conn.create_function('get_html_text', 1 , get_html_text)
conn.create_function('clean_text', 1 , clean_text)

def check_ntlk_dependencies():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download()

def get_articles_df(fetch_all=False):
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

def plot_tag_counts():
    tags = pd.read_sql('select * from v_all_labels limit 25', conn)
    plt.figure(figsize=(12,15)) 
    ax = sns.barplot(data=tags, x= "tag_count", y = "tags", palette='bright') 
    ax.set(ylabel = 'Count') 
    plt.show()

def plot_word_frequencies(terms = 30):     
    clean_non_ascii = lambda wrd: re.sub(r"[^{}]".format(string.ascii_letters), " ", wrd.lower())
    remove_stop_words = lambda text: ' '.join([w for w in text.split() if not w in stop_words])
    df_col = pd.read_sql('select excerpt from text_features', conn).excerpt
    df_col = df_col.apply(clean_non_ascii).apply(remove_stop_words)
    all_words = ' '.join([text for text in df_col]) 
    all_words = all_words.split() 
    fdist = nltk.FreqDist(all_words) 
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())}) 

    words_df = words_df.nlargest(columns="count", n = terms) 

    # visualize words and frequencies
    plt.figure(figsize=(12,15)) 
    ax = sns.barplot(data=words_df, x= "count", y = "word", palette='bright') 
    ax.set(ylabel = 'Word') 
    plt.show()

def get_categories():
    return set(pd.read_sql('select tags from v_all_labels', conn).tags)
    
def save_models(models: dict):
    """Derived from example at: 
    https://gist.github.com/JonathanRaiman/aa0bdfd8e3511c59f3af"""
    
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
    for i in model_outputs_df.to_dict(orient='records'):
        item_id = i['item_id']
        del i['item_id']
        conn.execute("""insert into model_output 
                        (item_id, model_probabilities) 
                        values (?, ?)""", (item_id, json.dumps(i)))
    conn.commit()
    
def prep_dataframe(df, text_col):
    tdf = df[['item_id','tags', text_col]]
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
    # class_names = ['python', 'postgres', 'sql']
    class_names = tags

    text_col = 'combined_text'
    train, test = train_test_split(tdf, random_state=42, test_size=0.33, shuffle=True)

    # train = pd.read_csv('../train.csv').fillna(' ')#.sample(1000)
    # test = pd.read_csv('../test.csv').fillna(' ')#.sample(1000)

    train_text = train[text_col]
    test_text = test[text_col]
    all_text = pd.concat([train, test])

    train = train.loc[:,class_names]

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
    del train_char_features,train_word_features
    test_features = hstack([test_char_features, test_word_features])
    del test_char_features,test_word_features

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

            xgb_params = {'eta': 0.3, 
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
#         print('Total CV score is {} for the label: "{}"'.format(np.mean(cv_scores), class_name))

    output.to_csv('output.csv', index=False)
    save_models(models)
    save_model_output(output)
    return output
    #TODO: Save errors as well
    #TODO: Add xgb_params to save_models()
    #TODO: Research Error -- https://github.com/dmlc/xgboost/issues/505

def predict_new(df):
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
    models = {l:m for l, m in conn.execute("""select label, model from models""").fetchall()}
    for class_name, model in models.items():
        try:
            print('Predicting for the label: "{}"'.format(class_name))
            output[class_name] = model.predict(d_features)
#             models[class_name.replace(' ', '_')] = model
        except Exception as e:
            errors[class_name] = e
            print(e)
#         print('Total CV score is {} for the label: "{}"'.format(np.mean(cv_scores), class_name))
#     print(errors)
    return output

new_article_output = predict_new(prediction_df)

    #TODO: Save errors as well
    #TODO: Add xgb_params to save_models()
    #TODO: Research Error -- https://github.com/dmlc/xgboost/issues/505
    
    
# def plot_tag_counts(df):
#     to_dict = lambda x: ast.literal_eval(x)
#     tags = list(itertools.chain.from_iterable(
#         df[df.tags.notnull()].tags.apply(lambda x: list(to_dict(x).keys())).values.tolist()
#         ))
#     tags = pd.DataFrame.from_dict(Counter(tags), orient='index').reset_index()
#     tags.columns = ['Tag', 'Count']
    
# #     tags_df = pd.DataFrame({'Tag': list(tags.keys()), 'Count': list(tags.values())})
#     tags = tags.nlargest(columns="Count", n = 25) 
#     plt.figure(figsize=(12,15)) 
#     ax = sns.barplot(data=tags, x= "Count", y = "Tag", palette='bright') 
#     ax.set(ylabel = 'Count') 
#     plt.show()
    
# def plot_word_frequencies(df_col, terms = 30):     
#     df_col = df_col.apply(clean_non_ascii).apply(remove_stop_words)
#     all_words = ' '.join([text for text in df_col]) 
#     all_words = all_words.split() 
#     fdist = nltk.FreqDist(all_words) 
#     words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())}) 

#     words_df = words_df.nlargest(columns="count", n = terms) 

#     # visualize words and frequencies
#     plt.figure(figsize=(12,15)) 
#     ax = sns.barplot(data=words_df, x= "count", y = "word", palette='bright') 
#     ax.set(ylabel = 'Word') 
#     plt.show()
    
# def get_categories(df):
#     is_dict = lambda x: type(ast.literal_eval(x)) == dict
#     df = df[df.tags.notnull()]
#     return set(df[df.tags.apply(is_dict)].tags.apply(extract_tags).explode().unique())

# def get_articles_df(fetch_new_only=True):
# #     def get_all_articles_list():
# #         #https://your-pocket-oauth-token.glitch.me/
# #         consumer_key = keyring.get_password('POCKET_CONSUMER_KEY', 'jmnickerson05@gmail.com')
# #         access_token = keyring.get_password('POCKET_ACCESS_TOKEN', 'jmnickerson05@gmail.com')
# #         last_updated = conn.execute("""select CAST(strftime('%s', max(date_updated))
# #             as int) latest_date
# #         from articles;""").fetchone()[0]
# #         parameters = {
# #                     "consumer_key": consumer_key,
# #                     "access_token": access_token,
# #                     "sort": "oldest",
# #                     "state": "all",
# #                     "detailType": "complete",
# #                     "since": last_updated
# #            }
# #         response = requests.get("https://getpocket.com/v3/get", params=parameters)
# #         rest_dict = response.json()
# #         articles_dict = rest_dict["list"]
# #         articles_list = [i for i in articles_dict.values()]
# #         return articles_list
#     def get_all_articles_list():
#         wd = pathlib.Path(__file__).parent.absolute()
#         stmts = open('data/import_new_articles.text').read().format(wd=wd).split(';')
#         for sql in stmts:
#             conn.execute(sql)
#         pd.read_sql("", conn)
        
    
#     def try_get_html_text(row):
#         try:
#             return html2text.html2text(requests.get(row.given_url, timeout=3).text)
#         except Exception as e:
#             print(e)
#             errors[row.item_id] = e 
#             return None

#     def add_datetimes(df):
#         epoch_to_dt = lambda epoch: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(epoch)))
#         df['date_added'] = df.time_added.apply(epoch_to_dt)
#         df['date_updated'] = df.time_updated.apply(epoch_to_dt)
# #         df = df.replace(np.nan, '', regex=True)
#         return df
    
#     def create_text_feature(df):
#         df.excerpt = df.excerpt.replace(np.nan, '', regex=True)
#         df.html_text = df.html_text.replace(np.nan, '', regex=True)
#         df['combined_text'] = df.excerpt + df.html_text
#         return df

#     def clean_text(df):
#         df.combined_text = df[df.combined_text.notnull()].combined_text.apply(
#             clean_non_ascii
#         ).apply(remove_stop_words)
#         return df
    
#     def save_articles(df):
#         dict_cols = ['authors', 'domain_metadata', 'image', 'images', 'videos', 'tags']
#         for d in dict_cols:
#             df[d] = df[d].apply(lambda x: json.dumps(x) 
#                                 if x is not None else x)
#         for d in df.to_dict(orient='records'):
#             try:
#                 columns = ", ".join(list(d.keys()))
#                 var_binders = ('?,' * len(d)).rstrip(',')            
#                 values = tuple([v for v in d.values()])
#                 stmt = f'insert or replace into articles ({columns}) values ({var_binders})' 
#                 conn.execute(stmt, values)
#             except Exception as e:
#                 conn.execute("insert into errors (item_id, errmsg) values ('{}','{}')".format(
#                     d['item_id'], str(e).replace("'","''")
#                 ))
# #                 print(stmt, values, e)
#         conn.commit()
        
#     if fetch_new_only:
#         df = pd.DataFrame(get_all_articles_list())
#         errors = dict()
#         df['html_text'] = df[['item_id','given_url']].apply(try_get_html_text, axis=1)
#         errors = pd.DataFrame(errors.items(), columns=['item_id', 'errmsg'])
# #         errors['created_at'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
# #         errors.to_sql('html_download_errors', conn, if_exists='append')
#         errors.to_csv('data/errors.csv')       
        
#         save_articles(df)
# #         df.to_csv('data/articles.csv',index=False)
#     else:
# #         df = pd.read_csv('data/articles.csv')
#         df = pd.read_sql('select * from articles', conn)
#         dict_cols = ['authors', 'domain_metadata', 'image', 'images', 'videos', 'tags']
#         def try_load_json(x):
#             if x:
#                 try:
#                     return json.loads(x) 
#                 except Exception as e:
#                     print(e)
#                     print(x)
#                     return x
                
#         for d in dict_cols:
#             df[d] = df[d].apply(try_load_json)
    
#     df = add_datetimes(df)
#     df = create_text_feature(df)
#     df = clean_text(df)
    
#     return df
    
