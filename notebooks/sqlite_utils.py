import html2text
import functools
import json
import sqlite3
import xgboost as xgb
import pickle
import re
import nltk
# Download before importing -- NLTK doesn't download data by default.
nltk.download('stopwords')
from nltk.corpus import stopwords
import contextlib
"""
Registers automatic data type conversions between Python and SQLite.
These autoconvert Python dicts, lists, and tuples to json objects and arrays.
"""
adapt_dict = lambda data: json.loads(data, sort_keys=True)
adapt_json = lambda data: (json.dumps(data, sort_keys=True)).encode()
convert_json = lambda blob: json.loads(blob.decode())
sqlite3.register_converter("pickle", pickle.loads)
sqlite3.register_adapter(xgb.core.Booster, pickle.dumps)
sqlite3.register_adapter(dict, adapt_json)
sqlite3.register_adapter(dict, adapt_dict)
sqlite3.register_adapter(list, adapt_json)
sqlite3.register_adapter(tuple, adapt_json)
sqlite3.register_converter('JSON', convert_json)

def none_on_exception(fn):
    """Used to convert SQL functions errors in NULLS
    so that the queries don't fail a small number of records"""
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None
    return inner

"""
These decatorators are used for all SQLite functions
-- @none_on_exception
-- @functools.lru_cache(maxsize=128):
    - lru_cache caches function calls so that 
    the results of identical inputs reads from the cache 
"""

@none_on_exception
@functools.lru_cache(maxsize=128)
def get_html_text(url):
    exclusions = ['.pdf', '.mp4']
    if not any([True for ex in exclusions if ex in url]):
        return html2text.html2text(requests.get(url, timeout=3).text)

@none_on_exception
@functools.lru_cache(maxsize=128)
def clean_text(text):
    print('Processing')
    stop_words = set(stopwords.words('english'))
    clean_non_ascii = lambda wrd: re.sub(r"[^{}]".format(string.ascii_letters), " ", wrd.lower())
    remove_stop_words = lambda text: ' '.join([w for w in text.split() if not w in stop_words])
    return remove_stop_words(clean_non_ascii(text))

class SQLiteConn:
    """
    Example taken from:
    https://www.blog.pythonlibrary.org/2015/10/20/python-201-an-intro-to-context-managers/

    Example usage:
        db = 'example.db'
        with SQLiteConn(db) as conn:
            result = conn.execute('select * from sqlite_master').fetchall()
    """
    def __init__(self, db_name):
        """Constructor"""
        self.db_name = db_name

    def __enter__(self):
        """Open the database connection"""
        self.conn = sqlite3.connect('../data/pocket.db', detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.create_function("get_html_text", 1, get_html_text)
        self.conn.create_function('clean_text', 1, clean_text)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the connection"""
        self.conn.close()
