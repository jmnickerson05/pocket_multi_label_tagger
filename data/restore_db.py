import zipfile
import os
import sqlite3

with zipfile.ZipFile('pocket.sql.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

if os.path.exists('pocket.sql.zip'):
    os.remove('pocket.sql.zip')

if os.path.exists('pocket.db'):
    os.remove('pocket.db')

os.system('sqlite3 pocket.db < pocket.sql')

# conn = sqlite3.connect('pocket.db')

# for sql in open('pocket.sql', 'r').read().split(';'):
#     conn.execute(sql.replace("'", "''"))