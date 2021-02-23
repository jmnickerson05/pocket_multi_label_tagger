import sqlite3
import zipfile
import os

os.chdir('data/')
os.system('sqlite3 pocket.db -cmd ".output ./pocket.sql" ".dump" ".exit"')
os.system('sqlite3 pocket.db -cmd ".output ./pocket_schema.sql" ".schema" ".exit"')

zipfile.ZipFile('pocket.sql.zip', mode='w').write('pocket.sql')
os.remove('pocket.sql')

zipfile.ZipFile('pocket_schema.sql.zip', mode='w').write('pocket_schema.sql')
os.remove('pocket_schema.sql')