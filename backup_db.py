import sqlite3
import zipfile
import os

# conn = sqlite3.connect('pocket.db')
#  open('pocket.sql', 'w+').write("\r\n".join(conn.iterdump()))
os.chdir('data/')
os.system('sqlite3 pocket.db -cmd ".output ./pocket.sql" ".dump" ".exit"')

zipfile.ZipFile('pocket.sql.zip', mode='w').write('pocket.sql')

os.remove('pocket.sql')
