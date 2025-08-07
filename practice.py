import sys, pandas as pd, sqlite3
from src.utils import paths

modelingpath = paths.get_project_root() / "data" / "nfl_data.db"
conn = sqlite3.connect(modelingpath)
QUERY = '''
SELECT half_seconds_remaining, time FROM plays WHERE half_seconds_remaining < 30
'''
df = pd.read_sql(QUERY, conn)
conn.close()

print(df.head(50))



