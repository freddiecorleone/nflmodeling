import shelve, nflhtmlparsing as parser, pandas as pd, sqlite3, time, random as rand, re 



shelfFile = shelve.open('data/nfldata2.0')

games = shelfFile['finalgamedata']
game = games[3234]

print(game.week, game.year, game.awayteam, game.hometeam)
shelfFile.close()

'''print(len(shelfFile))
print(list(shelfFile.keys()))

print(len(shelfFile['nflgamedatawithplaybyplay']))

nflgames = shelfFile['nflgamedatawithplaybyplay']

print(type(nflgames[0]))

print(type(nflgames[0].playbyplay[1]))


plays = pd.DataFrame({'game_id': [], 'description': [],  'field_position': []})
plays.loc[len(plays)] = [1, 'Peyton Manning fumbles', 'IND 47']

print(plays)

conn = sqlite3.connect("nfl_data.db")
df_games = pd.read_sql('SELECT * FROM plays', conn)
print(df_games)

conn.close()'''




















