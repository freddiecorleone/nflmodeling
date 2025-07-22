import pandas as pd, shelve, sqlite3, playindentificationtools as pit, re




def column(matrix, i):
    return [row[i] for row in matrix]



def createPlaysDataFrame():
    shelf = shelve.open('data/nfldata2.0')
    nflgames = shelf['finalgamedata']
    shelf.close()


    regex = re.compile(r'(\D\D\D)\s(\d+)')
    playlist = []

    
    for i in range(0, len(nflgames)):
        hometeam = nflgames[i].hometeam
        homeabbrev = pit.TeamNameToAbbreviation(hometeam)
        print('game id: ' + str(i))
        playbyplay = nflgames[i].playbyplay
        
        currentquarter = 0
        booly = True
        prevplay = None
       
        for play in playbyplay:

            #update quarter info

            if play[0] == '15:00' and booly:
                        currentquarter += 1
                        booly = False
            elif play[0] != '15:00':
                booly = True

            #ignore timeouts
            if play[5]!= '':

                if play[2] != '':
                    listify = list(play)
                    #add quarters to play by play
                    

                
                    listify.insert(0, currentquarter)
                    listify.insert(0, i)

                    #modify field position info
                    reggie = regex.search(listify[5])
                    if listify[5] != '':
                        if reggie.group(1) == homeabbrev:
                            listify[5] = int(reggie.group(2))
                        else:
                            listify[5] = 100 - int(reggie.group(2))
                    #shift plays by one
                    if prevplay != None:
                        listify[6] = prevplay[4]
                        listify[7] = prevplay[5]
                    playlist.append(listify)
                #update previous play
                prevplay = play
            

                
                
        
        


            #plays.loc[len(plays)] =listify
                



    
    plays = pd.DataFrame({'game_id': column(playlist, 0), 'quarter': column(playlist, 1), 'time': column(playlist, 2), 'down': column(playlist, 3), 'to_go': column(playlist, 4), 'field_position': column(playlist, 5),
                          'awaypoints': column(playlist, 6), 'homepoints': column(playlist, 7), 'description': column(playlist, 8), 'EPB': column(playlist, 9), 'EPA': column(playlist, 10), 'home_possession': column(playlist, 11)})

        


    return plays




def createGamesDataFrame():
    return 



'''def addQuartersToAllPlayByPlays():
    shelf = shelve.open('data/nflgamedata2.0')
    nflgames = shelf['nflgamedatawithplaybyplay']
    for nflgame in nflgames:
        addQuartersToPlayByPlay(nflgame.playbyplay)


def amendPlayByPlay(playbyplay,gameindex ):
    plays = []

    currentquarter = 0
    booly = True

    for play in playbyplay:
        listify = list(play)
       
        if listify[0] == '15:00' and booly:
            currentquarter += 1
            booly = False
        elif listify[0] != '15:00':
            booly = True
        
        listify.insert(0, currentquarter)
        listify.insert(0, gameindex)
        plays.append(listify)

    return plays'''






'''shelf = shelve.open('nfldata2.0')
print(shelf.keys())
nflgames = shelf['nflgamedatawithplaybyplay']

playbyplay = nflgames[26].playbyplay

print(addQuartersToPlayByPlays(playbyplay))'''

def LoadPlaysDataFrameIntoSQL():
    plays = createPlaysDataFrame()
    print(plays)
    conn = sqlite3.connect("data/nfl_data.db")
    plays.to_sql("plays", conn, if_exists="replace", index=False)
    conn.close()


def createGamesDataFrame():
    shelf = shelve.open('data/nfldata2.0')
    nflgames = shelf['finalgamedata']
    shelf.close()

    gamelist = []

    for i in range(0, len(nflgames)):
        try:
            print(i)
            game = nflgames[i]
            qtrscores = reformatQuarterScore(game)

            listify = [i, game.week, game.year, game.awayteam, game.hometeam, 
                    game.getRoof(), game.getWeather(), game.getOverUnder(), game.getSpread()[0], float(game.getSpread()[1])]
            listify = listify + qtrscores
            if len(listify) != 22:
                print(len(listify))
                print(i)
            else:
                gamelist.append(listify)
        except:
            print('prob for game ' + str(i))
      
    games = pd.DataFrame({'game_id': column(gamelist, 0), 'week': column(gamelist, 1), 
                         'year': column(gamelist, 2), 'away_team': column(gamelist, 3),
                          'home_team': column(gamelist, 4),'roof': column(gamelist, 5),
                            'weather': column(gamelist, 6), 'over_under': column(gamelist, 7), 
                            'favorite': column(gamelist, 8),
                             'spread':   column(gamelist, 9), 'q1Away':  column(gamelist, 10),
                             'q2Away':  column(gamelist, 11), 'q3Away':  column(gamelist, 12),
                             'q4Away':  column(gamelist, 13), 'OTAway':  column(gamelist, 14),
                             'final_Away':  column(gamelist, 15), 'q1Home':  column(gamelist, 16),
                             'q2Home':  column(gamelist, 17), 'q3Home':  column(gamelist, 18),
                             'q4Home':  column(gamelist, 19), 'OTHome':  column(gamelist, 20),
                             'final_Home':  column(gamelist, 21)})
    shelf.close()
    
    return games
     

def reformatQuarterScore(game):
    qtrscores = list(game.getBoxScore())

    awayscores = []
    for score in qtrscores[0]:
        awayscores.append(int(score))
        
    homescores = []

    for score in qtrscores[1]:
        homescores.append(int(score))

    if len(awayscores) == 5:
        awayscores.insert(4, 'NA')
        homescores.insert(4, 'NA')
    
    listo = awayscores + homescores

    return listo


def LoadGamesDataFrameIntoSQL():
    games = createGamesDataFrame()
    print(games)
    conn = sqlite3.connect("data/nfl_data.db")
    games.to_sql("games", conn, if_exists="replace", index=False)
    conn.close()



LoadPlaysDataFrameIntoSQL()

