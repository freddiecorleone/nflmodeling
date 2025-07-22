import shelve, time, random
from src.nflhtmlparsing import PFRParser, PFRWeekParser, NFLGame


class NFLGameDataSetBuilder:
    def  __init__(self):
        return 

        
    def getGameData(tuple):
        games = []
        
        for i in range(len(tuple[2])):
            time.sleep(random.randrange(3, 6))
            game = NFLGame()

            link = tuple[2][i]
            try:
                parser = PFRParser(link)
                game.boxscore = parser.getQuarterScores()
                    
            except:
                print('error getting quarter scores data week' + str(tuple[0]) + ", year " + str(tuple[1]))

            try:
                teams = parser.getTeamNames()
                game.awayteam, game.hometeam = teams[0], teams[1]
                print(game.awayteam + " at " + game.hometeam + ' week' + str(tuple[0]) + ", year " + str(tuple[1]))
            except:
                print('error getting team names week' + str(tuple[0]) + ", year " + str(tuple[1]))

            try:
                game.gameinfo = parser.getGameInfo()
                game.week, game.year = tuple[0], tuple[1]
                print(game.gameinfo)
                    

            except:
                print(game.awayteam + ' ' + game.hometeam + ' error getting game info week' + str(tuple[0]) + ", year " + str(tuple[1]))


            try:
                game.playbyplay = parser.getPlaybyPlay()

            except:
                print(game.awayteam + ' ' + game.hometeam + ' error getting play by play week' + str(tuple[0]) + ", year " + str(tuple[1]))
            
            
            games.append(game)

        return games
        
    

            
         
         

    
    

    def gatherLinks(wklb, wkub, yrlb, yrub):
        linkweekyeartuples = []
        for week in range(wklb, wkub):
            for year in range(yrlb, yrub):
                try:
                    print(str(week) + ' ' + str(year))
                    weekparser = PFRWeekParser(week, year)
                    somelinks = weekparser.getGameURLS()

                    linkweekyeartuples.append([week, year, somelinks])
                    time.sleep(random.randrange(4, 9))
                except:
                    print("error gathering links for week " + str(week) + ", year " + str(year) )
            

        return linkweekyeartuples

            

            



#print(len(shelfFile['links']))
#tuples = shelfFile['links']

#print(shelfFile['links'][2])

#fullgameslist = shelfFile['nflgamedatawithplaybyplay']
#gameslist = []

#for i in range(430, 432):
    #shelfFile['index'] = i
    #games = NFLGameDataSetBuilder.getGameData(tuples[i])
    #print(games[0].boxscore)
    #gameslist.extend(games)

#fullgameslist.extend(gameslist)

#shelfFile['nflgamedatawithplaybyplay'] = fullgameslist

#shelfFile.close()















        
