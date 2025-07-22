import shelve, src.nflhtmlparsing as nflhtmlparsing, src.nfldatasetbuilder as dsb

def resetDownload():

    db = shelve.open('nfldata2.0')
    db['index'] = 0
    db['finalgamedata'] = []


def downloadGames(howmany):


    

    shelfFile = shelve.open('data/nfldata2.0')
    links = shelfFile['links']

    gameslist = []
    startpoint = shelfFile['index']
    stoppoint = min(len(links), startpoint + howmany)
    fullgameslist = shelfFile['finalgamedata']


    for i in range(startpoint, stoppoint):
        games = dsb.NFLGameDataSetBuilder.getGameData(links[i])
        gameslist.extend(games)

    fullgameslist.extend(gameslist)
    print('PROCESS FINISHED')
    print(stoppoint)
    print('Games Downloaded:' + str(len(fullgameslist)))

    shelfFile['index'] = stoppoint
    shelfFile['finalgamedata'] = fullgameslist
    shelfFile.close()
        


    return 

downloadGames()




