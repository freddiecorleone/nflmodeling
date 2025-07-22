import bs4, requests, re, os, urllib.request
from decimal import Decimal 



class PFRParser:
    def __init__(self, url):
        self.URL = url
        self.res = requests.get(url)
        #print(self.res.text)
        self.soup = bs4.BeautifulSoup(self.res.text, features="html.parser")
        self.teams = None
        self.hometeamquarters = None
        self.awayteamquarters = None
            

    def getTeamNames(self):
        elems = self.soup.find('div', class_='linescore_wrap')
        elems2 = elems.find('tbody')
        names = elems2.find_all('a')

        self.teams = [names[2].getText(), names[5].getText()]

        elems3 = elems2.find_all('td', class_='center')
       
        self.hometeamquarters = []
        self.awayteamquarters = []
        if(len(elems3)== 12):
            for i in range(1, 6):
                self.awayteamquarters.append(elems3[i].getText())
            for j in range(7, 12):
                 self.hometeamquarters.append(elems3[j].getText())
        if(len(elems3)== 14):
            for i in range(1, 7):
                self.awayteamquarters.append(elems3[i].getText())
            for j in range(8, 14):
                 self.hometeamquarters.append(elems3[j].getText())
  

        return self.teams
    
    def getQuarterScores(self):
        if self.hometeamquarters == None:
            self.getTeamNames()

        return self.awayteamquarters, self.hometeamquarters
    

    def getGameInfo(self):
        regexLine = re.compile(r'(Line|Roof|Under|Weather)<\/th><td .*?>(.*?)<')

     
        grps = regexLine.findall(self.res.text)
        return grps

    
   

    def getPlaybyPlay(self):
        regexline = re.compile(r'(\S+) to receive the opening|<tr class="(divider)"|href="#pbp_(\d*?).000">(\d*?:\d\d).*? data-stat="down" >(\d{0,1})</td>.*?data-stat="yds_to_go" >(\d{0,2})</td>.*?data-stat="location".*?>(.*?)</td>.*?data-stat="pbp_score_aw".*?>(.*?)</td>.*?data-stat="pbp_score_hm".*?>(.*?)</td>.*?data-stat="detail".*?>(.*?)</td>.*?data-stat="exp_pts_before".*?>(.*?)</td>.*?data-stat="exp_pts_after".*?>(.*?)</td>', re.DOTALL)
        elem = regexline.findall(self.res.text)

        linkregex = re.compile(r'</{0,1}a.*?>', re.DOTALL)



        del elem[0]
        del elem[0]

        cleanedplaybyplay = []

        possessionarrow = self.determineInitialPossession(elem)

        del elem[0]

       
        

        
        
        for i in range(0, len(elem)):

    
            yi = list(elem[i])
            if yi[1] == 'divider':
                if possessionarrow:
                    possessionarrow = False
                else:
                    possessionarrow = True
            else:
                yi[9] = linkregex.sub('', yi[9])
                del yi[0]
                del yi[0]
                del yi[0]
                try:
                    yi[1] = int(yi[1])
                    yi[2] = int(yi[2])
                    yi[4] = int(yi[4])
                    yi[5] = int(yi[5])
                    yi[7] = float(yi[7])
                    yi[8] = float(yi[8])
                except:
                    pass
                            
                yi.append(possessionarrow)
                cleanedplaybyplay.append(yi)

        
         

        return cleanedplaybyplay
    

    def determineInitialPossession(self, elems):
        teamnames = self.getTeamNames()
        print(teamnames)
        possessionarrow = None
        if elems[0][0]in teamnames[0]:
            possessionarrow = False
        if  elems[0][0] in teamnames[1] :
            possessionarrow = True

        return possessionarrow

    



    




            
class NFLGame:
    def __init__(self):
        self.parser = None
        self.boxscore = None
        self.spread = None
        self.overunder = None
        self.playbyplay = None
        self.gameinfo = None
        self.week = None
        self.year = None
        self.awayteam = None
        self.hometeam = None


   
    def getAwayTeam(self):
        return self.awayteam
    
    def getBoxScore(self):
        return self.boxscore
    
    def getSpread(self):

        for item in self.gameinfo:
            if item[0] == 'Line' and 'Pick' in item[1]:
                return [self.awayteam, 0]
            
            if item[0] == 'Line':
                x = item[1].split(' -')
                x[1] = - float(x[1])
                return x

                
            
    def getOverUnder(self):
        for item in self.gameinfo:
            if item[0] == 'Under':
                return float(item[1])
            
    def getWeather(self):
        weather = 'NA'
        for tup in self.gameinfo:
            if tup[0] == 'Weather':
                weather = tup[1]
        return weather
    def getRoof(self):
        return self.gameinfo[0][1]

            

    def getWinningMargin(self):

        favorite = self.getSpread()[0]
        awayfinal = float(self.boxscore[0][-1])
        homefinal = float(self.boxscore[1][-1])
        if favorite == self.awayteam :
            return [favorite, awayfinal-homefinal]
        if favorite == self.hometeam:
            return [self.hometeam, homefinal - awayfinal]
        

    def coverMargin(self):
        return self.getSpread()[1] + self.getWinningMargin()[1]
    
    def getWeek(self):
        return int(self.week)

            
        
        


            
    









class PFRWeekParser:
    def __init__(self, week, year):
        self.URL = 'https://www.pro-football-reference.com/years/' + str(year) + '/week_' + str(week) + '.htm'
        self.res = requests.get(self.URL)
        self.res.raise_for_status()
        print(os.getcwd())

        


    def getGameURLS(self):
        
        
        #linksoup = bs4.BeautifulSoup(self.res)

        

        #starter = linksoup.find_all('td')
        #print(len(starter))
        
        #gamelinkregex = re.compile(r'(<td)', re.DOTALL)
        gamelinkregex = re.compile(r'<td class="right gamelink">.*?<a href="(/boxscores/.*?.htm)"', re.DOTALL)
        links = gamelinkregex.findall(self.res.text)
        print('links acquired: ' + str(len(links)))
        fulllinks = []
        for link in links:
            fulllinks.append('https://www.pro-football-reference.com/' + link)

        return fulllinks
    




        
           

       

        
       
    



         
          
    

            

        



