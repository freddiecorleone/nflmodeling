import shelve, src.nflhtmlparsing as nflhtmlparsing, matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scipy import stats
import pandas as pd
from sklearn import linear_model

shelf = shelve.open('nfldata')

games = shelf['nflgamedata']


'''winningmargins = []

for game in games:
    mrg = abs(game.getWinningMargin()[1])
    winningmargins.append(mrg)

frequency = Counter(winningmargins)
margs = list(frequency.keys())
freqs = list(frequency.values())
for i in range(0, len(freqs)):
    freqs[i] = freqs[i]/len(games)

x = np.array(margs)
y = np.array(freqs)

ints = []
for i in range(0, 15):
    ints.append(i)



plt.bar(x, y)
plt.xlim(0, 15)
plt.xticks(ints)
plt.title('NFL margin of victory frequencies')
plt.xlabel('Margin of victory')
plt.ylabel('Likelihood')
plt.show()'''



'''

teasercover = []
ous = []

for game in games:
    spread = game.getSpread()[1] 
    covermargin = game.coverMargin()
    margin = game.getWinningMargin()[1]
    ou = game.getOverUnder()

    if spread == -3:
        
    
        if covermargin > - 6:
            teasercover.append(1) 
            ous.append(ou)
        if covermargin <= -6:
            teasercover.append(0)
            ous.append(ou)


print(teasercover)
slope, intercept, r, p, std_err = stats.linregress(ous, teasercover)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, ous))

plt.scatter(ous, teasercover)
plt.plot(ous, mymodel)
plt.show()
print('slope: ' + str(slope))
print('int: ' + str(intercept))
print('p-value: ' + str(p))
print('std err: ' + str(std_err))'''


teasercover = []
ous = []
spreads = []

for game in games:
    spread = game.getSpread()[1] 
    covermargin = game.coverMargin()
    margin = game.getWinningMargin()[1]
    ou = game.getOverUnder()

    if spread >= -8.5 and spread <= -7.5:
        
        spreads.append(spread)
        ous.append(ou)
        if covermargin > -6:
            teasercover.append(1) 
            
        if covermargin <= -6:
            teasercover.append(0)
            

print(len(teasercover))
print(len(ous))
print(len(spreads))
dict = {'Cover': teasercover, 'overunder': ous, 'spreads': spreads}
df = pd.DataFrame(dict)

X = df[['overunder', 'spreads']]
Y = df['Cover']
regr = linear_model.LinearRegression()
regr.fit(X, Y)
print(regr.coef_)
print(regr.intercept_)


'''print(teasercover)
slope, intercept, r, p, std_err = stats.linregress(ous, teasercover)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, ous))

plt.scatter(ous, teasercover)
plt.plot(ous, mymodel)
plt.show()
print(slope)
print(intercept)
print(p)
print(std_err)'''

'''  

covpercs = []
pushpercs = []



for i in range(-1, 5):
    totalWongGames = 0
    totalcovgames = 0
    totalpushgames = 0
    for game in games:
        spread = game.getSpread()[1] 
        ou = game.getOverUnder()
        week = game.getWeek()
        covermargin = game.coverMargin()
        if spread >= -8.5 and spread <= -7.5 and ou > (35 + 5*i) and ou <= (35 + 5*(i+1)):
            totalWongGames +=1
            if covermargin > -6:
                totalcovgames +=1
            if covermargin == -6:
                totalpushgames +=1
    covperc = totalcovgames/totalWongGames
    pushperc = totalpushgames/totalWongGames
    print(totalWongGames)
    covpercs.append(covperc)
    pushpercs.append(pushperc)
ous = [37, 42]
print(covpercs)
print(pushpercs)'''
shelf.close()






    
