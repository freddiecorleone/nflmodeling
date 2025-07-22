import pandas as pd







def abbrevationToTeamName(str):
    df = pd.read_csv('nfl_teams_pfr_abbreviations_final.csv')

    for index, row in df.iterrows():
        if row['PFR Abbreviation'] in str:
            return row['Team Name']
        

    return 'Not Found'


def TeamNameToAbbreviation(str):

     df = pd.read_csv('data/nfl_teams_pfr_abbreviations_final.csv')

     for index, row in df.iterrows():
       
        if row['Team Name'] in str or str in row['Team Name']:
            return row['PFR Abbreviation']
    

     return 'Not Found'
     
    







def isThereAPlay(play):
    if play[6].contains('no play'):
        return False
    return True


print(TeamNameToAbbreviation('Oakland Raiders'))