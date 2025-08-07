import GameSimulatorModels as gsm, joblib, pandas as pd
from utils import paths as paths
import numpy as np
import pandas as pd
from scipy.stats import truncnorm, expon


class MonteCarloSimulator():
    def __init__(self, gamecondition):
        self.passmodel = gsm.PassPlayResultModel.load()
        self.runmodel = gsm.RunPlayResultModel.load()
        self.playdecisionmodel  = gsm.PlayDecisionModel.load()
        self.fourthdowndecisionmodel = gsm.FourthDownDecisionModel.load()
        self.gamecon = gamecondition 
        self.fieldgoalmodel =gsm.FieldGoalModel()

        ''' dataframe with columns:
        home_team
        away_team
        home_score
        away_score
        game_seconds_remaining:
        down
        ydstogo
        yardline_100
        home_possession 
        home_timeouts
        away_timeouts
        margin
        play_type 'normal', 'extra_point', 'kick_off'
        clock_stopped
        offense_mode 'hurry_up', 'kill_clock', or 'kill_and_kick'
          '''
        if game_seconds_remaining > 0:
            self.gameover = False
        else:
            self.gameover = True
        self.initialgamecon = gamecondition
        self.playsdf = gamecondition
        return 
    



    def SimulateGame(self):


        
      

        while  not self.game_over:
            self.runPlay()

        return 


    def runPlay(self):

      


        if self.get('play_type') == 'normal': # not an extra point or kickoff 
            self.runNormalPlay()

            




        
        return 
    
    def runNormalPlay(self):

        offenseteam = self.teamOnOffense()
        defenseteam = self.teamOnDefense()
        currentfieldposition = self.get('yardline_100')




        

        #should the team on offense kick a last second field goal to tie or win the game?
        if self.lastSecondFG():
            print(f'{self.offenseteam} attempts {self.get('yardline_100')+ 17} yard field goal')
            result = fieldgoalmodel.simulate(self.get('yardline_100'))
            if result == 'make':
                print('It is good!')
                self.updateScore('field_goal')
                self.set('play_type', 'kick_off')
                self.print({offenseteam})
                print(f'{self.get('home_team')}: {self.get('home_score')}, {self.get('away_team')}: {self.get('away_score')}')
                
            else:
                print(f'It is no good! {defenseteam} gets possession.')
                self.switchPossession()
                self.set('down', 1)
                self.set('yardstogo', 10)
            
            self.set('clock_stopped', True)
            play_type = 'field_goal_attempt'

        
        else:
            #MAKE DECISION punt kick pass or run? 
             choicedf = self.formatDataFrameForDecisionModel()
             if self.get('down') == 4:
                play_type == self.fourthdowndecisionmodel.make_decision(choicedf)
        
            if self.get('down') < 4:
                play_type = self.playdecisionmodel.simulate_decision(choicedf) #simulate play choice (run or pass)
                resultdf = self.formatDataFrameForResultModel()  #gather data in right format for model
                

            #get result of play
            if play_type == 'run':

                play_result = self.runmodel.simulate_play(resultdf)
                self.set('clock_stopped', False)
                print(f'{offenseteam} runs for {np.min(play_result, yardline_100)} yards')

            if play_type == 'pass':
                play_result = self.runmodel.simulate_play(resultdf)
                if play_result == 0:
                    self.set('clock_stopped', True)
                    print('Incomplete Pass')
                else:
                    self.set('clock_stopped', False)
                    print(f'{offenseteam} completes pass for {np.min(play_result, yardline_100)} yards')

               
                #check if a touchdown was scored
            if self.scoredTouchdown():
                print(f'Touchdown for {offenseteam}!')
                self.updateScore('touchdown')
                self.set('play_type', 'extra_point')
                self.set('clock_stopped', True)

            #check if a safety was scored
            elif self.scoredSafety():
                print(f'Safety for {defenseteam}')
                self.updateScore('safety')
                self.set('play_type', 'kick_off')
                self.set('clock_stopped', True)
            #check for turnover on downs
            elif self.turnoverOnDowns():


            #if there was no score:
            else:
                #check for first down, update accordingly
                if self.gotFirstDown():
                    print('First Down!')
                    self.set('down', 1)
                    self.set('yardstogo', 10)
                else:
                    current_down = self.get('down')
                    self.set('down', currentdown + 1)
                    currentyardstogo = self.get('yardstogo')
                    self.set('yardstogo', currentyardstogo - play_result)
                
                    self.set('yardline_100', currentfieldposition - play_result)
                    print(f'{self.get('down')} down, {self.get('yardstogo')} yards to go, and {self.get('yardline_100')} yards from goal line.')

         
            
        
        self.updateClock(play_type)

            
                    


        return 


    
    def updateClock(self, play_type):


        to, to_team = self.timeoutdecision()

        
        #updatenumberoftimeouts
        if (home_possession == 1 and to_team == 'offense'):
            current_tos = self.get('home_timeouts')
            self.set('home_timeouts', current_tos - 1)
            print(f'{self.teamOnOffense()} calls timeout. They have {current_tos -1} timeouts remaining. ')
        if (home_possession == 0 and to_team == 'defense'):
            current_tos = self.get('home_timeouts')
            self.set('home_timeouts', current_tos - 1)
            print(f'{self.teamOnDefense()} calls timeout. They have {current_tos -1} timeouts remaining. ')
        if (home_possession == 0 and to_team == 'offense'):
            current_tos = self.get('away_timeouts')
            self.set('away_timeouts', current_tos - 1)
            print(f'{self.teamOnOffense()} calls timeout. They have {current_tos -1} timeouts remaining. ')
          if (home_possession == 1 and to_team == 'defense'):
            current_tos = self.get('away_timeouts')
            self.set('away_timeouts', current_tos - 1)
            print(f'{self.teamOnDefense()} calls timeout. They have {current_tos -1} timeouts remaining. ')

        if to or self.get('clock_stopped'):
            play_runoff = 6
        
        elif self.get('offense_mode') == 'hurry_up':
            play_runoff = 20
        else:
            play_runoff = 40

        self.set('game_seconds_remaining',self.get('game_seconds)_remaining') - play_runoff)

        self.gameover = gameEnded()
        

        return

    def updateScore(self, type):
        currentmargin = self.get('margin')
        homescore = self.get('home_score')
        away_score = self.get('away_score')

        if type == 'touchdown':
            if home_possession == 1: 
                self.set('margin',currentmargin + 6 ) 
                self.set('home_score',homescore + 6 )
            else:
                self.set('margin', currentmargin - 6)
                self.set('away_score', away_score + 6)
        if type == 'field_goal':
             if home_possession == 1: 
                self.set('margin',currentmargin + 3 ) 
                self.set('home_score',homescore + 3 )
            else:
                self.set('margin', currentmargin - 3)
                self.set('away_score', away_score + 3)
        if type == 'safety':
             if home_possession == 1: 
                self.set('margin',currentmargin -2 ) 
                self.set('away_score',awayscore + 2 )
            else:
                self.set('margin', currentmargin + 2)
                self.set('home_score', home_score + 2)
        if type == 'xp':
            if home_possession == 1: 
                self.set('margin',currentmargin + 1 ) 
                self.set('home_score',homescore + 1 )
            else:
                self.set('margin', currentmargin - 1)
                self.set('away_score', away_score + 1)
        if type == '2xp':
             if home_possession == 1: 
                self.set('margin',currentmargin + 1 ) 
                self.set('home_score',homescore + 1 )
            else:
                self.set('margin', currentmargin - 1)
                self.set('away_score', away_score + 1)
    

        return 

   


    def timeout_decision(self):
    """
    Rule-based timeout decision model with caller identification.
    
    game_state: dict with keys
        - seconds_left: int (seconds remaining in half)
        - margin: int (offense score - defense score)
        - clock_stopped: bool (did the clock stop on this play)
        - off_timeouts: int (timeouts remaining for offense)
        - def_timeouts: int (timeouts remaining for defense)
        - next_down: int
        - next_ydstogo: int
        - next_yardline: int
        - posteam: str (offense abbreviation)
        - defteam: str (defense abbreviation)
    
    Returns (called: bool, caller: str | None)
        - called: True if a timeout is used
        - caller: 'offense' / 'defense' / None
    """

    # Don’t call if clock already stopped
    if self.get('clock_stopped'):
        return False, None

    seconds = self.get('game_seconds_left')
    
    if home_possession == 1:
        margin = self.get('margin')
        off_to = self.get('home_timeouts')
        def_to = self.get('away_timeouts')
    else:
        margin = -self.get('margin')
        off_to = self.get('away_timeouts')
        def_to = self.get('home_timeouts')


    

    # --- Offense trailing in final 2 minutes ---
    if margin < 0 and seconds <= 120 and off_to > 0:
        return True, 'offense'

    # --- Defense leading/tied, stop the clock after 2nd/3rd downs ---
    if margin >= 0 and seconds <= 120 and def_to > 0 and down in [2,3]:

        return True, 'defense'

    # --- End-of-half desperation (both sides) ---
    if seconds <= 40:
        if margin < 0 and off_to > 0:
            return True, 'offense'
        if margin >= 0 and def_to > 0:
            return True, 'defense'

    return False, None



    def gameEnded(self):
        if self.get('game_seconds_remaining') <= 0:
            return True
        return False
    
    def scoredSafety(self):
        if self.get('yardline_100') - play_result >=100:
            self.return True
        return False
    


    def teamOnOffense(self):
        if home_possession ==1:
            return self.get('home_team')
        else:
            return self.get('away_team')

    def teamOnDefense(self):
        if home_possession == 0:
            return self.get('home_team')
        else:
            return self.get('away_team')

    
    def scoredTouchdown(self, play_result):
        if play_result >= yardline_100:
            return True
        return False

    def turnoverOnDowns(self, play_result):


    
    def get(self, s):
        return self.gamecon[s].iloc[0]


    def set(self, s, x):
        self.gamecon.loc[0, s] = x

    def gotFirstDown(play_result):
        if play_result >= self.get('ydstogo')
            return True
        
        return False

    def switchPossession(self):
        if self.get('home_possession' ) == 1:
            self.set('home_possession', 0)
        else:
            self.set('home_possession', 1)

        return 


    def lastSecondFG(self):
        if self.get('home_possession') == 1:
            off_tos = self.get('home_timeouts')
            margin = self.get('margin')
        else:
            off_tos = self.get('away_timeouts')
            margin = -self.get('margin')

        if off_tos = 0 and self.get('games_seconds_remaining') <= 20 and margin >= -3 and margin <= 0 and self.get('yardline_100') < 45:
            return True

        return False
        

        

    def formatDataFrameForDecisionModel(self):

        columns = ['game_seconds_remaining', 'down', 'ydstogo', 'yardline_100', 'pos_team_timeouts', 'def_team_timeouts', 'margin']

       
        if self.get('home_possession') == 1:
            pos_team_timeouts = self.get('home_timeouts')
            def_team_timeouts = self.get('away_timeouts')
            margin = self.get('margin')

        else:
            pos_team_timeouts = self.get('away_timeouts')
            def_team_timeouts = self.get('home_timeouts')
            margin = -self.get('margin')

        test = pd.DataFrame([[
            self.get('game_seconds_remaining'), # secs remaining
            self.get('down'), #down
            self.get('ydstogo'), # to_go
            self.get('yardline_100'), #distance to goal line
            pos_team_timeouts, # pos timeouts remaining
            def_team_timeouts, #defending team timeouts remaining
            margin, #margin

        ]], columns=columns)


     
        
        return test
    
    def formatDataFrameForResultModel(self):
        columns = ['game_seconds_remaining', 'down', 'ydstogo', 'yardline_100', 'redzone', 'goal_to_go', 'margin', 'pos_team_timeouts', 'def_team_timeouts']

        if self.get('yardline_100') <= 20:
            redzone = 1
        else:
            redzone = 0

        if self.get('yardline_100') <= 5:
            goal_to_go = 1
        else:
            goal_to_go = 0



        if self.get('home_possession') == 1:
            pos_team_timeouts = self.get('home_timeouts')
            def_team_timeouts = self.get('away_timeouts')
            margin = self.get('margin')

        else:
            pos_team_timeouts = self.get('away_timeouts')
            def_team_timeouts = self.get('home_timeouts')
            margin = -self.get('margin')

        test = pd.DataFrame([[
            self.get('game_seconds_remaining'), # secs remaining
            self.get('down'), #down
            self.get('ydstogo'), # to_go
            self.get('yardline_100'), #distance to goal line
            redzone, #1 if in redzone
            goal_to_go, #1 if inside 5 yardline
            margin, #margin
            pos_team_timeouts, # pos timeouts remaining
            def_team_timeouts, #defending team timeouts remaining
        
        ]], columns=columns)

        return test
        