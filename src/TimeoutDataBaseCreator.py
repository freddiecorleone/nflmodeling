import sqlite3
import pandas as pd
import utils.paths as paths

def create_timeout_model_db():
    dbpath = paths.get_project_root() / "data" / "nfl_data.db"
    modelingpath = paths.get_project_root() / "data" / "modeling.db"

    query = """
    SELECT
        game_id, play_id, qtr, half_seconds_remaining, fourth_down_failed,
        posteam, defteam, score_differential, fumble_lost,
        posteam_timeouts_remaining, defteam_timeouts_remaining, desc,
        play_type, down, ydstogo, yardline_100, yards_gained,
        complete_pass, out_of_bounds, qb_spike, qb_kneel,
        timeout, timeout_team, end_clock_time, interception
    FROM plays
    WHERE qtr IN (2, 4)
    AND half_seconds_remaining <= 180
    """

    with sqlite3.connect(dbpath) as conn:
        df = pd.read_sql(query, conn)

    # Sort properly
    df = df.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    

    df['clock_stopped'] = (
        (df['timeout'] == 1) |
        ((df['play_type'] == 'pass') & (df['complete_pass'] == 0)) |
        (df['out_of_bounds'] == 1) |
        (df['qb_spike'] == 1) |
        (df['play_type'].isin(['field_goal','punt','kickoff', 'no_play'])) |
        ((df['half_seconds_remaining'] <= 140) & (df['half_seconds_remaining'] > 120)) |
        ((df['fourth_down_failed']==1)) |
        ((df['interception']==1))  |
        ((df['fumble_lost']==1)) 
        ).astype(int)

    

    df['next_play_type'] = df.groupby('game_id')['play_type'].shift(-1)
    df['temp_next_down'] = df.groupby('game_id')['down'].shift(-1)
    df['temp_next_ydstogo'] = df.groupby('game_id')['ydstogo'].shift(-1)
    df['temp_next_yardline'] = df.groupby('game_id')['yardline_100'].shift(-1)
    df['temp_next_posteam'] = df.groupby('game_id')['posteam'].shift(-1)


       # --- 7. Timeout label ---
    df['timeout_called_after'] = df.groupby('game_id')['timeout'].shift(-1).fillna(0).astype(int)


    df['next_down'] = df.apply(lambda row: row['down'] if (row['next_play_type'] == 'no_play') 
    else row['temp_next_down'], axis = 1)
    df['next_ydstogo'] = df.apply(lambda row: row['ydstogo'] if row['next_play_type'] == 'no_play' else row['temp_next_ydstogo'], axis = 1)
    df['next_yardline'] = df.apply(lambda row: row['yardline_100'] if row['next_play_type'] == 'no_play' else row['temp_next_yardline'], axis = 1)
    df['next_posteam'] = df.apply(lambda row: row['posteam'] if row['next_play_type'] == 'no_play' else row['temp_next_posteam'], axis = 1)

    
    




    df['margin'] = df['score_differential']
    df['off_timeouts'] = df['posteam_timeouts_remaining']
    df['def_timeouts'] = df['defteam_timeouts_remaining']

    df['next_timeout_team'] = df.groupby('game_id')['timeout_team'].shift(-1)


 

    df['timeout_team_role'] = df.apply(
        lambda row: 'offense' if (row['timeout_called_after'] == 1 and row['next_timeout_team'] == row['next_posteam'])
        else ('defense' if row['timeout_called_after'] == 1 else 'No_timeout'),
        axis=1
    )


 

   





     # --- 8. Extra engineered predictors ---
    # One-score game
    df['one_score_game'] = (df['margin'].abs() <= 8).astype(int)

    # Field position bins
    def field_zone(yardline):
        if pd.isna(yardline):
            return None
        if yardline <= 35:
            return 'field_goal_range'
        elif yardline <= 50:
            return 'midfield'
        else:
            return 'own_side'
    df['field_zone'] = df['next_yardline'].apply(field_zone)


    # Is trailing team on offense?
    df['trailing_offense'] = ((df['margin'] < 0) & (df['timeout_team_role'] == 'offense')).astype(int)

    # --- 9. Remove plays where next play is a scoring event ---
    scoring_types = ['extra_point','field_goal','kickoff','punt','safety', 'no_play']
    df = df[~df['play_type'].isin(scoring_types) | df['clock_stopped'] == 1]

    # --- 10. Final model features ---
    df_model = df[['qtr','half_seconds_remaining', 'next_yardline','next_down','next_ydstogo','margin', 
                   'off_timeouts','def_timeouts',
                   'field_zone','one_score_game',
                   'trailing_offense','timeout_team_role']].dropna()
    # Save to modeling.db
    with sqlite3.connect(modelingpath) as conn:
        df_model.to_sql("plays_for_timeout_model", conn, if_exists="replace", index=False)

    print(f"✓ Timeout model dataset created at {modelingpath}, table 'plays_for_timeout_model'")






   



    return df_model


create_timeout_model_db()

modelingpath = paths.get_project_root() / "data" / "modeling.db"



'''for i in range(100):
    print(model['desc'].iloc[i])
    print(f"Down after play: {model['next_down'].iloc[i]}")
    print(model['margin'].iloc[i])
    print(f"Time out called {model['timeout_called_after'].iloc[i] == 1}")'''





