import joblib, sqlite3, data_loader, utils.paths as paths, xgboost as xgb, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GroupShuffleSplit 
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def column(matrix, i):
    return [row[i] for row in matrix]


def createPlayResultDatabase():
    QUERY = '''CREATE TABLE modeling.play_result AS
    SELECT game_id, 
    game_seconds_remaining, 
    down, 
    ydstogo,
    yardline_100,
    CASE 
    WHEN yardline_100 <= 20
    THEN 1
    ELSE 0
    END AS redzone,
    CASE 
    WHEN yardline_100 <= 5
    THEN 1
    ELSE 0
    END AS goal_to_go,
    CASE 
    WHEN home_team = posteam
    THEN total_home_score - total_away_score
    ELSE total_away_score - total_home_score 
    END AS margin,
    CASE 
    WHEN home_team = posteam
    THEN home_timeouts_remaining
    ELSE away_timeouts_remaining
    END AS pos_team_timeouts,
    CASE 
    WHEN home_team = posteam
    THEN away_timeouts_remaining
    ELSE home_timeouts_remaining
    END AS def_team_timeouts,
    CASE 
    WHEN play_type = 'pass'
    THEN 1
    ELSE 0
    END AS play_type,
    yards_gained
    FROM plays
    WHERE play_type = 'pass' OR play_type = 'run'
    '''

    model_db = paths.get_project_root() / 'data' / 'modeling.db'


    with sqlite3.connect(data_loader.NFLSQLDatabasePath()) as conn:
        conn.execute(f"ATTACH DATABASE '{model_db}' AS modeling;")
        conn.execute("DROP TABLE IF EXISTS modeling.play_result;")
        conn.execute(QUERY)
        conn.execute("DETACH DATABASE modeling;")
        conn.commit()

    print("✓  modeling.db created with table 'play_result'")
    return 



    




#builds database for model that predicts whether a team will run, pass, 
def createPlayDecisionDatabase():
    QUERY = '''CREATE TABLE modeling.play_decision AS
    SELECT game_id, 
    game_seconds_remaining, 
    down, 
    ydstogo,
    yardline_100,
    CASE 
    WHEN home_team = posteam
    THEN home_timeouts_remaining
    ELSE away_timeouts_remaining
    END AS pos_team_timeouts,
    CASE 
    WHEN home_team = posteam
    THEN away_timeouts_remaining
    ELSE home_timeouts_remaining
    END AS def_team_timeouts,
    CASE 
    WHEN home_team = posteam
    THEN total_home_score - total_away_score
    ELSE total_away_score - total_home_score 
    END AS margin,
    play_type
    FROM plays

    '''

    model_db = paths.get_project_root() / 'data' / 'modeling.db'


    with sqlite3.connect(data_loader.NFLSQLDatabasePath()) as conn:
        conn.execute(f"ATTACH DATABASE '{model_db}' AS modeling;")
        conn.execute("DROP TABLE IF EXISTS modeling.play_decision;")
        conn.execute(QUERY)
        conn.execute("DETACH DATABASE modeling;")
        conn.commit()

    print("✓  modeling.db created with table 'play_decision'")
    return 


def label_timeouts(df):
    
    
    """
    Label plays with whether a timeout was called immediately after,
    and normalize timeout_team to 'offense' or 'defense'.
    """
    df = df.sort_values(["game_id", "play_id"]).reset_index(drop=True)

    # Whether the NEXT play is a timeout
    df['next_is_timeout'] = df.groupby('game_id')['timeout'].shift(-1).fillna(0).astype(int)

    # Grab which team called it
    df['next_timeout_team'] = df.groupby('game_id')['timeout_team'].shift(-1)

    # Attach labels
    df['timeout_called_after'] = df['next_is_timeout']
     # Map abbreviation -> offense/defense
    def map_team(row):
        if row['timeout_called_after'] == 1:
            if row['next_timeout_team'] == row['posteam']:
                return 'offense'
            else:
                return 'defense'
        return None


   

    df['timeout_team'] = df.apply(map_team, axis=1)

    conditions = ((df['incomplete_pass'] == 1) | 
    (df['out_of_bounds'] == 1) | (df['penalty'] == 1) | 
    (df['timeout'] ==1))

    df['clock_stopped'] = conditions.astype(int)

    return df



def createTimeoutModelDatabase():
    
    

    query = """
    SELECT game_id, play_id, qtr, game_seconds_remaining, half_seconds_remaining, down, ydstogo
       play_type, yards_gained, score_differential as margin, incomplete_pass, out_of_bounds, penalty,  
       posteam, defteam, timeout_team,
       posteam_timeouts_remaining, defteam_timeouts_remaining, timeout
    FROM plays
    """
   

    dbpath = paths.get_project_root() / "data" / "nfl_data.db"
    with sqlite3.connect(dbpath) as conn:
         df = pd.read_sql(query, conn)
    # Label timeouts
    print(df['play_type'].value_counts())
    df_labeled = label_timeouts(df)

    # Path to modeling.db
    modelingpath = paths.get_project_root() / "data" / "modeling.db"

    with sqlite3.connect(modelingpath) as conn:
        df_labeled.to_sql("plays_for_timeout_model", conn, if_exists="replace", index=False)

    print(f"✓ Saved labeled timeout data to {modelingpath} in table 'timeouts_labeled'")

    return




    





def createModelingDatabase():
    QUERY = '''CREATE TABLE modeling.plays_for_model AS
    SELECT game_id,
    game_seconds_remaining, 
    down, 
    ydstogo,
    yardline_100, 
    CASE 
    WHEN posteam = home_team
    THEN 1
    ELSE 
    0 
    END AS home_possession,
    total_home_score - total_away_score AS margin,
    total_line, 
    spread_line, 
    CASE
    WHEN result > 0
    THEN 1
    ELSE
    0 
    END AS result
    FROM plays
    WHERE down >=1 and ydstogo >=1
    '''
    model_db = paths.get_project_root() / 'data' / 'modeling.db'


    with sqlite3.connect(data_loader.NFLSQLDatabasePath()) as conn:
        conn.execute(f"ATTACH DATABASE '{model_db}' AS modeling;")
        conn.execute("DROP TABLE IF EXISTS modeling.plays_for_model;")
        conn.execute(QUERY)
        conn.execute("DETACH DATABASE modeling;")
        conn.commit()

    print("✓  modeling.db created with table 'plays_for_model'")
    return 



def createXGBModel():
    modelingpath = paths.get_project_root() / "data" / "modeling.db"
    conn = sqlite3.connect(modelingpath)
    df = pd.read_sql("SELECT * FROM plays_for_model", conn)
    conn.close()
    X = df.drop(columns=["result", "game_id"])  # exclude result and game_id
    y = df["result"]

    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df["game_id"]))
    print(X.columns.tolist())

    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        use_label_encoder=False,
        eval_metric="logloss",
        monotone_constraints= '(0, 0, 0, 0, 0, 1, 0, 1)'
    )

    model.fit(X_train, y_train)
    
    joblib.dump(model, paths.get_project_root() / "models" / "win_prob_model.pkl")


    plot_calibration_curve(model, X_val, y_val)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("ROC AUC:", roc_auc_score(y_val, y_proba))

    '''pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    #new_data = X_val.iloc[:5]  # just an example — could be live play inputs
    #probs = model.predict_proba(new_data)[:, 1]
    

    selected_rows = df[df['game_id'] == 3234].drop(columns = ["result", "game_id"])

    probs = model.predict_proba(selected_rows)
    probsdf = pd.DataFrame({'winprob': column(probs, 0)})
   
    #print(selected_rows)
    gamedf = pd.concat([selected_rows.reset_index(), probsdf], axis=1).drop(columns = ["index", "ou"])
    print(gamedf)
    pl = plt.plot(gamedf['time'], gamedf['winprob'])
    plt.show()'''
    
    



    return 





def plot_calibration_curve(model, X_test, y_test, label_encoder=None, n_bins=10):
    """
    Plot calibration curve for a binary classifier like PlayDecisionModel.
    """
    # Predict probabilities
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of class 1
    
    # If labels were encoded, make sure y_test is numeric
    if label_encoder is not None and not np.issubdtype(y_test.dtype, np.number):
        y_test = label_encoder.transform(y_test)

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=n_bins)

    # Plot
    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker='o', label="XGB Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.show()



createXGBModel()

def createBinModelingDatabase():
    QUERY = '''CREATE TABLE modeling.plays_for_model AS
    SELECT game_id,
    game_seconds_remaining, 
    down, 
    ydstogo,
    yardline_100, 
    CASE 
    WHEN posteam = home_team
    THEN 1
    ELSE 
    0 
    END AS home_possession,
    total_home_score - total_away_score AS margin,
    total_line, 
    spread_line, 
    result,
    FROM plays
    WHERE down >=1 and ydstogo >=1'''
    model_db = paths.get_project_root() / 'data' / 'modeling.db'

    with sqlite3.connect(data_loader.NFLSQLDatabasePath()) as conn:
        conn.execute(f"ATTACH DATABASE '{model_db}' AS modeling;")
        conn.execute("DROP TABLE IF EXISTS modeling.plays_for_bin_model;")
        conn.execute(QUERY)
        conn.execute("DETACH DATABASE modeling;")
        conn.commit()

        print("✓  modeling.db created with table 'plays_for_bin_model'")
    return 


def createBins():
    bin_edges = [-np.inf,-22, -11,  0, 10, 21, np.inf]
    n_bins = len(bin_edges)-1
    return bin_edges, n_bins

def createBinXGBModel():

    modelingpath = paths.get_project_root() / "data" / "modeling.db"
    conn = sqlite3.connect(modelingpath)
    df = pd.read_sql("SELECT * FROM plays_for_bin_model", conn)
    conn.close()
    bin_edges, n_bins = createBins()
    X = df.drop(columns=["result", "game_id"])  # exclude ID and non-numeric columns
    y = pd.cut(df['result'], bins=bin_edges, labels=False, right=False)

    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df["game_id"]))

    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

    

    
    
    model = XGBClassifier(
    objective='multi:softprob',
    num_class=n_bins,
    learning_rate=0.1,
    n_estimators=500,
    max_depth=6,
    min_child_weight=1,
    subsample=1.0,
    colsample_bytree=1.0,
    gamma=0
    )

    model.fit(X_train, y_train) 
    joblib.dump(model,  paths.get_project_root() / "models" / "bin_model.pkl")

    X_val.to_pickle(paths.get_project_root() / "data" / "X_valuation_data.pkl")
    y_val.to_pickle(paths.get_project_root() / "data" / "y_valuation_data.pkl")

    



    return 



























        # shape: (n_samples, n_bins)
'''y_pred = np.argmax(y_proba, axis=1)          # predicted bin index
acc = accuracy_score(y_val, y_pred)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_val, y_pred))
#print(confusion_matrix(y_val, y_pred))'''
'''for i in range(2014, 2105):
    x = X_val.iloc[[i]]
    print(x)
    y_proba = model.predict_proba(x)
    print(x)
    print(y_proba)'''



'''columns = ['time', 'possession', 'to_go', 'down','field_position', 'margin', 'spread', 'ou']
sample = pd.DataFrame([[
    10,     # time
    1,     # possession
    9,     # to go
    4,     # down
    85,    # field_position 1-50 home, 51-99 away
    -10,     #margin home -away
    3.5,  #pre game spread
    42.5   # pregame ou

]], columns=columns)
sample2 = pd.DataFrame([[
    10,     # time
    0,     # possession
    9,     # to go
    4,     # down
    15,    # field_position 1-50 home, 51-99 away
    10,     #margin home -away
    -3.5,  #pre game spread
    42.5   # pregame ou

]], columns=columns)

y_proba, y2 = model.predict_proba(sample), model.predict_proba(sample2)

bin_edges, n_bins = createBins()
bin_labels = []
for i in range(len(bin_edges) - 1):
    left = bin_edges[i]
    right = bin_edges[i + 1]
    if np.isinf(-left):
        label = f"< {right}"
    elif np.isinf(right):
        label = f"> {left+1}"
    else:
        label = f"{int(left) +1} to {int(right) }"
    bin_labels.append(label)


for i in range(len(y_proba[0])):
    print(f'{bin_labels[i]}: {int(y_proba[0][i]*100)} %')

for i in range(len(y2[0])):
    print(f'{bin_labels[i]}: {int(y2[0][i]*100)} %')'''






'''margin = game.at[1199, 'margin']

bin_edges, n_bins = createBins()
bin_edges = [margin + bin_edges]
print(bin_edges)

bin_labels = []
for i in range(len(bin_edges) - 1):
    left = bin_edges[i]
    right = bin_edges[i + 1]
    print(right)
    if np.isinf(-left):
        label = f"< {right}"
    elif np.isinf(right):
        label = f"> {left+1}"
    else:
        label = f"{int(left) +1} to {int(right) }"
    bin_labels.append(label)
print(bin_labels)
print(y_proba[0])
print(type(y_proba[0]))
print(len(y_proba[0]))     # should be num_bins
print(len(bin_labels))

# Step 2: plot
plt.figure(figsize=(10, 4))
plt.bar(range(len(y_proba[0])), y_proba[0], tick_label=bin_labels)
plt.xticks(rotation=45)
plt.ylabel("Predicted Probability")
plt.title("Predicted Distribution Over Margin Bins")
plt.tight_layout()
plt.show()'''





