import joblib, sqlite3, data_loader, utils.paths as paths, xgboost as xgb, pandas as pd, matplotlib.pyplot as plt, numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GroupShuffleSplit 
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report, confusion_matrix

def column(matrix, i):
    return [row[i] for row in matrix]
#


def createModelingDatabase():
    QUERY = ''' CREATE TABLE modeling.plays_for_model AS
    SELECT * FROM
    (SELECT g.game_id, 
    (15 - CAST(substr(p.time, 1, instr(p.time, ':') - 1) AS FLOAT) - CAST(substr(p.time, instr(p.time, ':') + 1) AS FLOAT) / 60) + 15*(p.quarter - 1) AS time, 
    p.home_possession as possession, CAST(p.to_go AS INTEGER) as to_go, CAST(p.down AS INTEGER) as down, CAST(p.field_position AS INTEGER) as field_position, p.homepoints - p.awaypoints as margin,  
    CASE
    WHEN g.home_team = g.favorite 
    THEN  g.spread
    WHEN g.away_team = g.favorite  
    THEN -g.spread
    ELSE 0
    END AS spread, 
    g.over_under as ou,
    CASE 
    WHEN g.final_Home > g.final_Away
    THEN 1
    WHEN g.final_Home <= final_Away
    THEN 0
    END AS result
    FROM plays as p
    JOIN games as g ON p.game_id = g.game_id)
     WHERE  to_go >= 1 AND down >= 1'''
    model_db = paths.get_project_root() / 'data' / 'modeling.db'

    with sqlite3.connect(data_loader.NFLSQLDatabasePath()) as conn:
        conn.execute(f"ATTACH DATABASE '{model_db}' AS modeling;")
        conn.execute("DROP TABLE IF EXISTS modeling.plays_for_model;")
        conn.execute(QUERY)
        conn.execute("DETACH DATABASE modeling;")
        conn.commit()

    print("✓  modeling.db created with table 'plays_for_model'")

    




def createXGBModel():

    


    modelingpath = paths.get_project_root() / "data" / "modeling.db"
    conn = sqlite3.connect(modelingpath)
    df = pd.read_sql("SELECT * FROM plays_for_model", conn)
    conn.close()
    X = df.drop(columns=["result", "game_id"])  # exclude ID and non-numeric columns
    y = df["result"]

    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df["game_id"]))

    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.01,
        use_label_encoder=False,
        eval_metric="logloss",
        monotone_constraints="(0, 0, 0, 0, 1, 1, -1, 0)"
    )

    model.fit(X_train, y_train)
    
    joblib.dump(model, paths.get_project_root() / "data" / "win_prob_model.pkl")




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




def createBinModelingDatabase():
    QUERY = ''' CREATE TABLE modeling.plays_for_bin_model AS
    SELECT * FROM
    (SELECT g.game_id, 
    (15 - CAST(substr(p.time, 1, instr(p.time, ':') - 1) AS FLOAT)  - (CAST(substr(p.time, instr(p.time, ':') + 1) AS FLOAT) / 60)) + 15*(p.quarter - 1) AS time, 
    p.home_possession as possession, CAST(p.to_go AS INTEGER) as to_go, CAST(p.down AS INTEGER) as down, CAST(p.field_position AS INTEGER) as field_position, p.homepoints - p.awaypoints as margin,  
    CASE
    WHEN g.home_team = g.favorite 
    THEN  g.spread
    WHEN g.away_team = g.favorite  
    THEN -g.spread
    ELSE 0
    END AS spread, 
    g.over_under as ou,
    g.final_Home - g.final_Away - (p.homepoints - p.awaypoints) AS result
    FROM plays as p
    JOIN games as g ON p.game_id = g.game_id)
     WHERE to_go >= 1 AND down >= 1'''
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








#X_val = pd.read_pickle(paths.get_project_root() /"data"/ "X_valuation_data.pkl")
#y_val = pd.read_pickle(paths.get_project_root() /"data"/ "y_valuation_data.pkl")
#model = joblib.load(paths.get_project_root() / "models" / "bin_model.pkl")


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

createXGBModel()





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






