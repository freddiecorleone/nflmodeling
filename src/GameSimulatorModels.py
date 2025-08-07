import joblib, sqlite3, utils.paths as paths, xgboost as xgb, pandas as pd, numpy as np, types as t
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GroupShuffleSplit 
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
    
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

class FourthDownDecisionModel():
    def __init__(self, model = None, label_encoder = None):
        self.model = model
        self.label_encoder = label_encoder
        self.defaultpath = paths.get_project_root() / "models" / "4th_down_decision_model.pkl"
        return 
    
    def fit(self, X= None, y = None):

        if X == None:
            modelingpath = paths.get_project_root() / "data" / "modeling.db"
            with sqlite3.connect(modelingpath) as conn:
                df = pd.read_sql("SELECT * FROM play_decision WHERE time < 1800 AND down = 4 AND play_type IN ('punt', 'field_goal', 'run', 'pass')", conn)
                
            X = df.drop(columns=["play_type", "game_id"])  # exclude result and game_id
            pre_y = df["play_type"]
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(pre_y)


            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )


        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            num_class=4,
            max_depth=7,
            n_estimators=500,
            learning_rate=0.05
        )

        self.model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)

        # Evaluate
        print("Accuracy:", accuracy_score(y_test, self.model.predict(X_test)))
        print("Log loss:", log_loss(y_test, y_pred_proba))



        return 
    
    def save(self, path = None):
        if path == None:
            path = self.defaultpath

        joblib.dump({
            "model": self.model,
            "label_encoder": self.label_encoder,
        }, path)
        return 

    @classmethod
    def load(cls, path = None):
        if path == None:
            path = paths.get_project_root() / "models" / "4th_down_decision_model.pkl"

        data = joblib.load(path)
        return cls(model=data["model"], label_encoder = data['label_encoder'])
    

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    

    def make_decision(self, X):
        choice = self.label_encoder.inverse_transform(self.model.predict(X))
        return choice
    



class FieldGoalModel():
    def __init(self, model = None, label_encoder = None):
        return

    def predict_proba(self, yardline): # X is just distance for now or dataframe, maybe later
        
        probability = self.field_goal_success_prob(yardline)

        return [1- probability, probability]

    def field_goal_success_prob(self, yardline_100):
        """
        Estimate FG make probability given yardline_100 (distance from opponent's EZ).
        """
        fg_distance = yardline_100 + 17  # LOS + 17 yards for snap/goalposts
        
        # Logistic model parameters tuned to NFL averages
        alpha, beta = 7.8, -0.13 # alpha=intercept, beta=slope
        
        z = alpha + beta * fg_distance
        prob = 1 / (1 + np.exp(-z))
        return round(prob, 3)

    

    def simulate(self, X):
        probs = self.predict_proba(X)

        fgresult = np.random.choice(['miss', 'make'], size =1, p = probs) 

        return fgresult




            

model = FieldGoalModel()

print(model.simulate(30))

        


        


class PlayDecisionModel():
    def __init__(self, model=None, label_encoder =None):
        self.model = model
        self.label_encoder = label_encoder
        self.defaultpath = paths.get_project_root() / "models" / "play_decision_model.pkl"
        return


    def fit(self, X= None, y = None):

        if X == None:
            QUERY = '''SELECT *
            FROM play_decision WHERE play_type = 'pass' OR play_type = 'run'
            '''
            modelingpath = paths.get_project_root() / "data" / "modeling.db"

            with sqlite3.connect(modelingpath) as conn:
                df = pd.read_sql(QUERY, conn)
            
            X = df.drop(columns=["play_type", "game_id"])  # exclude result and game_id, replace yardline with redzone indicator
            pre_y = df["play_type"]


            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(pre_y)
            print(self.label_encoder.classes_)

        
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
        )

        # Initialize the XGBoost classifier
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            max_depth=5,
            use_label_encoder = False,
            n_estimators=400,
            learning_rate=0.1
        )

        # Train the model
        self.model.fit(X_train, y_train)

        # Predict probabilities and class labels
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)

        # Evaluate performance
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Accuracy: {accuracy:.3f}")
        print(f"ROC AUC: {roc_auc:.3f}")

        
        return 
    

    def predict_proba(self, X):
        """Predict probabilities for yardage bins."""
        return self.model.predict_proba(X)
    
    def save(self, path = None):
       
        """Save the model and metadata."""
        if path == None:
            path = self.defaultpath
        joblib.dump({
            "model": self.model,
            "label_encoder": self.label_encoder,
        }, path)

    @classmethod
    def load(cls, path = None):
        if path == None:
            path = paths.get_project_root() / "models" / "play_decision_model.pkl"
        """Load a saved RunModel."""
        data = joblib.load(path)
        return cls(model=data["model"], label_encoder = data['label_encoder'])
    

    def simulate_decision(self, X):
        choice_probs = self.model.predict_proba(X) #get probs of run or pass
        print(choice_probs)
        play_type = np.random.choice(self.label_encoder.classes_, size =1, p = choice_probs[0]) 

        return play_type




            
class TimeoutDecisionModel():
    def __init__(self, model = None, label_encoder = None):
        self.model = model
        self.label_encoder = None
        self.defaultpath = paths.get_project_root() /  "models" / "timeout_model.pkl"


    def fit(self):
        modelingpath = paths.get_project_root() /  "data" / "modeling.db"

        conn = sqlite3.connect(modelingpath)
        df = pd.read_sql("SELECT * from plays_for_timeout_model", conn)

        conn.close()



        X = df.drop(columns = ['timeout_team_role'])
        #y = df['timeout_team_role']

        self.label_encoder = LabelEncoder()
        X['field_zone'] = self.label_encoder.fit_transform(X['field_zone'])
        print(self.label_encoder.classes_)
        y= self.label_encoder.fit_transform(df['timeout_team_role'])
        print(1234)
        print(self.label_encoder.classes_)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
        )

        # Initialize the XGBoost classifier
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='logloss',
            num_class = 3,
            max_depth=6,
            use_label_encoder = False,
            n_estimators=650,
            learning_rate=0.05,
            enable_categorical = True
        )

        # Train the model
        self.model.fit(X_train, y_train)

        # Predict probabilities and class labels
        self.model.fit(X_train, y_train)

        # Predict probabilities for each bin
        y_pred_proba = self.model.predict_proba(X_test)

        # Evaluate
        print("Accuracy:", accuracy_score(y_test, self.model.predict(X_test)))
        print("Log loss:", log_loss(y_test, y_pred_proba))   

        
        return 

    def save(self, path = None):
    
        """Save the model and metadata."""
        if path == None:
            path = self.defaultpath
        joblib.dump({
            "model": self.model,
        }, path)

    @classmethod
    def load(cls, path = None):
        if path == None:
            path = paths.get_project_root() / "models" / "timeout_model.pkl"

        data = joblib.load(path)
        return cls(model=data["model"])

    
    def predict_proba(self, X):
        """Predict probabilities for yardage bins."""
        return self.model.predict_proba(X)





class PassPlayResultModel():
    def __init__(self, model = None, bins = None, labels = None):
        self.bins = bins or [-np.inf, -10, -1, 0, 3, 7, 15, 30, np.inf]
        self.labels = labels or [0, 1, 2, 3, 4, 5, 6, 7] 
        self.model = model
        self.defaultpath = paths.get_project_root() / "models" / "play_result_model.pkl"
        return 
    

    def fit(self, X = None, y = None):

        if X == None or y == None:
            modelingpath = paths.get_project_root() / "data" / "modeling.db"
            with sqlite3.connect(modelingpath) as conn:
                df = pd.read_sql("SELECT * FROM play_result WHERE play_type = 1", conn)
            X = df.drop(columns=["yards_gained","play_type", "game_id"])  # exclude result and game_id
            print(X.columns.to_list())
            y = df["yards_gained"]
            y_binned = pd.cut(y, bins=self.bins, labels=self.labels).astype(int)


        

        X_train, X_test, y_train, y_test = train_test_split(
        X, y_binned, test_size=0.2, random_state=42
        )

        # Number of classes = number of bins
        num_classes = len(np.unique(y_binned))

        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            num_class=num_classes,
            max_depth=5,
            n_estimators=500,
            learning_rate=0.05
        )

        self.model.fit(X_train, y_train)

        # Predict probabilities for each bin
        y_pred_proba = self.model.predict_proba(X_test)

        # Evaluate
        print("Accuracy:", accuracy_score(y_test, self.model.predict(X_test)))
        print("Log loss:", log_loss(y_test, y_pred_proba))   
            

        return
    

    def save(self, path = None):
        if path == None:
            path = self.defaultpath

        joblib.dump({
            "model": self.model,
            "labels": self.labels,"bins": self.bins
        }, path)
        return 
    

    def simulate_play(self, X):

        if isinstance(X, np.ndarray) and X.ndim == 1:
            X = X.reshape(1, -1)
        
        # 1. Predict the bin probabilities
        probs = self.model.predict_proba(X)[0]
        
        # 2. Choose a bin at random, weighted by predicted probabilities
        chosen_bin = np.random.choice(self.labels, p=probs)
        
        # 3. Find the yardage range corresponding to that bin
        lower, upper = self.bins[chosen_bin] + 1, self.bins[chosen_bin + 1]
        
        # 4. Sample a yardage value within that range
        if np.isinf(upper):
            yards_gained = int(np.random.normal(loc=lower + 5, scale=3))
        elif np.isinf(lower):
            yards_gained = int(np.random.normal(loc = upper, scale = 3))
        else:
            yards_gained = np.random.randint(lower, upper + 1)
        
        return yards_gained, chosen_bin


    @classmethod
    def load(cls, path = None):
        if path == None:
            path = paths.get_project_root() / "models" / "play_result_model.pkl"

        data = joblib.load(path)
        return cls(model=data["model"], labels = data['labels'], bins = data['bins'])

    

class RunPlayResultModel():
    def __init__(self, model = None, bins = None, labels = None):
        self.bins = bins or [-np.inf, 0, 2, 4, 8, 15, np.inf]
        self.labels = labels or [0, 1, 2, 3, 4, 5]  
        self.model = model
        self.defaultpath = paths.get_project_root() / "models" / "run_play_result_model.pkl"
        return 
    

    def fit(self, X = None, y = None):

        if X == None or y == None:
            modelingpath = paths.get_project_root() / "data" / "modeling.db"
            with sqlite3.connect(modelingpath) as conn:
                df = pd.read_sql("SELECT * FROM play_result WHERE play_type = 0", conn)
            X = df.drop(columns=["yards_gained","play_type", "game_id"])  # exclude result and game_id
            print(X.columns.to_list())
            y = df["yards_gained"]
            y_binned = pd.cut(y, bins=self.bins, labels=self.labels).astype(int)


        

        X_train, X_test, y_train, y_test = train_test_split(
        X, y_binned, test_size=0.2, random_state=42
        )

        # Number of classes = number of bins
        num_classes = len(np.unique(y_binned))

        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            num_class=num_classes,
            max_depth=5,
            n_estimators=500,
            learning_rate=0.05
        )

        self.model.fit(X_train, y_train)

        # Predict probabilities for each bin
        y_pred_proba = self.model.predict_proba(X_test)

        # Evaluate
        print("Accuracy:", accuracy_score(y_test, self.model.predict(X_test)))
        print("Log loss:", log_loss(y_test, y_pred_proba))   
            

        return
    

    def save(self, path = None):
        if path == None:
            path = self.defaultpath

        joblib.dump({
            "model": self.model,
            "labels": self.labels,"bins": self.bins
        }, path)
        return 
    

    def simulate_play(self, X):

        if isinstance(X, np.ndarray) and X.ndim == 1:
            X = X.reshape(1, -1)
        
        # 1. Predict the bin probabilities
        probs = self.model.predict_proba(X)[0]
        
        # 2. Choose a bin at random, weighted by predicted probabilities
        chosen_bin = np.random.choice(self.labels, p=probs)
        
        # 3. Find the yardage range corresponding to that bin
        lower, upper = self.bins[chosen_bin] + 1, self.bins[chosen_bin + 1]
        
        # 4. Sample a yardage value within that range
        if np.isinf(upper):
            yards_gained = int(np.random.normal(loc=lower + 5, scale=3))
        elif np.isinf(lower):
            yards_gained = int(np.random.normal(loc = upper, scale = 3))
        else:
            yards_gained = np.random.randint(lower, upper + 1)
        
        return yards_gained, chosen_bin


    @classmethod
    def load(cls, path = None):
        if path == None:
            path = paths.get_project_root() / "models" / "run_play_result_model.pkl"

        data = joblib.load(path)
        return cls(model=data["model"], labels = data['labels'], bins = data['bins'])