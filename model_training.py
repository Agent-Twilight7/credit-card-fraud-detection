from sklearn.ensemble import IsolationForest
import pickle

def train_isolation_forest(X_train):
    model = IsolationForest(n_estimators=100, contamination=0.002, random_state=42)
    model.fit(X_train)
    with open('isolation_forest_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    return model
