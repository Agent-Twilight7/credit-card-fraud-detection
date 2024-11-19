import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=['Class'])
    y = data['Class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test
