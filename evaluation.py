from sklearn.metrics import classification_report, accuracy_score
import pickle

def evaluate_model(X_test, y_test, model_path='isolation_forest_model.pkl'):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    y_pred = model.predict(X_test)
    y_pred_binary = [1 if pred == -1 else 0 for pred in y_pred]
    print("Accuracy:", accuracy_score(y_test, y_pred_binary))
    print("Classification Report:\n", classification_report(y_test, y_pred_binary))
