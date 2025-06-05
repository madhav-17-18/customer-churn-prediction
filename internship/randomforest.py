from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocess import load_and_preprocess

def run_random_forest():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"[Random Forest] Accuracy: {acc:.4f}")
    return acc
