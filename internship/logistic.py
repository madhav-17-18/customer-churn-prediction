from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocess import load_and_preprocess

def run_logistic_model():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"[Logistic Regression] Accuracy: {acc:.4f}")
    return acc
