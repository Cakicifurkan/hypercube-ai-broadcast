
import time

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_svm(X_train, X_test, y_train, y_test):
    print("[NODE 11 - SVM] Training started...")
    start_time = time.perf_counter()

    model = SVC(kernel="rbf", C=10, gamma="scale")
    model.fit(X_train, y_train)

    elapsed_time = round(time.perf_counter() - start_time, 4)
    y_pred = model.predict(X_test)

    metrics = {
        "node": "11 (Thread-11 / SVM)",
        "model": "SVM",
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "training_time_sec": elapsed_time,
    }

    print(f"[NODE 11 - SVM] Training completed in {elapsed_time} sec")
    return metrics


if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X = digits.data.astype("float32") / 16.0
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = train_svm(X_train, X_test, y_train, y_test)
    print("\nSVM PERFORMANCE METRICS (NODE 11)")
    for key, value in results.items():
        print(f"{key}: {value}")
