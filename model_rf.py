import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_rf(X_train, X_test, y_train, y_test, n_estimators=100, random_state=42):
    start_time = time.time()
    print("[NODE 10 - RANDOM FOREST] Training started on Sklearn Digits...")

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    if X_train.ndim > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
    if X_test.ndim > 2:
        X_test = X_test.reshape(X_test.shape[0], -1)

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    elapsed_time = round(time.time() - start_time, 4)

    metrics = {
        "node": "10 (Thread-10 / Bagging Classifier)",
        "model": "Random Forest (Sklearn Digits 8x8)",
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "training_time_sec": elapsed_time,
        "egitim_suresi_sn": elapsed_time
    }

    print(f"[NODE 10 - RANDOM FOREST] Training completed in {elapsed_time} sec")
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

    rf_metrics = train_rf(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 60)
    print("RANDOM FOREST PERFORMANCE METRICS (NODE 10)")
    print("=" * 60)
    for key, value in rf_metrics.items():
        print(f"• {key:18s}: {value}")
    print("=" * 60)
