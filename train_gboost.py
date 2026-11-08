import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_gboost(X_train, X_test, y_train, y_test):
    """Thread 2: Ortak veriyi kullanarak Gradient Boosting modelini egitir."""
    print("[T2 - Gradient Boosting] Egitim basladi.")

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    baslangic = time.perf_counter()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    egitim_suresi = round(time.perf_counter() - baslangic, 4)

    sonuc = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "training_time_sec": egitim_suresi
    }

    print(f"[T2 - Gradient Boosting] Tamamlandi: {egitim_suresi} saniye.")
    return sonuc


if __name__ == "__main__":
    # Burayi bagimsiz test icin yaptim ana bolumu calistirmaz
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X = digits.data.astype("float32") / 16.0
    y = digits.target
    veri = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    metrikler = train_gboost(*veri)
    print("\nThread 2 sonuclari:")
    for isim, deger in metrikler.items():
        print(f"{isim}: {deger}")
