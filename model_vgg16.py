import time
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def prepare_data(test_size=0.2, random_state=42):
    print("[DATA PREP] Loading Sklearn Digits (8x8) dataset...")
    digits = load_digits()

    X = digits.data.astype("float32") / 16.0
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[DATA READY] Total: {len(X)} | X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_vgg16(X_train, X_test, y_train, y_test, epochs=10, batch_size=32):
    start_time = time.time()
    print("[NODE 00 - VGG16] Training started on Sklearn Digits...")

    X_train_8x8 = X_train.reshape(-1, 8, 8)
    X_test_8x8 = X_test.reshape(-1, 8, 8)

    X_train_32 = np.repeat(np.repeat(X_train_8x8, 4, axis=1), 4, axis=2)
    X_test_32 = np.repeat(np.repeat(X_test_8x8, 4, axis=1), 4, axis=2)

    X_train_img = np.stack([X_train_32] * 3, axis=-1)
    X_test_img = np.stack([X_test_32] * 3, axis=-1)

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(32, 32, 3)
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train_img, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    probabilities = model.predict(X_test_img, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    elapsed_time = round(time.time() - start_time, 4)

    metrics = {
        "node": "00 (Thread-00 / Coordinator)",
        "model": "VGG16 (Sklearn Digits 8x8)",
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "training_time_sec": elapsed_time
    }

    print(f"[NODE 00 - VGG16] Training completed in {elapsed_time} sec")
    return metrics


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    vgg16_metrics = train_vgg16(X_train, X_test, y_train, y_test, epochs=10)

    print("\n" + "=" * 60)
    print("VGG16 PERFORMANCE METRICS (NODE 00)")
    print("=" * 60)
    for key, value in vgg16_metrics.items():
        print(f"• {key:18s}: {value}")
    print("=" * 60)