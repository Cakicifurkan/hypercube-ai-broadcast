import sys
sys.path.insert(0, r"C:\ai_env\Lib\site-packages")  

import time
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def veriyi_hazirla(test_orani=0.2, random_state=42):
    print("[1. KİŞİ - VERİ HAZIRLIĞI] Sklearn Digits (8x8) veri seti yükleniyor...")
    digits = load_digits()

    X = digits.data.astype("float32") / 16.0
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_orani, random_state=random_state, stratify=y
    )

    print(f"[VERİ HAZIR] Toplam: {len(X)} | X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test



def egit_vgg16(X_train, X_test, y_train, y_test, epochs=10, batch_size=32):
    baslangic_zamani = time.time()
    print("[THREAD 1 (00) - VGG16] Sklearn Digits üzerinde eğitim başladı...")

    
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

    olasiliklar = model.predict(X_test_img, verbose=0)
    y_pred = np.argmax(olasiliklar, axis=1)

    bitis_zamani = time.time()
    gecen_sure = round(bitis_zamani - baslangic_zamani, 4)

    sonuc = {
        "dugum": "00 (Thread 1 - Sözcü)",
        "model_adi": "VGG16 (Sklearn Digits 8x8)",
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "egitim_suresi_sn": gecen_sure
    }

    print(f"[THREAD 1 (00) - VGG16] Eğitim tamamlandı! Süre: {gecen_sure} sn")
    return sonuc


if __name__ == "__main__":
    print("=" * 65)
    print("1. KİŞİ MODÜL TESTİ: SKLEARN DIGITS (1.797 ÖRNEK) VE VGG16")
    print("=" * 65)

    X_train, X_test, y_train, y_test = veriyi_hazirla()
    vgg16_sonuc = egit_vgg16(X_train, X_test, y_train, y_test, epochs=10)

    print("\n" + "=" * 65)
    print("📊 VGG16 PERFORMANS SONUÇLARI (DÜĞÜM 00)")
    print("=" * 65)
    for anahtar, deger in vgg16_sonuc.items():
        print(f"• {anahtar:18s}: {deger}")
    print("=" * 65)