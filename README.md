# 🧊 2D Hypercube (Q2) One-to-All Broadcasting & Paralel AI Analizi

Bu proje, **2 Boyutlu Hiperküp (Q2 - 4 Düğüm/Thread)** topolojisi üzerinde **One-to-All Broadcasting** *(Aynı Veri, Farklı İş)* mimarisini simüle etmek ve 4 farklı Yapay Zeka / Makine Öğrenmesi modelini eşzamanlı (paralel) olarak eğitip performanslarını karşılaştırmak amacıyla geliştirilmiştir.

---

## 🌐 Hiperküp (Q2) Topolojisi ve Düğüm Mimarisi

Hiperküp kuralları gereği yalnızca aralarında **1 bit fark** olan komşu düğümler doğrudan haberleşebilir (`XOR` adresleme):

```text
      [Düğüm 00: SÖZCÜ] ─────────── (1. Adım: 00 -> 01) ───────────► [Düğüm 01]
       (VGG16 + Veri)                                                (GBoost)
             │                                                           │
             │ (2. Adım: 00 -> 10)                         (2. Adım: 01 -> 11)
             ▼                                                           ▼
        [Düğüm 10]      ─────────── (Komşu: 1 Bit Fark) ───────────  [Düğüm 11]
     (Random Forest)                                                   (SVM)
```

### 🔄 İletişim ve Çalışma Adımları
1. **Veri Hazırlığı (`Düğüm 00 - Sözcü`):** `Sklearn Digits` (`1.797` örnek, `8x8 = 64` piksel/özellik) veri seti yüklenir, normalize edilir ve `%80 Eğitim / %20 Test` olarak ayrılır.
2. **One-to-All Broadcast (2 Adımda Veri Dağıtımı):**
   * **1. Adım (Sağ Bit Değişimi):** `00 -> 01`
   * **2. Adım (Sol Bit Değişimi - Eşzamanlı):** `00 -> 10` ve `01 -> 11`
3. **Eşzamanlı (Paralel) Model Eğitimi:** 4 Thread aynı veri seti üzerinde aynı anda kendi AI modelini eğitir.
4. **Ters Yönde Sonuç Toplama (Reverse Aggregation):**
   * **1. Adım (Eşzamanlı):** `11 -> 01` ve `10 -> 00`
   * **2. Adım:** `01 -> 00` *(Tüm metrikler Sözcü Düğümde toplanır ve raporlanır).*

---

## 🧩 Sistem Mimarisi ve Modül Dağılımı (5 Modüllü Yapı)

| Modül / Mimari Rol | Hiperküp Düğümü | Dosya Adı | Sorumluluk ve Teknik Görev Tanımı |
| :--- | :---: | :--- | :--- |
| **Modül 1:** Veri Hattı & Derin Öğrenme *(Sözcü)* | `00` *(Thread 1)* | `model_vgg16.py` | Ortak veri setinin hazırlanması (`veriyi_hazirla`) ve `8x8 -> 32x32x3` dönüşümü ile **VGG16** derin öğrenme modelinin eğitilmesi. |
| **Modül 2:** Boosting Sınıflandırıcı | `01` *(Thread 2)* | `model_gboost.py` | Dağıtılan ortak veri seti üzerinde **Gradient Boosting** modelinin eğitilmesi ve performans metriklerinin üretilmesi. |
| **Modül 3:** Bagging Sınıflandırıcı | `10` *(Thread 3)* | `model_rf.py` | Dağıtılan ortak veri seti üzerinde **Random Forest** modelinin eğitilmesi ve performans metriklerinin üretilmesi. |
| **Modül 4:** Çekirdek Tabanlı Sınıflandırıcı | `11` *(Thread 4)* | `model_svm.py` | Dağıtılan ortak veri seti üzerinde **SVM (Destek Vektör Makineleri)** modelinin eğitilmesi ve performans metriklerinin üretilmesi. |
| **Modül 5:** Hiperküp Ağ & Orkestrasyon Motoru | **Q2 Topoloji** | `main_hypercube.py` | Q2 Hiperküp Thread/Senkronizasyon yapısının kurulması, 2 adımlı dağıtım/toplama işlemlerinin yürütülmesi ve **Paralel vs. Seri Zaman ($T_{iletişim} + T_{işlem} < T_{seri}$)** analizinin raporlanması. |

---

## 📊 Ölçülen Performans Metrikleri
Her düğüm eğitimi tamamladığında Sözcü Düğüme (`00`) aşağıdaki standart sözlük (dictionary) yapısını döndürür:
* `accuracy` (Doğruluk)
* `precision` (Kesinlik)
* `recall` (Duyarlılık)
* `f1_score` (F1 Skoru)
* `egitim_suresi_sn` (Eğitim Süresi - Saniye)