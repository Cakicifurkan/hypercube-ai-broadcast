
import importlib
import queue
import threading
import time
from datetime import datetime


BOYUT = 2
THREAD_SAYISI = 2 ** BOYUT

VERI_MODULU = "model_vgg16"
VERI_FONKSIYONU = "prepare_data"

MODELLER = {
    0: ("model_vgg16", "train_vgg16"),
    1: ("model_gboost", "train_gboost"),
    2: ("model_rf", "train_rf"),
    3: ("model_svm", "train_svm")
}


def log_yaz(mesaj):
    zaman = datetime.now().strftime("%H:%M:%S")
    print(f"[{zaman}] {mesaj}", flush=True)


def thread_adi(dugum):
    return f"T{dugum + 1}"


def ikili_kod(dugum):
    return f"{dugum:0{BOYUT}b}"


def xor_komsusu(dugum, bit):
    return dugum ^ (1 << bit)


class HypercubeSystem:

    def __init__(self):
        self.kuyruklar = [
            queue.Queue() for _ in range(THREAD_SAYISI)
        ]

        self.threadler = []
        self.veriler = [None] * THREAD_SAYISI
        self.sonuclar = [{} for _ in range(THREAD_SAYISI)]

    def baslat(self):
        for dugum in range(THREAD_SAYISI):
            thread = threading.Thread(
                target=self.thread_calistir,
                args=(dugum,),
                name=thread_adi(dugum)
            )

            thread.start()
            self.threadler.append(thread)

            komsular = [
                thread_adi(xor_komsusu(dugum, bit))
                for bit in range(BOYUT)
            ]

            log_yaz(
                f"{thread_adi(dugum)} ({ikili_kod(dugum)}) "
                f"başlatıldı. Komşular: {', '.join(komsular)}"
            )

    def thread_calistir(self, dugum):
        while True:
            mesaj = self.kuyruklar[dugum].get()

            if mesaj is None:
                self.kuyruklar[dugum].task_done()
                break

            try:
                islem = mesaj["islem"]

                if islem == "VERI_BASLAT":
                    self.veriler[dugum] = mesaj["veri"]

                    log_yaz(
                        f"{thread_adi(dugum)} başlangıç verisini aldı."
                    )

                elif islem == "VERI_GONDER":
                    hedef = mesaj["hedef"]
                    tamamlandi = threading.Event()

                    aktarim = {
                        "islem": "VERI_AL",
                        "veri": self.veriler[dugum],
                        "tamamlandi": tamamlandi
                    }

                    self.kuyruklar[hedef].put(aktarim)
                    tamamlandi.wait()

                    if "hata" in aktarim:
                        raise aktarim["hata"]

                    log_yaz(
                        f"{thread_adi(dugum)} -> "
                        f"{thread_adi(hedef)}: Veri gönderildi."
                    )

                elif islem == "VERI_AL":
                    self.veriler[dugum] = mesaj["veri"]

                    log_yaz(
                        f"{thread_adi(dugum)} veriyi aldı."
                    )

                elif islem == "MODEL_CALISTIR":
                    self.model_calistir(dugum)

                elif islem == "SONUC_GONDER":
                    hedef = mesaj["hedef"]
                    tamamlandi = threading.Event()

                    aktarim = {
                        "islem": "SONUC_BIRLESTIR",
                        "veri": self.sonuclar[dugum].copy(),
                        "tamamlandi": tamamlandi
                    }

                    self.kuyruklar[hedef].put(aktarim)
                    tamamlandi.wait()

                    if "hata" in aktarim:
                        raise aktarim["hata"]

                    log_yaz(
                        f"{thread_adi(dugum)} -> "
                        f"{thread_adi(hedef)}: Sonuç gönderildi."
                    )

                elif islem == "SONUC_BIRLESTIR":
                    self.sonuclar[dugum].update(mesaj["veri"])

                    log_yaz(
                        f"{thread_adi(dugum)} sonuçları birleştirdi."
                    )

            except Exception as hata:
                mesaj["hata"] = hata

            finally:
                if "tamamlandi" in mesaj:
                    mesaj["tamamlandi"].set()

                self.kuyruklar[dugum].task_done()

    def komutlari_gonder(self, gorevler):
        mesajlar = []

        for hedef, islem, veri in gorevler:
            mesaj = {
                "islem": islem,
                **veri,
                "tamamlandi": threading.Event()
            }

            self.kuyruklar[hedef].put(mesaj)
            mesajlar.append(mesaj)

        for mesaj in mesajlar:
            mesaj["tamamlandi"].wait()

        hatalar = [
            mesaj["hata"]
            for mesaj in mesajlar
            if "hata" in mesaj
        ]

        if hatalar:
            raise RuntimeError(
                f"Thread işlemi başarısız: {hatalar[0]}"
            ) from hatalar[0]

    def veri_dagit(self, veri):
        self.komutlari_gonder([
            (0, "VERI_BASLAT", {"veri": veri})
        ])

        for bit in range(BOYUT):
            gorevler = []

            for kaynak in range(THREAD_SAYISI):
                if kaynak & (1 << bit):
                    continue

                if (kaynak >> (bit + 1)) != 0:
                    continue

                hedef = xor_komsusu(kaynak, bit)

                gorevler.append(
                    (kaynak, "VERI_GONDER", {"hedef": hedef})
                )

            self.komutlari_gonder(gorevler)

        log_yaz("Veri dağıtımı tamamlandı.")

    def model_calistir(self, dugum):
        modul_adi, fonksiyon_adi = MODELLER[dugum]

        modul = importlib.import_module(modul_adi)
        fonksiyon = getattr(modul, fonksiyon_adi)

        X_train, X_test, y_train, y_test = self.veriler[dugum]

        log_yaz(
            f"{thread_adi(dugum)} - {modul_adi} eğitimi başladı."
        )

        baslangic = time.time()

        sonuc = fonksiyon(
            X_train,
            X_test,
            y_train,
            y_test
        )

        gecen_sure = round(time.time() - baslangic, 4)

        if not isinstance(sonuc, dict):
            raise TypeError(
                f"{modul_adi} sözlük (dict) döndürmelidir."
            )

        sonuc.setdefault("training_time_sec", gecen_sure)

        self.sonuclar[dugum][modul_adi] = sonuc

        log_yaz(
            f"{thread_adi(dugum)} - {modul_adi} eğitimi tamamlandı."
        )

    def modelleri_calistir(self):
        gorevler = [
            (dugum, "MODEL_CALISTIR", {})
            for dugum in range(THREAD_SAYISI)
        ]

        log_yaz("Dört model eşzamanlı olarak başlatılıyor.")

        self.komutlari_gonder(gorevler)

        log_yaz("Bütün model eğitimleri tamamlandı.")

    def sonuclari_topla(self):
        for bit in reversed(range(BOYUT)):
            gorevler = []

            for kaynak in range(THREAD_SAYISI):
                if not (kaynak & (1 << bit)):
                    continue

                if (kaynak >> (bit + 1)) != 0:
                    continue

                hedef = xor_komsusu(kaynak, bit)

                gorevler.append(
                    (kaynak, "SONUC_GONDER", {"hedef": hedef})
                )

            self.komutlari_gonder(gorevler)

        log_yaz("Bütün sonuçlar T1 üzerinde toplandı.")

    def sonuclari_goster(self):
        print("\n" + "=" * 55)
        print("HYPERCUBE MODEL SONUÇLARI")
        print("=" * 55)

        for model, metrikler in self.sonuclar[0].items():
            print(f"\n{model}")

            for anahtar, deger in metrikler.items():
                print(f"{anahtar:20}: {deger}")

        print("=" * 55)

    def kapat(self):
        for kuyruk in self.kuyruklar:
            kuyruk.put(None)

        for thread in self.threadler:
            thread.join()

        log_yaz("Bütün threadler kapatıldı.")


def veri_yukle():
    modul = importlib.import_module(VERI_MODULU)
    fonksiyon = getattr(modul, VERI_FONKSIYONU)

    veri = fonksiyon()

    if not isinstance(veri, (tuple, list)) or len(veri) != 4:
        raise ValueError(
            "Veri hazırlama fonksiyonu "
            "(X_train, X_test, y_train, y_test) döndürmelidir."
        )

    return tuple(veri)


def main():
    sistem = HypercubeSystem()

    try:
        veri = veri_yukle()

        log_yaz("Veri seti hazır.")

        sistem.baslat()
        sistem.veri_dagit(veri)
        sistem.modelleri_calistir()
        sistem.sonuclari_topla()
        sistem.sonuclari_goster()

    finally:
        sistem.kapat()


if __name__ == "__main__":
    main()
