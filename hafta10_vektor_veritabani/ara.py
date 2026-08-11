"""
10. Hafta: komut satirindan esik kontrollu arama.

Esigin altinda kalan sorularda hicbir chunk dondurulmuyor - LLM'e verilecek
bir sey olmadigi icin uydurma ihtimali de ortadan kalkiyor.

Calistirma:
    ../.venv/bin/python hafta10_vektor_veritabani/ara.py "guatr belirtileri nelerdir"
    ../.venv/bin/python hafta10_vektor_veritabani/ara.py --esik 0.50 "sepsis tedavisi"
    ../.venv/bin/python hafta10_vektor_veritabani/ara.py          # interaktif
"""
import argparse

from arama import ara, cevapla
from ayarlar import BENZERLIK_ESIGI, VARSAYILAN_K


def yazdir(soru: str, k: int, esik: float) -> None:
    sonuc = cevapla(soru, k=k, esik=esik)
    print(f"\n> {soru}")
    print(f"  en yüksek benzerlik: {sonuc['en_iyi_skor']:.4f}  (eşik {esik:.2f})")

    if not sonuc["cevaplandi"]:
        print(f"  RET: {sonuc['mesaj']}")
        # Neyin elendigini gormek faydali: skor esigin hemen altindaysa soru
        # aslinda kapsamda olabilir, esik gozden gecirilmeli.
        en_yakin = ara(soru, k=1)[0]
        print(f"  (en yakın kayıt: {en_yakin['title'][:60]} — {en_yakin['benzerlik']:.4f})")
        return

    print(f"  {len(sonuc['kaynaklar'])} kaynak eşiği geçti:\n")
    for sira, kaynak in enumerate(sonuc["kaynaklar"], start=1):
        print(f"  [{sira}] {kaynak['benzerlik']:.4f}  {kaynak['title']}")
        if kaynak["bolum_basligi"]:
            print(f"      bölüm: {kaynak['bolum_basligi']}")
        print(f"      {kaynak['chunk_text'][:220]}")
        print(f"      {kaynak['url']}\n")


def main() -> None:
    ayristirici = argparse.ArgumentParser(description="Eşik kontrollü vektör arama")
    ayristirici.add_argument("soru", nargs="*", help="aranacak soru")
    ayristirici.add_argument("-k", type=int, default=VARSAYILAN_K, help="kaç sonuç")
    ayristirici.add_argument(
        "--esik", type=float, default=BENZERLIK_ESIGI, help="benzerlik eşiği"
    )
    arg = ayristirici.parse_args()

    if arg.soru:
        yazdir(" ".join(arg.soru), arg.k, arg.esik)
        return

    print("Soru yazın (boş satır veya Ctrl-D ile çıkış).")
    while True:
        try:
            soru = input("\nsoru> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not soru:
            return
        yazdir(soru, arg.k, arg.esik)


if __name__ == "__main__":
    main()
