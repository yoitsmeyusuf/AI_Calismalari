"""
Asama 2: benzerlik esigi analizi.

Esik bastan secilmiyor, supurulerek bulunuyor. Test seti uc parcali:

  turetilmis pozitif : chunk'lardan OTOMATIK uretiliyor, gold_chunk_id var
                       -> recall goz karari degil olcum
  dogal pozitif      : elle yazilan gercekci kullanici mesajlari (sorular.json)
  negatif            : alan_disi / konu_yok / yakin_kacirma (sorular.json)

10. haftadan devralinan ders: hepsi "Fenerbahce kac sampiyonluk" tipinde
olsaydi analiz sahte cikardi - 0.2 de 0.6 da ayni sonucu verirdi. Esigi
fiilen belirleyen `yakin_kacirma` grubu.

    ../../.venv/bin/python 2_esik_analizi.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BURASI = Path(__file__).resolve().parent
sys.path.insert(0, str(BURASI))
sys.path.insert(0, str(BURASI.parent))

import ayarlar  # noqa: E402
from retriever import Retriever  # noqa: E402

GORSELLER = BURASI / "gorseller"
GORSELLER.mkdir(exist_ok=True)
SONUC_JSON = BURASI / "esik_sonuclari.json"

YUZEY, MURK, IKINCIL = "#fcfcfb", "#0b0b0b", "#52514e"
SERI = ["#2a78d6", "#eb6834", "#1baf7a"]

DURAK = {
    "bir", "bu", "ve", "de", "da", "ile", "için", "gibi", "ama", "ben", "sen",
    "o", "biz", "siz", "her", "çok", "daha", "en", "ne", "mi", "mı", "var",
    "yok", "olan", "olur", "oldu", "ki", "ya", "hem", "kadar", "sonra", "önce",
    "ise", "diye", "şey", "beni", "bana", "senin", "benim", "onu", "ona",
}


def turetilmis_pozitifler(adet: int = 20, tohum: int = 42) -> list[dict]:
    """
    Chunk'lardan sorgu turetir: chunk'a OZGU (korpusta nadir) kelimeleri
    secip bir arama ifadesi kurar. Boylece her sorgunun dogru cevabi
    bellidir (gold_chunk_id) ve recall olculebilir.

    Soru yazip cevabini aramak degil, cevaptan soru turetmek - 10. haftadaki
    yontemin aynisi.
    """
    chunklar = [json.loads(s) for s in (BURASI / "chunklar.jsonl").read_text(
        encoding="utf-8").splitlines()]

    # Korpus geneli kelime frekansi
    genel = Counter()
    for c in chunklar:
        genel.update(re.findall(r"\w+", c["chunk_text"].lower()))

    rastgele = random.Random(tohum)
    aday = [c for c in chunklar if c["karakter"] >= 200]
    rastgele.shuffle(aday)

    sorgular = []
    for c in aday:
        kelimeler = [k for k in re.findall(r"\w+", c["chunk_text"].lower())
                     if len(k) >= 5 and k not in DURAK]
        if len(set(kelimeler)) < 6:
            continue
        # Korpusta en NADIR gecen kelimeler chunk'i en iyi ayirt edenler
        ozgun = sorted(set(kelimeler), key=lambda k: genel[k])[:5]
        sorgular.append({
            "sorgu": " ".join(ozgun),
            "gold_chunk_id": c["id"],
            "sarki": c["sarki"],
        })
        if len(sorgular) >= adet:
            break
    return sorgular


def main() -> None:
    r = Retriever()
    sorular = json.loads((BURASI / "sorular.json").read_text(encoding="utf-8"))

    turetilmis = turetilmis_pozitifler()
    dogal = sorular["dogal_pozitif"]
    negatifler = sorular["negatif"]
    print(f"turetilmis pozitif : {len(turetilmis)}")
    print(f"dogal pozitif      : {len(dogal)}")
    print(f"negatif            : {len(negatifler)}")

    kayitlar: list[dict] = []

    for t in turetilmis:
        sonuclar = r.ara(t["sorgu"], k=5)
        ilk5 = [s.id for s in sonuclar]
        kayitlar.append({
            "tip": "turetilmis_pozitif", "grup": "turetilmis",
            "sorgu": t["sorgu"], "sarki": t["sarki"],
            "en_iyi": sonuclar[0].benzerlik,
            "gold_ilk1": ilk5[0] == t["gold_chunk_id"],
            "gold_ilk5": t["gold_chunk_id"] in ilk5,
        })

    for mesaj in dogal:
        sonuclar = r.ara(mesaj, k=5)
        kayitlar.append({"tip": "dogal_pozitif", "grup": "dogal",
                         "sorgu": mesaj, "en_iyi": sonuclar[0].benzerlik})

    for n in negatifler:
        sonuclar = r.ara(n["soru"], k=5)
        kayitlar.append({"tip": "negatif", "grup": n["grup"],
                         "sorgu": n["soru"], "en_iyi": sonuclar[0].benzerlik})

    # --- Grup dagilimlari ---
    print(f"\n{'grup':<22}{'n':>4}{'min':>9}{'medyan':>9}{'max':>9}")
    print("-" * 53)
    gruplar: dict[str, list[float]] = {}
    for k in kayitlar:
        ad = k["tip"] if k["tip"] != "negatif" else f"negatif/{k['grup']}"
        gruplar.setdefault(ad, []).append(k["en_iyi"])
    for ad, deger in gruplar.items():
        d = sorted(deger)
        print(f"{ad:<22}{len(d):>4}{d[0]:>9.4f}{d[len(d)//2]:>9.4f}{d[-1]:>9.4f}")

    # --- Recall (yalnizca turetilmisler icin olculebilir) ---
    tp = [k for k in kayitlar if k["tip"] == "turetilmis_pozitif"]
    print(f"\ngold chunk ilk 1'de : {sum(k['gold_ilk1'] for k in tp)}/{len(tp)}")
    print(f"gold chunk ilk 5'te : {sum(k['gold_ilk5'] for k in tp)}/{len(tp)}")

    # --- Esik supurmesi ---
    # Iki supurme yapiliyor. Sebep yukaridaki recall olcumu: turetilmis
    # sorgular (nadir kelime torbasi) bu korpusta gold chunk'i iyi
    # bulamiyor ve skorlari dogal mesajlardan belirgin dusuk. Onlari
    # pozitif saymak esigi yapay olarak asagi cekiyor.
    #
    # Uygulamanin gercek girdisi dogal kullanici mesaji, o yuzden ISLETME
    # esigi ikinci supurmeden aliniyor; birincisi karsilastirma icin duruyor.
    negatif_skor = [k["en_iyi"] for k in kayitlar if k["tip"] == "negatif"]
    alt, ust, adim = ayarlar.ESIK_TARAMA

    def supur(pozitif_skorlar: list[float]) -> tuple[list[dict], dict, list[dict], int]:
        s_list = []
        e = alt
        while e <= ust + 1e-9:
            dp = sum(1 for s in pozitif_skorlar if s >= e)
            dn = sum(1 for s in negatif_skor if s < e)
            s_list.append({"esik": round(e, 3), "pozitif_dogru": dp,
                           "negatif_dogru": dn, "toplam": dp + dn})
            e += adim
        en_iyi = max(s["toplam"] for s in s_list)
        plt_ = [s for s in s_list if s["toplam"] == en_iyi]
        return s_list, plt_[len(plt_) // 2], plt_, en_iyi

    hepsi_skor = [k["en_iyi"] for k in kayitlar if k["tip"].endswith("pozitif")]
    dogal_skor = [k["en_iyi"] for k in kayitlar if k["tip"] == "dogal_pozitif"]

    supurme_hepsi, secilen_hepsi, _, en_iyi_hepsi = supur(hepsi_skor)
    supurme, secilen, plato, en_iyi_toplam = supur(dogal_skor)
    pozitifler = dogal_skor

    print(f"\nIki supurme:")
    print(f"  butun pozitifler (turetilmis+dogal, {len(hepsi_skor)}) -> esik "
          f"{secilen_hepsi['esik']:.2f}  ({en_iyi_hepsi}/{len(hepsi_skor)+len(negatif_skor)})")
    print(f"  yalniz dogal mesajlar ({len(dogal_skor)})            -> esik "
          f"{secilen['esik']:.2f}  ({en_iyi_toplam}/{len(dogal_skor)+len(negatif_skor)})")

    print(f"\n{'esik':>7}{'pozitif':>10}{'negatif':>10}{'toplam':>9}")
    print("-" * 36)
    n_top = len(pozitifler) + len(negatif_skor)
    for s in supurme:
        if abs(s["esik"] * 100 - round(s["esik"] * 100)) < 1e-6 and \
                round(s["esik"] * 100) % 5 == 0:
            isaret = "  <-- secilen" if s["esik"] == secilen["esik"] else ""
            print(f"{s['esik']:>7.2f}{s['pozitif_dogru']:>7}/{len(pozitifler):<3}"
                  f"{s['negatif_dogru']:>7}/{len(negatif_skor):<3}"
                  f"{s['toplam']:>6}/{n_top}{isaret}")

    print(f"\nplato: {plato[0]['esik']:.2f} - {plato[-1]['esik']:.2f} "
          f"({en_iyi_toplam}/{n_top})")
    print(f"SECILEN ESIK: {secilen['esik']:.2f}")

    elenen = [s for s in negatif_skor if s < secilen["esik"]]
    gecen_poz = [s for s in pozitifler if s >= secilen["esik"]]
    if elenen and gecen_poz:
        print(f"\n  en yuksek dogru elenen negatif : {max(elenen):.4f}")
        print(f"                          esik   : {secilen['esik']:.4f}")
        print(f"  en dusuk gecen pozitif         : {min(gecen_poz):.4f}")

    SONUC_JSON.write_text(json.dumps(
        {"kayitlar": kayitlar,
         "supurme_dogal": supurme, "supurme_hepsi": supurme_hepsi,
         "secilen_esik": secilen["esik"],
         "esik_hepsi_dahil": secilen_hepsi["esik"],
         "plato": [plato[0]["esik"], plato[-1]["esik"]],
         "en_iyi_toplam": en_iyi_toplam, "toplam_soru": n_top,
         "recall_ilk1": sum(k["gold_ilk1"] for k in tp),
         "recall_ilk5": sum(k["gold_ilk5"] for k in tp),
         "turetilmis_adet": len(tp)},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Gorseller ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor=YUZEY)

    sira = ["turetilmis_pozitif", "dogal_pozitif",
            "negatif/yakin_kacirma", "negatif/konu_yok", "negatif/alan_disi"]
    renk = {"turetilmis_pozitif": SERI[0], "dogal_pozitif": SERI[2],
            "negatif/yakin_kacirma": SERI[1], "negatif/konu_yok": SERI[1],
            "negatif/alan_disi": SERI[1]}
    for i, ad in enumerate([a for a in sira if a in gruplar]):
        d = gruplar[ad]
        ax1.scatter(d, [i] * len(d), s=70, color=renk[ad], alpha=0.75,
                    edgecolors=YUZEY, linewidths=1.5, zorder=3)
    ax1.axvline(secilen["esik"], color=MURK, linewidth=1.5, linestyle="--", zorder=4)
    ax1.annotate(f"eşik {secilen['esik']:.2f}", xy=(secilen["esik"], len(gruplar) - 0.4),
                 xytext=(6, 0), textcoords="offset points", color=MURK,
                 fontsize=9, weight="bold")
    ax1.set_yticks(range(len([a for a in sira if a in gruplar])))
    ax1.set_yticklabels([a.replace("negatif/", "neg: ") for a in sira if a in gruplar],
                        fontsize=8)
    ax1.set_xlabel("en iyi chunk benzerligi", color=IKINCIL, fontsize=9)
    ax1.set_title("Skor dagilimi", color=MURK, fontsize=11, loc="left", pad=10)

    esikler = [s["esik"] for s in supurme]
    for i, (anahtar, etiket) in enumerate([("pozitif_dogru", "pozitif dogru"),
                                           ("negatif_dogru", "negatif dogru")]):
        deger = [s[anahtar] for s in supurme]
        ax2.plot(esikler, deger, linewidth=2, color=SERI[i], label=etiket, zorder=3)
    ax2.plot(esikler, [s["toplam"] for s in supurme], linewidth=2.5,
             color=MURK, label="toplam", zorder=4)
    ax2.axvline(secilen["esik"], color=MURK, linewidth=1.5, linestyle="--", zorder=2)
    ax2.set_xlabel("esik", color=IKINCIL, fontsize=9)
    ax2.set_ylabel("dogru sayisi", color=IKINCIL, fontsize=9)
    ax2.set_title(f"Esik supurmesi (en iyi {en_iyi_toplam}/{n_top})",
                  color=MURK, fontsize=11, loc="left", pad=10)
    ax2.legend(frameon=False, fontsize=8, labelcolor=IKINCIL, loc="center left")

    for ax in (ax1, ax2):
        ax.set_facecolor(YUZEY)
        ax.tick_params(colors=IKINCIL, labelsize=8, length=3)
        for k in ("top", "right"):
            ax.spines[k].set_visible(False)
        for k in ("left", "bottom"):
            ax.spines[k].set_color("#d8d7d2")
        ax.grid(True, color="#e8e7e3", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(GORSELLER / "esik_analizi.png", dpi=150, facecolor=YUZEY)
    plt.close(fig)
    print(f"\n  -> gorseller/esik_analizi.png")
    print(f"  -> {SONUC_JSON.name}")


if __name__ == "__main__":
    main()
