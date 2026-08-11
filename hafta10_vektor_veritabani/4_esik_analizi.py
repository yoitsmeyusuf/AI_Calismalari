"""
10. Hafta - 4. adim: 30 soruluk benchmark ve esik (threshold) analizi.

Esigi bastan "0.7 olsun" diye secmiyoruz; olcuyoruz. Her esik degeri icin iki
hata turu sayiliyor:

    esik cok yuksek -> cevabi olan soruya "bilmiyorum" denir  (kacirma)
    esik cok dusuk  -> alakasiz soruya uydurma cevap uretilir (halusinasyon)

Cikti:
    veri/esik_sonuclari.json
    gorseller/skor_dagilimi.png    pozitif vs negatif top-1 skorlari
    gorseller/esik_supurme.png     esige gore dogruluk egrileri

Ayrica Chroma (HNSW, yaklasik) ile numpy (kesin kosinus) sonuclari
karsilastiriliyor: esik analizi yaklasik aramaya dayaniyorsa raporun anlami
kalmaz.

Calistirma:
    ../.venv/bin/python hafta10_vektor_veritabani/4_esik_analizi.py
"""
import json

import numpy as np

from arama import ara, tam_kosinus
from ayarlar import ESIK_SONUC_JSON, ESIK_TARAMA, GORSELLER, SORULAR_JSON, VERI

K = 5


def sorulari_oku() -> dict:
    with open(SORULAR_JSON, encoding="utf-8") as f:
        return json.load(f)


def sorulari_calistir(sorular: dict) -> dict:
    pozitif, negatif, sapma = [], [], 0

    for kayit in sorular["pozitif"]:
        sonuclar = ara(kayit["soru"], k=K)
        idler = [s["chunk_id"] for s in sonuclar]
        kesin = tam_kosinus(kayit["soru"], k=K)
        if kesin[0]["chunk_id"] != idler[0]:
            sapma += 1
        # parent isabeti: ayni bolumden baska bir chunk da kabul edilebilir cevap
        gold = kayit["gold_chunk_id"]
        gold_parent = gold.rsplit("-", 1)[0]
        pozitif.append(
            {
                "soru": kayit["soru"],
                "gold_chunk_id": gold,
                "top1_skor": sonuclar[0]["benzerlik"],
                "top1_id": idler[0],
                "gold_sirasi": idler.index(gold) + 1 if gold in idler else None,
                "parent_sirasi": next(
                    (i + 1 for i, s in enumerate(sonuclar) if s["parent_id"] == gold_parent),
                    None,
                ),
                "kesin_top1_skor": kesin[0]["benzerlik"],
            }
        )

    for kayit in sorular["negatif"]:
        sonuclar = ara(kayit["soru"], k=K)
        kesin = tam_kosinus(kayit["soru"], k=K)
        if kesin[0]["chunk_id"] != sonuclar[0]["chunk_id"]:
            sapma += 1
        negatif.append(
            {
                "soru": kayit["soru"],
                "grup": kayit["grup"],
                "top1_skor": sonuclar[0]["benzerlik"],
                "top1_baslik": sonuclar[0]["title"],
                "kesin_top1_skor": kesin[0]["benzerlik"],
            }
        )

    return {"pozitif": pozitif, "negatif": negatif, "hnsw_sapmasi": sapma}


def esik_supur(sonuc: dict) -> list[dict]:
    """Her esik degeri icin dogru cevaplanan / dogru reddedilen soru sayisi."""
    bas, son, adim = ESIK_TARAMA
    poz = sonuc["pozitif"]
    neg = sonuc["negatif"]

    tablo = []
    for esik in np.arange(bas, son + adim / 2, adim):
        # Pozitif dogru: esigi gecti VE gold chunk ilk K icinde.
        #
        # Olcut bilerek "ilk K" - "ilk sirada" degil. LLM'e ilk K chunk birden
        # veriliyor; gold'un 1. mi 3. mu oldugu cevabin dogrulugunu degistirmiyor.
        # Ayrica ayni makalenin kardes chunk'lari cogu zaman gold kadar gecerli
        # kaynak oluyor, onlari "hata" saymak esigi degil erisimi olcerdi.
        poz_dogru = sum(
            1 for p in poz if p["top1_skor"] >= esik and p["gold_sirasi"] is not None
        )
        poz_dogru_top1 = sum(1 for p in poz if p["top1_skor"] >= esik and p["gold_sirasi"] == 1)
        neg_dogru = sum(1 for n in neg if n["top1_skor"] < esik)
        tablo.append(
            {
                "esik": round(float(esik), 2),
                "pozitif_dogru": poz_dogru,
                "pozitif_dogru_top1": poz_dogru_top1,
                "negatif_dogru": neg_dogru,
                "toplam_dogru": poz_dogru + neg_dogru,
                "dogruluk": (poz_dogru + neg_dogru) / (len(poz) + len(neg)),
            }
        )
    return tablo


def grafik_ciz(sonuc: dict, tablo: list[dict], en_iyi: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GORSELLER.mkdir(exist_ok=True)
    poz = [p["top1_skor"] for p in sonuc["pozitif"]]

    # --- 1. Skor dagilimi ---
    fig, ax = plt.subplots(figsize=(9, 5))
    gruplar = {
        "pozitif (20)": (poz, "#2a9d8f", "o"),
        "yakın-kaçırma (4)": (
            [n["top1_skor"] for n in sonuc["negatif"] if n["grup"] == "yakin_kacirma"],
            "#e76f51",
            "s",
        ),
        "konu yok (3)": (
            [n["top1_skor"] for n in sonuc["negatif"] if n["grup"] == "konu_yok"],
            "#f4a261",
            "^",
        ),
        "alan dışı (3)": (
            [n["top1_skor"] for n in sonuc["negatif"] if n["grup"] == "alan_disi"],
            "#8d99ae",
            "v",
        ),
    }
    for y, (ad, (skorlar, renk, isaret)) in enumerate(gruplar.items()):
        ax.scatter(
            skorlar,
            np.full(len(skorlar), y) + np.random.default_rng(0).normal(0, 0.05, len(skorlar)),
            c=renk, marker=isaret, s=70, alpha=0.85, label=ad, edgecolors="white", linewidths=0.6,
        )
    ax.axvline(en_iyi, color="#264653", ls="--", lw=1.6, label=f"eşik = {en_iyi:.2f}")
    ax.set_yticks(range(len(gruplar)))
    ax.set_yticklabels(gruplar.keys())
    ax.set_xlabel("top-1 kosinüs benzerliği")
    ax.set_title("Soru gruplarına göre en yüksek benzerlik skoru")
    ax.grid(axis="x", alpha=0.25)
    ax.set_ylim(-0.6, len(gruplar) - 0.4)
    # Pozitifler sagda, alan disi solda ust sirada toplaniyor; sol-alt tek bos
    # bolge, aciklama kutusu noktalari ortmesin diye oraya konuyor.
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(GORSELLER / "skor_dagilimi.png", dpi=150)
    plt.close(fig)

    # --- 2. Esik supurme ---
    esikler = [t["esik"] for t in tablo]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(esikler, [t["pozitif_dogru"] / 20 for t in tablo], label="pozitif doğru (20)", color="#2a9d8f", lw=2)
    ax.plot(esikler, [t["negatif_dogru"] / 10 for t in tablo], label="negatif doğru (10)", color="#e76f51", lw=2)
    ax.plot(esikler, [t["dogruluk"] for t in tablo], label="toplam doğruluk (30)", color="#264653", lw=2.4, ls="-")
    ax.axvline(en_iyi, color="#264653", ls="--", lw=1.4)
    ax.annotate(f"en iyi eşik = {en_iyi:.2f}", (en_iyi, 0.05), xytext=(6, 0), textcoords="offset points", fontsize=9)
    ax.set_xlabel("benzerlik eşiği")
    ax.set_ylabel("oran")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Eşik süpürmesi: doğru cevaplama ve doğru reddetme dengesi")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(GORSELLER / "esik_supurme.png", dpi=150)
    plt.close(fig)
    print(f"OK  grafikler -> {GORSELLER}")


def main() -> None:
    sorular = sorulari_oku()
    sonuc = sorulari_calistir(sorular)
    tablo = esik_supur(sonuc)

    # En iyi esik: toplam dogruluk maksimum; esitlikte marji en genis olan
    # (yani pozitiflerin en dusugu ile negatiflerin en yuksegi arasi) secilir.
    en_yuksek = max(t["toplam_dogru"] for t in tablo)
    adaylar = [t for t in tablo if t["toplam_dogru"] == en_yuksek]
    en_iyi = adaylar[len(adaylar) // 2]["esik"]  # plato varsa ortasi

    poz = [p["top1_skor"] for p in sonuc["pozitif"]]
    neg = [n["top1_skor"] for n in sonuc["negatif"]]

    print(f"\n--- Pozitif sorular ({len(poz)}) ---")
    for p in sorted(sonuc["pozitif"], key=lambda x: x["top1_skor"]):
        isaret = "OK " if p["gold_sirasi"] == 1 else ("~" + str(p["gold_sirasi"]) if p["gold_sirasi"] else "MISS")
        print(f"  {p['top1_skor']:.4f} [{isaret:>4}] {p['soru'][:66]}")

    print(f"\n--- Negatif sorular ({len(neg)}) ---")
    for n in sorted(sonuc["negatif"], key=lambda x: -x["top1_skor"]):
        print(f"  {n['top1_skor']:.4f} [{n['grup']:>14}] {n['soru'][:56]}")

    print(
        f"\n--- Ozet ---\n"
        f"  pozitif top-1 skor : min {min(poz):.4f} | medyan {np.median(poz):.4f} | max {max(poz):.4f}\n"
        f"  negatif top-1 skor : min {min(neg):.4f} | medyan {np.median(neg):.4f} | max {max(neg):.4f}\n"
        f"  ayrim marji        : {min(poz) - max(neg):+.4f}\n"
        f"  gold top-1 isabeti : {sum(1 for p in sonuc['pozitif'] if p['gold_sirasi'] == 1)}/{len(poz)}\n"
        f"  gold top-5 isabeti : {sum(1 for p in sonuc['pozitif'] if p['gold_sirasi'])}/{len(poz)}\n"
        f"  parent top-5       : {sum(1 for p in sonuc['pozitif'] if p['parent_sirasi'])}/{len(poz)}\n"
        f"  HNSW sapmasi       : {sonuc['hnsw_sapmasi']}/30 (Chroma vs kesin kosinus)\n"
        f"  EN IYI ESIK        : {en_iyi:.2f} -> {en_yuksek}/30 dogru"
    )

    VERI.mkdir(exist_ok=True)
    with open(ESIK_SONUC_JSON, "w", encoding="utf-8") as f:
        json.dump({"en_iyi_esik": en_iyi, "tablo": tablo, **sonuc}, f, ensure_ascii=False, indent=2)
    print(f"OK  sonuclar -> {ESIK_SONUC_JSON}")

    grafik_ciz(sonuc, tablo, en_iyi)


if __name__ == "__main__":
    main()
