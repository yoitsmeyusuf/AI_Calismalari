"""
10. Hafta - 3. adim: vektorlu chunk'lari Hugging Face Hub'a Dataset olarak pushlar.

Zorunlu uc sutun (odev): url, chunk_text, chunk_vector.
Ek meta: chunk_id, title, bolum_basligi, parent_id, __source, token_sayisi.

Token `hf auth login` cache'inden geliyor (write izni gerekli). Repo id
ayarlar.py'de sabit.

Calistirma:
    ../.venv/bin/python hafta10_vektor_veritabani/3_push_dataset.py
    ../.venv/bin/python hafta10_vektor_veritabani/3_push_dataset.py --kuru  # push etmeden dogrula
"""
import argparse
import json
import statistics

import pyarrow.parquet as pq

from ayarlar import (
    BENZERLIK_ESIGI,
    CHROMA_MESAFE,
    DATASET_REPO_ID,
    EMBEDDING_MODELI,
    ESIK_SONUC_JSON,
    GORSELLER,
    HASTANELER,
    HERE,
    KAYNAK_DATASET,
    VEKTOR_BOYUTU,
    VEKTOR_PARQUET,
)
from chunklama import MAX_TOK, MIN_TOK, OVERLAP_TOK

ZORUNLU_SUTUNLAR = ["url", "chunk_text", "chunk_vector"]


def dogrula(tablo) -> dict:
    eksik = [s for s in ZORUNLU_SUTUNLAR if s not in tablo.schema.names]
    if eksik:
        raise SystemExit(f"zorunlu sutun eksik: {eksik}")

    vektorler = tablo.column("chunk_vector").to_pylist()
    boylar = {len(v) for v in vektorler}
    if boylar != {VEKTOR_BOYUTU}:
        raise SystemExit(f"vektor boyutlari tutarsiz: {sorted(boylar)[:5]}")

    metinler = tablo.column("chunk_text").to_pylist()
    if any(not m.strip() for m in metinler):
        raise SystemExit("bos chunk_text var")
    if any(not u.strip() for u in tablo.column("url").to_pylist()):
        raise SystemExit("bos url var")

    tokenlar = sorted(tablo.column("token_sayisi").to_pylist())
    basliklar = tablo.column("bolum_basligi").to_pylist()
    print(
        f"OK  {tablo.num_rows} satir, {VEKTOR_BOYUTU} boyut\n"
        f"    token: medyan {statistics.median(tokenlar):.0f} | max {max(tokenlar)}\n"
        f"    makale: {len(set(tablo.column('parent_id').to_pylist()))} bolum, "
        f"{len(set(u for u in tablo.column('url').to_pylist()))} benzersiz url"
    )
    return {
        "satir": tablo.num_rows,
        "url": len(set(tablo.column("url").to_pylist())),
        "token_med": statistics.median(tokenlar),
        "token_p90": tokenlar[int(len(tokenlar) * 0.9)],
        "token_max": max(tokenlar),
        "baslikli": 100 * sum(1 for b in basliklar if b) / len(basliklar),
    }


def _esik_ozeti() -> dict:
    """Esik analizinin ciktisindan README icin sayilari cikarir."""
    if not ESIK_SONUC_JSON.exists():
        raise SystemExit(
            f"{ESIK_SONUC_JSON} yok - once 4_esik_analizi.py calistirin.\n"
            "README'de esik analizi bolumu zorunlu."
        )
    with open(ESIK_SONUC_JSON, encoding="utf-8") as f:
        sonuc = json.load(f)

    poz = [p["top1_skor"] for p in sonuc["pozitif"]]
    esik = sonuc["en_iyi_esik"]
    satir = next(t for t in sonuc["tablo"] if abs(t["esik"] - esik) < 1e-9)

    gruplar = {}
    for kayit in sonuc["negatif"]:
        gruplar.setdefault(kayit["grup"], []).append(kayit["top1_skor"])

    # Karar bosluğu: esigin altinda kalan en yuksek negatif ile en dusuk pozitif
    elenen = [n["top1_skor"] for n in sonuc["negatif"] if n["top1_skor"] < esik]
    # Esikle elenemeyen negatifler - README'de aciklanmasi gereken kisim
    kacan = [n for n in sonuc["negatif"] if n["top1_skor"] >= esik]

    return {
        "esik": esik,
        "dogru": satir["toplam_dogru"],
        "poz_dogru": satir["pozitif_dogru"],
        "neg_dogru": satir["negatif_dogru"],
        "poz_min": min(poz),
        "poz_med": statistics.median(poz),
        "poz_max": max(poz),
        "gruplar": gruplar,
        "en_yuksek_elenen": max(elenen) if elenen else 0.0,
        "kacan": kacan,
        "tablo": sonuc["tablo"],
        "top5": sum(1 for p in sonuc["pozitif"] if p["gold_sirasi"]),
        "top1": sum(1 for p in sonuc["pozitif"] if p["gold_sirasi"] == 1),
        "hnsw_sapmasi": sonuc["hnsw_sapmasi"],
    }


GRUP_ADI = {
    "alan_disi": "alan disi",
    "konu_yok": "konu yok",
    "yakin_kacirma": "yakin-kacirma",
}


def kart_yaz(ozet: dict) -> str:
    e = _esik_ozeti()
    ham = f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/main"

    # Esik supurme tablosu: platonun etrafindaki ilginc araligi goster
    supurme = "\n".join(
        f"| {t['esik']:.2f} | {t['pozitif_dogru']}/20 | {t['negatif_dogru']}/10 | "
        f"{t['toplam_dogru']}/30 |"
        for t in e["tablo"]
        if t["esik"] in (0.40, 0.50, 0.53, 0.55, 0.56, 0.57, 0.60, 0.63, 0.70)
    )

    grup_satirlari = "\n".join(
        f"| {GRUP_ADI.get(ad, ad)} ({len(skorlar)}) | {min(skorlar):.4f} | "
        f"{statistics.median(skorlar):.4f} | {max(skorlar):.4f} | "
        f"{sum(1 for s in skorlar if s < e['esik'])}/{len(skorlar)} |"
        for ad, skorlar in sorted(e["gruplar"].items())
    )

    kacan_satirlari = "\n".join(
        f"- **\"{k['soru']}\"** — {k['top1_skor']:.4f}" for k in e["kacan"]
    )

    return f"""---
license: cc-by-4.0
language:
- tr
task_categories:
- feature-extraction
- question-answering
tags:
- medical
- turkish
- rag
- vector-database
- embeddings
size_categories:
- 1K<n<10K
---

# Turkce Tibbi Makale Vektor Veri Seti

[{KAYNAK_DATASET}](https://huggingface.co/datasets/{KAYNAK_DATASET}) veri
setinden secilen **{', '.join(HASTANELER)}** hastane kaynaklarindan
{ozet['url']} makale parcalanip
[`{EMBEDDING_MODELI}`](https://huggingface.co/{EMBEDDING_MODELI}) ile
vektorlestirildi. Toplam **{ozet['satir']} chunk**.

## Sutunlar

| sutun | tip | aciklama |
|---|---|---|
| `chunk_id` | string | Parcanin kimligi (`hastane-makale-bolum-parca`) |
| `url` | string | Parcanin ait oldugu orijinal makalenin kaynak baglantisi |
| `chunk_text` | string | Parcalanmis metin icerigi (ham; baslik eklenmemis) |
| `chunk_vector` | list[float] | {VEKTOR_BOYUTU} boyutlu, L2-normalize embedding |
| `title` | string | Makale basligi |
| `bolum_basligi` | string | Parcanin ait oldugu bolumun basligi |
| `parent_id` | string | Ayni bolumden cikan chunk'lari baglar (parent-child) |
| `__source` | string | Kaynak hastane |
| `token_sayisi` | int | Chunk'in token uzunlugu |

## 1. Chunking stratejisi

**Yontem: bolum tabanli (yapisal) parcalama.**

```
1. metni \\n ile bloklara ayir
2. baslik isaretle:  len<120 & basinda "- " yok & sonunda ":" yok
                     & ("?" ile bitiyor | "." ya da "!" ile bitmiyor)
3. icindekiler menusunu sil (govdede baslik olarak da gecen "- X" satirlari)
4. bolum = baslik + kendisinden sonraki icerik bloklari
5. chunk uret:
      bolum <= {MAX_TOK} token  ->  oldugu gibi                (~%89)
      bolum >  {MAX_TOK} token  ->  blok sinirindan bol, {OVERLAP_TOK} overlap
                              tek blok asiyorsa cumleden bol
      bolum <  {MIN_TOK} token  ->  komsu bolume yapistir
```

**Neden bu yontem secildi.** Once sabit token + overlap denendi, olcum sonucu
iki varsayim da cokdu:

- **`\\n\\n` yok.** Kaynak veri setinin 14 split'i olculdu; `liv` disinda hicbirinde
  cift newline gecmiyor. Ayrac tek `\\n`. Ama metin duz blok da degil: makaleler
  "baslik + o basligin altindaki icerik" seklinde diziliyor. Baslik tespiti
  kurali 14 split'in **hepsinde bolumlerin %84-94'unde** isabet ediyor.
- **350 token hedefi cok buyuktu.** Modelin tokenizer'i Turkce'ye ozel oldugu
  icin **5,77 karakter/token** veriyor (cok dilli modellerde ~3,2). Bolumlerin
  medyani sadece ~90 token, p99'u ~400. 350 token'lik hedef bolumlerin
  %99'undan buyuk oldugu icin 3-5 alakasiz bolumu tek chunk'a tikardi.

Sonucta **chunk sinirini tokenizer degil, belgenin kendi yapisi belirliyor.**
Overlap chunk'larin yalnizca ~%11'inde devreye giriyor: overlap keyfi noktadan
kesince anlamin ikiye bolunmesini telafi eden bir yamadir, biz keyfi noktadan
degil baslik sinirindan kestigimiz icin sadece istisnai buyuk bolumlerde
gerekiyor.

**Neden {MAX_TOK} token tavan.** Teknik zorunluluk degil - model 8192 token
aliyor. Sebep arama kalitesi: bir chunk ne kadar cok konu icerirse vektoru o
kadar ortalamaya kayar, pozitif ve negatif sorularin skorlari birbirine
yaklasir, esik ayirt edemez hale gelir. Tavan konu safligini koruyan sinir.

**Sonuc:** {ozet['url']} makale -> **{ozet['satir']} chunk**, medyan
{ozet['token_med']:.0f} token, p90 {ozet['token_p90']}, max {ozet['token_max']}
(tavani asan: 0), %{ozet['baslikli']:.0f}'i baslik tasiyor.

## 2. Embedding modeli

[`{EMBEDDING_MODELI}`](https://huggingface.co/{EMBEDDING_MODELI})

| | |
|---|---|
| **Boyut (dimension)** | **{VEKTOR_BOYUTU}**, L2-normalize |
| Context window | 8.192 token |
| Parametre | ~200M (Gemma3 govdeli) |
| Pooling | mean + 2 dense ({VEKTOR_BOYUTU} -> 3072 -> {VEKTOR_BOYUTU}) |

**Neden secildi:** Turkce'ye ozel uyarlanmis (cross-lingual tokenizer surgery +
offline distillation), 8K context uzun bolumleri kesmeden aliyor, 200M parametre
8 GB VRAM'e rahat sigiyor - {ozet['satir']} chunk 18 saniyede gomuldu.
Ciktilar L2-normalize oldugu icin **kosinus benzerligi = nokta carpimi**.

**Model asimetrik - prompt kullanimi zorunlu.** `config_sentence_transformers.json`:

```
"query":    "task: search result | query: "
"document": "title: none | text: "
```

Chunk'lar `document`, sorular `query` prompt'u ile gomuluyor. Bu atlanirsa model
yine calisir ama skor dagilimi kayar ve esik analizi anlamsizlasir.

**Baslik enjeksiyonu:** `document` prompt'undaki `title: none` alanina gercek
baslik konuyor. Hastane metinleri anafora dolu ("Bu hastalikta...", "Tedavi
sureci..."); tek basina chunk cogu zaman neyden bahsettigini soylemiyor.
`chunk_text` sutunu **ham kaliyor** - baslik yalnizca vektor uretiminde
kullanildi.

**float32 tercih edildi:** model varsayilan olarak bfloat16 yukleniyor ve o
hassasiyette L2 normlar 0,996-1,004 arasinda kayiyor. `normalize_embeddings=True`
de cozmuyor (normalizasyon da bf16 icinde). float32'de normlar tam 1,0.

## Kullanim

```python
from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer

veri = load_dataset("{DATASET_REPO_ID}", split="train")
vektorler = np.array(veri["chunk_vector"], dtype=np.float32)

model = SentenceTransformer("{EMBEDDING_MODELI}")
soru = model.encode(["safra kesesi tasi nasil tedavi edilir"], prompt_name="query")[0]

skorlar = vektorler @ soru          # L2-normalize -> nokta carpimi = kosinus
en_iyi = int(np.argmax(skorlar))
if skorlar[en_iyi] < {BENZERLIK_ESIGI}:
    print("Bu sorunun cevabi dokumanlarimda yer almamaktadir.")
else:
    print(veri[en_iyi]["chunk_text"], veri[en_iyi]["url"])
```

## 3. Esik (threshold) analizi

**Secilen esik: {e['esik']:.2f} — 30 soruluk test setinde {e['dogru']}/30
({e['poz_dogru']}/20 pozitif, {e['neg_dogru']}/10 negatif).**

Esik bastan secilmedi, **0,30-0,90 arasi 0,01 adimla supuruldu**. Her esik
degeri iki hata turunu dengeliyor: esik cok yuksekse cevabi olan soruya
"bilmiyorum" denir, cok dusukse alakasiz soruya uydurma cevap uretilir.

| esik | pozitif | negatif | toplam |
|---|---|---|---|
{supurme}

**Test seti.** 20 pozitif soru chunk'lardan *turetildi* (soru yazip cevabini
aramak degil): her sorunun `gold_chunk_id`'si kayitli, boylece erisim goz karari
degil olcum. 10 negatif soru bilerek uc gruba ayrildi - hepsi "Fenerbahce kac
sampiyonluk aldi" tipinde olsaydi esik analizi sahte cikardi, 0,2 de 0,6 da ayni
sonucu verirdi.

| grup | min | medyan | max | {e['esik']:.2f}'da dogru |
|---|---|---|---|---|
| pozitif (20) | {e['poz_min']:.4f} | {e['poz_med']:.4f} | {e['poz_max']:.4f} | {e['poz_dogru']}/20 |
{grup_satirlari}

![skor dagilimi]({ham}/gorseller/skor_dagilimi.png)
![esik supurmesi]({ham}/gorseller/esik_supurme.png)

**Neden tam {e['esik']:.2f}.** Plato ortasi secildi; bu ayni zamanda karar
bosluğunun ortasi:

```
en yuksek dogru elenen negatif : {e['en_yuksek_elenen']:.4f}
                        esik   : {e['esik']:.4f}
en dusuk pozitif               : {e['poz_min']:.4f}
```

**Erisim (recall):** gold chunk **{e['top5']}/20** soruda ilk 5 sonuc icinde
(top-1'de {e['top1']}/20). Olcut bilerek "ilk 5" - LLM'e ilk K chunk birden
veriliyor ve ayni bolumun kardes chunk'lari cogu zaman gold kadar gecerli kaynak.

**Vektor veritabani:** ChromaDB, `hnsw:space={CHROMA_MESAFE}`. Iki tuzak:
Chroma'nin varsayilan mesafesi L2'dir (acikca `cosine` verilmeli) ve Chroma
*distance* dondurur, benzerlik degil -> `benzerlik = 1 - distance`. HNSW
yaklasik arama yaptigi icin 30 sorunun tamami numpy ile kesin kosinus
hesabiyla karsilastirildi: **sapma {e['hnsw_sapmasi']}/30**.

### Esigin yapisal siniri

Esikle elenemeyen {len(e['kacan'])} negatif soru var ve sebepleri ogretici:

{kacan_satirlari}

Birincisinde korpusta sitma yok ama *"Sivrisinek Isirigina Ne Iyi Gelir?"*
makalesi var - vektor konu yakinligini goruyor. Ikincisinde kolesistektomi
korpusta **var**, sadece **fiyat bilgisi** yok.

Yani esik **konu yoklugunu** filtreliyor, **bilgi yoklugunu** filtrelemiyor.
Kosinus benzerligi "bu metin bu soruyla ayni konuda mi" sorusunu cevapliyor,
"bu metin bu sorunun cevabini iceriyor mu" sorusunu degil. Bu vektor aramanin
yapisal siniri, ayar hatasi degil; kapatmanin yolu ikinci bir kapi (LLM'e
"verilen metinde cevap yoksa bilmiyorum de" talimati + ciktinin kaynaga karsi
dogrulanmasi).

## Kod

Veri setini ureten ve olcen kodlar bu reponun `kod/` klasorunde:

| dosya | is |
|---|---|
| `ayarlar.py` | model, hastane, esik, yol sabitleri |
| `chunklama.py` | parcalama mantigi (tek basina calisirsa ornek doker) |
| `1_veri_hazirla.py` | indir -> ornekle -> chunk'la |
| `2_gom_ve_indeksle.py` | gom -> ChromaDB + parquet |
| `3_push_dataset.py` | bu repoyu olusturur |
| `4_esik_analizi.py` | 30 soruyu kos -> esik supur -> gorseller |
| `arama.py` / `ara.py` | arama katmani + komut satiri arayuzu |
| `sorular.json` | 20 pozitif (gold_chunk_id'li) + 10 negatif |
| `check_env.py` | paket / erisim / cikti kontrolu |

```bash
python 1_veri_hazirla.py && python 2_gom_ve_indeksle.py && python 4_esik_analizi.py
python ara.py "safra kesesi tasi nasil tedavi edilir"
```

Kaynak veri seti **gated**: once
[dataset sayfasinda](https://huggingface.co/datasets/{KAYNAK_DATASET})
"Agree and access" tiklanmali.
"""


def main() -> None:
    ayristirici = argparse.ArgumentParser()
    ayristirici.add_argument("--kuru", action="store_true", help="push etme, sadece dogrula")
    arg = ayristirici.parse_args()

    if not VEKTOR_PARQUET.exists():
        raise SystemExit(f"{VEKTOR_PARQUET} yok - once 2_gom_ve_indeksle.py calistirin.")

    tablo = pq.read_table(VEKTOR_PARQUET)
    ozet = dogrula(tablo)
    kart = kart_yaz(ozet)

    if arg.kuru:
        print("\n--- kuru calistirma: push edilmedi ---")
        print(kart[:600] + "...")
        return

    from datasets import Dataset
    from huggingface_hub import HfApi

    api = HfApi()

    veri = Dataset(tablo)
    print(f"\n{DATASET_REPO_ID} reposuna push ediliyor...")
    veri.push_to_hub(DATASET_REPO_ID, private=False)

    # Odev veri setiyle birlikte kodlarin da teslimini istiyor: script'ler
    # kod/ altina, esik analizi gorselleri gorseller/ altina.
    for dosya in sorted(HERE.glob("*.py")) + [HERE / "sorular.json", HERE / "requirements.txt"]:
        api.upload_file(
            path_or_fileobj=dosya,
            path_in_repo=f"kod/{dosya.name}",
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
        )
    print(f"OK  kod/ ({len(list(HERE.glob('*.py'))) + 2} dosya)")

    for gorsel in sorted(GORSELLER.glob("*.png")):
        api.upload_file(
            path_or_fileobj=gorsel,
            path_in_repo=f"gorseller/{gorsel.name}",
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
        )
    print(f"OK  gorseller/ ({len(list(GORSELLER.glob('*.png')))} dosya)")

    # README en son: gorseller yuklenmeden yazilirsa kart kirik resim gosterir.
    api.upload_file(
        path_or_fileobj=kart.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
    )
    print(f"OK  https://huggingface.co/datasets/{DATASET_REPO_ID}")


if __name__ == "__main__":
    main()
