"""Embedding modellerini olcer ve karsilastirir.

Ogrenen bellegin tek kritik sorusu su: kullanici ayni seyi BASKA KELIMELERLE
sordugunda kayittaki cevabi buluyor muyuz? Bunu "hissederek" anlayamazsiniz;
olcmeniz gerekir. Bu betik iki soru sorar:

    1. ISABET: bellekteki bir kaydin parafrazi sorulunca DOGRU kayit ilk
               sirada geliyor mu?
    2. AYRIM : parafrazlarin en dusuk skoru, bellekle ilgisiz sorularin
               en yuksek skorundan buyuk mu?

(2) negatifse hicbir esik degeri ise yaramaz: ya benzer sorulari kacirip her
seferinde internete cikarsiniz, ya da alakasiz bir kayittan cevap uretirsiniz.
Ikincisi daha kotudur — asistan emin emin yanlis cevap verir.

Olcum GERCEK bellegi kirletmez: kendi deneme koleksiyonuna tohum kayitlar yazar.

Kullanim:
    python3 olcum_karsilastirma.py                    # tum modeller
    python3 olcum_karsilastirma.py --model gemma bge
"""

import argparse

import code_rag
import ollama_client

# Gercek bellek yerine ayri bir koleksiyona yaz (bkz. code_rag.COLLECTION_PREFIX).
code_rag.COLLECTION_PREFIX = "olcum"

# Bellege tohum olarak yazilacak soru -> cevap ciftleri. Internete cikmadan,
# elle yazildi; olcumun her makinede ayni sonucu vermesi icin sabit.
TOHUM = [
    ("Python'da ModuleNotFoundError hatasi nasil cozulur?",
     "Paket kurulu degildir ya da yanlis sanal ortam aktiftir: 'pip install <paket>' calistirin "
     "ve 'which python' ile dogru venv'de oldugunuzu dogrulayin."),
    ("C# NullReferenceException neden alinir?",
     "Deger atanmamis (null) bir referans uzerinden uye erisimi yapilmistir. Erisimden once "
     "null kontrolu ekleyin ya da null-conditional operatoru (?.) kullanin."),
    ("Git'te son commit nasil geri alinir?",
     "Degisiklikleri korumak icin 'git reset --soft HEAD~1', tamamen atmak icin "
     "'git reset --hard HEAD~1' kullanilir."),
    ("Docker container neden hemen kapaniyor?",
     "Container'in ana sureci bitince container da biter. Uzun sureli calisan bir surec "
     "baslatin ya da 'docker logs <id>' ile cikis sebebine bakin."),
    ("JavaScript'te CORS hatasi nedir?",
     "Tarayici, farkli kaynaktaki bir API'ye yapilan istegi sunucu izin vermedigi icin engeller. "
     "Cozum sunucu tarafinda Access-Control-Allow-Origin basligini dondurmektir."),
    ("Postgres'te 'too many connections' hatasi neden olur?",
     "Acik baglanti sayisi max_connections sinirini asmistir. Baglanti havuzu kullanin ya da "
     "sizinti yapan baglantilari kapatin."),
]

# Tohumdaki sorularin PARAFRAZI — kullanici ayni seyi boyle sorarsa da bulmaliyiz.
# (parafraz, beklenen tohum sorusunun icindeki bir parca)
PARAFRAZ = [
    ("python modulu bulunamiyor hatasi aliyorum ne yapmaliyim", "ModuleNotFoundError"),
    ("c sharp null referans istisnasi nasil duzeltilir", "NullReferenceException"),
    ("yanlislikla commit attim geri almak istiyorum", "son commit"),
    ("docker konteynerim aninda duruyor sebebi ne", "Docker container"),
    ("tarayicida cross origin engeli aliyorum", "CORS"),
    ("veritabanina cok fazla baglanti hatasi", "too many connections"),
]

# Bellekle hicbir ilgisi olmayan sorular: esigin ALTINDA kalmalilar ki
# asistan bunlar icin internete ciksin.
#
# Ilk grup kolay (baska bir dunya). Asil sinav ikinci grup: AYNI ALANDAN ama
# bellekte OLMAYAN sorular. Bir soru-cevap bellegini bozan sey alakasiz soruya
# cevap vermesi degil, "yakin ama ayni olmayan" bir kayittan cevap uydurmasidir
# — kullanici Rust sorunca Python cevabini almasi gibi.
ALAKASIZ = [
    "Bugun Istanbul'da hava nasil?",
    "Dolar kuru ne kadar?",
    "Besiktas kac kupa kazandi?",
    "Sagopa Kajmer'in en iyi albumu hangisi?",
    "Kuantum bilgisayar nedir?",
    # yakin ama bellekte yok:
    "Rust borrow checker hatasi nasil cozulur?",
    "Kubernetes pod'u Pending durumunda kaliyor neden?",
    "npm install EACCES izin hatasi veriyor",
    "Java'da OutOfMemoryError nasil giderilir?",
]

parser = argparse.ArgumentParser(description="Embedding modellerini karsilastir.")
parser.add_argument(
    "--model",
    nargs="+",
    default=list(ollama_client.EMBED_MODELS),
    choices=list(ollama_client.EMBED_MODELS),
    help="Olculecek modeller",
)
args = parser.parse_args()

for embed_key in args.model:
    config = ollama_client.EMBED_MODELS[embed_key]
    print(f"\n=== {embed_key}  ({config['name']}) ===")

    # Her olcumde temiz baslangic: deneme koleksiyonunu sifirla ve tohumla.
    code_rag.forget_all(embed_key)
    for question, answer in TOHUM:
        code_rag.remember(question, answer, source="tohum", embed_key=embed_key)

    hits = 0
    in_scores = []
    for question, expected in PARAFRAZ:
        top = code_rag.search(question, embed_key, k=1)[0]
        correct = expected.lower() in top["question"].lower()
        hits += correct
        in_scores.append(top["similarity"])
        print(f"  {'OK ' if correct else 'YANLIS'} {top['similarity']:.3f}  "
              f"{question[:38]:40} -> {top['question'][:36]}")

    out_scores = [(code_rag.search(q, embed_key, k=1)[0]["similarity"], q) for q in ALAKASIZ]
    out_scores.sort(reverse=True)
    print("  --- bellekte olmamasi gerekenlerin en yuksek 3 skoru")
    for score, question in out_scores[:3]:
        print(f"     {score:.3f}  {question}")

    en_yuksek_alakasiz = out_scores[0][0]
    gap = min(in_scores) - en_yuksek_alakasiz
    # Guvenli esik: hicbir alakasiz soru gecmesin. Bunun bedeli, esigin altinda
    # kalan parafrazlar — onlar icin internete cikariz, yani hatanin ucuz olani.
    guvenli = en_yuksek_alakasiz + 0.02
    kacan = sum(1 for s in in_scores if s < guvenli)
    print(f"  ---")
    print(f"  Isabet@1         : {hits}/{len(PARAFRAZ)}")
    print(f"  Parafraz en dusuk: {min(in_scores):.3f}  (ortalama {sum(in_scores)/len(in_scores):.3f})")
    print(f"  Alakasiz en yuks.: {en_yuksek_alakasiz:.3f}")
    print(f"  AYRIM            : {gap:+.3f}  ->  "
          f"{'tek esik ayirir' if gap > 0 else 'tek esik AYIRAMAZ (ustteki listeye bak)'}")
    print(f"  Guvenli esik     : {guvenli:.2f}  "
          f"({kacan}/{len(PARAFRAZ)} parafraz kacar, su anki: {config['min_similarity']})")
