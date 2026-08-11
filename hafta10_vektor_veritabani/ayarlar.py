"""
10. Hafta: tum adimlarin paylastigi sabitler.

Model, hastane secimi ve dosya yollari tek yerde dursun ki 1-4 arasindaki
script'ler birbirinden ayrilmasin (ornegin arama, indeksleme ile ayni prompt'u
kullanmak zorunda).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERI = HERE / "veri"
GORSELLER = HERE / "gorseller"
CHROMA_DIZINI = HERE / "chroma_db"

CHUNK_JSONL = VERI / "chunks.jsonl"
VEKTOR_PARQUET = VERI / "chunks_vektorlu.parquet"
SORULAR_JSON = HERE / "sorular.json"
ESIK_SONUC_JSON = VERI / "esik_sonuclari.json"

# --- Kaynak veri ---
KAYNAK_DATASET = "umutertugrul/turkish-hospital-medical-articles"

# Uc farkli yapisal profil secildi (14 split olculdukten sonra):
#   guven    -> icindekiler menulu, doktor alintili, makale basina 7.9 bolum
#   medipol  -> menusuz, ince bolumlu (bolum medyani 301 karakter)
#   florence -> menusuz, kalin bolumlu (bolum medyani 568 karakter)
# liv elendi: 2836 satirin sadece 982'si benzersiz (%65 tekrar).
HASTANELER = ["guven", "medipol", "florence"]
HASTANE_BASINA_MAKALE = 134  # 3 x 134 = 402 makale (odev araligi 100-1000)
ORNEKLEM_TOHUMU = 42

# --- Embedding ---
# 768 boyutlu, L2-normalize cikti; 8192 token context. Turkce'ye ozel tokenizer
# oldugu icin 5.77 karakter/token veriyor (cok dilli modellerde ~3.2).
EMBEDDING_MODELI = "magibu/embeddingmagibu-200m"
VEKTOR_BOYUTU = 768

# Model asimetrik: soru ve dokuman farkli prompt'lardan geciyor.
#   query    -> "task: search result | query: "
#   document -> "title: none | text: "
# Bu atlanirsa model yine calisir ama skor dagilimi kayar ve esik analizi
# anlamsizlasir. Prompt adlari modelin config_sentence_transformers.json'undan.
SORGU_PROMPT = "query"
DOKUMAN_PROMPT = "document"

GOMME_BATCH = 64  # 200M model, 8 GB VRAM: rahat siger

# Model varsayilan olarak bfloat16 yukleniyor; o hassasiyette L2 normlar
# 0.996-1.004 arasinda kayiyor ve "kosinus = nokta carpimi" varsayimi tam
# tutmuyor. 200M parametre float32'de ~800 MB - 8 GB VRAM'de sorun degil,
# karsiliginda skorlar tekrarlanabilir oluyor (bf16 ile farki 7e-5 kosinus).
GOMME_DTYPE = "float32"

# --- Vektor veritabani ---
KOLEKSIYON_ADI = "tibbi_chunklar"
# Chroma'nin varsayilan mesafesi L2; kosinus icin acikca belirtmek gerekiyor.
# Ayrica Chroma "distance" dondurur, benzerlik degil: benzerlik = 1 - distance.
CHROMA_MESAFE = "cosine"

# --- Arama / esik ---
VARSAYILAN_K = 5
ESIK_TARAMA = (0.30, 0.90, 0.01)

# 4_esik_analizi.py'nin olctugu deger. 30 soruluk sette 0.55-0.56 platosu
# 28/30 veriyor (20/20 pozitif, 8/10 negatif); plato ortasi secildi ve bu
# ayni zamanda karar bosluginin ortasi:
#     en yuksek dogru elenen negatif : 0.5438 (kuduz asisi dozu)
#     en dusuk pozitif               : 0.5696 (sepsis ilk mudahale)
# Kalan 2 hata esikle cozulemiyor - detay README'de.
BENZERLIK_ESIGI = 0.56

RET_MESAJI = "Bu sorunun cevabı dokümanlarımda yer almamaktadır."

# --- Teslim ---
DATASET_REPO_ID = "yoitsmeyusuf/turkce-tibbi-vektor-veri-seti"
