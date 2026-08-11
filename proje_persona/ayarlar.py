"""
Proje: Kendi Karakteristik AI'ni Yarat - tum asamalarin paylastigi sabitler.

Dort alt klasor (1_fastapi, 2_transformer, 3_rag_persona, 4_mini_gpt) ayni
korpusu ve ayni sabitleri kullaniyor. Klasor adlari rakamla basladigi icin
paket olarak import edilemiyorlar; alt projeler birbirinin modulunu
sys.path uzerinden aliyor (bkz. 4_mini_gpt/model.py).
"""
from pathlib import Path

KOK = Path(__file__).resolve().parent
VERI = KOK / "veri"

FASTAPI = KOK / "1_fastapi"
TRANSFORMER = KOK / "2_transformer"
RAG = KOK / "3_rag_persona"
MINI_GPT = KOK / "4_mini_gpt"

# --- Kaynak veri ---------------------------------------------------------
# Scraping yerine hazir veri seti kullaniliyor (odev ikisine de izin veriyor).
# Olculen durum:
#   genius.com, lyricstranslate.com, sarkisozlerihd.com -> Cloudflare bot duvari
#   alternatifim.com -> robots.txt erisime acik AMA
#                       "Content-Signal: ai-train=no" ile AI egitimi icin
#                       veri toplamayi acikca reddediyor.
# Bu yuzden MIT lisansli bir HF veri seti tercih edildi. Lisans derlemeye ait;
# sarki sozlerinin telifi hak sahiplerinde, o yuzden korpus git'e girmiyor ve
# HF'ye push EDILMIYOR (9. ve 10. haftadan bilincli ayrilma).
KAYNAK_DATASET = "metncelik/turkish-song-lyrics"

# Iki CSV var ve ayrimi kritik:
#   songs.csv             -> islenmis, satir sonlari SILINMIS (medyan 1 satir)
#   songs-unprocessed.csv -> ham, satir yapisi KORUNMUS (medyan 24 satir)
# datasets-server API'si "default" config'de ikisini birlestirip sunuyor;
# o yuzden sanatci basina satir sayisi iki katina cikmis gorunuyor (263 -> 526).
# Bar sinirindan chunk'lamak da, MiniGPT'nin okunabilir metin uretmesi de
# satir yapisina bagli oldugu icin ham dosya kullaniliyor.
KAYNAK_DOSYA = "songs-unprocessed.csv"

SANATCI = "Sagopa Kajmer"

# Olculen korpus profili (bkz. README "Korpus" bolumu):
#   262 sarki, 268.183 karakter (262 KB), tekrar yok
#   satir/sarki : medyan 24, p10 20, max 82
#   satir uzunlugu : medyan 39, p90 54 karakter
#   vocab_size : 115, bunun 29'u 10'dan az geciyor
#   yapisal etiket ([Nakarat], (x2), bos satirli kita ayrimi) YOK
KORPUS_TXT = VERI / "sagopa.txt"
KORPUS_JSONL = VERI / "sarkilar.jsonl"

# --- Chunking (3_rag_persona) -------------------------------------------
# Kita ayrimi olmadigi icin sinir belgeden okunamiyor, kurulmak zorunda.
# Bar medyani 39 karakter oldugundan 6 barlik grup ~230 karaktere denk
# geliyor; odevin istedigi 200-400 araligina dogal olarak oturuyor.
CHUNK_BAR = 6
CHUNK_BAR_MIN = 4
CHUNK_KARAKTER_HEDEFI = (200, 400)

# --- Embedding / vektor DB ----------------------------------------------
# 10. haftadan devir: Turkce'ye uyarlanmis, 768 boyut, asimetrik prompt.
EMBEDDING_MODELI = "magibu/embeddingmagibu-200m"
VEKTOR_BOYUTU = 768
SORGU_PROMPT = "query"
DOKUMAN_PROMPT = "document"
GOMME_DTYPE = "float32"  # bf16'da L2 normlar kayiyor (10. hafta bulgusu)

CHROMA_DIZINI = RAG / "chroma_db"
KOLEKSIYON_ADI = "sagopa_sozler"
CHROMA_MESAFE = "cosine"  # varsayilan L2; benzerlik = 1 - distance

VARSAYILAN_K = 3  # odev "en alakali 2-3 sarki sozu" diyor
ESIK_TARAMA = (0.20, 0.90, 0.01)

# 2_esik_analizi.py'nin olctugu deger. Iki supurme yapiliyor:
#   butun pozitifler (turetilmis + dogal, 30) -> 0.35  (37/41)
#   yalniz dogal kullanici mesajlari (10)     -> 0.38  (18/21)
# Isletme esigi ikincisinden aliniyor: turetilmis sorgular (nadir kelime
# torbasi) bu korpusta gold chunk'i iyi bulamiyor (ilk5'te 7/20) ve skorlari
# dogal mesajlardan dusuk; onlari pozitif saymak esigi yapay olarak asagi
# cekiyor. Uygulamanin gercek girdisi dogal mesaj.
# Karar boslugu:
#   en yuksek dogru elenen negatif : 0.3495
#                           esik   : 0.3800
#   en dusuk gecen pozitif         : 0.4141
BENZERLIK_ESIGI = 0.38

# Guardrail: yanit, getirilen chunk'lardan bu kadar karakter birebir
# kopyalarsa reddedilir. Sagopa'nin bar medyani 39 karakter; 40 "tam bir
# dize" demek. Odev "sarki sozlerini dogrudan yapistirmadan yorumla" diyor -
# bu kural prompt'ta degil, harness'ta.
KOPYA_ESIGI = 40
GUARDRAIL_DENEME = 2

# --- MiniGPT (4_mini_gpt) -----------------------------------------------
# Odevin onerdigi hiperparametreler. Korpus 262 KB, yani odevin onerdigi
# 500 KB-1 MB'in yarisi: 5000 iter x 64 batch x 128 block = ~41M token, bu
# da 236K karakterlik egitim verisi uzerinde ~174 epoch demek. Ezber
# bekleniyor; bu yuzden ikinci bir kosu veriye olcekli ayarlarla yapiliyor.
GPT_TABAN = dict(n_embd=128, n_head=4, n_layer=3, block_size=128, dropout=0.0)
GPT_OLCEKLI = dict(n_embd=96, n_head=4, n_layer=2, block_size=128, dropout=0.2)

GPT_BATCH = 64
GPT_ITER = 5000
GPT_LR = 1e-3
GPT_DEGERLENDIRME_ARALIGI = 500
GPT_EGITIM_ORANI = 0.9
# 10'dan az gecen 29 karakter olu embedding satiri olur; <NADIR> ile birlesir.
GPT_NADIR_ESIGI = 10

# --- LLM arka ucu (3_rag_persona) ---------------------------------------
# 7. ve 8. haftadaki desen: HF Inference Providers uzerinden OpenAI uyumlu
# cagri. .env'deki TOOL_MODEL / HF_TOKEN ile ayarlanir.
LLM_MODEL_VARSAYILAN = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

# Kredi bitince (402) ya da TOOL_BACKEND=yerel iken kullanilan model.
# ONCEDEN 4bit'lenmis surum: 7. ve 8. haftadan cache'te hazir (5.2 GB) ve
# 8 GB VRAM'e rahat siger. Tam hassasiyetli Qwen/Qwen2.5-7B-Instruct'i
# indirip yerelde kirpmak ~15 GB indirme demekti.
LLM_MODEL_YEREL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

# --- FastAPI (1_fastapi ve 3_rag_persona) -------------------------------
API_HOST = "127.0.0.1"
API_PORT = 8000
CORS_DEMO_PORT = 8080  # farkli origin olsun diye ayri port
