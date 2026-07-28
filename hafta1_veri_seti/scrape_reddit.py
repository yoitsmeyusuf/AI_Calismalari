"""
1. Hafta: r/felsefe'den Playwright (headless Chromium) ile Türkçe soru-cevap
verisi toplar. Hesap/OAuth app kaydı GEREKMİYOR.

Neden Playwright: Reddit artık hem OAuth'suz .json uç noktalarını hem de
old.reddit.com'u 403 ile kapatıyor; yeni www.reddit.com ise JS ile çözülen
bir "please wait for verification" duvarına sahip (plain `requests` bunu
geçemez, gerçek bir tarayıcı JS motoru gerekir). Headless Chromium bu
duvarı otomatik geçiyor.

Neden r/felsefe: İlk denemede r/Turkey kullanıldı ama "top" listesi ağırlıklı
siyasi gündem/tepki-avı çıktı (hakaret içeren yorumlar dahil) — soru-cevap
formatına uymuyordu. r/AskTurkey de denendi ama ağırlıklı İngilizce çıktı.
r/felsefe test edildi: gerçek Türkçe felsefi tartışma var, siyasi çamur yok.

Kaynaklar (SOURCES): genel "top" listesine ek olarak, community'nin kendi
flair'lediği üç alt-küme de ayrı birer kaynak: "bilim • philosophy of
science", "düşünürler, düşünceler, düşünmeler", "eseme • logic". Bunlar
zaten insan-etiketli (moderasyon/community tarafından konuya sabitlenmiş)
olduğu için genel listeden daha az gürültülü çıkıyor — ilk temizlik turunda
genel "top" listesinden gelen 33 satırın 27'sini atmıştık (siyaset, flört
sohbeti vb.), flair'ler bu riski azaltır. Flair sayfalarındaki başlıklar her
zaman "?" ile bitmiyor (ör. "Kalabalığın Bilgeliği"), bu yüzden
REQUIRE_QUESTION_FORMAT sadece "top" kaynağı için zorunlu (bkz. SOURCES'daki
require_question_title); flair kaynaklarında başlık soru değilse post
atlanmıyor, sadece o başlık top-level "soru" olarak kullanılmıyor — yorum
ağacından soru-cevap madenciliği (extract_pairs) yine de çalışıyor.

r/felsefe küçük bir subreddit (top/new sıralamalarının ikisi de ~25 gönderide
düzlüğe çıkıyor, yani bu subreddit'ten çekilebilecek gönderi sayısı sınırlı).
Bu yüzden her gönderiden sadece 1 çift (başlık + en iyi yorum) almak yerine,
yorum ağacının içine de iniyoruz: bir yorum soru gibiyse (is_question) ve ona
verilmiş bir cevap varsa, bunu da ayrı bir (soru, cevap) çifti olarak alıyoruz
— aynı ~25 gönderiden çok daha fazla çift çıkarmış oluyoruz.

Çıktı: data/raw/scraped_turkish_qa.jsonl (satır başına {"soru","cevap","source_url"}
— kullanıcı adı gibi kişisel bilgi tutulmuyor).

Kurulum (bir kere):
    uv pip install --python ../.venv/bin/python playwright
    ../.venv/bin/python -m playwright install chromium

Çalıştırma:
    .venv/bin/python hafta1_veri_seti/scrape_reddit.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playwright.sync_api import sync_playwright

SUBREDDIT = "felsefe"
TIME_FILTER = "all"  # sort=top için: hour/day/week/month/year/all

# Her kaynak: {name, url, require_question_title}. require_question_title=True
# olan kaynaklarda başlık soru gibi görünmüyorsa post tamamen atlanır (genel
# "top" listesi için gerekli — konu dışı sızıntıları önler, test ettik).
# Flair kaynaklarında bu kapalı: community zaten konuyu sabitlemiş, başlık
# soru olmasa bile yorum ağacından soru-cevap çifti çıkarmaya değer.
def _flair_url(encoded_name: str) -> str:
    return f"https://www.reddit.com/r/{SUBREDDIT}/?f=flair_name%3A%22{encoded_name}%22"


SOURCES = [
    {
        "name": "top-all",
        "url": f"https://www.reddit.com/r/{SUBREDDIT}/top/?t={TIME_FILTER}",
        "require_question_title": True,
    },
    {
        # "top" sadece en çok oylanan gönderileri gösteriyor — az oylanmış ama
        # gerçek felsefi içeriği kaçırmamak için "new" sıralaması da eklendi.
        "name": "new",
        "url": f"https://www.reddit.com/r/{SUBREDDIT}/new/",
        "require_question_title": True,
    },
    {
        "name": "flair-bilim",
        "url": _flair_url("bilim%20%E2%80%A2%20philosophy%20of%20science"),
        "require_question_title": False,
    },
    {
        "name": "flair-dusunurler",
        "url": _flair_url(
            "d%C3%BC%C5%9F%C3%BCn%C3%BCrler%2C%20d%C3%BC%C5%9F%C3%BCnceler%2C%20d%C3%BC%C5%9F%C3%BCnmeler"
        ),
        "require_question_title": False,
    },
    {
        "name": "flair-eseme-logic",
        "url": _flair_url("eseme%20%E2%80%A2%20logic"),
        "require_question_title": False,
    },
    {
        "name": "flair-bilgi-epistemology",
        "url": _flair_url("bilgi%20%E2%80%A2%20epistemology"),
        "require_question_title": False,
    },
    {
        "name": "flair-yasamin-icinden-axiology",
        "url": _flair_url("ya%C5%9Fam%C4%B1n%20i%C3%A7inden%20%E2%80%A2%20axiology"),
        "require_question_title": False,
    },
    {
        "name": "flair-inanc-philosophy-of-religion",
        "url": _flair_url("inan%C3%A7%20%E2%80%A2%20philosophy%20of%20religion"),
        "require_question_title": False,
    },
    {
        # Siyaset felsefesi (Panarşizm, Marx vb.) — güncel siyaset/parti gündemi
        # değil; yine de OFFTOPIC_WORDS güncel-siyaset kelimeleri için güvenlik ağı.
        "name": "flair-yonetim-philosophy-of-politics",
        "url": _flair_url("y%C3%B6netim%20%E2%80%A2%20philosophy%20of%20politics"),
        "require_question_title": False,
    },
    {
        "name": "flair-iyilik-ethics",
        "url": _flair_url("%C2%ABiyilik%C2%BB%20%C3%BCzerine%20%E2%80%A2%20ethics"),
        "require_question_title": False,
    },
    {
        "name": "flair-varlik-ontology",
        "url": _flair_url("varl%C4%B1k%20%E2%80%A2%20ontology"),
        "require_question_title": False,
    },
    {
        "name": "flair-guzellik-aesthetics",
        "url": _flair_url("%C2%ABg%C3%BCzellik%C2%BB%20%C3%BCzerine%20%E2%80%A2%20aesthetics"),
        "require_question_title": False,
    },
    {
        # "güldürü" (mizah) flair'i bilerek dışarıda bırakıldı: çok küçük (~3
        # gönderi) ve şaka/sarkazm ağırlıklı, kalite riski faydasından yüksek.
        "name": "flair-subreddit-meta",
        "url": _flair_url("%2Fr%2Ffelsefe%E2%80%99ye%20de%C4%9Fgin"),
        "require_question_title": False,
    },
]
POST_LIMIT = 100  # kaynak başına üst sınır (flair sayfaları zaten ~28 gönderide düzlüğe çıkıyor)
MAX_SCROLL_ROUNDS = 40
SOURCE_DELAY_MS = 4000  # kaynaklar arası bekleme — çok sayıda kaynağı art arda çekince throttling'e girdik, test ettik

MIN_COMMENT_SCORE = 5  # 3 -> 5: düşük onaylı, tek kelimelik şaka cevapları eler (temizlikte gördük)
MIN_COMMENT_LEN = 20
MIN_TITLE_LEN = 10
MAX_QUESTION_LEN = 300  # gerçek sorular kısadır; bundan uzunu genelde rant/monolog (soru değil) çıkıyor
MAX_ANSWER_CHARS = 6000  # trainer'daki MAX_SEQ_LENGTH (2048 token) için güvenli üst sınır
MAX_THREAD_PAIRS_PER_POST = 6  # tek bir uzun tartışmanın veri setine hakim olmasını önler

OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "raw" / "scraped_turkish_qa.jsonl"
# Manuel kalite incelemesinde elenmiş id'ler — dosyadan silinseler bile
# scraper'ın aynı id'yi tekrar eklememesi için ayrıca burada tutulur (id sadece
# OUTPUT_PATH'te "şu an var mı" diye kontrol edilseydi, temizlik sonrası tekrar
# scrape çalıştırıldığında elenen satırlar geri gelirdi — canlı testte gördük).
REJECTED_IDS_PATH = Path(__file__).resolve().parent / "data" / "raw" / "rejected_ids.txt"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")
TURKISH_STOPWORDS = {
    "bir", "ve", "bu", "de", "da", "için", "ile", "gibi", "çok", "ne",
    "nasıl", "neden", "mi", "mı", "mu", "mü", "var", "yok", "ben", "sen",
    "ama", "her", "en", "daha", "kadar", "olan", "olarak", "diye",
}

# r/Turkey'nin "top" listesi ağırlıklı olarak siyasi gündem/haber başlıklarından
# oluşuyor (soru değil, tepki-avı) — bu yüzden gerçekten SORU olan başlıkları
# ayrıca filtreliyoruz.
QUESTION_WORDS = {
    "nasıl", "neden", "niye", "niçin", "ne", "hangi", "kaç", "kim",
    "nerede", "nereye", "nereden", "mı", "mi", "mu", "mü",
}

# Kaba/hakaret içeren yorumları elemek için kaba bir blocklist (tam kapsamlı
# değil, ama en azından açık hakaret/küfür içeren yanıtları eler).
BLOCKLIST_WORDS = {
    "sürtük", "orospu", "piç", "pic", "yavşak", "şerefsiz", "namussuz",
    "ibne", "gerizekalı", "kaltak", "amk", "aq", "sikim", "siktir", "sikik",
}

# İlk temizlikte gördük: r/felsefe küçük bir topluluk olduğu için güncel
# siyaset, flört/görünüş sohbeti, oyun fiyatı gibi konu dışı thread'ler de
# "soru gibi" görünüp sızabiliyor. Bunları başlık/yorum seviyesinde eleriz —
# domain'i (felsefe) gerçekten kaydırmak için bu filtre önemli.
OFFTOPIC_WORDS = {
    "iktidar", "muhalefet", "seçim", "cumhurbaşkanı", "parti", "chp", "akp",
    "zam", "enflasyon", "lira", "fiyat", "dolar",
    "flört", "sevgili", "çekicilik", "görünüş", "kıyafet",
    "meme",
}

# looks_turkish() Türkçe karakter/stopword VARLIĞINA bakıyor ama uzun bir
# İngilizce alıntı bloğu (ör. bir kaynaktan kopyalanmış paragraf) içeren bir
# yanıt da birkaç Türkçe kelimeyle bu testi geçebiliyor — test ettik, bir
# Machiavelli alıntısı İngilizce olarak sızdı. Bu yüzden ayrıca İngilizce
# kirliliğini de kontrol ediyoruz.
ENGLISH_STOPWORDS = {
    "the", "and", "of", "is", "are", "his", "her", "that", "this", "will",
    "always", "have", "with", "for", "not", "was", "were", "their", "they",
}

WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def has_english_contamination(text: str) -> bool:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    hits = sum(1 for w in words if w in ENGLISH_STOPWORDS)
    return hits >= 4


def looks_turkish(text: str) -> bool:
    if has_english_contamination(text):
        return False
    if any(ch in TURKISH_CHARS for ch in text):
        return True
    words = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+", text.lower())
    hits = sum(1 for w in words if w in TURKISH_STOPWORDS)
    return hits >= 2


def is_question(title: str) -> bool:
    t = title.strip()
    if t.endswith("?"):
        return True
    words = set(re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+", t.lower()))
    return bool(words & QUESTION_WORDS)


def is_strict_question(text: str) -> bool:
    """is_question()'dan daha sıkı: yorum ağacından "soru" çıkarırken kullanılır.
    Sadece bir kelime içermek (ör. gövdenin içinde bir yerde "mi" geçmesi)
    uzun bir rant/monoloğu soru sanmamıza yol açıyordu (temizlikte gördük) —
    burada gerçekten "?" ile biten VE kısa (gerçek bir soru gibi) metin isteriz.
    """
    t = text.strip()
    return t.endswith("?") and len(t) <= MAX_QUESTION_LEN


def has_blocked_words(text: str) -> bool:
    words = set(re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+", text.lower()))
    return bool(words & BLOCKLIST_WORDS)


def has_offtopic_words(text: str) -> bool:
    words = set(re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+", text.lower()))
    return bool(words & OFFTOPIC_WORDS)


def trim_to_sentence(text: str, limit: int = MAX_ANSWER_CHARS) -> str:
    """Cevabı limit'e kadar keser, son cümle sonunda durur (trainer'daki
    MAX_SEQ_LENGTH token bütçesini aşan çok uzun cevapları önlemek için)."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    cut_points = [m.end() for m in re.finditer(r"[.!?]\s", window)]
    cut = cut_points[-1] if cut_points else limit
    return window[:cut].strip()


def collect_posts(page, url: str) -> list[dict]:
    # Flair/arama filtreli sayfalar (?f=flair_name%3A...) düz /top/ listesine
    # göre daha yavaş/dengesiz hidrate oluyor (sanal liste), tek deneme bazen
    # timeout'a giriyor (canlı testte gördük) — bu yüzden bir kez retry var.
    for attempt in range(2):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_selector("shreddit-post", timeout=20000)
            break
        except Exception:
            if attempt == 1:
                raise
            page.wait_for_timeout(5000)  # retry'den önce throttling'in geçmesi için bekle
    page.wait_for_timeout(1500)

    prev_count = -1
    stale_rounds = 0
    for _ in range(MAX_SCROLL_ROUNDS):
        count = page.eval_on_selector_all("shreddit-post", "els => els.length")
        if count >= POST_LIMIT:
            break
        stale_rounds = stale_rounds + 1 if count == prev_count else 0
        if stale_rounds >= 3:
            break  # küçük subreddit'lerde daha fazla gönderi yok, boşuna scroll etme
        prev_count = count
        page.mouse.wheel(0, 3000)
        page.wait_for_timeout(1000)

    posts = []
    for el in page.query_selector_all("shreddit-post"):
        attrs = page.evaluate(
            "(el) => ({id: el.getAttribute('id'), permalink: el.getAttribute('permalink'), "
            "title: el.getAttribute('post-title'), lang: el.getAttribute('post-language'), "
            "stickied: el.hasAttribute('is-created-from-ads-ui')})",
            el,
        )
        posts.append(attrs)
        if len(posts) >= POST_LIMIT:
            break
    return posts


def collect_comments(page) -> list[dict]:
    """Sayfadaki yorumları thingid/parentid ile birlikte (thread yapısını
    kaybetmeden) toplar ve temel kalite filtrelerini uygular."""
    raw = page.evaluate(
        """() => Array.from(document.querySelectorAll('shreddit-comment')).map(el => {
            const slot = el.querySelector('[slot="comment"]');
            return {
                thingid: el.getAttribute('thingid'),
                parentid: el.getAttribute('parentid'),
                author: el.getAttribute('author'),
                score: el.getAttribute('score'),
                body: slot ? slot.innerText : ''
            };
        })"""
    )

    comments = []
    for r in raw:
        if not r.get("thingid") or r.get("author") in (None, "AutoModerator"):
            continue
        body = clean_text(r.get("body") or "")
        if not body or body in ("[deleted]", "[removed]") or len(body) < MIN_COMMENT_LEN:
            continue
        if has_blocked_words(body) or not looks_turkish(body):
            continue
        try:
            score = int(r.get("score") or 0)
        except (TypeError, ValueError):
            score = 0
        comments.append({"thingid": r["thingid"], "parentid": r.get("parentid"), "score": score, "body": body})
    return comments


def extract_pairs(page, permalink: str) -> list[tuple[str | None, str, str]]:
    """(soru, cevap, id) çiftleri döner. soru=None -> çağıran, post başlığını
    kullanmalı (en iyi top-level yorum, gönderiye doğrudan cevaptır).
    Ayrıca thread içinde soru soran yorumlar + en iyi cevapları da eklenir."""
    try:
        page.goto(f"https://www.reddit.com{permalink}", wait_until="domcontentloaded", timeout=20000)
        page.wait_for_selector("shreddit-comment", timeout=10000)
    except Exception:
        return []  # bu gönderiyi atla, tüm scrape'i durdurma
    page.wait_for_timeout(1000)

    comments = collect_comments(page)
    if not comments:
        return []

    by_parent: dict[str | None, list[dict]] = {}
    for c in comments:
        by_parent.setdefault(c["parentid"], []).append(c)

    pairs: list[tuple[str | None, str, str]] = []

    top_level = sorted(by_parent.get(None, []), key=lambda c: c["score"], reverse=True)
    if (
        top_level
        and top_level[0]["score"] >= MIN_COMMENT_SCORE
        and not has_offtopic_words(top_level[0]["body"])
    ):
        pairs.append((None, trim_to_sentence(top_level[0]["body"]), top_level[0]["thingid"]))

    thread_pairs = 0
    for c in comments:
        if thread_pairs >= MAX_THREAD_PAIRS_PER_POST:
            break
        if not is_strict_question(c["body"]) or has_offtopic_words(c["body"]):
            continue
        replies = sorted(by_parent.get(c["thingid"], []), key=lambda r: r["score"], reverse=True)
        if not replies or replies[0]["score"] < MIN_COMMENT_SCORE or has_offtopic_words(replies[0]["body"]):
            continue
        pairs.append((c["body"], trim_to_sentence(replies[0]["body"]), replies[0]["thingid"]))
        thread_pairs += 1

    return pairs


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    seen_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    if REJECTED_IDS_PATH.exists():
        with open(REJECTED_IDS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    seen_ids.add(line)

    written = 0
    seen_post_ids: set[str] = set()  # aynı post birden fazla kaynakta (top + flair) çıkabilir
    with sync_playwright() as p, open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)

        for source in SOURCES:
            page.wait_for_timeout(SOURCE_DELAY_MS)  # kaynaklar arası kibar bir bekleme (rate-limit'i tetiklememek için)
            try:
                posts = collect_posts(page, source["url"])
            except Exception as e:
                # Bir kaynağın (ör. rate-limit/timeout) çökmesi diğer kaynakları
                # engellemesin — canlı testte peş peşe çok istekten sonra bir
                # kaç flair sayfası timeout verdi, muhtemelen geçici throttling.
                print(f"[{source['name']}] UYARI: gönderi listesi alınamadı ({e!r}), atlanıyor")
                continue
            print(f"[{source['name']}] {len(posts)} gönderi bulundu, işleniyor...")

            for post in posts:
                if post.get("stickied"):
                    continue
                post_id = post.get("id")
                if post_id:
                    if post_id in seen_post_ids:
                        continue
                    seen_post_ids.add(post_id)

                post_title = clean_text(post["title"] or "")
                if len(post_title) < MIN_TITLE_LEN:
                    continue
                title_is_question = is_question(post_title)
                if source["require_question_title"] and not title_is_question:
                    continue
                if post.get("lang") != "tr" and not looks_turkish(post_title):
                    continue
                if has_offtopic_words(post_title):
                    continue
                if has_blocked_words(post_title):
                    continue

                pairs = extract_pairs(page, post["permalink"])

                for soru, cevap, pair_id in pairs:
                    if pair_id in seen_ids:
                        continue
                    if soru is None:
                        if not title_is_question:
                            continue  # başlık soru gibi değilse post başlığını "soru" olarak kullanma
                        soru = post_title
                    row = {
                        "id": pair_id,
                        "soru": soru,
                        "cevap": cevap,
                        "source_url": f"https://www.reddit.com{post['permalink']}",
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    written += 1
                    seen_ids.add(pair_id)

                if written % 10 == 0:
                    print(f"  {written} çift toplandı...")

        browser.close()

    print(f"Toplam {written} yeni soru-cevap çifti eklendi -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
