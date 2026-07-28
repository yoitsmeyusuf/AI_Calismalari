"""
1. Hafta: Ekşi Sözlük'ten (hesap/login/API GEREKMİYOR) Türkçe soru-cevap verisi toplar.

Domain: Felsefe (düşünürler, akımlar, temel kavramlar) — r/Turkey denemesinde
çıkan siyasi çamur atma sorununun aksine, bu başlıklardaki içerik gerçekten
teorik/kavramsal (test ettik: 27/27 aday başlık gerçek içerikle döndü).

Neden Ekşi Sözlük: Reddit'in hem OAuth API'si (hesap+app kaydı ister) hem de
anonim .json görünümü (artık login'siz istekleri genel olarak 403 ile
engelliyor — test ettik) hesapsız kullanım için uygun değil. Ekşi Sözlük'ün
normal başlık sayfaları herkese açık ve login istemiyor.

Mantık: TOPIC_SLUGS içindeki her başlık (ör. "marksizm--83324") bir "soru"
(başlık metni) olarak, o başlıktaki en çok favorilenen entry'ler de "cevap"
olarak alınır. "(bkz: ...)" gibi salt-referans entry'ler ve çok kısa
olanlar elenir.

Çıktı: data/raw/scraped_turkish_qa.jsonl (satır başına {"soru","cevap","source_url"}
— yazar adı gibi kişisel bilgi tutulmuyor).

Çalıştırma:
    .venv/bin/python hafta1_veri_seti/scrape_eksisozluk.py
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from bs4 import BeautifulSoup

# Felsefe — düşünürler, akımlar, temel kavramlar. Ekşi Sözlük başlıkları
# genelde çıplak isim/kavramdır ("nietzsche", "etik"), gerçek bir soru değil
# — bu yüzden her slug için "soru" metnini elle eşliyoruz. Slug'ı bulmak
# için: eksisozluk.com'da başlığı açın, URL'nin sonundaki "baslik-adi--12345"
# kısmını kopyalayın. Genişletmek isterseniz: "spinoza", "hume", "locke",
# "hobbes", "confucius", "budizm", "taoizm", "pragmatizm", "fenomenoloji",
# "postmodernizm", "michel foucault" gibi başlıklar da mevcut.
TOPIC_SLUGS = {
    "nietzsche--34013": "Nietzsche kimdir?",
    "sokrates--33212": "Sokrates kimdir?",
    "platon--35529": "Platon kimdir?",
    "aristoteles--117051": "Aristoteles kimdir?",
    "immanuel-kant--37714": "Immanuel Kant kimdir?",
    "descartes--33070": "Descartes kimdir?",
    "hegel--143670": "Hegel kimdir?",
    "schopenhauer--47883": "Schopenhauer kimdir?",
    "kierkegaard--61875": "Kierkegaard kimdir?",
    "wittgenstein--52557": "Wittgenstein kimdir?",
    "heidegger--33845": "Heidegger kimdir?",
    "jean-paul-sartre--2417296": "Jean-Paul Sartre kimdir?",
    "albert-camus--34048": "Albert Camus kimdir?",
    "varolusculuk--68863": "Varoluşçuluk nedir?",
    "epistemoloji--47982": "Epistemoloji nedir?",
    "ontoloji--47983": "Ontoloji nedir?",
    "etik--34015": "Etik nedir?",
    "metafizik--75703": "Metafizik nedir?",
    "nihilizm--32896": "Nihilizm nedir?",
    "stoacilik--138260": "Stoacılık nedir?",
    "absurdizm--52783": "Absürdizm nedir?",
    "rasyonalizm--50071": "Rasyonalizm nedir?",
    "determinizm--86797": "Determinizm nedir?",
    "ozgur-irade--53825": "Özgür irade nedir?",
    "faydacilik--371971": "Faydacılık nedir?",
    "spinoza--106853": "Spinoza kimdir?",
    # --- ikinci tur genişletme (çeşitliliği artırmak için) ---
    "john-locke--136852": "John Locke kimdir?",
    "thomas-hobbes--111729": "Thomas Hobbes kimdir?",
    "konfucyus--255939": "Konfüçyüs kimdir?",
    "michel-foucault--132126": "Michel Foucault kimdir?",
    "jean-jacques-rousseau--105448": "Jean-Jacques Rousseau kimdir?",
    "machiavelli--67083": "Machiavelli kimdir?",
    "simone-de-beauvoir--51989": "Simone de Beauvoir kimdir?",
    "hannah-arendt--215366": "Hannah Arendt kimdir?",
    "karl-popper--108061": "Karl Popper kimdir?",
    "john-stuart-mill--124187": "John Stuart Mill kimdir?",
    "jeremy-bentham--283133": "Jeremy Bentham kimdir?",
    "karl-marx--50376": "Karl Marx kimdir?",
    "michel-de-montaigne--597587": "Michel de Montaigne kimdir?",
    "voltaire--71294": "Voltaire kimdir?",
    "budizm--72891": "Budizm nedir?",
    "taoizm--85397": "Taoizm nedir?",
    "pragmatizm--37563": "Pragmatizm nedir?",
    "fenomenoloji--66268": "Fenomenoloji nedir?",
    "postmodernizm--49078": "Postmodernizm nedir?",
    "aydinlanma-cagi--71295": "Aydınlanma çağı nedir?",
    "epikurosculuk--6008275": "Epikürosçuluk nedir?",
    "hedonizm--37562": "Hedonizm nedir?",
    "agnostisizm--118192": "Agnostisizm nedir?",
    "ateizm--41569": "Ateizm nedir?",
    "deizm--47065": "Deizm nedir?",
    "panteizm--47067": "Panteizm nedir?",
    "solipsizm--118036": "Solipsizm nedir?",
    "mutluluk--40751": "Mutluluk nedir?",
    "zihin-felsefesi--313078": "Zihin felsefesi nedir?",
    "dil-felsefesi--220671": "Dil felsefesi nedir?",
    "siyaset-felsefesi--457691": "Siyaset felsefesi nedir?",
    "ahlak-felsefesi--539948": "Ahlak felsefesi nedir?",
}

ENTRIES_PER_TOPIC = 8
PAGES_PER_TOPIC = 2  # her sayfada ~10 entry var, arttırırsanız daha çok veri toplarsınız
MIN_FAVORITE_COUNT = 0
MIN_ENTRY_LEN = 20
MAX_ENTRY_LEN = 6000  # common/lora_trainer.py MAX_SEQ_LENGTH (2048 token) için güvenli üst sınır
REQUEST_DELAY = 1.5  # saniye — kibar bir tempo

OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "raw" / "scraped_turkish_qa.jsonl"
# scrape_reddit.py ile paylaşılan, manuel kalite incelemesinde elenmiş id'ler
# (bkz. o dosyadaki aynı isimli sabitin açıklaması) — Ekşi Sözlük entry id'leri
# sayısal, Reddit'inkiler "t1_" önekli olduğu için çakışma riski yok.
REJECTED_IDS_PATH = Path(__file__).resolve().parent / "data" / "raw" / "rejected_ids.txt"

BASE_URL = "https://eksisozluk.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

BKZ_RE = re.compile(r"^\s*\(bkz:.*\)\s*$", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    return WHITESPACE_RE.sub(" ", text).strip()


def trim_to_sentence(text: str, limit: int = MAX_ENTRY_LEN) -> str:
    """Çok uzun (bazı başlıklarda 10k+ karakter) entry'leri son cümle
    sonunda keser — trainer'ın token bütçesini aşan cevapları önler."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    cut_points = [m.end() for m in re.finditer(r"[.!?]\s", window)]
    cut = cut_points[-1] if cut_points else limit
    return window[:cut].strip()


def fetch_topic_entries(slug: str) -> list[dict]:
    entries_by_id: dict[str, dict] = {}
    for page in range(1, PAGES_PER_TOPIC + 1):
        resp = requests.get(f"{BASE_URL}/{slug}", headers=HEADERS, params={"p": page}, timeout=15)
        time.sleep(REQUEST_DELAY)
        if resp.status_code != 200:
            print(f"  UYARI: {slug} (sayfa {page}) -> HTTP {resp.status_code}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        for li in soup.select("li[data-id]"):
            content_tag = li.select_one(".content")
            if content_tag is None:
                continue
            body = clean_text(content_tag.get_text(" ", strip=True))
            if not body or BKZ_RE.match(body) or len(body) < MIN_ENTRY_LEN:
                continue
            try:
                favorite_count = int(li.get("data-favorite-count", 0))
            except ValueError:
                favorite_count = 0
            entries_by_id[li["data-id"]] = {
                "id": li["data-id"],
                "body": trim_to_sentence(body),
                "favorite_count": favorite_count,
            }

    entries = [e for e in entries_by_id.values() if e["favorite_count"] >= MIN_FAVORITE_COUNT]
    entries.sort(key=lambda e: e["favorite_count"], reverse=True)
    return entries[:ENTRIES_PER_TOPIC]


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

    if not TOPIC_SLUGS:
        raise SystemExit("TOPIC_SLUGS boş. En az bir başlık slug'ı ekleyin.")

    written = 0
    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        for slug, soru in TOPIC_SLUGS.items():
            entries = fetch_topic_entries(slug)
            for entry in entries:
                if entry["id"] in seen_ids:
                    continue
                row = {
                    "id": entry["id"],
                    "soru": soru,
                    "cevap": entry["body"],
                    "source_url": f"{BASE_URL}/{slug}",
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                seen_ids.add(entry["id"])

            print(f"  '{soru}' -> {len(entries)} entry eklendi")

    print(f"Toplam {written} yeni soru-cevap çifti eklendi -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
