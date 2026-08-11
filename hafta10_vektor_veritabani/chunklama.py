"""
10. Hafta: parcalama (chunking) mantigi.

Hastane makalelerinde `\\n\\n` yok - paragraflar tek `\\n` ile ayrilmis. Ama
metin duz bir blok da degil: her makale "baslik + o basligin altindaki icerik"
seklinde bolumlere dizilmis. Ornek (guven, pankreas ameliyati makalesi):

    Pankreas ameliyati denince pek cok kisinin aklina...       <- giris
    - Ameliyat Gerektirebilecek Pankreas Hastaliklari...       <- icindekiler
    - Her Lezyon Ameliyat Gerektirir mi?                       <- icindekiler
    Ameliyat Gerektirebilecek Pankreas Hastaliklari Nelerdir?  <- BASLIK
    Pankreas Kistleri ve IPMN: Pankreasta rastlanan...         <- icerik
    Her Lezyon Ameliyat Gerektirir mi?                         <- BASLIK
    Kesinlikle hayir. Pankreas lezyonlarinin bir bolumu...     <- icerik

Bu yuzden kor token paketlemesi yerine bolum sinirlarini kullaniyoruz: chunk
sinirini tokenizer degil, belgenin kendi yapisi belirliyor. Olculen sonuc, 14
split'in hepsinde bolumlerin %84-94'unde baslik yakalandigi.

Ikinci olcum: tokenizer Turkce'ye ozel oldugu icin 5.77 karakter/token veriyor
(cok dilli modellerde bu oran ~3.2). Yani bolumlerin medyani sadece ~90 token,
p99'u ~400. Bastan planladigimiz 350 token'lik hedef bolumlerin %99'undan buyuk
oldugu icin 3-5 alakasiz bolumu tek chunk'a tikardi - tam da kacinmak istedigimiz
konu sulanmasi. Hedefi bu olcume gore asagi cektik.

MAX_TOK bir kalite siniri, teknik zorunluluk degil: model 8192 token aliyor.
Chunk ne kadar cok konu icerirse vektoru o kadar ortalamaya kayar, pozitif ve
negatif sorularin skorlari birbirine yaklasir, esik ayirt edemez hale gelir.

Kendi basina calistirilirsa ornek bir makaleyi chunk'layip doker:
    ../.venv/bin/python hafta10_vektor_veritabani/chunklama.py
"""
import re

HEDEF_TOK = 256   # buyuk bolumu bolerken hedeflenen boy
MIN_TOK = 48      # bunun altindaki chunk tek basina soru cevaplamaz -> birlestir
MAX_TOK = 320     # mutlak tavan; asan chunk uretilmez
OVERLAP_TOK = 48  # sadece bolum kesildiginde devreye girer

# Cumle sinirini ".!?" + bosluk + buyuk harf/rakam olarak aliyoruz. Turkce buyuk
# harfler ASCII disinda oldugu icin acikca yaziliyor.
_CUMLE_SINIRI = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9])")


def bloklara_ayir(metin: str) -> list[str]:
    """Ham `text` alanini satir bloklarina ayirir (bos satirlar atilir)."""
    duz = (metin or "").replace("\r\n", "\n").replace("\n\n", "\n")
    return [satir.strip() for satir in duz.split("\n") if satir.strip()]


def baslik_mi(blok: str) -> bool:
    """Bir blogun bolum basligi olup olmadigini karar verir.

    Dayanak: basliklar cumle degildir - kisadirlar ve nokta ile bitmezler.

    - "- " ile baslayan   -> madde isareti, baslik degil
    - 120 karakterden uzun -> paragraf
    - ":" ile biten        -> "En yaygin belirtiler sunlardir:" gibi giris cumlesi;
                              baslik gibi gorunur ama altindaki listeye baglidir
    - "?" ile biten        -> "X Nedir?" kalibi; bu sitelerin ana baslik bicimi
    - "." veya "!" ile bitmiyorsa -> baslik
    """
    if blok.startswith("- ") or len(blok) > 120 or blok.endswith(":"):
        return False
    if blok.endswith("?"):
        return True
    return not blok.rstrip().endswith((".", "!"))


def bolumlere_ayir(metin: str) -> list[dict]:
    """Makaleyi {baslik, bloklar} bolumlerine ayirir, icindekiler menusunu atar.

    "- " isareti iki farkli is icin kullanilmis: icindekiler menusu ve gercek
    madde listesi. Ayirt etme kurali: "- X" satiri, X govdede ayrica baslik
    olarak da geciyorsa menudur. Gecmiyorsa gercek icerik, korunur.
    """
    bloklar = bloklara_ayir(metin)
    basliklar = {b for b in bloklar if baslik_mi(b)}
    menu = {b[2:].strip() for b in bloklar if b.startswith("- ")} & basliklar
    govde = [b for b in bloklar if not (b.startswith("- ") and b[2:].strip() in menu)]

    bolumler: list[dict] = []
    acik = {"baslik": "", "bloklar": []}
    for blok in govde:
        if baslik_mi(blok):
            if acik["bloklar"]:
                bolumler.append(acik)
            acik = {"baslik": blok, "bloklar": []}
        else:
            acik["bloklar"].append(blok)
    if acik["bloklar"]:
        bolumler.append(acik)
    return bolumler


def _cumlelere_bol(blok: str, say) -> list[str]:
    """Tek basina MAX_TOK'u asan blogu cumle sinirindan boler.

    Blok sinirindan bolmek yetmiyor: bazi makalelerde tek bir dev paragraf var
    ve bolunmezse 1000+ token'lik chunk cikiyor.
    """
    parcalar, birikim, boy = [], [], 0
    for cumle in _CUMLE_SINIRI.split(blok):
        tok = say(cumle)
        if birikim and boy + tok > HEDEF_TOK:
            parcalar.append(" ".join(birikim))
            birikim, boy = [], 0
        birikim.append(cumle)
        boy += tok
    if birikim:
        parcalar.append(" ".join(birikim))
    return parcalar


def _kisa_bolumleri_birlestir(bolumler: list[dict], say) -> list[dict]:
    """MIN_TOK altindaki bolumu bir sonrakine yapistirir."""
    sonuc: list[dict] = []
    bekleyen: dict | None = None
    for bolum in bolumler:
        if bekleyen is not None:
            bolum = {
                "baslik": bekleyen["baslik"] or bolum["baslik"],
                "bloklar": bekleyen["bloklar"] + bolum["bloklar"],
            }
            bekleyen = None
        if say(" ".join(bolum["bloklar"])) < MIN_TOK:
            bekleyen = bolum
        else:
            sonuc.append(bolum)

    if bekleyen is not None:
        # Son bolum kisa kaldi: yapisacak bir sonraki yok, oncekine ekle.
        # Hicbiri yoksa (makalenin tamami kisa) oldugu gibi birak.
        if sonuc:
            sonuc[-1]["bloklar"].extend(bekleyen["bloklar"])
        else:
            sonuc.append(bekleyen)
    return sonuc


def chunkla(metin: str, say) -> list[dict]:
    """Makale metnini chunk'lara boler.

    `say`: metin -> token sayisi (embedding modelinin kendi tokenizer'i).

    Donen her chunk: {bolum, parca, baslik, metin}. `bolum` ayni makale icindeki
    bolum sirasi (parent kimligini kurmak icin), `parca` bolum kesildiyse kacinci
    parca oldugu - kesilmediyse 0.
    """
    bolumler = _kisa_bolumleri_birlestir(bolumlere_ayir(metin), say)

    chunklar: list[dict] = []
    for sira, bolum in enumerate(bolumler):
        govde = " ".join(bolum["bloklar"])
        if say(govde) <= MAX_TOK:
            chunklar.append(
                {"bolum": sira, "parca": 0, "baslik": bolum["baslik"], "metin": govde}
            )
            continue

        # Bolum tavani asiyor: blok sinirindan bol. Tek blok da asiyorsa once
        # onu cumlelere bol, sonra paketle.
        parcalanabilir = [
            parca
            for blok in bolum["bloklar"]
            for parca in (_cumlelere_bol(blok, say) if say(blok) > MAX_TOK else [blok])
        ]

        birikim, boy, parca_no = [], 0, 0
        for blok in parcalanabilir:
            tok = say(blok)
            if birikim and boy + tok > HEDEF_TOK:
                chunklar.append(
                    {
                        "bolum": sira,
                        "parca": parca_no,
                        "baslik": bolum["baslik"],
                        "metin": " ".join(birikim),
                    }
                )
                parca_no += 1
                # Overlap: son bloklardan geriye sararak kuyruk tasi. Kuyruk +
                # yeni blok tavani asacaksa overlap'ten vazgec - yoksa tavan
                # delinir (olculdu: max 320 yerine 509 cikiyordu).
                kuyruk, kuyruk_boy = [], 0
                for onceki in reversed(birikim):
                    if kuyruk_boy >= OVERLAP_TOK or kuyruk_boy + say(onceki) + tok > MAX_TOK:
                        break
                    kuyruk.insert(0, onceki)
                    kuyruk_boy += say(onceki)
                birikim, boy = kuyruk, kuyruk_boy
            birikim.append(blok)
            boy += tok

        if birikim:
            chunklar.append(
                {
                    "bolum": sira,
                    "parca": parca_no,
                    "baslik": bolum["baslik"],
                    "metin": " ".join(birikim),
                }
            )
    return chunklar


def gomulecek_metin(makale_basligi: str, bolum_basligi: str, chunk_metni: str) -> str:
    """Embedding'e verilecek metin.

    Model asimetrik: dokuman tarafinda `document` prompt'u
    "title: none | text: " sablonunu basiyor. Gercek basligi oraya koyabiliyoruz;
    hastane metinleri anafora dolu ("Bu hastalikta...", "Tedavi sureci..."), tek
    basina chunk cogu zaman neyden bahsettigini soylemiyor.

    `chunk_text` sutununa ham metin yaziliyor - baslik sadece vektor uretiminde
    kullaniliyor.
    """
    basliklar = " — ".join(x for x in (makale_basligi, bolum_basligi) if x)
    return f"{basliklar} | {chunk_metni}" if basliklar else chunk_metni


def _ornek_calistir() -> None:
    from transformers import AutoTokenizer

    from ayarlar import EMBEDDING_MODELI

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODELI)
    say = lambda s: len(tokenizer.encode(s, add_special_tokens=False))  # noqa: E731

    # Gercek makalelerin yapisini taklit eden ornek: giris, icindekiler menusu,
    # baslikli bolumler. Bolumler MIN_TOK'u gececek uzunlukta tutuldu ki
    # birlestirme degil bolumleme gorunsun.
    ornek = (
        "Tiroid bezi boynun on kisminda, girtlagin hemen altinda yer alan "
        "kelebek seklinde bir ic salgi bezidir. Vucudun metabolizma hizini "
        "duzenleyen tiroid hormonlarini salgilar. Bu hormonlar kalp hizindan "
        "vucut isisina, kilo kontrolunden sindirim hizina kadar pek cok sureci "
        "etkiler. Bezin az ya da cok calismasi genis bir belirti yelpazesine "
        "yol acabilir.\n"
        "- Guatr Nedir?\n"
        "- Guatr Belirtileri Nelerdir?\n"
        "Guatr Nedir?\n"
        "Guatr, tiroid bezinin normalden buyuk olmasi durumudur. Dunya genelinde "
        "en sik nedeni iyot eksikligidir; iyotlu tuz kullaniminin yayginlasmasi "
        "bu oranlari belirgin sekilde dusurmustur. Her guatr kanser anlamina "
        "gelmez, buyuk cogunlugu iyi huyludur. Bezin buyumesi bazen hormon "
        "duzeyleri tamamen normalken de gorulebilir.\n"
        "Guatr Belirtileri Nelerdir?\n"
        "En sik gorulen belirtiler sunlardir:\n"
        "- Boyun on kisminda gozle gorulur sislik\n"
        "- Yutma guclugu ve bogazda baski hissi\n"
        "- Ses kisikligi veya seste kalinlasma\n"
        "Buyuk guatrlar soluk borusuna basi yaparak nefes darligina yol "
        "acabilir. Bu durumda cerrahi degerlendirme gerekir."
    )
    print(f"MIN_TOK={MIN_TOK} HEDEF_TOK={HEDEF_TOK} MAX_TOK={MAX_TOK}\n")
    for i, chunk in enumerate(chunkla(ornek, say)):
        print(f"[chunk {i}] bolum={chunk['bolum']} parca={chunk['parca']} "
              f"tok={say(chunk['metin'])}")
        print(f"    baslik: {chunk['baslik'] or '(yok — makale girisi)'}")
        print(f"    metin : {chunk['metin'][:130]}...\n")


if __name__ == "__main__":
    _ornek_calistir()
