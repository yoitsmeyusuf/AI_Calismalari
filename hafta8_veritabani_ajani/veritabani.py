"""
8. Hafta: veritabani katmani (SQLite).

Senaryo: kucuk bir felsefe kitapcisi. Iki tablo yetiyor:

    kitaplar    -> katalog + stok (araclar buradan okur, siparis stogu duser)
    siparisler  -> verilen siparisler (araclar buraya yazar)

Bu dosya SQL'i tek yerde topluyor; `araclar.py` sadece fonksiyon cagiriyor.
Boylece "model ne yapabilir" (arac semalari) ile "veri nasil saklaniyor" (SQL)
birbirine karismiyor.

Yazma islemleri (`siparis_olustur`, `siparis_iptal`) tek bir `BEGIN IMMEDIATE`
islemi icinde donuyor: stok kontrolu ile stok dusumu arasinda baska bir istek
araya giremiyor. Stok sutununda ayrica `CHECK (stok >= 0)` var - uygulama
katmani hata yapsa bile veritabani negatif stoga izin vermiyor.

Veritabani dosyasi yoksa ilk erisimde semasi kurulup ornek katalogla doldurulur
(`db_kur()`), yani depoyu klonlayan biri ek bir adim yapmadan calistirabiliyor.

Calistirma (veritabanini sifirdan kur ve icerigini dok):
    .venv/bin/python hafta8_veritabani_ajani/veritabani.py
    .venv/bin/python hafta8_veritabani_ajani/veritabani.py --sifirla
"""
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Space'te kalici disk (/data) varsa oraya yaz: Space yeniden baslatildiginda
# siparisler silinmesin. Yoksa klasorun yanina.
def _varsayilan_db_yolu() -> Path:
    ortam = os.environ.get("KITAPCI_DB")
    if ortam:
        return Path(ortam)
    kalici = Path("/data")
    if kalici.is_dir() and os.access(kalici, os.W_OK):
        return kalici / "kitapci.db"
    return HERE / "kitapci.db"


DB_YOLU = _varsayilan_db_yolu()

DURUMLAR = ("hazırlanıyor", "kargoda", "teslim edildi", "iptal edildi")
IPTAL_EDILEBILIR = ("hazırlanıyor", "kargoda")

SEMA = """
CREATE TABLE IF NOT EXISTS kitaplar (
    id     INTEGER PRIMARY KEY,
    baslik TEXT    NOT NULL,
    yazar  TEXT    NOT NULL,
    akim   TEXT    NOT NULL,
    yil    INTEGER,                       -- negatif deger = M.O.
    fiyat  REAL    NOT NULL CHECK (fiyat > 0),
    stok   INTEGER NOT NULL CHECK (stok >= 0)
);

CREATE TABLE IF NOT EXISTS siparisler (
    id          INTEGER PRIMARY KEY,
    kod         TEXT    NOT NULL UNIQUE,
    musteri     TEXT    NOT NULL,
    kitap_id    INTEGER NOT NULL REFERENCES kitaplar(id),
    adet        INTEGER NOT NULL CHECK (adet > 0),
    birim_fiyat REAL    NOT NULL,
    toplam      REAL    NOT NULL,
    durum       TEXT    NOT NULL DEFAULT 'hazırlanıyor',
    olusturma   TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_kitaplar_yazar ON kitaplar(yazar);
CREATE INDEX IF NOT EXISTS ix_siparisler_kod ON siparisler(kod);
"""

# (baslik, yazar, akim, yil, fiyat, stok)
# Stoklar bilincli olarak cesitli: 0 olan (tukendi) ve 2 olan (yetersiz stok)
# kayitlar araclarin hata yollarini canli demoda da gosterebilmek icin var.
KATALOG = [
    ("Böyle Buyurdu Zerdüşt", "Friedrich Nietzsche", "nihilizm", 1885, 145.0, 7),
    ("Ahlakın Soykütüğü Üstüne", "Friedrich Nietzsche", "nihilizm", 1887, 120.0, 3),
    ("Varlık ve Zaman", "Martin Heidegger", "varoluşçuluk", 1927, 380.0, 2),
    ("Varlık ve Hiçlik", "Jean-Paul Sartre", "varoluşçuluk", 1943, 420.0, 0),
    ("İkinci Cinsiyet", "Simone de Beauvoir", "varoluşçuluk", 1949, 310.0, 6),
    ("Sisifos Söyleni", "Albert Camus", "absürdizm", 1942, 110.0, 9),
    ("Devlet", "Platon", "antik felsefe", -375, 165.0, 11),
    ("Nikomakhos'a Etik", "Aristoteles", "antik felsefe", -340, 160.0, 5),
    ("Kendime Düşünceler", "Marcus Aurelius", "stoacılık", 180, 95.0, 12),
    ("Söylevler", "Epiktetos", "stoacılık", 125, 88.0, 6),
    ("Metot Üzerine Konuşma", "René Descartes", "rasyonalizm", 1637, 75.0, 8),
    ("Arı Usun Eleştirisi", "Immanuel Kant", "aydınlanma", 1781, 460.0, 4),
    ("İnsan Doğası Üzerine Bir İnceleme", "David Hume", "empirizm", 1739, 340.0, 3),
    ("Tractatus Logico-Philosophicus", "Ludwig Wittgenstein", "analitik felsefe", 1921, 130.0, 5),
    ("Hapishanenin Doğuşu", "Michel Foucault", "postyapısalcılık", 1975, 290.0, 4),
]

# Bos bir veritabaninda da `siparis_durumu` sorulabilsin diye iki ornek siparis.
# (kitap_id, musteri, adet, durum)
ORNEK_SIPARISLER = [
    (1, "Yusuf K.", 1, "teslim edildi"),
    (6, "Elif D.", 2, "kargoda"),
]


class VeritabaniHatasi(Exception):
    """Arac katmanina tasinacak, kullaniciya gosterilebilir veritabani hatasi."""


class KitapYokHatasi(VeritabaniHatasi):
    """Istenen kitap_id katalogda yok - genelde model id'yi uydurmustur."""


def baglan(yol: Path | str | None = None) -> sqlite3.Connection:
    """Yeni bir baglanti acar.

    Her arac cagrisi kendi baglantisini acip kapatiyor: Gradio istekleri ayri
    thread'lerde kosuyor ve SQLite baglantilari thread'ler arasinda paylasilmaz.
    `isolation_level=None` ile otomatik islem yonetimi kapatiliyor; yazma
    araclari `BEGIN IMMEDIATE`'i kendileri aciyor.
    """
    hedef = Path(yol or DB_YOLU)
    hedef.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(hedef, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def baglanti(yol: Path | str | None = None):
    """Okuma sorgulari icin `with` destekli baglanti (cikista kapanir).

    sqlite3.Connection'in kendi `with` blogu islemi commit/rollback eder ama
    baglantiyi *kapatmaz*; okumalarda baglanti sizmasin diye bu sarmalayici var.
    """
    conn = baglan(yol)
    try:
        yield conn
    finally:
        conn.close()


def db_kur(yol: Path | str | None = None, sifirla: bool = False) -> Path:
    """Semayi kurar, veritabani bossa ornek katalogla doldurur.

    `sifirla=True` mevcut dosyayi silip sifirdan kurar (demo sonrasi temizlik).
    Idempotent: tablolar ve satirlar varsa dokunmaz, tekrar tekrar cagrilabilir.
    """
    hedef = Path(yol or DB_YOLU)
    if sifirla and hedef.exists():
        hedef.unlink()

    with baglanti(hedef) as conn:
        conn.executescript(SEMA)
        (kitap_sayisi,) = conn.execute("SELECT COUNT(*) FROM kitaplar").fetchone()
        if kitap_sayisi == 0:
            conn.executemany(
                "INSERT INTO kitaplar (baslik, yazar, akim, yil, fiyat, stok) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                KATALOG,
            )
            for kitap_id, musteri, adet, durum in ORNEK_SIPARISLER:
                satir = conn.execute(
                    "SELECT fiyat FROM kitaplar WHERE id = ?", (kitap_id,)
                ).fetchone()
                conn.execute(
                    "INSERT INTO siparisler "
                    "(kod, musteri, kitap_id, adet, birim_fiyat, toplam, durum, olusturma) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        _siparis_kodu(conn),
                        musteri,
                        kitap_id,
                        adet,
                        satir["fiyat"],
                        round(satir["fiyat"] * adet, 2),
                        durum,
                        _simdi(),
                    ),
                )
                # Ornek siparislerin stogu da dusuyor ki katalog tutarli olsun.
                conn.execute(
                    "UPDATE kitaplar SET stok = stok - ? WHERE id = ?", (adet, kitap_id)
                )
    return hedef


def _simdi() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _siparis_kodu(conn: sqlite3.Connection) -> str:
    """Siradaki siparis kodunu uretir: SIP-1001, SIP-1002, ...

    Kod modelin uyduramayacagi tek anahtar: `siparis_durumu` yalnizca burada
    uretilmis bir kodu taniyor.
    """
    (enb,) = conn.execute("SELECT COALESCE(MAX(id), 0) FROM siparisler").fetchone()
    return f"SIP-{1001 + enb}"


def yil_metni(yil: int | None) -> str | None:
    """Negatif yillari 'M.Ö. 375' olarak yazar (Platon, Aristoteles)."""
    if yil is None:
        return None
    return f"M.Ö. {abs(yil)}" if yil < 0 else str(yil)


def _kitap_sozlugu(satir: sqlite3.Row) -> dict:
    return {
        "kitap_id": satir["id"],
        "baslik": satir["baslik"],
        "yazar": satir["yazar"],
        "akim": satir["akim"],
        "yil": yil_metni(satir["yil"]),
        "fiyat_tl": round(satir["fiyat"], 2),
        "stok": satir["stok"],
        # Sayiyi yorumlamayi modele birakmiyoruz: stok=0 goren model bunu
        # "kitap katalogda yok" diye ozetleyip yerine baska kitap onerebiliyor.
        "stok_durumu": "stokta" if satir["stok"] > 0 else "tükendi (katalogda var, stok yok)",
    }


def _siparis_sozlugu(satir: sqlite3.Row) -> dict:
    return {
        "siparis_kodu": satir["kod"],
        "musteri": satir["musteri"],
        "kitap": satir["baslik"],
        "yazar": satir["yazar"],
        "adet": satir["adet"],
        "birim_fiyat_tl": round(satir["birim_fiyat"], 2),
        "toplam_tl": round(satir["toplam"], 2),
        "durum": satir["durum"],
        "siparis_tarihi": satir["olusturma"],
    }


# --------------------------------------------------------------------------
# Okuma
# --------------------------------------------------------------------------


def kitap_ara(
    metin: str | None = None,
    yazar: str | None = None,
    en_fazla_fiyat: float | None = None,
    sadece_stokta: bool = False,
    limit: int = 8,
) -> list[dict]:
    """Katalogda arama. Bos filtre = tum katalog (limitli)."""
    kosullar: list[str] = []
    degerler: list = []

    if metin:
        kosullar.append("(baslik LIKE ? OR yazar LIKE ? OR akim LIKE ?)")
        desen = f"%{metin.strip()}%"
        degerler += [desen, desen, desen]
    if yazar:
        kosullar.append("yazar LIKE ?")
        degerler.append(f"%{yazar.strip()}%")
    if en_fazla_fiyat is not None:
        kosullar.append("fiyat <= ?")
        degerler.append(float(en_fazla_fiyat))
    if sadece_stokta:
        kosullar.append("stok > 0")

    sql = "SELECT * FROM kitaplar"
    if kosullar:
        sql += " WHERE " + " AND ".join(kosullar)
    sql += " ORDER BY stok = 0, baslik LIMIT ?"  # tukenenler listenin sonunda
    degerler.append(int(limit))

    with baglanti() as conn:
        return [_kitap_sozlugu(s) for s in conn.execute(sql, degerler)]


def kitap_getir(kitap_id: int) -> dict | None:
    with baglanti() as conn:
        satir = conn.execute("SELECT * FROM kitaplar WHERE id = ?", (kitap_id,)).fetchone()
    return _kitap_sozlugu(satir) if satir else None


def siparis_getir(kod: str) -> dict | None:
    with baglanti() as conn:
        satir = conn.execute(
            "SELECT s.*, k.baslik, k.yazar FROM siparisler s "
            "JOIN kitaplar k ON k.id = s.kitap_id WHERE s.kod = ?",
            (kod.strip().upper(),),
        ).fetchone()
    return _siparis_sozlugu(satir) if satir else None


def siparisleri_listele(limit: int = 10) -> list[dict]:
    """Arayuzdeki 'son siparisler' paneli icin (modele acilan bir arac degil)."""
    with baglanti() as conn:
        return [
            _siparis_sozlugu(s)
            for s in conn.execute(
                "SELECT s.*, k.baslik, k.yazar FROM siparisler s "
                "JOIN kitaplar k ON k.id = s.kitap_id ORDER BY s.id DESC LIMIT ?",
                (int(limit),),
            )
        ]


def katalogu_listele() -> list[dict]:
    """Arayuzdeki stok paneli icin tum katalog."""
    with baglanti() as conn:
        return [
            _kitap_sozlugu(s)
            for s in conn.execute("SELECT * FROM kitaplar ORDER BY baslik")
        ]


# --------------------------------------------------------------------------
# Yazma
# --------------------------------------------------------------------------


def siparis_olustur(kitap_id: int, adet: int, musteri: str) -> dict:
    """Siparisi kaydeder ve stogu duser. Tek islem, ya hep ya hic.

    `BEGIN IMMEDIATE` yazma kilidini hemen aliyor: stogu okuyup dusurene kadar
    baska bir siparis araya giremez (ayni kitabin son kopyasini iki kisiye
    satmayalim).
    """
    conn = baglan()
    try:
        conn.execute("BEGIN IMMEDIATE")
        kitap = conn.execute(
            "SELECT * FROM kitaplar WHERE id = ?", (kitap_id,)
        ).fetchone()
        if kitap is None:
            raise KitapYokHatasi(f"{kitap_id} numaralı kitap katalogda yok.")
        if kitap["stok"] < adet:
            raise VeritabaniHatasi(
                f"'{kitap['baslik']}' için yeterli stok yok: istenen {adet}, "
                f"mevcut {kitap['stok']}."
            )

        toplam = round(kitap["fiyat"] * adet, 2)
        kod = _siparis_kodu(conn)
        zaman = _simdi()
        conn.execute(
            "INSERT INTO siparisler "
            "(kod, musteri, kitap_id, adet, birim_fiyat, toplam, durum, olusturma) "
            "VALUES (?, ?, ?, ?, ?, ?, 'hazırlanıyor', ?)",
            (kod, musteri.strip(), kitap_id, adet, kitap["fiyat"], toplam, zaman),
        )
        conn.execute("UPDATE kitaplar SET stok = stok - ? WHERE id = ?", (adet, kitap_id))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    return {
        "siparis_kodu": kod,
        "musteri": musteri.strip(),
        "kitap": kitap["baslik"],
        "yazar": kitap["yazar"],
        "adet": adet,
        "birim_fiyat_tl": round(kitap["fiyat"], 2),
        "toplam_tl": toplam,
        "durum": "hazırlanıyor",
        "kalan_stok": kitap["stok"] - adet,
        "siparis_tarihi": zaman,
    }


def siparis_iptal(kod: str) -> dict:
    """Siparisi iptal eder ve stogu geri ekler (yine tek islem)."""
    kod = kod.strip().upper()
    conn = baglan()
    try:
        conn.execute("BEGIN IMMEDIATE")
        siparis = conn.execute(
            "SELECT s.*, k.baslik FROM siparisler s JOIN kitaplar k ON k.id = s.kitap_id "
            "WHERE s.kod = ?",
            (kod,),
        ).fetchone()
        if siparis is None:
            raise VeritabaniHatasi(f"'{kod}' kodlu bir sipariş bulunamadı.")
        if siparis["durum"] not in IPTAL_EDILEBILIR:
            raise VeritabaniHatasi(
                f"'{kod}' kodlu siparişin durumu '{siparis['durum']}', iptal edilemez. "
                f"Yalnızca {' veya '.join(IPTAL_EDILEBILIR)} durumundaki siparişler iptal edilir."
            )

        conn.execute("UPDATE siparisler SET durum = 'iptal edildi' WHERE id = ?", (siparis["id"],))
        conn.execute(
            "UPDATE kitaplar SET stok = stok + ? WHERE id = ?",
            (siparis["adet"], siparis["kitap_id"]),
        )
        (kalan,) = conn.execute(
            "SELECT stok FROM kitaplar WHERE id = ?", (siparis["kitap_id"],)
        ).fetchone()
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    return {
        "siparis_kodu": kod,
        "kitap": siparis["baslik"],
        "adet": siparis["adet"],
        "onceki_durum": siparis["durum"],
        "durum": "iptal edildi",
        "iade_edilen_tutar_tl": round(siparis["toplam"], 2),
        "kalan_stok": kalan,
    }


if __name__ == "__main__":
    sifirla = "--sifirla" in sys.argv
    yol = db_kur(sifirla=sifirla)
    print(f"Veritabanı: {yol} ({'sıfırdan kuruldu' if sifirla else 'hazır'})\n")

    print("--- Katalog ---")
    for k in katalogu_listele():
        print(
            f"  #{k['kitap_id']:>2} {k['baslik'][:38]:<38} {k['yazar'][:22]:<22} "
            f"{k['fiyat_tl']:>7.2f} TL  stok={k['stok']}"
        )

    print("\n--- Siparişler ---")
    for s in siparisleri_listele():
        print(
            f"  {s['siparis_kodu']}  {s['musteri'][:14]:<14} {s['kitap'][:30]:<30} "
            f"x{s['adet']}  {s['toplam_tl']:>7.2f} TL  {s['durum']}"
        )
