"""
8. Hafta: modele acilan araclar (function/tool definitions) + implementasyonlari.

Dort arac var; ikisi veritabanindan okuyor, ikisi yaziyor:

    search_books      (oku)   katalog + stok sorgusu
    create_order      (yaz)   siparis kaydeder, stogu duser
    get_order_status  (oku)   siparis kodundan durum sorgular
    cancel_order      (yaz)   siparisi iptal eder, stogu geri ekler

Butun SQL `veritabani.py`'de; burada sadece (1) modelin gonderdigi argumanlari
dogrulama, (2) sonucu modelin okuyacagi sozluge cevirme, (3) hata yollarini
modele geri bildirme isi yapiliyor.

Halusinasyon acisindan onemli iki tasarim karari:

  - `create_order` yalnizca **veritabanindaki bir kitap_id** ile calisir. Model
    kitap adini kendi belleginden uydurup siparis veremez; once `search_books`
    cagirip gercek id'yi almak zorunda. Bu, odevdeki "iki turlu zincir".
  - Her arac ciktisina, modelin tam o anda baktigi yere bir `not` alani
    ekleniyor ("yalnizca bu listedeki kitaplar mevcut..."). 7. haftada sistem
    promptundaki ayni kuralin tek basina yetmedigini gormustuk.

Tek basina calistirilirsa butun araclari (hata yollari dahil) test eder:
    .venv/bin/python hafta8_veritabani_ajani/araclar.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import veritabani as db  # noqa: E402

MAKS_ADET = 10  # tek sipariste satilabilecek en fazla kopya

KATALOG_NOTU = (
    "kitapçının stoğu yalnızca bu listedekilerden ibarettir; listede olmayan bir "
    "kitabı, fiyatı veya stok sayısını kendin uydurma. Sipariş vermek için "
    "buradaki kitap_id değerini create_order aracına ver"
)
SIPARIS_NOTU = (
    "bu sipariş veritabanına kaydedildi; kullanıcıya sipariş kodunu bu çıktıdaki "
    "haliyle bildir, kod uydurma"
)

# `ilk_cumle()` icin cumle sonu deseni; "örn." / "vb." / "vs." sonrasindaki
# noktalar cumle sonu sayilmiyor.
CUMLE_SONU_DESENI = re.compile(
    r"(?<!\börn)(?<!\bvb)(?<!\bvs)\.\s+(?=[\"'“*A-ZÇĞİÖŞÜ])", re.IGNORECASE
)


class AracHatasi(Exception):
    """Arac calistirilirken olusan, modele geri bildirilecek hata."""


def _tam_sayi(deger, ad: str, en_az: int = 1, en_fazla: int | None = None) -> int:
    """Modelin string olarak yollayabilecegi sayilari da kabul eder."""
    try:
        sayi = int(str(deger).strip())
    except (TypeError, ValueError):
        raise AracHatasi(f"{ad} tam sayı olmalı, gelen: {deger!r}") from None
    if sayi < en_az:
        raise AracHatasi(f"{ad} en az {en_az} olmalı, gelen: {sayi}")
    if en_fazla is not None and sayi > en_fazla:
        raise AracHatasi(f"{ad} en fazla {en_fazla} olabilir, gelen: {sayi}")
    return sayi


# --------------------------------------------------------------------------
# Araclar
# --------------------------------------------------------------------------


def search_books(
    query: str | None = None,
    author: str | None = None,
    max_price: float | None = None,
    only_in_stock: bool = False,
) -> dict:
    """Katalogda kitap arar (okuma). Filtre verilmezse tum katalogu dondurur."""
    if max_price is not None:
        try:
            max_price = float(str(max_price).replace(",", "."))
        except (TypeError, ValueError):
            raise AracHatasi(f"max_price sayısal olmalı, gelen: {max_price!r}") from None

    sonuclar = db.kitap_ara(
        metin=query,
        yazar=author,
        en_fazla_fiyat=max_price,
        sadece_stokta=bool(only_in_stock),
    )
    if not sonuclar:
        filtreler = {
            "query": query,
            "author": author,
            "max_price": max_price,
            "only_in_stock": only_in_stock,
        }
        return {
            "bulunan": 0,
            "kitaplar": [],
            "not": (
                f"Bu filtrelerle katalogda kitap yok ({filtreler}). Kullanıcıya "
                "kitabın stokta olmadığını söyle; başka bir kitap uydurma."
            ),
        }

    return {
        "bulunan": len(sonuclar),
        "kitaplar": sonuclar,
        "not": KATALOG_NOTU,
    }


def create_order(
    book_id: int,
    quantity: int = 1,
    customer_name: str = "Misafir",
) -> dict:
    """Siparis olusturur ve stogu duser (yazma)."""
    kitap_id = _tam_sayi(book_id, "book_id")
    adet = _tam_sayi(quantity, "quantity", en_az=1, en_fazla=MAKS_ADET)
    musteri = (customer_name or "Misafir").strip() or "Misafir"

    try:
        sonuc = db.siparis_olustur(kitap_id, adet, musteri)
    except db.KitapYokHatasi as e:
        # Arac adini hata mesajina burada ekliyoruz: veritabani katmani modele
        # hangi araclarin acildigini bilmek zorunda degil.
        raise AracHatasi(f"{e} Doğru kitap_id için önce search_books aracını çağır.") from e
    except db.VeritabaniHatasi as e:
        raise AracHatasi(str(e)) from e

    sonuc["not"] = SIPARIS_NOTU
    return sonuc


def get_order_status(order_code: str) -> dict:
    """Siparis kodundan siparisin durumunu sorgular (okuma)."""
    kod = (order_code or "").strip()
    if not kod:
        raise AracHatasi("order_code boş olamaz (örn. 'SIP-1001').")

    siparis = db.siparis_getir(kod)
    if siparis is None:
        raise AracHatasi(
            f"'{kod.upper()}' kodlu bir sipariş kaydı yok. Kodu kullanıcıya "
            "doğrulat; sipariş bilgisi uydurma."
        )
    return siparis


def cancel_order(order_code: str) -> dict:
    """Siparisi iptal eder ve stogu geri ekler (yazma)."""
    kod = (order_code or "").strip()
    if not kod:
        raise AracHatasi("order_code boş olamaz (örn. 'SIP-1001').")

    try:
        return db.siparis_iptal(kod)
    except db.VeritabaniHatasi as e:
        raise AracHatasi(str(e)) from e


# --------------------------------------------------------------------------
# Modele verilen JSON semalari (Tool / Function Definition)
# --------------------------------------------------------------------------

ARAC_SEMALARI = [
    {
        "type": "function",
        "function": {
            "name": "search_books",
            "description": (
                "Kitapçının veritabanında kitap arar ve her kitabın kitap_id, başlık, "
                "yazar, akım, fiyat (TL) ve stok bilgisini döndürür. Kullanıcı bir "
                "kitabı, yazarı, felsefe akımını, fiyatı veya stok durumunu sorduğunda "
                "MUTLAKA bu aracı çağır — katalog bilgisini kendi belleğinden verme. "
                "Sipariş oluşturmadan önce de doğru kitap_id'yi bu araçla bulmalısın."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Serbest arama metni; kitap adı, yazar veya akım içinde "
                            "aranır. Örn. 'Zerdüşt', 'stoacılık', 'Camus'."
                        ),
                    },
                    "author": {
                        "type": "string",
                        "description": "Yalnızca bu yazarın kitaplarını getirir.",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Fiyat üst sınırı (TL).",
                    },
                    "only_in_stock": {
                        "type": "boolean",
                        "description": "True ise stoğu tükenmiş kitapları listeleme.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": (
                "Bir kitap için sipariş oluşturur: siparişi veritabanına yazar ve "
                "stoktan düşer. Kullanıcı bir kitabı satın almak/sipariş etmek "
                "istediğinde çağır. book_id'yi UYDURMA — önce search_books ile kitabı "
                "bul, dönen kitap_id'yi buraya ver. Stok yetersizse araç hata döndürür, "
                "sipariş oluşmaz."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "book_id": {
                        "type": "integer",
                        "description": "search_books çıktısındaki kitap_id değeri.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": f"Kaç adet sipariş edilecek (1-{MAKS_ADET}). Belirtilmezse 1.",
                        "minimum": 1,
                        "maximum": MAKS_ADET,
                    },
                    "customer_name": {
                        "type": "string",
                        "description": (
                            "Müşterinin adı. Kullanıcı adını söylediyse onu kullan, "
                            "söylemediyse bu alanı boş bırak."
                        ),
                    },
                },
                "required": ["book_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Sipariş kodundan (örn. 'SIP-1001') siparişin durumunu, kitabını, "
                "adedini ve tutarını döndürür. 'Siparişim nerede', 'SIP-1002 ne durumda' "
                "gibi sorularda çağır. Sipariş bilgisini asla kendin uydurma."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_code": {
                        "type": "string",
                        "description": "Sipariş kodu, örn. 'SIP-1001'.",
                    }
                },
                "required": ["order_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": (
                "Bir siparişi iptal eder ve kitapları stoğa geri ekler. Yalnızca "
                "'hazırlanıyor' veya 'kargoda' durumundaki siparişler iptal edilebilir; "
                "teslim edilmiş sipariş iptal edilemez (araç hata döndürür)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_code": {
                        "type": "string",
                        "description": "İptal edilecek siparişin kodu, örn. 'SIP-1003'.",
                    }
                },
                "required": ["order_code"],
            },
        },
    },
]

ARAC_TABLOSU = {
    "search_books": search_books,
    "create_order": create_order,
    "get_order_status": get_order_status,
    "cancel_order": cancel_order,
}

# Veritabanina yazan araclar. Ajan bunlari izde ayrica isaretliyor (kullanici
# hangi cagrinin kalici bir degisiklik yaptigini gorsun).
YAZAN_ARACLAR = {"create_order", "cancel_order"}


def ilk_cumle(metin: str) -> str:
    """Arac aciklamasinin ilk cumlesi (arayuz ve Space kartindaki ozetler icin).

    Nokta ile bolmek yetmiyor: aciklamalarda "(örn. 'SIP-1001')" gibi kisaltmalar
    var. Cumle sonu = "nokta + bosluk + buyuk harf/tirnak", kisaltmalardan sonra
    gelen noktalar haric.
    """
    parcalar = CUMLE_SONU_DESENI.split(metin.strip(), maxsplit=1)
    return parcalar[0].rstrip(".") + "."


def araci_calistir(ad: str, argumanlar: dict) -> dict:
    """Model tarafindan istenen araci calistirir.

    Hata durumunda exception firlatmaz; modelin okuyup toparlanabilecegi bir
    {"hata": "..."} sozlugu dondurur (tool calling'de hatayi modele geri
    beslemek, zinciri kirmaktan iyidir).
    """
    fonksiyon = ARAC_TABLOSU.get(ad)
    if fonksiyon is None:
        return {"hata": f"'{ad}' adlı bir araç yok. Mevcut araçlar: {list(ARAC_TABLOSU)}"}
    try:
        db.db_kur()  # veritabani yoksa ilk cagrida kurulur (idempotent)
        return fonksiyon(**(argumanlar or {}))
    except AracHatasi as e:
        return {"hata": str(e)}
    except TypeError as e:
        return {"hata": f"'{ad}' için geçersiz argümanlar: {e}"}
    except Exception as e:  # beklenmeyen durumda da zinciri kirmiyoruz
        return {"hata": f"'{ad}' çalıştırılırken beklenmeyen hata: {type(e).__name__}: {e}"}


if __name__ == "__main__":
    import json

    db.db_kur(sifirla=True)
    print(f"Veritabanı sıfırdan kuruldu: {db.DB_YOLU}\n")

    ornekler = [
        ("search_books", {"query": "Nietzsche"}),
        ("search_books", {"author": "Camus"}),
        ("search_books", {"max_price": 120, "only_in_stock": True}),
        ("create_order", {"book_id": 6, "quantity": 2, "customer_name": "Yusuf"}),
        ("get_order_status", {"order_code": "SIP-1003"}),
        ("cancel_order", {"order_code": "SIP-1003"}),
        # --- hata yollari ---
        ("search_books", {"query": "Simulakrlar ve Simulasyon"}),  # katalogda yok
        ("create_order", {"book_id": 4, "quantity": 1}),           # stok 0
        ("create_order", {"book_id": 3, "quantity": 5}),           # stok yetersiz (2)
        ("create_order", {"book_id": 999, "quantity": 1}),         # olmayan kitap
        ("create_order", {"book_id": 1, "quantity": 99}),          # adet siniri
        ("get_order_status", {"order_code": "SIP-9999"}),          # olmayan siparis
        ("cancel_order", {"order_code": "SIP-1001"}),              # teslim edilmis
        ("bilinmeyen_arac", {}),
    ]
    for ad, argumanlar in ornekler:
        print(f"-> {ad}({argumanlar})")
        print(f"<- {json.dumps(araci_calistir(ad, argumanlar), ensure_ascii=False)}\n")

    print("--- Son stok durumu (yazma araçlarının etkisi) ---")
    for k in db.katalogu_listele():
        if k["kitap_id"] in (3, 4, 6):
            print(f"  #{k['kitap_id']} {k['baslik']}: stok={k['stok']}")
