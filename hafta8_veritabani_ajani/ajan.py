"""
8. Hafta: veritabanina baglanan tool calling ajani (agentic loop).

Dongu 7. haftadakiyle ayni iskelet:

    1. Kullanici sorusu + arac semalari modele gonderilir.
    2. Model arac cagirdiysa araclar calistirilir, sonuclar `role="tool"`
       mesajlari olarak gecmise eklenir ve model tekrar cagrilir.
    3. Model artik arac cagirmayip metin dondurdugunde dongu biter.

Bu haftaya ozel kisim **halusinasyon guardrail'i** (`_dogrulanmamis_veri`).
Sistem promptunda "veritabaninda olmayan kitabi varmis gibi sunma" demek tek
basina yetmiyor; kural donguye tasindi. Nihai yanit yayinlanmadan once sunlar
denetleniyor:

  - Katalog/siparis sorusuna **hic arac cagirmadan** cevap verilmis mi?
  - "Siparisiniz alindi" denmis ama veritabanina yazan arac hic calismamis mi?
  - Cevaptaki siparis kodlari (SIP-...), tirnak icindeki kitap adlari ve
    fiyatlar (TL) arac ciktilarindan mi geliyor, yoksa model uydurmus mu?

Ihlal varsa modelden bir kez duzeltme isteniyor ve o turda arac cagrisi zorunlu
tutuluyor (`arac_zorla=True`). Mudahale iz panelinde `[!] harness uyarisi`
satiri olarak gorunur - guardrail de kullanicidan gizlenmiyor.

Model katmani (`modeller.py`) 7. haftadan oldugu gibi yeniden kullaniliyor;
arka uc secimi (ZeroGPU/transformers veya HF Inference Providers) oradaki
`TOOL_BACKEND` ile yapiliyor.

Calistirma (yerel test, tek soru):
    .venv/bin/python hafta8_veritabani_ajani/ajan.py
    .venv/bin/python hafta8_veritabani_ajani/ajan.py "SIP-1002 ne durumda?"
    TOOL_BACKEND=api .venv/bin/python hafta8_veritabani_ajani/ajan.py
"""
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

HERE = Path(__file__).resolve().parent
# modeller.py 7. haftadan geliyor (kopyalamak yerine ayni modulu kullaniyoruz).
# Space'e deploy edilirken dosya app.py'nin yanina kopyalandigi icin orada
# dogrudan HERE'den cozulur; asagidaki sira once kendi klasorune bakar.
sys.path.insert(0, str(HERE.parent / "hafta7_tool_calling"))
sys.path.insert(0, str(HERE))

import veritabani as db  # noqa: E402
from araclar import ARAC_SEMALARI, YAZAN_ARACLAR, araci_calistir  # noqa: E402
from modeller import model_yukle  # noqa: E402

MAKS_TUR = 5  # search_books -> create_order -> nihai yanit + guardrail payi

SISTEM_PROMPTU = """Sen bir felsefe kitapçısının sipariş asistanısın. Kitapçının kataloğuna, \
stok bilgisine ve sipariş kayıtlarına YALNIZCA sana verilen araçlar üzerinden erişebilirsin.

Kurallar:
- Kitap, yazar, akım, fiyat veya stok içeren her soruda mutlaka search_books aracını çağır. \
Katalogda hangi kitapların olduğunu kendi belleğinden bildiğini varsayma; bu kitapçının \
stoğu genel felsefe bilginle aynı değildir.
- Araç çıktısında olmayan bir kitabı, fiyatı, stok sayısını veya sipariş kodunu ASLA yazma. \
Aradığı kitap listede yoksa "kitapçıda yok" de, benzer bir kitap uydurma.
- Sipariş oluşturmak için önce search_books ile kitabın kitap_id'sini bul, sonra o id ile \
create_order çağır. kitap_id'yi tahmin etme.
- Sipariş durumu sorulduğunda get_order_status, iptal istendiğinde cancel_order çağır. \
Sipariş kodunu kullanıcı vermediyse iste.
- Bir araç hata döndürürse (stok yok, sipariş bulunamadı gibi) hatayı kullanıcıya sakin bir \
dille aktar; uydurma veriyle devam etme.
- Araçlardan gelen veriyi topladıktan sonra Türkçe, kısa ve net bir cevap yaz. Fiyatları \
TL ile, sipariş kodunu araç çıktısındaki haliyle ver.
- Kitapçılık dışında bir konu sorulursa araç çağırmadan kısaca yanıtla ve ne yapabileceğini söyle.

Sipariş iki adımda verilir. Doğru davranış:
  Kullanıcı: "Sisifos Söyleni'nden 2 tane sipariş ver."
  1. adım: search_books(query="Sisifos Söyleni")  ->  kitap_id: 6, fiyat_tl: 110.0, stok: 7
  2. adım: create_order(book_id=6, quantity=2)    ->  siparis_kodu: "SIP-1003", toplam_tl: 220.0
  3. adım: cevabı yaz -> "Siparişiniz alındı: 2 adet Sisifos Söyleni, 220 TL. Kodunuz SIP-1003."
Yanlış davranış: search_books'u atlayıp kitap_id'yi tahmin etmek ya da sipariş kodunu kendin \
uydurmak."""

# Kullanicinin sorusu katalog/siparis konusuna mi giriyor?
KATALOG_ISTEGI_DESENI = re.compile(
    r"kitap|kitab|stok|fiyat|kaç para|kaça|ucuz|pahalı|sipariş|siparis|sepet|satın al|"
    r"yazar|eser|SIP-\d+|₺|\bTL\b",
    re.IGNORECASE,
)
# Cevapta gecen para tutari ("145 TL", "220,50 ₺").
FIYAT_DESENI = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:TL|₺|lira)\b", re.IGNORECASE)
# Cevapta gecen siparis kodu.
KOD_DESENI = re.compile(r"\bSIP-\d+\b", re.IGNORECASE)
# Cevapta "siparisiniz alindi" gibi bir *yazma* iddiasi. Olumsuz cekimler
# ("verilemedi", "iptal edilemedi") bilincli olarak eslesmiyor.
YAZMA_IDDIASI_DESENI = re.compile(
    r"sipariş\w*\s+(?:\S+\s+){0,3}?(?:alındı|verildi|oluşturuldu|kaydedildi|tamamlandı)"
    r"|iptal\s+edildi",
    re.IGNORECASE,
)
# Cevapta tirnak icine alinmis ifade - modeller kitap adlarini boyle yaziyor.
# Turkce kesme isareti ("Camus'nun") yuzunden tek tirnak ayrac sayilmiyor.
TIRNAK_DESENI = re.compile(r"[\"“”«»]([^\"“”«»]{4,70})[\"“”«»]")


def _normalize(metin: str) -> str:
    """Karsilastirma icin sadelestirir: Turkce'ye duyarli kucuk harf + tek boşluk.

    `str.lower()` Turkce'de 'I' -> 'i' yapiyor ('ı' degil); kitap adlari
    ("Arı Usun Eleştirisi") bu yuzden once elle esleniyor.
    """
    metin = metin.replace("İ", "i").replace("I", "ı")
    return " ".join(metin.lower().split())


# Tirnak icinde gecse de kitap adi sayilmayacak ifadeler: siparis durumlari ve
# arac adlari. (Model "durumu 'kargoda'" yazdiginda guardrail tetiklenmesin.)
TIRNAK_MUAFLARI = {_normalize(d) for d in db.DURUMLAR} | {
    _normalize(s["function"]["name"]) for s in ARAC_SEMALARI
}

DUZELTME_ISTEGI = (
    "Bu cevapta veritabanından gelmeyen bilgi var: {ihlal}. Kitapçının kataloğu ve "
    "sipariş kayıtları hakkında yalnızca araçlardan dönen veriyi kullanabilirsin. "
    "Şimdi gerekli aracı çağır ve cevabı yalnızca aracın döndürdüğü veriyle yeniden yaz."
)

# Duzeltme istendigi halde model israr ederse cevap sessizce gecmiyor: dogrulanmamis
# oldugu kullaniciya yazili olarak soyleniyor. (Kucuk modellerde gorulen durum -
# guardrail bir arac cagrisini zorlayabiliyor ama *dogru* araci sectiremiyor.)
DOGRULANMADI_UYARISI = (
    "⚠️ **Doğrulanmadı:** {ihlal}. Yukarıdaki cevabın bu kısmı kitapçının "
    "veritabanından teyit edilemedi, lütfen esas almayın."
)


@dataclass
class AracIzi:
    """Tek bir arac cagrisi ve donen sonuc."""

    ad: str
    argumanlar: dict
    sonuc: dict | None = None

    @property
    def yazma(self) -> bool:
        """Veritabanina yazan bir cagri mi (izde ayrica isaretlenir)."""
        return self.ad in YAZAN_ARACLAR

    @property
    def cagri_metni(self) -> str:
        argv = ", ".join(f"{k}={v!r}" for k, v in (self.argumanlar or {}).items())
        return f"{self.ad}({argv})"

    @property
    def sonuc_metni(self) -> str:
        if self.sonuc is None:
            return "(çalışıyor...)"
        return json.dumps(self.sonuc, ensure_ascii=False)


@dataclass
class TurIzi:
    """Modelin bir turu: cagirdigi araclar, varsa ara metni ve dusunce blogu."""

    no: int
    araclar: list[AracIzi] = field(default_factory=list)
    ara_metin: str = ""
    dusunce: str = ""


@dataclass
class AjanDurumu:
    """Ajanin o anki durumu. Generator her adimda bunu yayinlar."""

    soru: str
    turlar: list[TurIzi] = field(default_factory=list)
    nihai_yanit: str | None = None
    nihai_dusunce: str = ""
    hata: str | None = None
    model: str = ""
    arka_uc: str = ""
    duzeltme_notu: str = ""

    @property
    def arac_cagri_sayisi(self) -> int:
        return sum(len(t.araclar) for t in self.turlar)

    @property
    def yazma_sayisi(self) -> int:
        return sum(1 for t in self.turlar for iz in t.araclar if iz.yazma)

    @property
    def basarili_yazma_sayisi(self) -> int:
        """Veritabanini gercekten degistiren cagri sayisi (hata donenler haric)."""
        return sum(
            1
            for iz in self.arac_izleri()
            if iz.yazma and iz.sonuc is not None and "hata" not in iz.sonuc
        )

    def arac_izleri(self) -> list[AracIzi]:
        return [iz for t in self.turlar for iz in t.araclar]


def _dusunmeyi_ayikla(metin: str | None) -> tuple[str, str]:
    """<think>...</think> blogunu metinden ayirir.

    Qwen3 gibi "thinking" modelleri muhakemeyi cevabin icine gomer; kullaniciya
    gosterilen yanitta bunu istemiyoruz ama iz panelinde gostermek isabetli.
    ("temiz metin", "dusunce") doner.
    """
    if not metin:
        return "", ""
    dusunceler = re.findall(r"<think>(.*?)</think>", metin, flags=re.DOTALL)
    temiz = re.sub(r"<think>.*?</think>", "", metin, flags=re.DOTALL)
    kapanmamis = re.search(r"<think>(.*)", temiz, flags=re.DOTALL)
    if kapanmamis:
        dusunceler.append(kapanmamis.group(1))
        temiz = temiz[: kapanmamis.start()]
    return temiz.strip(), "\n".join(d.strip() for d in dusunceler if d.strip()).strip()


# --------------------------------------------------------------------------
# Halusinasyon guardrail'i
# --------------------------------------------------------------------------


def _sayilari_topla(veri, kova: set[float]) -> None:
    """Arac ciktisindaki butun sayilari (ic ice sozluk/liste dahil) toplar."""
    if isinstance(veri, bool):
        return
    if isinstance(veri, (int, float)):
        kova.add(float(veri))
    elif isinstance(veri, dict):
        for deger in veri.values():
            _sayilari_topla(deger, kova)
    elif isinstance(veri, list):
        for deger in veri:
            _sayilari_topla(deger, kova)


def _arac_gercekleri(durum: AjanDurumu) -> tuple[set[float], set[str], str]:
    """Arac ciktilarindan gelen sayilar, siparis kodlari ve ham metin.

    Bu uclu, cevabin "dogrulanmis veri" kumesi: cevapta gecen her fiyat, her
    siparis kodu ve her kitap adi bunlardan biriyle aciklanabilmeli.
    """
    sayilar: set[float] = set()
    kodlar: set[str] = set()
    metinler: list[str] = []
    for iz in durum.arac_izleri():
        if iz.sonuc is None:
            continue
        _sayilari_topla(iz.sonuc, sayilar)
        ham = json.dumps(iz.sonuc, ensure_ascii=False)
        kodlar.update(k.upper() for k in KOD_DESENI.findall(ham))
        metinler.append(ham)
    return sayilar, kodlar, _normalize(" ".join(metinler))



def _fiyat_dogrulanmis(deger: float, arac_sayilari: set[float]) -> bool:
    """Cevaptaki bir tutar arac ciktisiyla aciklanabiliyor mu?

    Birebir esitligin yani sira "birim fiyat x adet" carpimlarina da izin
    veriyoruz: kullanici "3 tane alsam kaç eder?" diye sordugunda model henuz
    siparis olusturmadan (yani toplam_tl araca donmeden) dogru hesabi yapabilir.
    Uydurulan katalog fiyatlari bu kapiya takilmaya devam eder.
    """
    for gercek in arac_sayilari:
        if abs(deger - gercek) <= 0.01:
            return True
        for adet in range(2, 11):  # araclardaki MAKS_ADET ile ayni ust sinir
            if abs(deger - gercek * adet) <= 0.01:
                return True
    return False


def _dogrulanmamis_veri(soru: str, yanit: str, durum: AjanDurumu) -> str:
    """Nihai yanitta veritabanindan gelmeyen bilgi var mi?

    Bos string = sorun yok. Dolu string, ihlalin kisa aciklamasi (hem duzeltme
    mesajinda hem iz panelinde kullaniliyor).
    """
    yanit = yanit or ""
    katalog_sorusu = bool(KATALOG_ISTEGI_DESENI.search(soru))

    # 1) Katalog/siparis sorusuna hic araca gitmeden cevap verilmis.
    if katalog_sorusu and not durum.arac_cagri_sayisi and len(yanit.split()) > 3:
        return "hiç araç çağrılmadan katalog/sipariş bilgisi verildi"

    # 2) "Siparişiniz alındı" deniyor ama veritabanina yazan bir cagri basarili
    #    olmamis. (Qwen2.5-0.5B, create_order'i hic cagirmadan "2 adet sipariş
    #    verildi" yazarken bu kurala takildi - en tehlikeli halusinasyon turu,
    #    cunku kullanici olmayan bir siparisi beklemeye baslar.)
    if YAZMA_IDDIASI_DESENI.search(yanit) and not durum.basarili_yazma_sayisi:
        return "sipariş/iptal yapıldığı söylendi ama veritabanına yazan araç çalışmadı"

    arac_sayilari, arac_kodlari, arac_metni = _arac_gercekleri(durum)

    # 3) Cevaptaki siparis kodu hicbir arac ciktisinda gecmiyor.
    uydurma_kodlar = {
        k.upper() for k in KOD_DESENI.findall(yanit) if k.upper() not in arac_kodlari
    }
    if uydurma_kodlar:
        return f"araç çıktısında olmayan sipariş kodu: {', '.join(sorted(uydurma_kodlar))}"

    # 4) Cevapta tirnak icine alinmis bir kitap adi var ama ne arac ciktisinda
    #    ne de kullanicinin sorusunda geciyor -> model kitabi uydurmus.
    #    (Qwen2.5-7B, stogu tukenmis "Varlık ve Hiçlik" yerine olmayan bir
    #    "Yaratıcılık ve Hiçlik" kitabini onerirken bu kurala takildi.)
    if durum.arac_cagri_sayisi:
        soru_metni = _normalize(soru)
        for ham_ad in TIRNAK_DESENI.findall(yanit):
            ad = _normalize(ham_ad)
            if ad in TIRNAK_MUAFLARI:  # kitap adi degil, alan/durum adi
                continue
            if ad and ad not in arac_metni and ad not in soru_metni:
                return f"araç çıktısında olmayan kitap/ifade: “{ham_ad.strip()}”"

    # 5) Cevaptaki tutar hicbir arac ciktisiyla aciklanamiyor.
    for ham in FIYAT_DESENI.findall(yanit):
        try:
            deger = float(ham.replace(".", "").replace(",", ".") if "," in ham else ham)
        except ValueError:
            continue
        if not _fiyat_dogrulanmis(deger, arac_sayilari):
            return f"araç çıktısında olmayan fiyat: {ham} TL"

    return ""


def _model_hatasi(model: str, e: Exception) -> str:
    """Model cagrisi hatasini kullaniciya anlamli bir mesaja cevirir."""
    metin = str(e)
    if "402" in metin or "credits" in metin.lower():
        return (
            f"HF Inference Providers aylık ücretsiz kredisi tükenmiş ({model}).\n\n"
            "Çözüm seçenekleri:\n"
            "- `TOOL_BACKEND=yerel` ile modeli ZeroGPU/yerel GPU üzerinde çalıştırın "
            "(kredi harcamaz),\n"
            "- kredinin aylık yenilenmesini bekleyin,\n"
            "- veya `TOOL_BASE_URL` + `TOOL_API_KEY` ile OpenAI uyumlu başka bir "
            "sağlayıcıya yönlendirin."
        )
    if "401" in metin or "403" in metin:
        return (
            f"Model çağrısı yetkilendirilemedi ({model}). HF_TOKEN tanımlı mı ve "
            "token'da 'Make calls to Inference Providers' izni açık mı kontrol edin.\n\n"
            f"Ham hata: {metin}"
        )
    if "404" in metin:
        return (
            f"'{model}' bulunamadı. TOOL_MODEL değerini kontrol edin (tool calling "
            "destekleyen bir sohbet modeli olmalı).\n\n"
            f"Ham hata: {metin}"
        )
    if "out of memory" in metin.lower() or "CUDA" in metin:
        return (
            f"Model belleğe sığmadı ya da GPU hatası ({model}): {metin}\n\n"
            "Daha küçük bir TOOL_MODEL deneyin (örn. Qwen/Qwen2.5-3B-Instruct)."
        )
    return f"Model çağrısı başarısız ({model}): {type(e).__name__}: {metin}"


def ajan_akisi(
    soru: str,
    gecmis: list[dict] | None = None,
    model=None,
    maks_tur: int = MAKS_TUR,
) -> Iterator[AjanDurumu]:
    """Ajani calistirir ve her adimda guncel durumu yayinlar (generator).

    `gecmis`: arayuzdeki sohbet gecmisi ({"role": "user"/"assistant", "content"}).
    Arac mesajlari gecmise tasinmaz; her soru kendi arac zincirini kurar.
    `model`: hazir bir model nesnesi (modeller.model_yukle() cikisi). Verilmezse
    TOOL_BACKEND'e gore olusturulur - Space'te model bir kez yuklenip yeniden
    kullanilsin diye disaridan verilebiliyor.
    """
    durum = AjanDurumu(soru=soru)

    try:
        db.db_kur()  # veritabani yoksa semayi kur + ornek katalogu yaz
        sohbet_modeli = model if model is not None else model_yukle()
    except Exception as e:
        durum.hata = f"Ajan hazırlanamadı: {type(e).__name__}: {e}"
        yield durum
        return

    durum.model = sohbet_modeli.ad
    durum.arka_uc = type(sohbet_modeli).__name__

    mesajlar: list[dict] = [{"role": "system", "content": SISTEM_PROMPTU}]
    mesajlar += gecmis or []
    mesajlar.append({"role": "user", "content": soru})

    duzeltme_istendi = False
    arac_zorla = False

    for _ in range(maks_tur):
        try:
            yanit = sohbet_modeli.tamamla(mesajlar, ARAC_SEMALARI, arac_zorla=arac_zorla)
        except Exception as e:
            durum.hata = _model_hatasi(sohbet_modeli.ad, e)
            yield durum
            return

        metin, dusunce = _dusunmeyi_ayikla(yanit.metin)

        # Model arac cagirmadi -> bu, nihai yanit (guardrail temizse).
        if not yanit.arac_cagrilari:
            ihlal = _dogrulanmamis_veri(soru, metin, durum)
            if ihlal and not duzeltme_istendi:
                # Ricayla yetinmiyoruz: sonraki tur arac cagrisi zorunlu.
                duzeltme_istendi = True
                arac_zorla = True
                durum.duzeltme_notu = f"{ihlal}; araç çağrısı zorunlu tutularak düzeltme istendi"
                mesajlar.append({"role": "assistant", "content": metin})
                mesajlar.append({"role": "user", "content": DUZELTME_ISTEGI.format(ihlal=ihlal)})
                yield durum
                continue

            if ihlal:
                # Duzeltme istendi ama model israr etti. Dongude kilitlenmiyoruz;
                # cevabi veriyoruz ama dogrulanmadigini acikca yaziyoruz.
                durum.duzeltme_notu = (
                    f"{ihlal}; düzeltme istendi, model ısrar etti — cevap uyarıyla işaretlendi"
                )
                metin = (metin + "\n\n" if metin else "") + DOGRULANMADI_UYARISI.format(ihlal=ihlal)

            durum.nihai_dusunce = dusunce
            durum.nihai_yanit = metin or "Model boş bir yanıt döndürdü."
            yield durum
            return

        # Tur numarasi arac cagiran turlara gore verilir: duzeltme istenen
        # (arac cagrilmayan) deneme numara tuketmesin, iz paneli 1,2,3... aksin.
        tur = TurIzi(no=len(durum.turlar) + 1, ara_metin=metin, dusunce=dusunce)
        durum.turlar.append(tur)
        arac_zorla = False  # zorlama yalnizca duzeltme turu icin gecerli
        mesajlar.append(
            {"role": "assistant", "content": metin, "arac_cagrilari": yanit.arac_cagrilari}
        )

        for cagri in yanit.arac_cagrilari:
            iz = AracIzi(ad=cagri.ad, argumanlar=cagri.argumanlar)
            tur.araclar.append(iz)
            yield durum  # "-> search_books(query='Camus')" satiri hemen gorunsun

            iz.sonuc = araci_calistir(iz.ad, iz.argumanlar)
            mesajlar.append(
                {
                    "role": "tool",
                    "cagri_id": cagri.id,
                    "ad": cagri.ad,
                    "content": json.dumps(iz.sonuc, ensure_ascii=False),
                }
            )
            yield durum  # "<- {...}" satiri

    durum.hata = (
        f"Model {maks_tur} turda araç çağırmayı bitirmedi, döngü durduruldu. "
        "Soruyu daha basit ifade etmeyi deneyin."
    )
    yield durum


def ajan_calistir(soru: str, **kwargs) -> AjanDurumu:
    """Generator'u sonuna kadar tuketip son durumu dondurur (senkron kullanim)."""
    durum = AjanDurumu(soru=soru)
    for durum in ajan_akisi(soru, **kwargs):
        pass
    return durum


def _tek_satir(metin: str, sinir: int = 300) -> str:
    """Cok satirli dusunce blogunu iz panelinde tek satira sigdirir."""
    duz = " ".join(metin.split())
    return duz if len(duz) <= sinir else duz[: sinir - 1] + "…"


def izi_metne_cevir(durum: AjanDurumu) -> str:
    """Arac cagri izini odevde istenen seffaf adim formatinda dokur."""
    satirlar: list[str] = []
    if durum.duzeltme_notu:
        satirlar.append(f"[!] harness uyarısı: {durum.duzeltme_notu}")
        satirlar.append("")
    for tur in durum.turlar:
        satirlar.append(f"[Tur {tur.no}] Araç Çağrıları:")
        if tur.dusunce:
            satirlar.append(f"   # model düşüncesi: {_tek_satir(tur.dusunce)}")
        if tur.ara_metin:
            satirlar.append(f"   # model notu: {_tek_satir(tur.ara_metin)}")
        for iz in tur.araclar:
            etiket = " [DB YAZMA]" if iz.yazma else ""
            satirlar.append(f"   -> {iz.cagri_metni}{etiket}")
            satirlar.append(f"   <- {iz.sonuc_metni}")
        satirlar.append("")

    if durum.hata:
        satirlar.append(f"[HATA] {durum.hata}")
    elif durum.nihai_yanit:
        satirlar.append(f"[Tur {len(durum.turlar) + 1}] Nihai Yanıt:")
        if durum.nihai_dusunce:
            satirlar.append(f"   # model düşüncesi: {_tek_satir(durum.nihai_dusunce)}")
        satirlar.append(durum.nihai_yanit)
    else:
        satirlar.append("(model yanıtı bekleniyor...)")

    return "\n".join(satirlar).strip()


def _guardrail_testi() -> None:
    """Guardrail'i modele hic gitmeden test eder (`ajan.py --guardrail`).

    Vakalar gercek calistirmalardan derlendi; ozellikle "uydurma kitap" satiri
    Qwen2.5-7B'nin canli olarak yaptigi halusinasyonun birebir kaydi.
    """
    arama = (
        "search_books",
        {"query": "Varlık ve Hiçlik"},
        {"kitaplar": [{"kitap_id": 4, "baslik": "Varlık ve Hiçlik", "fiyat_tl": 420.0, "stok": 0}]},
    )
    camus = (
        "search_books",
        {"author": "Camus"},
        {"kitaplar": [{"kitap_id": 6, "baslik": "Sisifos Söyleni", "fiyat_tl": 110.0, "stok": 7}]},
    )
    siparis = (
        "get_order_status",
        {"order_code": "SIP-1002"},
        {"siparis_kodu": "SIP-1002", "kitap": "Sisifos Söyleni", "toplam_tl": 220.0,
         "durum": "kargoda"},
    )
    yazma = (
        "create_order",
        {"book_id": 6, "quantity": 2},
        {"siparis_kodu": "SIP-1003", "kitap": "Sisifos Söyleni", "adet": 2,
         "birim_fiyat_tl": 110.0, "toplam_tl": 220.0, "kalan_stok": 5},
    )

    # (ad, soru, modelin yaniti, arac cagrilari, ihlal bekleniyor mu)
    vakalar = [
        ("araç çağrılmadı", "Nietzsche'nin kitabı var mı?",
         "Evet, Böyle Buyurdu Zerdüşt 145 TL.", [], True),
        ("alakasız soru, araç yok", "Fransa'nın başkenti neresi?",
         "Paris. Ben kitapçı asistanıyım.", [], False),
        ("uydurma kitap adı", "Varlık ve Hiçlik'ten bir tane istiyorum.",
         'Yok. Ancak "Yaratıcılık ve Hiçlik" öneriyorum, 420 TL.', [arama], True),
        ("gerçek kitap adı", "Varlık ve Hiçlik lazım",
         '"Varlık ve Hiçlik" katalogda var ama stoğu tükenmiş (420 TL).', [arama], False),
        ("tırnakta durum adı", "SIP-1002 ne durumda?",
         'Siparişiniz "kargoda" görünüyor, tutar 220 TL.', [siparis], False),
        ("yapılmamış sipariş iddiası", "Camus'dan 2 tane sipariş ver.",
         'Kitap "Sisifos Söyleni" için 2 adet sipariş verildi.', [camus], True),
        ("gerçek sipariş", "Camus'dan 2 tane sipariş ver.",
         "Siparişiniz alındı: 2 adet Sisifos Söyleni, 220 TL. Kodunuz SIP-1003.",
         [camus, yazma], False),
        ("sipariş verilemedi", "Varlık ve Hiçlik'ten bir tane.",
         "Stok tükendiği için sipariş verilemedi.", [arama], False),
        ("uydurma sipariş kodu", "sipariş ver",
         "Siparişiniz alındı, kodunuz SIP-4242.", [camus, yazma], True),
        ("uydurma fiyat", "Camus'nun kitabı kaç para?",
         "Sisifos Söyleni 89 TL.", [camus], True),
        ("birim fiyat x adet", "3 tane alsam kaç eder?",
         "3 adet 330 TL eder.", [camus], False),
    ]

    basarisiz = 0
    for ad, soru, yanit, cagrilar, ihlal_bekleniyor in vakalar:
        durum = AjanDurumu(soru=soru)
        if cagrilar:
            tur = TurIzi(no=1)
            tur.araclar = [AracIzi(ad=a, argumanlar=g, sonuc=s) for a, g, s in cagrilar]
            durum.turlar.append(tur)
        ihlal = _dogrulanmamis_veri(soru, yanit, durum)
        gecti = bool(ihlal) == ihlal_bekleniyor
        basarisiz += not gecti
        print(f"  [{'OK ' if gecti else 'HATA'}] {ad:<24} -> {ihlal or 'temiz'}")

    print(f"\n{len(vakalar) - basarisiz}/{len(vakalar)} vaka geçti.")
    if basarisiz:
        raise SystemExit(1)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    if "--guardrail" in sys.argv:
        print("Halüsinasyon guardrail'i (model çağrısı yok):\n")
        _guardrail_testi()
        raise SystemExit(0)

    soru = " ".join(sys.argv[1:]) or (
        "Camus'nun kitabı var mı? Varsa 2 tane sipariş ver, adım Yusuf."
    )
    print(f"Soru: {soru}")
    print(f"Arka uç: {os.environ.get('TOOL_BACKEND', 'yerel')}")
    print(f"Veritabanı: {db.DB_YOLU}\n")
    durum = ajan_calistir(soru)
    print(izi_metne_cevir(durum))
    print(
        f"\n(model: {durum.model} / {durum.arka_uc}, "
        f"araç çağrısı: {durum.arac_cagri_sayisi}, DB yazma: {durum.yazma_sayisi})"
    )
