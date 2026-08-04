"""
7. Hafta: Tool Calling ajan dongusu (agentic loop).

Dongu, modelin nasil cagrildigindan bagimsiz (bkz. modeller.py):

    1. Kullanici sorusu + arac semalari modele gonderilir.
    2. Model arac cagirdiysa araclar calistirilir, sonuclar `role="tool"`
       mesajlari olarak gecmise eklenir ve model tekrar cagrilir.
    3. Model artik arac cagirmayip metin dondurdugunde dongu biter.

Her adim `TurIzi` / `AracIzi` olarak kaydedilir; `izi_metne_cevir()` bunu odevde
istenen "hangi araci neden cagirdim" formatinda dokuyor. `ajan_akisi()` bir
generator oldugu icin Gradio arayuzu adimlari olustukca ekrana basabiliyor.

Mesajlar arka uctan bagimsiz "kanonik" bicimde tutulur; her arka uc bunu kendi
tel formatina cevirir:
    {"role": "system"|"user", "content": str}
    {"role": "assistant", "content": str, "arac_cagrilari": [IstenenCagri]}
    {"role": "tool", "cagri_id": str, "ad": str, "content": json_str}

Calistirma (yerel test, tek soru):
    .venv/bin/python hafta7_tool_calling/ajan.py
    .venv/bin/python hafta7_tool_calling/ajan.py "Izmir'de yarin yagmur var mi?"
    TOOL_BACKEND=api .venv/bin/python hafta7_tool_calling/ajan.py   # API arka ucu
"""
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from araclar import ARAC_SEMALARI, araci_calistir  # noqa: E402
from modeller import model_yukle  # noqa: E402

MAKS_TUR = 4

SISTEM_PROMPTU = """Sen hava durumu ve hava kalitesi sorularını yanıtlayan bir asistansın. \
Verilere yalnızca sana verilen araçlar üzerinden erişebilirsin.

Kurallar:
- Hava, sıcaklık, tahmin veya hava kalitesi içeren her soruda mutlaka ilgili aracı çağır. \
Kendi belleğinden sayı uydurma, tahmin etme.
- Birden fazla şehir sorulduysa her şehir için ayrı çağrı yap.
- Fahrenheit veya Kelvin isteniyorsa önce Celsius değeri al, sonra convert_temperature \
aracıyla çevir. Çevirimi kafadan hesaplama; convert_temperature çağırmadan cevabında \
Fahrenheit veya Kelvin sayısı yazma.
- Bir araç hata döndürürse hatayı kullanıcıya sakin bir dille aktar, uydurma veriyle devam etme.
- Araçlardan gelen verileri topladıktan sonra Türkçe, kısa ve net bir cevap yaz. \
Sayıları birimleriyle birlikte ver.
- Hava dışında bir konu sorulursa araç çağırmadan kısaca yanıtla ve neyi yapabileceğini söyle.

Birim çevirimi iki adımda yapılır. Doğru davranış:
  Kullanıcı: "Ankara kaç Fahrenheit?"
  1. adım: get_weather(city="Ankara") çağır  ->  sicaklik_c: 27
  2. adım: convert_temperature(value=27, to_unit="F") çağır  ->  sonuc: 80.6
  3. adım: cevabı yaz  ->  "Ankara 80.6°F."
Yanlış davranış: 2. adımı atlayıp 80.6 sayısını kendin hesaplamak. Cevabında bir \
Fahrenheit veya Kelvin sayısı geçiyorsa, o sayı mutlaka convert_temperature \
çıktısından gelmiş olmalı."""


# Kullanicinin baska bir birim istedigini yakalayan desen.
BIRIM_ISTEGI_DESENI = re.compile(r"fahrenheit|kelvin|°\s*f\b|\bF['’]?a\b", re.IGNORECASE)
# Cevapta gecen Fahrenheit/Kelvin sayisi (ör. "60.4°F", "300 K").
BIRIM_SAYISI_DESENI = re.compile(r"\d+(?:[.,]\d+)?\s*°?\s*(?:F\b|K\b)")

DUZELTME_ISTEGI = (
    "Bu cevapta convert_temperature aracını çağırmadan Fahrenheit/Kelvin değeri "
    "yazdın. Çevirimi kendin hesaplamak yasak. Şimdi her sıcaklık için "
    "convert_temperature aracını çağır ve cevabı yalnızca aracın döndürdüğü "
    "değerlerle yeniden yaz."
)


@dataclass
class AracIzi:
    """Tek bir arac cagrisi ve donen sonuc."""

    ad: str
    argumanlar: dict
    sonuc: dict | None = None

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


def _dusunmeyi_ayikla(metin: str | None) -> tuple[str, str]:
    """<think>...</think> blogunu metinden ayirir.

    Qwen3 gibi "thinking" modelleri muhakemeyi cevabin icine gomer. Kullaniciya
    gosterilen yanitta bunu istemiyoruz ama iz panelinde gostermek isabetli
    (odev "adimlarini acikca goster" diyor). ("temiz metin", "dusunce") doner.
    """
    if not metin:
        return "", ""
    dusunceler = re.findall(r"<think>(.*?)</think>", metin, flags=re.DOTALL)
    temiz = re.sub(r"<think>.*?</think>", "", metin, flags=re.DOTALL)
    # Kapanmamis blok: gerisini dusunce say.
    kapanmamis = re.search(r"<think>(.*)", temiz, flags=re.DOTALL)
    if kapanmamis:
        dusunceler.append(kapanmamis.group(1))
        temiz = temiz[: kapanmamis.start()]
    return temiz.strip(), "\n".join(d.strip() for d in dusunceler if d.strip()).strip()


def _cevirim_atlandi_mi(soru: str, yanit: str, durum: AjanDurumu) -> bool:
    """Model, convert_temperature'i atlayip cevirimi kendisi mi yapti?

    Sadece prompt'a guvenmek yetmedi: Qwen2.5-7B ve 14B-Instruct sistem
    promptundaki kurala, arac cikisindaki nota ve promptdaki ornege ragmen
    Fahrenheit'i kafadan hesapliyor. Bu yuzden kural donguye tasindi - ihlali
    yakalayip modelden bir kez duzeltme istiyoruz (harness guardrail'i).
    """
    if not BIRIM_ISTEGI_DESENI.search(soru):
        return False  # kullanici zaten baska birim istemedi
    if any(iz.ad == "convert_temperature" for t in durum.turlar for iz in t.araclar):
        return False  # arac cagrilmis, sorun yok
    return bool(BIRIM_SAYISI_DESENI.search(yanit or ""))


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
        sohbet_modeli = model if model is not None else model_yukle()
    except Exception as e:
        durum.hata = f"Model hazırlanamadı: {type(e).__name__}: {e}"
        yield durum
        return

    durum.model = sohbet_modeli.ad
    durum.arka_uc = type(sohbet_modeli).__name__

    mesajlar: list[dict] = [{"role": "system", "content": SISTEM_PROMPTU}]
    mesajlar += gecmis or []
    mesajlar.append({"role": "user", "content": soru})

    duzeltme_istendi = False
    arac_zorla = False

    for tur_no in range(1, maks_tur + 1):
        try:
            yanit = sohbet_modeli.tamamla(mesajlar, ARAC_SEMALARI, arac_zorla=arac_zorla)
        except Exception as e:
            durum.hata = _model_hatasi(sohbet_modeli.ad, e)
            yield durum
            return

        metin, dusunce = _dusunmeyi_ayikla(yanit.metin)

        # Model arac cagirmadi -> bu, nihai yanit (ihlal yoksa).
        if not yanit.arac_cagrilari:
            if not duzeltme_istendi and _cevirim_atlandi_mi(soru, metin, durum):
                # Duzeltmeyi rica etmek yetmiyor (bkz. _cevirim_atlandi_mi):
                # sonraki tur arac cagrisi zorunlu tutuluyor.
                duzeltme_istendi = True
                arac_zorla = True
                durum.duzeltme_notu = (
                    "convert_temperature çağrılmadan Fahrenheit/Kelvin yazıldı; "
                    "araç çağrısı zorunlu tutularak düzeltme istendi"
                )
                mesajlar.append({"role": "assistant", "content": metin})
                mesajlar.append({"role": "user", "content": DUZELTME_ISTEGI})
                yield durum
                continue

            durum.nihai_dusunce = dusunce
            durum.nihai_yanit = metin or "Model boş bir yanıt döndürdü."
            yield durum
            return

        # Tur numarasi arac cagiran turlara gore verilir: duzeltme istenen
        # (arac cagrilmayan) deneme numara tuketmesin, iz paneli 1,2,3... aksin.
        tur = TurIzi(no=len(durum.turlar) + 1, ara_metin=metin, dusunce=dusunce)
        durum.turlar.append(tur)
        arac_zorla = False  # zorlama yalnizca duzeltme turu icin geçerli
        mesajlar.append(
            {"role": "assistant", "content": metin, "arac_cagrilari": yanit.arac_cagrilari}
        )

        for cagri in yanit.arac_cagrilari:
            iz = AracIzi(ad=cagri.ad, argumanlar=cagri.argumanlar)
            tur.araclar.append(iz)
            yield durum  # "-> get_weather(city='Ankara')" satiri hemen gorunsun

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
            satirlar.append(f"   -> {iz.cagri_metni}")
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


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    soru = " ".join(sys.argv[1:]) or (
        "Ankara mı daha sıcak Londra mı, ve bu değerler Fahrenheit olarak kaç eder?"
    )
    print(f"Soru: {soru}")
    print(f"Arka uç: {os.environ.get('TOOL_BACKEND', 'yerel')}\n")
    durum = ajan_calistir(soru)
    print(izi_metne_cevir(durum))
    print(
        f"\n(model: {durum.model} / {durum.arka_uc}, "
        f"araç çağrısı: {durum.arac_cagri_sayisi})"
    )
