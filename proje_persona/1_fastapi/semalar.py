"""
Pydantic modelleri - gelen ve giden verinin tip sozlesmesi.

Buradaki her model README'deki bir iddiayi karsiliyor:
  SarkiOlustur  -> gelen veri dogrulamasi (kisit, ozel dogrulayici, extra=forbid)
  SarkiYanit    -> giden veri dogrulamasi (ic alanlar disariya sizmaz)
  SarkiGuncelle -> kismi guncelleme (butun alanlar opsiyonel)
  HataYaniti    -> elle uretilen hatalarin da sabit bir sekli olsun diye

Bu demo sadece sarki KUNYESI tutuyor (baslik, albüm, yil, sure). Sarki sozu
metni burada yok; o 3_rag_persona'nin isi.
"""
from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class Tur(str, Enum):
    """Literal yerine Enum: /docs'ta acilir liste olarak render ediliyor."""

    RAP = "rap"
    SKIT = "skit"
    ENSTRUMANTAL = "enstrumantal"


# Annotated ile kisitlari tekrar tekrar yazmak yerine tip takma adi uretiyoruz.
Baslik = Annotated[str, Field(min_length=1, max_length=120, description="Sarki adi")]
Yil = Annotated[int, Field(ge=1994, le=2100, description="Yayin yili")]
Saniye = Annotated[int, Field(gt=0, le=3600, description="Sure (saniye)")]


class SarkiTemel(BaseModel):
    # extra="forbid": gonderilen ama modelde olmayan alan sessizce yutulmaz,
    # 422 doner. Varsayilan davranis ("ignore") yazim hatalarini gizler:
    # {"basslik": "..."} gonderirsen alan bos kalir ve fark etmezsin.
    model_config = ConfigDict(extra="forbid")

    baslik: Baslik
    album: Annotated[str, Field(min_length=1, max_length=120)]
    yil: Yil
    sure_sn: Saniye
    tur: Tur = Tur.RAP

    @field_validator("baslik", "album")
    @classmethod
    def bosluk_kirp(cls, v: str) -> str:
        # Dogrulayici min_length'ten SONRA calisir, o yuzden "   " (3 bosluk)
        # min_length=1'i gecer ama burada bos stringe dusup hata verir.
        v = v.strip()
        if not v:
            raise ValueError("sadece bosluktan olusamaz")
        return v

    @field_validator("yil")
    @classmethod
    def gelecek_olamaz(cls, v: int) -> int:
        # ge/le ile ifade edilemeyen kural: "gelecek yil" degisken bir sinir.
        simdi = datetime.now().year
        if v > simdi:
            raise ValueError(f"gelecekteki bir yil olamaz (su an {simdi})")
        return v


class SarkiOlustur(SarkiTemel):
    """POST /sarkilar gövdesi."""


class SarkiGuncelle(BaseModel):
    """PUT /sarkilar/{id} gövdesi - kismi guncelleme, hepsi opsiyonel."""

    model_config = ConfigDict(extra="forbid")

    baslik: Baslik | None = None
    album: Annotated[str, Field(min_length=1, max_length=120)] | None = None
    yil: Yil | None = None
    sure_sn: Saniye | None = None
    tur: Tur | None = None


class SarkiKayit(SarkiTemel):
    """
    Sunucunun icerideki hali. `ic_not` ve `kaynak_ip` disariya cikmamali.
    Yanit modeli olarak ASLA bu kullanilmiyor - bkz. SarkiYanit.
    """

    id: int
    ic_not: str = ""
    kaynak_ip: str = ""


class SarkiYanit(BaseModel):
    """
    GET/POST yanitlari. SarkiKayit'in yalnizca disari acilan alanlari.

    response_model olarak bunu vermek "giden veri dogrulamasi"nin ta kendisi:
    endpoint yanlislikla SarkiKayit dondurse bile FastAPI ic alanlari kirpar.

    from_attributes: FastAPI response_model uzerinden donerken bu ayar
    gerekmiyor (kendisi hallediyor) ama SayfaliYanit'i endpoint icinde ELLE
    kurarken gerekiyor - orada duz Pydantic dogrulamasi calisiyor ve
    SarkiKayit NESNESI dict olmadigi icin reddedilir.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    baslik: str
    album: str
    yil: int
    sure_sn: int
    tur: Tur

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sure_mmss(self) -> str:
        """Turetilmis alan: istemcinin ayrica hesaplamasi gerekmesin."""
        return f"{self.sure_sn // 60}:{self.sure_sn % 60:02d}"


class SayfaliYanit(BaseModel):
    """Liste uclari icin sarmalayici - toplam sayi olmadan sayfalama yapilamaz."""

    toplam: int
    limit: int
    offset: int
    kayitlar: list[SarkiYanit]


class HataYaniti(BaseModel):
    detay: str
