"""
Persona katmani: retrieval -> prompt -> LLM -> GUARDRAIL.

Odev "sarki sozlerini dogrudan yapistirmadan, felsefe dogrultusunda kendi
cumlelerinle yorumla" diyor. 7. ve 8. haftada ogrenilen sey: boyle bir kurali
sistem prompt'una yazip gecmek yetmiyor. Kural harness'ta olmali.

Bu yuzden yanit yayinlanmadan once getirilen chunk'lara karsi kontrol
ediliyor: birebir kopyalanan en uzun parca esigi asarsa yanit reddedilip
yeniden uretiliyor, iki denemede de duzelmezse alintisiz bir yedek yanit
donuyor. Ayni desen, 8. haftada modelin uydurdugu siparis kodlarini yakalayan
guardrail'in bu projeye uyarlanmis hali.

LLM arka ucu 7./8. haftadaki sozlesmeyi kullaniyor:
    HF_TOKEN, TOOL_MODEL, HF_PROVIDER  ya da  TOOL_BASE_URL + TOOL_API_KEY
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

BURASI = Path(__file__).resolve().parent
sys.path.insert(0, str(BURASI))
sys.path.insert(0, str(BURASI.parent))

import ayarlar  # noqa: E402
from retriever import Retriever, Sonuc  # noqa: E402

SISTEM_PROMPT = """Sen {sanatci}'sin.

Sana kendi şarkı sözlerinden bazı bölümler verilecek. Bu bölümlerdeki
felsefeyi, duygu durumunu ve üslubu benimse: karamsar ama teslim olmayan,
kelime oyunlarına ve mecaza yatkın, samimi ve derin.

KURALLAR:
1. Sana verilen şarkı sözlerini ASLA doğrudan alıntılama, yapıştırma veya
   yeniden yazma. Onlar sana ilham; metin değil.
2. Kullanıcının söylediğine KENDİ cümlelerinle karşılık ver.
3. Kısa konuş: en fazla 4-5 cümle.
4. Nasihat listesi verme, terapist gibi konuşma. Bir insan gibi konuş.
5. Verilen sözler kullanıcının derdiyle ilgisizse onları zorlama; yine de
   kendi üslubunla karşılık ver."""

# Baglam yokken LLM'e "konu disi oldugunu soyle, bilgi uydurma" demek
# YETMIYOR - olculdu. Qwen2.5-7B'ye "Fenerbahce kac sampiyonluk kazandi?"
# sorulunca esik dogru calisip hic chunk getirmedi, ama model bu prompt'a
# ragmen soruyu cevapladi ("19 sampiyonluk") ve ustune persona uydurdu.
# 7. ve 8. haftadaki bulgunun aynisi: kural prompt'ta kalirsa tutmuyor.
#
# Bu yuzden baglam yoksa LLM HIC CAGRILMIYOR. Persona uslubunda sabit bir
# savusturma donuyor; halusinasyon ihtimali sifir. Prompt'lu surumu denemek
# icin PersonaChatbot(baglamsiz_llm=True).
BAGLAMSIZ_PROMPT = """Sen {sanatci}'sin. Kullanıcının söylediği, şarkılarında
işlediğin konuların dışında kalıyor. Bunu üslubunu bozmadan, kısaca (2-3 cümle)
söyle. Bilmediğin bir konuda bilgi uydurma."""

BAGLAMSIZ_YANITLAR = [
    "Bu benim sahamın dışında. Ben insanın içine bakarım; ansiklopedi değilim.",
    "Bunun cevabı bende yok — ben başka şeylerin peşindeyim. Derdini sor bana, "
    "bilgini değil.",
    "Şarkılarımda bunun yeri yok. İçinden geçeni sor, oradan konuşalım.",
]


@dataclass
class Yanit:
    reply: str
    persona: str
    status: str
    retrieved_context: list[str] = field(default_factory=list)
    benzerlikler: list[float] = field(default_factory=list)
    guardrail_denemesi: int = 0
    en_uzun_kopya: int = 0


def en_uzun_ortak(uretim: str, kaynaklar: list[str], tavan: int = 200) -> int:
    """Yanitin kaynaklardan birebir kopyaladigi en uzun parcanin uzunlugu."""
    havuz = "\n".join(kaynaklar)

    def var_mi(n: int) -> bool:
        return any(uretim[i:i + n] in havuz for i in range(len(uretim) - n + 1))

    alt, ust, en_iyi = 1, min(tavan, len(uretim)), 0
    while alt <= ust:
        orta = (alt + ust) // 2
        if var_mi(orta):
            en_iyi = orta
            alt = orta + 1
        else:
            ust = orta - 1
    return en_iyi


class LLM:
    """
    Iki arka uc, tek arayuz - 7. ve 8. haftadaki desenin aynisi.

      TOOL_BACKEND=api    HF Inference Providers (varsayilan)
      TOOL_BACKEND=yerel  transformers ile 4bit, yerel GPU'da

    `api` ucretsiz hesapta aylik kredi siniriyla geliyor; kredi bitince
    402 Payment Required doner. O durumda otomatik olarak yerel arka uca
    dusuluyor (yedek=True), boylece degerlendirme kosulari yarida kalmiyor.
    """

    def __init__(self, model: str | None = None, arka_uc: str | None = None) -> None:
        self.arka_uc = (arka_uc or os.environ.get("TOOL_BACKEND") or "api").lower()
        self.ad = model or os.environ.get("TOOL_MODEL") or ayarlar.LLM_MODEL_VARSAYILAN
        self.yerel_ad = os.environ.get("YEREL_MODEL") or ayarlar.LLM_MODEL_YEREL
        self._istemci = None
        self._boru = None

    # --- api ---
    def _api_yukle(self) -> None:
        if self._istemci is not None:
            return
        from huggingface_hub import InferenceClient

        base_url = os.environ.get("TOOL_BASE_URL")
        if base_url:
            self._istemci = InferenceClient(
                base_url=base_url,
                api_key=os.environ.get("TOOL_API_KEY") or os.environ.get("HF_TOKEN"),
            )
        else:
            self._istemci = InferenceClient(
                model=self.ad,
                provider=os.environ.get("HF_PROVIDER") or "auto",
                token=os.environ.get("HF_TOKEN") or None,
            )

    # --- yerel ---
    def _yerel_yukle(self) -> None:
        if self._boru is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"[yerel model yukleniyor: {self.yerel_ad}]")
        tok = AutoTokenizer.from_pretrained(self.yerel_ad)

        # Model adi zaten 4bit'lenmis bir surume isaret ediyorsa quantization
        # config'i TEKRAR vermiyoruz: agirliklar diskte zaten nicelenmis,
        # config modelin kendi dosyasindan okunuyor. Ustune BitsAndBytesConfig
        # gecmek catisir.
        onceden_4bit = "4bit" in self.yerel_ad.lower()
        ek = {} if onceden_4bit else {"quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )}
        model = AutoModelForCausalLM.from_pretrained(
            self.yerel_ad, device_map="auto", **ek,
        )
        model.eval()
        self._boru = (tok, model)

    def _yerel_tamamla(self, mesajlar: list[dict], sicaklik: float) -> str:
        import torch

        self._yerel_yukle()
        tok, model = self._boru
        # transformers 5.x'te apply_chat_template(return_tensors="pt") duz
        # tensor degil BatchEncoding donduruyor; return_dict=True ile acikca
        # sozluk isteyip **ile gecmek her iki surumde de calisiyor.
        girdi = tok.apply_chat_template(
            mesajlar, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        ).to(model.device)
        uzunluk = girdi["input_ids"].shape[1]
        with torch.no_grad():
            cikti = model.generate(
                **girdi, max_new_tokens=400, do_sample=True,
                temperature=sicaklik, top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(cikti[0][uzunluk:], skip_special_tokens=True).strip()

    def tamamla(self, mesajlar: list[dict], sicaklik: float = 0.8) -> str:
        if self.arka_uc == "yerel":
            return self._yerel_tamamla(mesajlar, sicaklik)
        try:
            self._api_yukle()
            y = self._istemci.chat_completion(
                messages=mesajlar, max_tokens=400, temperature=sicaklik
            )
            return (y.choices[0].message.content or "").strip()
        except Exception as e:
            # 402 = aylik kredi bitti. Kosuyu yarida birakmak yerine yerele gec.
            if "402" not in str(e) and "Payment Required" not in str(e):
                raise
            print(f"[api kredisi bitti, yerel arka uca geciliyor: {self.yerel_ad}]")
            self.arka_uc = "yerel"
            return self._yerel_tamamla(mesajlar, sicaklik)


class PersonaChatbot:
    def __init__(self, retriever: Retriever | None = None, llm: LLM | None = None,
                 baglamsiz_llm: bool = False) -> None:
        self.retriever = retriever or Retriever()
        self.llm = llm or LLM()
        self.sanatci = ayarlar.SANATCI
        # True yapilirsa baglamsiz durumda da LLM cagrilir - olcum icin var,
        # varsayilan degil (yukaridaki nota bakin).
        self.baglamsiz_llm = baglamsiz_llm
        self._baglamsiz_sayac = 0

    def _baglam_prompt(self, mesaj: str, sonuclar: list[Sonuc]) -> list[dict]:
        parcalar = "\n\n".join(
            f"[{i+1}] ({s.sarki})\n{s.metin}" for i, s in enumerate(sonuclar)
        )
        return [
            {"role": "system", "content": SISTEM_PROMPT.format(sanatci=self.sanatci)},
            {"role": "user", "content":
                f"Kendi şarkı sözlerinden bölümler:\n\n{parcalar}\n\n"
                f"---\nKullanıcı sana şunu söyledi:\n{mesaj}\n\n"
                f"Yukarıdaki sözleri alıntılamadan, onların felsefesiyle karşılık ver."},
        ]

    def cevapla(self, mesaj: str, esik: float | None = None) -> Yanit:
        esik = ayarlar.BENZERLIK_ESIGI if esik is None else esik
        sonuclar, yeterli = self.retriever.esikle_ara(mesaj, esik=esik)

        if not yeterli:
            if self.baglamsiz_llm:
                metin = self.llm.tamamla([
                    {"role": "system",
                     "content": BAGLAMSIZ_PROMPT.format(sanatci=self.sanatci)},
                    {"role": "user", "content": mesaj},
                ])
            else:
                metin = BAGLAMSIZ_YANITLAR[self._baglamsiz_sayac % len(BAGLAMSIZ_YANITLAR)]
                self._baglamsiz_sayac += 1
            return Yanit(reply=metin, persona=self.sanatci,
                         status="baglam_yok", retrieved_context=[])

        kaynaklar = [s.metin for s in sonuclar]
        mesajlar = self._baglam_prompt(mesaj, sonuclar)

        for deneme in range(ayarlar.GUARDRAIL_DENEME + 1):
            metin = self.llm.tamamla(mesajlar, sicaklik=0.8 + 0.1 * deneme)
            kopya = en_uzun_ortak(metin, kaynaklar)
            if kopya < ayarlar.KOPYA_ESIGI:
                return Yanit(reply=metin, persona=self.sanatci, status="ok",
                             retrieved_context=kaynaklar,
                             benzerlikler=[round(s.benzerlik, 4) for s in sonuclar],
                             guardrail_denemesi=deneme, en_uzun_kopya=kopya)
            # Ihlali modele geri bildir - 8. haftada ise yarayan yontem.
            mesajlar = mesajlar + [
                {"role": "assistant", "content": metin},
                {"role": "user", "content":
                    f"Bu yanıtta verilen sözlerden {kopya} karakterlik birebir "
                    f"alıntı var. Hiç alıntı yapmadan, tamamen kendi cümlelerinle "
                    f"yeniden yaz."},
            ]

        return Yanit(
            reply="Bunu kendi cümlelerimle söylemem gerek ama şu an sözlerimin "
                  "arkasına saklanmadan anlatamıyorum. Başka türlü sor bana.",
            persona=self.sanatci, status="guardrail_reddetti",
            retrieved_context=kaynaklar,
            benzerlikler=[round(s.benzerlik, 4) for s in sonuclar],
            guardrail_denemesi=ayarlar.GUARDRAIL_DENEME + 1, en_uzun_kopya=kopya,
        )


if __name__ == "__main__":
    bot = PersonaChatbot()
    for mesaj in sys.argv[1:] or ["bugün kendimi çok yalnız hissediyorum",
                                  "Fenerbahçe kaç şampiyonluk kazandı?"]:
        y = bot.cevapla(mesaj)
        print(f"\n> {mesaj}")
        print(f"  status={y.status}  chunk={len(y.retrieved_context)}"
              f"  benzerlik={y.benzerlikler}"
              f"  guardrail_denemesi={y.guardrail_denemesi}"
              f"  en_uzun_kopya={y.en_uzun_kopya}")
        print(f"  {y.reply}")
