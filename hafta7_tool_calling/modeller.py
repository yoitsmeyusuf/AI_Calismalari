"""
7. Hafta: model katmani - ajan dongusunu modelden ayirir.

Iki arka uc var, ikisi de ayni arayuzu (`tamamla()`) konusur:

  YerelModel  (TOOL_BACKEND=yerel, varsayilan)
      Modeli transformers ile dogrudan calistirir. HF Spaces'te **ZeroGPU**
      (ucretsiz H200 dilimi) uzerinde donuyor: `app.py` uretim dongusunu
      `@spaces.GPU` ile sarar, GPU sadece istek suresince tutulur.
      Arac cagrilari modelin kendi sohbet sablonuyla kuruluyor
      (Qwen2.5 -> Hermes tarzi `<tool_call>{...}</tool_call>` blogu),
      cikti bu bloklardan ayristiriliyor.

  ApiModel    (TOOL_BACKEND=api)
      HF Inference Providers uzerinden OpenAI uyumlu `chat.completions`.
      GPU gerekmez ama ucretsiz hesaplarda aylik kredi siniri var.

Ajan, mesajlari arka uctan bagimsiz bir bicimde tutar (asagidaki `Mesaj`
sozlukleri); her arka uc bunu kendi tel formatina cevirir.

Tek basina calistirilirsa arac cagrisi ayristiricisini test eder:
    .venv/bin/python hafta7_tool_calling/modeller.py
"""
import json
import os
import re
import uuid
from dataclasses import dataclass, field

VARSAYILAN_YEREL_MODEL = "Qwen/Qwen2.5-7B-Instruct"
VARSAYILAN_API_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
MAKS_YENI_TOKEN = 512

# Qwen2.5 / Hermes tarzi arac cagrisi blogu.
TOOL_CALL_DESENI = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Kapanis etiketi uretilmeden token biterse diye gevsek yedek desen.
TOOL_CALL_KAPANMAMIS = re.compile(r"<tool_call>\s*(\{.*)", re.DOTALL)


@dataclass
class IstenenCagri:
    """Modelin calistirilmasini istedigi arac cagrisi."""

    ad: str
    argumanlar: dict
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")


@dataclass
class ModelYaniti:
    metin: str
    arac_cagrilari: list[IstenenCagri] = field(default_factory=list)


def _argumanlari_coz(ham) -> dict:
    if isinstance(ham, dict):
        return ham
    if not ham:
        return {}
    try:
        cozulmus = json.loads(ham)
    except (json.JSONDecodeError, TypeError):
        return {"_cozulemeyen_arguman": str(ham)}
    return cozulmus if isinstance(cozulmus, dict) else {"value": cozulmus}


def arac_cagrilarini_ayikla(uretilen: str) -> ModelYaniti:
    """Uretilen metinden <tool_call> bloklarini ayirir.

    Donen `metin` bloklardan arindirilmis kullaniciya gosterilebilir kisim.
    Bozuk JSON'lar sessizce yutulmaz; modelin duzeltebilmesi icin arac adi
    olarak isaretlenip ajan tarafina hata olarak gider.
    """
    cagrilar: list[IstenenCagri] = []
    bloklar = TOOL_CALL_DESENI.findall(uretilen)
    metin = TOOL_CALL_DESENI.sub("", uretilen)

    if not bloklar:
        kapanmamis = TOOL_CALL_KAPANMAMIS.search(uretilen)
        if kapanmamis:
            bloklar = [kapanmamis.group(1)]
            metin = uretilen[: kapanmamis.start()]

    for blok in bloklar:
        try:
            veri = json.loads(blok)
        except json.JSONDecodeError:
            cagrilar.append(
                IstenenCagri(ad="_bozuk_arac_cagrisi", argumanlar={"ham": blok[:400]})
            )
            continue
        cagrilar.append(
            IstenenCagri(
                ad=str(veri.get("name", "_isimsiz_arac")),
                argumanlar=_argumanlari_coz(veri.get("arguments")),
            )
        )

    return ModelYaniti(metin=metin.strip(), arac_cagrilari=cagrilar)


# --------------------------------------------------------------------------
# Yerel arka uc: transformers (+ HF Spaces ZeroGPU)
# --------------------------------------------------------------------------


class YerelModel:
    """Modeli transformers ile yerel/ZeroGPU uzerinde calistirir."""

    def __init__(self, model_adi: str | None = None):
        self.ad = model_adi or os.environ.get("TOOL_MODEL") or VARSAYILAN_YEREL_MODEL
        self._tokenizer = None
        self._model = None

    def yukle(self):
        """Model ve tokenizer'i (bir kez) yukler.

        ZeroGPU'da modeli import aninda GPU'ya tasimak dogru degil; `device_map`
        vermeyip modeli varsayilan cihazda tutuyor, uretim aninda `@spaces.GPU`
        altinda cuda'ya tasiyoruz (bkz. `tamamla`).
        """
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.ad)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.ad,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self._model.eval()

    def _istem_kur(self, mesajlar: list[dict], arac_semalari: list[dict]) -> str:
        """Kanonik mesajlari modelin sohbet sablonuna cevirir."""
        sablon_mesajlari = []
        for m in mesajlar:
            rol = m["role"]
            if rol == "assistant" and m.get("arac_cagrilari"):
                sablon_mesajlari.append(
                    {
                        "role": "assistant",
                        "content": m.get("content", ""),
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": c.ad, "arguments": c.argumanlar},
                            }
                            for c in m["arac_cagrilari"]
                        ],
                    }
                )
            elif rol == "tool":
                sablon_mesajlari.append(
                    {"role": "tool", "name": m.get("ad", ""), "content": m["content"]}
                )
            else:
                sablon_mesajlari.append({"role": rol, "content": m.get("content", "")})

        try:  # Qwen3 gibi thinking modellerinde muhakemeyi kapat
            return self._tokenizer.apply_chat_template(
                sablon_mesajlari,
                tools=arac_semalari,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                sablon_mesajlari,
                tools=arac_semalari,
                add_generation_prompt=True,
                tokenize=False,
            )

    def tamamla(
        self,
        mesajlar: list[dict],
        arac_semalari: list[dict],
        arac_zorla: bool = False,
    ) -> ModelYaniti:
        """Bir tur uretir.

        `arac_zorla=True` iken modelin metin yazmasina izin vermeyiz: uretim
        istemine `<tool_call>` acilis etiketini biz ekliyoruz, model zorunlu
        olarak bir arac cagrisi tamamliyor (hangi arac ve hangi argumanlar
        oldugunu yine model seciyor). API'lerdeki `tool_choice="required"`nin
        yerel karsiligi; kucuk modeller kurala uymadiginda kullaniliyor.
        """
        import torch

        self.yukle()
        if torch.cuda.is_available() and self._model.device.type != "cuda":
            self._model.to("cuda")

        istem = self._istem_kur(mesajlar, arac_semalari)
        onek = "<tool_call>\n" if arac_zorla else ""
        girdi = self._tokenizer(istem + onek, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            cikti = self._model.generate(
                **girdi,
                max_new_tokens=MAKS_YENI_TOKEN,
                do_sample=False,  # greedy: ayni soru ayni araci cagirsin
                pad_token_id=self._tokenizer.eos_token_id,
            )
        uretilen = self._tokenizer.decode(
            cikti[0][girdi["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        # Zorlama modunda acilis etiketini biz yazdik; ayristirici gorebilsin.
        return arac_cagrilarini_ayikla(onek + uretilen)


# --------------------------------------------------------------------------
# API arka ucu: HF Inference Providers (OpenAI uyumlu)
# --------------------------------------------------------------------------


class ApiModel:
    """Modeli HF Inference Providers (ya da OpenAI uyumlu baska bir endpoint) ile cagirir."""

    def __init__(self, model_adi: str | None = None):
        self.ad = model_adi or os.environ.get("TOOL_MODEL") or VARSAYILAN_API_MODEL
        self._istemci = None

    def yukle(self):
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

    def tamamla(
        self,
        mesajlar: list[dict],
        arac_semalari: list[dict],
        arac_zorla: bool = False,
    ) -> ModelYaniti:
        self.yukle()

        tel_mesajlari = []
        for m in mesajlar:
            rol = m["role"]
            if rol == "assistant" and m.get("arac_cagrilari"):
                tel_mesajlari.append(
                    {
                        "role": "assistant",
                        "content": m.get("content", ""),
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {
                                    "name": c.ad,
                                    "arguments": json.dumps(c.argumanlar, ensure_ascii=False),
                                },
                            }
                            for c in m["arac_cagrilari"]
                        ],
                    }
                )
            elif rol == "tool":
                tel_mesajlari.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.get("cagri_id", ""),
                        "name": m.get("ad", ""),
                        "content": m["content"],
                    }
                )
            else:
                tel_mesajlari.append({"role": rol, "content": m.get("content", "")})

        def cagir(secim: str):
            return self._istemci.chat.completions.create(
                messages=tel_mesajlari,
                tools=arac_semalari,
                # "required": modelin bu turda mutlaka bir arac cagirmasi istenir
                # (yerel arka uctaki <tool_call> onekinin API karsiligi).
                tool_choice=secim,
                max_tokens=900,
                temperature=0.2,
            )

        try:
            yanit = cagir("required" if arac_zorla else "auto")
        except Exception as e:
            # Bazi saglayicilar tool_choice="required"i desteklemiyor ve
            # 422 "failed to compile grammar" donuyor (TGI grammar derleyicisi).
            # Zorlamayi kaybediyoruz ama duzeltme turu hic calismamasindan iyi;
            # ihlal surerse guardrail zaten cevabi oldugu gibi birakiyor.
            if arac_zorla and "422" in str(e):
                yanit = cagir("auto")
            else:
                raise
        mesaj = yanit.choices[0].message
        cagrilar = [
            IstenenCagri(
                id=tc.id or f"call_{uuid.uuid4().hex[:12]}",
                ad=tc.function.name,
                argumanlar=_argumanlari_coz(tc.function.arguments),
            )
            for tc in (mesaj.tool_calls or [])
        ]
        return ModelYaniti(metin=mesaj.content or "", arac_cagrilari=cagrilar)


def model_yukle(arka_uc: str | None = None, model_adi: str | None = None):
    """TOOL_BACKEND'e gore arka ucu secer ('yerel' | 'api')."""
    arka_uc = (arka_uc or os.environ.get("TOOL_BACKEND") or "yerel").strip().lower()
    if arka_uc in ("api", "inference", "hf"):
        return ApiModel(model_adi)
    if arka_uc in ("yerel", "local", "transformers", "zerogpu"):
        return YerelModel(model_adi)
    raise ValueError(f"Bilinmeyen TOOL_BACKEND: {arka_uc!r} (geçerli: yerel, api)")


if __name__ == "__main__":
    ornekler = {
        "tek cagri": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Ankara"}}\n</tool_call>',
        "iki cagri + metin": (
            "Bakıyorum.\n"
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Ankara"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "London"}}\n</tool_call>'
        ),
        "cagri yok": "Ankara 16°C ve açık.",
        "kapanmamis etiket": '<tool_call>\n{"name": "get_forecast", "arguments": {"city": "İzmir", "days": 3}}',
        "bozuk json": "<tool_call>\n{bu json degil}\n</tool_call>",
    }
    for ad, ham in ornekler.items():
        y = arac_cagrilarini_ayikla(ham)
        print(f"[{ad}]")
        print(f"  metin  : {y.metin!r}")
        for c in y.arac_cagrilari:
            print(f"  cagri  : {c.ad}({c.argumanlar})")
        print()
