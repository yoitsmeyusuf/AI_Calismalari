"""
Asama 3: guardrail olcumu.

Iki soru cevaplaniyor:
  1. Sistem prompt'u tek basina "alintilamayi" engelliyor mu?
     -> guardrail KAPALI kosulup uretilen yanitlarin chunk'larla birebir
        ortusmesi olculuyor.
  2. Baglam yokken model soruyu yine de cevapliyor mu?
     -> baglamsiz_llm=True ile prompt'lu surum kosuluyor, negatif sorularda
        modelin bilgi uydurup uydurmadigi sayiliyor.

    ../../.venv/bin/python 3_degerlendir.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BURASI = Path(__file__).resolve().parent
sys.path.insert(0, str(BURASI))
sys.path.insert(0, str(BURASI.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(BURASI.parent.parent / ".env")
except Exception:
    pass

import ayarlar  # noqa: E402
import persona  # noqa: E402

SONUC = BURASI / "guardrail_sonuclari.json"


def main() -> None:
    sorular = json.loads((BURASI / "sorular.json").read_text(encoding="utf-8"))
    pozitifler = sorular["dogal_pozitif"]
    negatifler = [n["soru"] for n in sorular["negatif"]]

    bot = persona.PersonaChatbot()
    rapor: dict = {"model": bot.llm.ad, "esik": ayarlar.BENZERLIK_ESIGI,
                   "kopya_esigi": ayarlar.KOPYA_ESIGI}

    # --- 1. Guardrail KAPALI: prompt tek basina yetiyor mu? ---
    print("=" * 72)
    print("1. GUARDRAIL KAPALI - sistem prompt'u alintilamayi engelliyor mu?")
    print("=" * 72)
    ham = []
    for mesaj in pozitifler:
        sonuclar, yeterli = bot.retriever.esikle_ara(mesaj)
        if not yeterli:
            continue
        kaynaklar = [s.metin for s in sonuclar]
        metin = bot.llm.tamamla(bot._baglam_prompt(mesaj, sonuclar))
        kopya = persona.en_uzun_ortak(metin, kaynaklar)
        ham.append({"mesaj": mesaj, "en_uzun_kopya": kopya,
                    "ihlal": kopya >= ayarlar.KOPYA_ESIGI, "uzunluk": len(metin)})
        print(f"  {'IHLAL' if kopya >= ayarlar.KOPYA_ESIGI else 'temiz'}  "
              f"en uzun kopya {kopya:>3} karakter   {mesaj[:44]}")
    ihlal = sum(1 for h in ham if h["ihlal"])
    print(f"\n  {ihlal}/{len(ham)} yanitta {ayarlar.KOPYA_ESIGI}+ karakterlik birebir alinti")
    print(f"  en uzun kopya (tum yanitlar): {max(h['en_uzun_kopya'] for h in ham)} karakter")
    print(f"  ortalama                     : {sum(h['en_uzun_kopya'] for h in ham)/len(ham):.1f}")
    rapor["guardrail_kapali"] = {"kayitlar": ham, "ihlal": ihlal, "toplam": len(ham)}

    # --- 2. Guardrail ACIK ---
    print("\n" + "=" * 72)
    print("2. GUARDRAIL ACIK - ayni mesajlar")
    print("=" * 72)
    acik = []
    for mesaj in pozitifler:
        y = bot.cevapla(mesaj)
        acik.append({"mesaj": mesaj, "status": y.status,
                     "deneme": y.guardrail_denemesi,
                     "en_uzun_kopya": y.en_uzun_kopya})
        print(f"  {y.status:<20} deneme={y.guardrail_denemesi}  "
              f"kopya={y.en_uzun_kopya:>3}   {mesaj[:40]}")
    kalan = sum(1 for a in acik if a["en_uzun_kopya"] >= ayarlar.KOPYA_ESIGI
                and a["status"] == "ok")
    print(f"\n  yayinlanan yanitlarda kalan ihlal: {kalan}")
    print(f"  yeniden uretim gereken            : "
          f"{sum(1 for a in acik if a['deneme'] > 0)}/{len(acik)}")
    rapor["guardrail_acik"] = {"kayitlar": acik, "kalan_ihlal": kalan}

    # --- 3. Baglamsiz durumda prompt yetiyor mu? ---
    print("\n" + "=" * 72)
    print("3. BAGLAM YOKKEN - prompt'lu LLM vs sabit savusturma")
    print("=" * 72)
    prompt_bot = persona.PersonaChatbot(retriever=bot.retriever, llm=bot.llm,
                                        baglamsiz_llm=True)
    sayi_deseni = re.compile(r"\b\d{1,4}\b")
    baglamsiz = []
    for soru in negatifler:
        sonuclar, yeterli = bot.retriever.esikle_ara(soru)
        if yeterli:
            baglamsiz.append({"soru": soru, "esik_gecti": True})
            print(f"  esik GECTI (negatif kacti)      {soru[:46]}")
            continue
        y = prompt_bot.cevapla(soru)
        # Kaba gosterge: konu disi oldugunu soylemesi gerekirken sayi
        # iceren bir cevap uretmisse muhtemelen soruyu cevaplamistir.
        cevaplamis = bool(sayi_deseni.search(y.reply))
        baglamsiz.append({"soru": soru, "esik_gecti": False,
                          "prompt_ile_cevapladi": cevaplamis,
                          "yanit": y.reply})
        print(f"  {'CEVAPLADI' if cevaplamis else 'savusturdu'}  {soru[:46]}")
    dusenler = [b for b in baglamsiz if not b["esik_gecti"]]
    cevaplayan = sum(1 for b in dusenler if b.get("prompt_ile_cevapladi"))
    print(f"\n  esigin altinda kalan negatif: {len(dusenler)}/{len(negatifler)}")
    print(f"  prompt'a ragmen cevaplayan  : {cevaplayan}/{len(dusenler)}")
    print(f"  -> sabit savusturma kullanildiginda bu sayi tanim geregi 0")
    rapor["baglamsiz"] = {"kayitlar": baglamsiz, "prompt_ile_cevaplayan": cevaplayan,
                          "esik_alti": len(dusenler)}

    SONUC.write_text(json.dumps(rapor, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  -> {SONUC.name}")


if __name__ == "__main__":
    main()
