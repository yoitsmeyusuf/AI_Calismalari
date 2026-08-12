"""Ogrenen RAG bellegi: ayni soru ikinci kez sorulunca internete cikilmasin.

Bu dosyanin tek bir amaci var: model daha once benzerini gordugu bir soruyu
kendi bellegindeki cevaptan uretsin; hic gormediyse internetten arayip ogrensin.
Uc kapi:

    1. ARAMA KAPISI  -> benzerlik esigin altindaysa bellekte yok sayilir ve
                        LLM'e internet arama sonuclari verilir.
    2. URETIM KAPISI -> LLM'e "sadece bu metinlerden cevapla" talimati verilir.
    3. KAYIT KAPISI  -> internetten uretilen cevap, soruyla birlikte kaydedilir;
                        bir sonraki benzer soruda 1. kapi artik acilir.

Bellek en basta BOSTUR. Her yeni soru onu biraz daha doldurur, yani asistanin
isabeti ve hizi kullandikca artar. Kaydedilen sey makale parcalari degil,
"soru -> dogrulanmis cevap" ciftleridir: aramayi soru metni uzerinden yaptigimiz
icin benzerlik karsilastirmasi soru-soru olur, soru-belge degil.

Kullanim (bellegi elle yoklamak icin):
    python3 code_rag.py --liste
    python3 code_rag.py --sor "python ModuleNotFoundError nasil cozulur"
    python3 code_rag.py --sifirla
"""

import hashlib
import os
import time

import chromadb

import ollama_client

# Bellek bu klasorde diske yazilir; silmek asistanin ogrendiklerini sifirlar.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# Kayitlarin raf omru. Bir soru-cevap bellegi eskiyebilir ("son surum hangisi",
# "hangi komut onerilir"); bu kadar gun once yazilmis kayit bellekte yok sayilir
# ve soru yeniden arastirilir. 0 verirsen omur sinirsiz olur.
MAX_AGE_DAYS = 30

# Koleksiyon adinin onu. olcum_karsilastirma.py bunu degistirerek gercek
# bellegi kirletmeden ayri bir deneme koleksiyonunda calisir.
COLLECTION_PREFIX = "ogrenilen_sorular"

# Kaynak yetmediginde verilecek sabit cevap. Kaydetmedigimiz tek cevap budur.
REFUSAL = "Bilmiyorum — bu soru icin guvenilir bir kaynak bulamadim."

SYSTEM_PROMPT = f"""Sen Turkce konusan bir bilgi asistanisin.

Kurallar:
1. SADECE asagida verilen kaynak metinlerdeki bilgiyi kullan.
2. Kendi ezberini KESINLIKLE kullanma. Metinde yoksa, senin icin yoktur.
3. Kaynaklar soruyu cevaplamaya yetmiyorsa, aynen su cumleyi yaz ve baska
   hicbir sey ekleme:
{REFUSAL}
4. Cevabini Turkce, kisa ve uygulanabilir yaz. Kod hatasi sorulduysa cozumu
   adim adim ver.
"""


def get_collection(embed_key: str = ollama_client.DEFAULT_EMBED):
    """Secilen embedding modeline ait Chroma koleksiyonunu acar/olusturur.

    Her model icin AYRI koleksiyon kullaniriz: iki modelin vektorleri farkli
    uzaylarda yasar, karistirilirsa arama sonuclari anlamsiz olur.
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        name=f"{COLLECTION_PREFIX}_{embed_key}",
        metadata={"hnsw:space": "cosine"},  # mesafe = 1 - kosinus benzerligi
        embedding_function=None,            # vektorleri biz uretiyoruz (Ollama ile)
    )


def _soru_kimligi(question: str) -> str:
    """Ayni sorunun bellekte iki kez durmamasi icin sabit bir kimlik uretir."""
    return hashlib.sha1(" ".join(question.lower().split()).encode("utf-8")).hexdigest()[:16]


def search(question: str, embed_key: str = ollama_client.DEFAULT_EMBED, k: int = 3) -> list[dict]:
    """Soruyu vektore cevirip bellekteki en yakin k soruyu getirir."""
    collection = get_collection(embed_key)
    if collection.count() == 0:
        return []

    query_vector = ollama_client.embed([question], embed_key, kind="query")[0]
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "question": text,
                "answer": meta.get("answer", ""),
                "source": meta.get("source", "-"),
                "created": meta.get("created", "-"),
                "age_days": (time.time() - float(meta.get("created_ts", 0))) / 86400,
                "similarity": 1.0 - distance,
            }
        )
    return hits


def remember(
    question: str,
    answer: str,
    source: str = "internet",
    embed_key: str = ollama_client.DEFAULT_EMBED,
) -> str:
    """3. KAPI: soru-cevap ciftini bellege yazar ve kimligini dondurur.

    Vektor SORU metninden uretilir (cevaptan degil): bir dahaki sefere
    karsilastirdigimiz sey de kullanicinin sorusu olacak.
    """
    vector = ollama_client.embed([question], embed_key, kind="doc")[0]
    doc_id = _soru_kimligi(question)
    # upsert: ayni soru tekrar ogrenilirse yeni kayit acmaz, ustune yazar.
    get_collection(embed_key).upsert(
        ids=[doc_id],
        documents=[question],
        embeddings=[vector],
        metadatas=[{
            "answer": answer,
            "source": source,
            "created": time.strftime("%Y-%m-%d %H:%M"),
            "created_ts": time.time(),  # raf omru hesabi icin
        }],
    )
    return doc_id


def forget_all(embed_key: str = ollama_client.DEFAULT_EMBED) -> None:
    """Bellegi sifirlar (koleksiyonu silip bos halde yeniden acar)."""
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(f"{COLLECTION_PREFIX}_{embed_key}")
    except Exception:
        pass  # zaten yoksa sifirlanacak bir sey de yok
    get_collection(embed_key)


def _uret(question: str, context: str) -> str:
    """2. KAPI: LLM'e sadece verilen metinlerden cevaplamasini soyler."""
    message = ollama_client.chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"KAYNAK METINLER:\n{context}\n\nSORU: {question}"},
        ],
        temperature=0.0,  # yaraticilik yok, sadece metne sadakat
    )
    return (message.get("content") or "").strip() or REFUSAL


def answer_with_memory(
    question: str,
    embed_key: str = ollama_client.DEFAULT_EMBED,
    k: int = 3,
    web_search=None,
) -> dict:
    """Soruyu once bellekten, olmazsa internetten cevaplar ve ogrendigini kaydeder.

    web_search: (soru -> metin) imzali bir arama fonksiyonu. Varsayilan olarak
    tools.web_lookup kullanilir; test ederken buraya sahte bir fonksiyon
    verilebilir (o zaman ag'a hic cikilmaz).
    """
    hits = search(question, embed_key, k)

    # --- 1. KAPI: arama ---
    # Esigi gecen bir kayit varsa internete hic cikmayiz. Esik, kullanilan
    # embedding modeline gore degisir (bkz. ollama_client.EMBED_MODELS).
    threshold = ollama_client.EMBED_MODELS[embed_key]["min_similarity"]
    relevant = [
        h for h in hits
        if h["similarity"] >= threshold
        and (MAX_AGE_DAYS <= 0 or h["age_days"] <= MAX_AGE_DAYS)
    ]

    if relevant:
        context = "\n\n".join(
            f"[{i}] Daha once sorulan: {h['question']}\nKaydedilen cevap: {h['answer']}"
            for i, h in enumerate(relevant, start=1)
        )
        answer = _uret(question, context)
        if not answer.lower().startswith("bilmiyorum"):
            return {
                "answer": answer,
                "answered_from": "bellek",
                "learned": False,
                "grounded": True,
                "hits": relevant,
            }
        # 2. kapi bellekteki kaydi reddetti: kayit esigi gecmis ama soruyu
        # gercekten cevaplamiyormus ("yakin ama ayni degil"). Kullaniciyi
        # "Bilmiyorum" ile bosa dondurmek yerine asagidaki internet yoluna
        # dusuruyoruz; dogru cevap zaten bellekte olmadigi icin ogrenilecek de.

    # Bellekte yok (ya da bellektekiler ise yaramadi): internetten ara,
    # ayni uretim kapisindan gecir.
    if web_search is None:
        from tools import web_lookup  # dairesel import olmasin diye burada
        web_search = web_lookup
    findings = web_search(question)
    answer = _uret(question, findings)
    grounded = not answer.lower().startswith("bilmiyorum")

    # --- 3. KAPI: kayit ---
    # "Bilmiyorum"u kaydetmeyiz: kaydedersek asistan bir daha o soruyu
    # arastirmaz, kendi bilgisizligini ezberlemis olur.
    if grounded:
        remember(question, answer, source="internet", embed_key=embed_key)

    return {
        "answer": answer,
        "answered_from": "internet",
        "learned": grounded,
        "grounded": grounded,
        "hits": hits,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ogrenen RAG bellegi.")
    parser.add_argument("--embed-model", default=ollama_client.DEFAULT_EMBED,
                        choices=list(ollama_client.EMBED_MODELS))
    parser.add_argument("--liste", action="store_true", help="Bellektekileri dok")
    parser.add_argument("--sor", metavar="SORU", help="Tek bir soruyu uc kapidan gecir")
    parser.add_argument("--sifirla", action="store_true", help="Bellegi bosalt")
    args = parser.parse_args()

    if args.sifirla:
        forget_all(args.embed_model)
        print(f"[{args.embed_model}] bellek sifirlandi.")

    if args.sor:
        result = answer_with_memory(args.sor, embed_key=args.embed_model)
        print(f"\n[kaynak: {result['answered_from']}"
              f"{', ogrenildi' if result['learned'] else ''}]\n{result['answer']}\n")

    if args.liste or not (args.sor or args.sifirla):
        collection = get_collection(args.embed_model)
        data = collection.get(include=["documents", "metadatas"])
        print(f"[{args.embed_model}] bellekte {collection.count()} soru var:")
        for doc, meta in zip(data["documents"], data["metadatas"]):
            print(f"  - {doc}")
            print(f"    {meta.get('created', '-')} / {meta.get('source', '-')} :"
                  f" {meta.get('answer', '')[:100].replace(chr(10), ' ')}...")
