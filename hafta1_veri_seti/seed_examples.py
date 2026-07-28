"""
1. Hafta için elle yazılan örnekler. Domain: Felsefe (düşünürler, akımlar,
temel kavramlar).

Format: her örnek {"soru": "...", "cevap": "..."} şeklinde bir dict.
build_dataset.py bunları data/raw/scraped_turkish_qa.jsonl'daki (Ekşi Sözlük +
r/felsefe'den çekilen, referans/ilham için kullanılabilecek) satırlarla
birleştirip push eder.

En az 10-20 örnek yazmanız önerilir (assignment'ın sentetik veri kuralı).
"""

DOMAIN = "Felsefe (düşünürler, akımlar, temel kavramlar)"

TURKISH_SEEDS = [
{"soru": "Varoluşçuluk nedir?", "cevap": "Varoluşçuluk (veya egzistansiyalizm), insanın dünyaya önceden belirlenmiş bir amaçla gelmediğini; kendi özünü ve hayatının anlamını kendi hür iradesiyle, yaptığı seçimlerle kendisinin yarattığını savunan felsefi bir akımdır. Akımın temel mottosu 'varoluş özden önce gelir'' şeklindedir.Bu derin felsefi akımın temel detayları şunlardır:1. 'Varoluş Özden Önce Gelir' Ne Demektir?Bu görüşe göre bir sandalye veya masa yapılmadan önce zihinde tasarlanır, yani önce 'özü' belirlenir ve sonra üretilir. Ancak insanda durum tam tersidir; insan önce dünyaya gelir (var olur), daha sonra yaşarken yaptığı tercihleriyle, eylemleriyle kendi karakterini, anlamını ve 'özünü' kendisi inşa eder.2. Temel KavramlarÖzgürlük ve Sorumluluk: İnsan kendi seçimlerini yapmakta tamamen özgürdür. Ancak bu özgürlük beraberinde büyük bir sorumluluk yükü getirir; insan yaptığı veya yapmadığı her şeyin sonucuna katlanmak zorundadır.Seçim Yapma Zorunluluğu: Varoluşçulara göre seçim yapmamak da bir seçimdir. İnsan sürekli bir karar mekanizması içindedir.Yabancılaşma ve Anlamsızlık (Absürt): Evrenin veya hayatın doğuştan gelen bir anlamı yoktur. Bu anlamsızlık karşısında insan boşluğa düşebilir ancak bu durum kişiyi kendi hayatının anlamını üretmeye iter.3. Öncüleri ve TemsilcileriAkım 20. yüzyılda, özellikle iki büyük dünya savaşı sonrasında büyük bir yankı uyandırmıştır:Jean-Paul Sartre: Akımın dünyadaki en büyük temsilcisidir. Eylemleri ve özgürlüğü merkeze alır.Albert Camus: İnsanın anlamsız bir dünyada anlam arayışını 'saçma' (absürt) kavramı üzerinden açıklamıştır.Martin Heidegger: İnsanın dünyadaki varoluşsal kaygılarını ve zaman kavramını irdelemiştir."},
{"soru": "Nietzsche'nin 'Tanrı öldü' sözüyle kastettiği nedir?", "cevap": "Nietzsche 'Tanrı öldü' derken, inancın bitişinden ziyade aydınlanma ve bilimle birlikte Batı toplumundaki 'mutlak değerler sisteminin ve nesnel ahlakın' çöktüğünü kasteder. İnsanın kendi aklıyla tüm evreni anlamlandırabileceğine ve artık eski yönlendirici dogmalara ihtiyacı kalmadığına işaret eder.Bu ünlü ifadeyi daha iyi anlamak için şu detaylara bakabiliriz:İnancın Yitimi: Aydınlanma çağı, akıl ve bilimsel gelişmelerin yükselişiyle birlikte; modern insanın hayatını ve evreni anlamlandırmak için artık Tanrı'ya veya ilahi kurallara ihtiyaç duymadığını ifade eder.Ahlaki Çöküş ve Nihilizm: Tanrı fikri, binlerce yıldır insanlara evrensel bir 'iyi' ve 'kötü' tanımı sunuyordu. Bu inancın yitirilmesiyle birlikte tüm ahlaki temeller sarsılmış ve hayatın anlamsızlaştığı bir boşluk (nihilizm) ortaya çıkmıştır.Sorumluluk ve Üstinsan (Übermensch): Nietzsche bu duruma üzülmez. Aksine, Tanrı'nın ölümüyle birlikte insanın kendi değerlerini, kendi kaderini ve anlamını kendisinin yaratması (Üstinsan vizyonu) için eşsiz bir özgürlük doğduğuna inanır."},
]
