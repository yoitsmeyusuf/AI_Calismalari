"""
6. Hafta: Felsefeye ozel 100 soruluk cok-secenekli (A-E) benchmark verisi.

hafta5_mmlu_benchmark/mmlu_benchmark.py'deki genel MMLU testinin aksine, bu liste
sadece felsefe konularina odaklanan, 13 alt kategoriye yayilmis 100 sorudan
olusan bir soru bankasidir. Format, 5. haftadaki degerlendirme mantigiyla
(harf esleme + anlamsal benzerlik fallback) dogrudan uyumlu olacak sekilde
HARFLER = ["A","B","C","D","E"] sirasina gore 0-indeksli "cevap" alani
kullanir.

Dogru cevabin konumu, pozisyon onyargisini (position bias) onlemek icin
A-E arasinda dengeli dagitilmistir (bkz. asagidaki __main__
dogrulamasindaki harf dagilimi).
"""

SORULAR = [
    # --- Antik Yunan Felsefesi ---
    {
        "id": 1, "kategori": "Antik Yunan Felsefesi",
        "soru": "Sokrates'in \"Hiçbir şey bilmediğimi biliyorum\" sözüyle ifade ettiği tavrın adı nedir?",
        "secenekler": ["Sokratik ironi / bilgece cehalet", "Septisizm", "Nihilizm", "Dogmatizm", "Rölativizm"],
        "cevap": 0,
    },
    {
        "id": 2, "kategori": "Antik Yunan Felsefesi",
        "soru": "Platon'a göre gerçek bilginin nesnesi nedir?",
        "secenekler": ["Duyularla algılanan nesneler", "İdealar (formlar)", "Rüyalar", "Yalnızca matematiksel semboller", "Toplumsal sözleşmeler"],
        "cevap": 1,
    },
    {
        "id": 3, "kategori": "Antik Yunan Felsefesi",
        "soru": "Aristoteles'in \"dört neden\" (nedensellik) öğretisinde YER ALMAYAN hangisidir?",
        "secenekler": ["Maddi neden", "Formel neden", "Rastlantısal neden", "Etkin neden", "Ereksel (gaye) neden"],
        "cevap": 2,
    },
    {
        "id": 4, "kategori": "Antik Yunan Felsefesi",
        "soru": "Herakleitos'un \"her şey akar\" görüşünü ifade eden ünlü söz/kavram hangisidir?",
        "secenekler": ["Arkhe", "Logos'un sabitliği", "Atomculuk", "Panta rhei", "Değişmezlik ilkesi"],
        "cevap": 3,
    },
    {
        "id": 5, "kategori": "Antik Yunan Felsefesi",
        "soru": "Demokritos ve Leukippos'un kurucusu olduğu, evrenin bölünemez parçacıklardan oluştuğunu savunan öğreti nedir?",
        "secenekler": ["İdealizm", "Stoacılık", "Septisizm", "Pisagorculuk", "Atomculuk"],
        "cevap": 4,
    },
    {
        "id": 6, "kategori": "Antik Yunan Felsefesi",
        "soru": "Epikuros'a göre insan yaşamının nihai amacı nedir?",
        "secenekler": ["Ataraksia (ruh dinginliği) ve haz", "Devlete hizmet etmek", "Tanrılara kurban sunmak", "Bilgiyi reddetmek", "Sürekli acı çekmek"],
        "cevap": 0,
    },
    {
        "id": 7, "kategori": "Antik Yunan Felsefesi",
        "soru": "Stoacı felsefenin temel öğretisi nedir?",
        "secenekler": ["Hazza göre yaşamak", "Doğaya uygun/erdemli yaşamak ve duygulara (pathos) hükmetmek", "Şüpheyi mutlak ilke yapmak", "Devletin ortadan kalkmasını savunmak", "Ruh göçünü reddetmek"],
        "cevap": 1,
    },
    {
        "id": 8, "kategori": "Antik Yunan Felsefesi",
        "soru": "Platon'un \"mağara alegorisi\" temel olarak neyi anlatır?",
        "secenekler": ["Bilginin yalnızca duyularla elde edildiğini", "Devletin ideal yönetim biçimini", "Gerçeklik ve bilgi katmanlarını, cehaletten aydınlanmaya geçişi", "Ruhun ölümsüz olmadığını", "Sanatın taklit (mimesis) olduğunu"],
        "cevap": 2,
    },
    {
        "id": 9, "kategori": "Antik Yunan Felsefesi",
        "soru": "Aristoteles'e göre \"eudaimonia\" en iyi nasıl tanımlanır?",
        "secenekler": ["Anlık haz", "Zenginlik", "Ün ve şöhret", "Erdeme uygun etkinlikle elde edilen insani gelişkinlik/mutluluk", "Bedensel güç"],
        "cevap": 3,
    },
    {
        "id": 10, "kategori": "Antik Yunan Felsefesi",
        "soru": "Pisagorcuların felsefesinde evrenin temel ilkesi nedir?",
        "secenekler": ["Su", "Ateş", "Hava", "Belirsiz olan (apeiron)", "Sayı ve sayısal oran/uyum"],
        "cevap": 4,
    },
    # --- Ortaçağ ve İslam Felsefesi ---
    {
        "id": 11, "kategori": "Ortaçağ ve İslam Felsefesi",
        "soru": "Farabi'nin \"El-Medinetü'l-Fazıla\" (Erdemli Şehir) eserinde ideal toplum düzenini kime benzeterek anlatır?",
        "secenekler": ["İnsan bedenine/organizmaya", "Bir orduya", "Bir pazar yerine", "Bir hapishaneye", "Bir okyanusa"],
        "cevap": 0,
    },
    {
        "id": 12, "kategori": "Ortaçağ ve İslam Felsefesi",
        "soru": "İbn Sina'nın \"uçan adam\" (tayyar) düşünce deneyi neyi göstermeyi amaçlar?",
        "secenekler": ["Tanrının varlığını", "Ruhun bedenden bağımsız var olabileceğini / öz bilincin doğrudanlığını", "Evrenin sonsuzluğunu", "Beş duyunun güvenilirliğini", "Atomların varlığını"],
        "cevap": 1,
    },
    {
        "id": 13, "kategori": "Ortaçağ ve İslam Felsefesi",
        "soru": "Gazali'nin \"Tehâfüt el-Felâsife\" (Filozofların Tutarsızlığı) eserinde eleştirdiği temel hedef kimlerdir?",
        "secenekler": ["Mutezile kelamcıları", "Hristiyan teologlar", "Farabi ve İbn Sina gibi meşşai (Aristotelesçi) filozoflar", "Antik Yunan sofistleri", "Sufiler"],
        "cevap": 2,
    },
    {
        "id": 14, "kategori": "Ortaçağ ve İslam Felsefesi",
        "soru": "Aziz Augustinus'a göre kötülüğün kaynağı nedir?",
        "secenekler": ["Tanrı'nın kötülüğü doğrudan yaratması", "Maddenin doğası gereği kötü olması", "Şeytanın Tanrı ile eşit güçte olması", "İnsan iradesinin özgürce iyilikten (Tanrı'dan) sapması; kötülüğün bir yokluk/eksiklik olması", "Kader"],
        "cevap": 3,
    },
    {
        "id": 15, "kategori": "Ortaçağ ve İslam Felsefesi",
        "soru": "Thomas Aquinas'ın Tanrı'nın varlığını kanıtlamak için sunduğu \"beş yol\"dan \"ilk hareket ettirici\" kanıtı hangi öncüle dayanır?",
        "secenekler": ["Ahlaki değerlerin var olduğu", "Evrenin güzel olduğu", "Kutsal kitabın otoritesi", "Rüyalarda Tanrı'nın görülmesi", "Her hareketin bir hareket ettiricisi olması gerektiği ve sonsuza dek geriye gidilemeyeceği"],
        "cevap": 4,
    },
    {
        "id": 16, "kategori": "Ortaçağ ve İslam Felsefesi",
        "soru": "İbn Rüşd (Averroes), akıl ile din/vahiy arasındaki ilişkiyi nasıl ele almıştır?",
        "secenekler": ["Hakikatin tek olduğunu, felsefe ve dinin aynı hakikate farklı yollardan ulaştığını savunmuştur", "İkisinin tamamen çeliştiğini, birinin seçilmesi gerektiğini savunmuştur", "Felsefeyi tümüyle reddetmiştir", "Dini tümüyle reddetmiştir", "Akıl ile vahyin konusunun hiç kesişmediğini savunmuştur"],
        "cevap": 0,
    },
    {
        "id": 17, "kategori": "Ortaçağ ve İslam Felsefesi",
        "soru": "Skolastik felsefede \"üniversaller (tümeller) problemi\" temel olarak neyi tartışır?",
        "secenekler": ["Evrenin sonlu mu sonsuz mu olduğunu", "Genel kavramların (insanlık, kırmızılık gibi) gerçekten var olup olmadığını", "Tanrı'nın birliğini", "Ruh göçünü", "Zamanın doğrusal mı döngüsel mi olduğunu"],
        "cevap": 1,
    },
    {
        "id": 18, "kategori": "Ortaçağ ve İslam Felsefesi",
        "soru": "Farabi'nin felsefesinde \"Vâcibü'l-Vücûd\" (İlk Sebep/Zorunlu Varlık) kavramı neyi ifade eder?",
        "secenekler": ["Evrenin rastlantısal başlangıcını", "İlk insanı", "Varlığı kendinden olan, zorunlu ilk varlığı (Tanrı)", "Aklın maddeden bağımsız olamayacağını", "Zamanın başlangıcını"],
        "cevap": 2,
    },
    {
        "id": 19, "kategori": "Ortaçağ ve İslam Felsefesi",
        "soru": "Anselmus'un \"ontolojik kanıtı\" Tanrı'nın varlığını nasıl temellendirir?",
        "secenekler": ["Evrendeki düzenden hareketle", "Mucizelerin gözlemlenmesiyle", "Ahlaki yasanın varlığıyla", "\"Kendisinden daha büyüğü tasavvur edilemeyen varlık\" kavramının var olmayı da içermesi gerektiği mantığıyla", "Deneysel gözlemle"],
        "cevap": 3,
    },
    # --- Modern Felsefe ---
    {
        "id": 20, "kategori": "Modern Felsefe",
        "soru": "Descartes'ın \"Cogito, ergo sum\" (Düşünüyorum, öyleyse varım) önermesi hangi yöntemin sonucudur?",
        "secenekler": ["Deneysel gözlem", "Diyalektik", "Sezgisel inanç", "Tümevarım", "Metodik şüphe"],
        "cevap": 4,
    },
    {
        "id": 21, "kategori": "Modern Felsefe",
        "soru": "Spinoza'nın felsefesinde \"Deus sive Natura\" (Tanrı ya da Doğa) ifadesi hangi görüşü yansıtır?",
        "secenekler": ["Panteizm — Tanrı ile doğanın/evrenin özdeşliği", "Düalizm", "Ateizm", "Politeizm", "Deizm"],
        "cevap": 0,
    },
    {
        "id": 22, "kategori": "Modern Felsefe",
        "soru": "Leibniz'in \"monad\" kavramı nedir?",
        "secenekler": ["Bölünebilir maddi atomlar", "Ruhsal, bölünemez, \"penceresiz\" temel varlık birimleri", "Fiziksel kuvvetler", "Matematiksel sabitler", "Devletin temel birimleri"],
        "cevap": 1,
    },
    {
        "id": 23, "kategori": "Modern Felsefe",
        "soru": "John Locke'a göre insan zihni doğuştan hangi durumdadır?",
        "secenekler": ["Doğuştan gelen idealarla dolu", "Tanrısal aydınlanmayla dolu", "Tabula rasa (boş levha) — bilgi deneyimle oluşur", "Mükemmel biçimde rasyonel", "Kolektif bilinçdışıyla dolu"],
        "cevap": 2,
    },
    {
        "id": 24, "kategori": "Modern Felsefe",
        "soru": "Hume'un \"nedensellik\" eleştirisi temel olarak neyi savunur?",
        "secenekler": ["Nedensellik mantıksal bir zorunluluktur", "Nedensellik Tanrı tarafından garanti edilir", "Nedensellik matematiksel bir aksiyomdur", "Nedensellik doğrudan gözlemlenemez; alışkanlık/birliktelik tecrübesinden çıkarsanır", "Nedensellik yalnızca rüyada deneyimlenir"],
        "cevap": 3,
    },
    {
        "id": 25, "kategori": "Modern Felsefe",
        "soru": "Kant'ın \"Kopernik devrimi\" olarak adlandırılan görüşü neyi ifade eder?",
        "secenekler": ["Güneş'in Dünya çevresinde döndüğünü", "Ahlakın toplumsal sözleşmeye dayandığını", "Uzayın sonsuz olduğunu", "Tanrı'nın evreni yarattığını", "Bilginin nesneye değil, zihnin kategorilerinin nesneyi biçimlendirmesine dayandığını"],
        "cevap": 4,
    },
    {
        "id": 26, "kategori": "Modern Felsefe",
        "soru": "Hobbes'a göre \"doğa durumu\" nasıl bir haldir?",
        "secenekler": ["\"Herkesin herkese karşı savaşı\" — güvensiz ve kaotik bir hal", "Barışçıl bir cennet hali", "Mükemmel adalet düzeni", "Tanrısal bir uyum", "Sınıfsız eşitlikçi bir toplum"],
        "cevap": 0,
    },
    {
        "id": 27, "kategori": "Modern Felsefe",
        "soru": "Rousseau'nun \"Toplum Sözleşmesi\" eserindeki \"genel irade\" (volonté générale) kavramı neyi ifade eder?",
        "secenekler": ["Kralın bireysel iradesini", "Toplumun ortak iyiliğini hedefleyen kolektif iradeyi", "Çoğunluğun anlık her isteğini", "Kilisenin iradesini", "Piyasanın görünmez elini"],
        "cevap": 1,
    },
    {
        "id": 28, "kategori": "Modern Felsefe",
        "soru": "Berkeley'in \"esse est percipi\" (var olmak algılanmaktır) ilkesi hangi görüşü savunur?",
        "secenekler": ["Materyalizm", "Determinizm", "Öznel idealizm — maddi nesnelerin ancak algılandıkları sürece var olduğu", "Ampirizmin reddi", "Düalizm"],
        "cevap": 2,
    },
    # --- Epistemoloji ---
    {
        "id": 29, "kategori": "Epistemoloji",
        "soru": "Rasyonalizm ile ampirizm arasındaki temel ayrım nedir?",
        "secenekler": ["Rasyonalizm dini, ampirizm bilimi savunur", "İkisi de aynı görüştedir", "Rasyonalizm sadece matematikle ilgilenir", "Rasyonalizm bilginin kaynağını akılda, ampirizm deneyimde/duyularda görür", "Ampirizm doğuştan idealara inanır"],
        "cevap": 3,
    },
    {
        "id": 30, "kategori": "Epistemoloji",
        "soru": "Gettier problemi, geleneksel \"gerekçelendirilmiş doğru inanç\" bilgi tanımına yönelik olarak neyi göstermeye çalışır?",
        "secenekler": ["Bilginin imkânsız olduğunu", "Doğru inancın gerekli olmadığını", "Gerekçelendirmenin gereksiz olduğunu", "Bilginin sadece algıdan geldiğini", "Gerekçelendirilmiş doğru inancın, bilgi için yeterli olmayabileceğini"],
        "cevap": 4,
    },
    {
        "id": 31, "kategori": "Epistemoloji",
        "soru": "Kartezyen şüphenin (Descartes'ın metodik şüphesinin) amacı nedir?",
        "secenekler": ["Kesin ve sarsılmaz bir bilgi temeli bulmak için şüphe edilebilecek her şeyi eleyerek ilerlemek", "Her şeyi reddederek nihilist olmak", "Bilimi tamamen reddetmek", "Sadece duyulara güvenmek", "Tanrı'yı inkâr etmek"],
        "cevap": 0,
    },
    {
        "id": 32, "kategori": "Epistemoloji",
        "soru": "Kant'a göre \"a priori\" bilgi nedir?",
        "secenekler": ["Sadece deneyimden sonra elde edilen bilgi", "Deneyimden bağımsız, ondan önce gelen bilgi", "Yanlış bilgi", "Sezgisel olmayan bilgi", "Sadece matematik dışı alanlarda geçerli bilgi"],
        "cevap": 1,
    },
    {
        "id": 33, "kategori": "Epistemoloji",
        "soru": "Septisizmin (kuşkuculuğun) felsefi tavrı temel olarak neyi savunur?",
        "secenekler": ["Her önermenin kesin doğru olduğunu", "Sadece dini bilginin geçerli olduğunu", "Kesin/mutlak bilgiye ulaşmanın mümkün olup olmadığının sorgulanması ve yargının askıya alınması", "Bilimin yanılmaz olduğunu", "Duyuların her zaman güvenilir olduğunu"],
        "cevap": 2,
    },
    {
        "id": 34, "kategori": "Epistemoloji",
        "soru": "Pragmatizmin bilgi/hakikat anlayışına göre bir önermenin doğruluğu neyle ölçülür?",
        "secenekler": ["Tanrısal vahiyle", "Sadece mantıksal tutarlılıkla", "Çoğunluğun oyuyla", "Pratikte işe yaraması, sonuçları ve eylemdeki faydasıyla", "Estetik güzellikle"],
        "cevap": 3,
    },
    {
        "id": 35, "kategori": "Epistemoloji",
        "soru": "Popper'a göre bilimsel bir teoriyi sözde-bilimden ayıran temel ölçüt nedir?",
        "secenekler": ["Doğrulanabilirlik", "Popülerlik", "Matematiksel olması", "Deneyle hiç ilgisi olmaması", "Yanlışlanabilirlik (falsifiabilite)"],
        "cevap": 4,
    },
    {
        "id": 36, "kategori": "Epistemoloji",
        "soru": "Edmund Husserl'in fenomenolojisinde \"epokhe\" (paranteze alma) yöntemi neyi amaçlar?",
        "secenekler": ["Doğal tutumdaki önyargı ve varsayımları paranteze alıp bilincin saf yapısına ulaşmayı", "Dış dünyanın varlığını kanıtlamayı", "Tanrının varlığını ispatlamayı", "Toplumsal sınıfları analiz etmeyi", "Dili çözümlemeyi"],
        "cevap": 0,
    },
    {
        "id": 37, "kategori": "Epistemoloji",
        "soru": "\"A priori\" bilginin karşıtı olan, deneyime dayanan bilgi türüne ne ad verilir?",
        "secenekler": ["Sentetik", "A posteriori", "Analitik", "Transandantal", "Ontolojik"],
        "cevap": 1,
    },
    # --- Metafizik ve Ontoloji ---
    {
        "id": 38, "kategori": "Metafizik ve Ontoloji",
        "soru": "Düalizm (ikicilik) felsefi görüşüne göre evren temelde kaç tür özden oluşur ve bunlar nelerdir?",
        "secenekler": ["Bir; sadece madde", "Üç; madde, enerji, zaman", "İki; zihin (ruh) ve madde", "Sonsuz sayıda monad", "Hiçbiri, her şey yanılsamadır"],
        "cevap": 2,
    },
    {
        "id": 39, "kategori": "Metafizik ve Ontoloji",
        "soru": "Materyalizm (maddecilik) felsefi görüşü neyi savunur?",
        "secenekler": ["Yalnızca zihinsel/ruhsal varlığın gerçek olduğunu", "Tanrının maddeden önce var olduğunu", "Sayıların bağımsız olarak gerçek olduğunu", "Var olan her şeyin temelde madde ve maddenin süreçlerinden ibaret olduğunu", "Zamanın var olmadığını"],
        "cevap": 3,
    },
    {
        "id": 40, "kategori": "Metafizik ve Ontoloji",
        "soru": "Determinizm felsefi görüşüne göre evrendeki olaylar nasıl gerçekleşir?",
        "secenekler": ["Tamamen rastlantısal olarak", "İnsan iradesiyle sınırsızca değiştirilerek", "Tanrısal kaprisle", "Hiçbir kurala bağlı olmadan", "Önceki nedenler tarafından zorunlu olarak belirlenerek"],
        "cevap": 4,
    },
    {
        "id": 41, "kategori": "Metafizik ve Ontoloji",
        "soru": "\"Özgür irade\" ile \"determinizm\"in bir arada mümkün olduğunu iddia eden görüşe ne ad verilir?",
        "secenekler": ["Uzlaşımcılık (compatibilism)", "Sert determinizm (hard determinism)", "Metafizik liberteryenizm", "Fatalizm", "Nihilizm"],
        "cevap": 0,
    },
    {
        "id": 42, "kategori": "Metafizik ve Ontoloji",
        "soru": "Ontolojide \"töz\" (substance) kavramı en yalın haliyle neyi ifade eder?",
        "secenekler": ["Değişen ve geçici olan görünüşleri", "Kendi başına var olan, başka bir şeye bağımlı olmayan temel varlığı", "Sadece sayıları", "Toplumsal kurumları", "Duyumları"],
        "cevap": 1,
    },
    {
        "id": 43, "kategori": "Metafizik ve Ontoloji",
        "soru": "Nominalizm, tümeller (üniversaller) problemine nasıl bir çözüm sunar?",
        "secenekler": ["Tümellerin nesnelerden bağımsız gerçek varlıklar olduğunu savunur", "Tümellerin Tanrı'nın zihninde var olduğunu savunur", "Tümellerin yalnızca isimler/dilsel kurgular olduğunu, gerçekte sadece tikel nesnelerin var olduğunu savunur", "Tümellerin matematiksel nesneler olduğunu savunur", "Tümellerin duyularla doğrudan algılandığını savunur"],
        "cevap": 2,
    },
    {
        "id": 44, "kategori": "Metafizik ve Ontoloji",
        "soru": "Bergson'un felsefesinde \"élan vital\" (yaşam atılımı) kavramı neyi ifade eder?",
        "secenekler": ["Mekanik nedensellik zincirini", "Toplumsal sınıf çatışmasını", "Matematiksel bir sabiti", "Canlıları yaratıcı evrim içinde sürükleyen içsel yaşamsal gücü", "Dini bir ritüeli"],
        "cevap": 3,
    },
    {
        "id": 45, "kategori": "Metafizik ve Ontoloji",
        "soru": "Heidegger'in \"Dasein\" kavramı temel olarak neyi ifade eder?",
        "secenekler": ["Salt biyolojik insan bedenini", "Evrensel tini", "Maddenin özünü", "Toplumsal sınıfı", "Kendi varlığını sorgulayabilen, \"orada-olma\" biçimindeki insan varoluşunu"],
        "cevap": 4,
    },
    {
        "id": 46, "kategori": "Metafizik ve Ontoloji",
        "soru": "\"Oluş\" (Herakleitos) ile \"Varlık\" (Parmenides) arasındaki antik metafizik tartışma temel olarak neyi konu edinir?",
        "secenekler": ["Gerçekliğin sürekli değişim mi yoksa değişmez/kalıcı bir öz mü olduğunu", "Devletin en iyi yönetim biçimini", "Tanrının varlığını", "Dilin kökenini", "Sanatın işlevini"],
        "cevap": 0,
    },
    # --- Etik ---
    {
        "id": 47, "kategori": "Etik",
        "soru": "Kant'ın \"kategorik buyruk\" (kategorik imperatif) ilkesine göre ahlaki eylem nasıl belirlenir?",
        "secenekler": ["Eylemin sonuçlarının fayda getirmesiyle", "Eylem maksiminin evrensel bir yasa olmasını isteyebilme koşuluyla", "Toplumun geleneklerine uygunlukla", "Tanrının buyruklarına körü körüne itaatle", "Bireysel hazza göre"],
        "cevap": 1,
    },
    {
        "id": 48, "kategori": "Etik",
        "soru": "Faydacılık (utilitarianism) ahlak teorisine göre bir eylemin ahlaki değeri neyle belirlenir?",
        "secenekler": ["Eylemi yapanın niyetiyle", "Dini kurallara uygunlukla", "En çok sayıda insan için en fazla mutluluk/faydayı üretmesiyle", "Geleneklere uygunlukla", "Sadece yasalara uygunlukla"],
        "cevap": 2,
    },
    {
        "id": 49, "kategori": "Etik",
        "soru": "Aristoteles'in erdem etiğinde \"altın orta yol\" (mesotes) ilkesi neyi ifade eder?",
        "secenekler": ["Herkesin eşit pay alması gerektiğini", "Ortalama zekaya sahip olmayı", "Tanrısal buyruklara uymayı", "Erdemin iki aşırı uç (ifrat ve tefrit) arasındaki dengede bulunduğunu", "Hazzın tamamen reddedilmesini"],
        "cevap": 3,
    },
    {
        "id": 50, "kategori": "Etik",
        "soru": "Nietzsche'nin \"efendi ahlakı\" ve \"köle ahlakı\" ayrımında köle ahlakı neyi temsil eder?",
        "secenekler": ["Güç, gurur ve yaratıcılık değerlerini", "Bilimsel rasyonaliteyi", "Antik Yunan aristokrasisini", "Üstinsan idealini", "Zayıfların güçlülere karşı geliştirdiği alçakgönüllülük, merhamet, itaat gibi değerleri"],
        "cevap": 4,
    },
    {
        "id": 51, "kategori": "Etik",
        "soru": "Ahlaki değerlerin nesnel/evrensel olmadığını, öznel tercihler veya duygusal tepkiler olduğunu savunan meta-etik görüşe ne ad verilir?",
        "secenekler": ["Ahlaki relativizm / emotivizm", "Ahlaki realizm", "Doğal hukuk teorisi", "Kategorik imperatif", "Erdem etiği"],
        "cevap": 0,
    },
    {
        "id": 52, "kategori": "Etik",
        "soru": "Hedonizm etik görüşüne göre iyi yaşamın (mutluluğun) ölçütü nedir?",
        "secenekler": ["Erdemli davranış", "Haz elde etme ve acıdan kaçınma", "Toplumsal statü", "Bilgi birikimi", "Dini bağlılık"],
        "cevap": 1,
    },
    {
        "id": 53, "kategori": "Etik",
        "soru": "Doğal hukuk (natural law) etik/hukuk anlayışına göre ahlaki/hukuki normların kaynağı nedir?",
        "secenekler": ["Sadece devletin koyduğu yasalar", "Bireysel hazlar", "İnsan doğasında ve akılda içkin olduğu düşünülen evrensel ilkeler", "Rastlantısal gelenekler", "Yalnızca dini metinlerin harfi anlamı"],
        "cevap": 2,
    },
    {
        "id": 54, "kategori": "Etik",
        "soru": "Sartre'ın varoluşçu etiğinde \"insan özgürlüğe mahkumdur\" sözü neyi ifade eder?",
        "secenekler": ["İnsanın önceden belirlenmiş bir özü olduğunu", "İnsanın kaderci olması gerektiğini", "Özgürlüğün bir yanılsama olduğunu", "İnsanın önceden verilmiş bir özü olmadan var olduğunu, seçimleriyle kendini yarattığını ve bu özgürlükten kaçamayacağını", "Toplumun bireyi tamamen belirlediğini"],
        "cevap": 3,
    },
    {
        "id": 55, "kategori": "Etik",
        "soru": "\"Sonuç mu yoksa ödev/kural mı ahlaki değerlendirmeyi belirlemeli\" tartışmasında deontolojik (ödev) etik hangi tarafı temsil eder?",
        "secenekler": ["Sonuçların önemli olduğunu savunan tarafı", "Faydacılığı", "Hedonizmi", "Erdem etiğini", "Eylemin kendisinin, ilke/kurala/ödeve uygunluğunun ahlaki değeri belirlediğini savunan tarafı"],
        "cevap": 4,
    },
    # --- Siyaset Felsefesi ---
    {
        "id": 56, "kategori": "Siyaset Felsefesi",
        "soru": "Platon'un \"Devlet\" (Politeia) eserinde ideal devleti kim yönetmelidir?",
        "secenekler": ["Filozof-krallar", "Askerler", "Tüccarlar", "Rastgele seçilen yurttaşlar", "Din adamları"],
        "cevap": 0,
    },
    {
        "id": 57, "kategori": "Siyaset Felsefesi",
        "soru": "Machiavelli'nin \"Prens\" eserinde savunduğu, siyasetin ahlaktan bağımsız pragmatik bir güç sanatı olarak ele alınması yaklaşımına ne denir?",
        "secenekler": ["İdealizm", "Machiavelizm / siyasal realizm", "Anarşizm", "Ütopyacılık", "Teokrasi"],
        "cevap": 1,
    },
    {
        "id": 58, "kategori": "Siyaset Felsefesi",
        "soru": "Locke'un siyaset felsefesinde devletin temel meşruiyet kaynağı nedir?",
        "secenekler": ["Tanrısal hak (kralların ilahi hakkı)", "Askeri güç", "Yönetilenlerin rızası ve doğal hakların (yaşam, özgürlük, mülkiyet) korunması", "Gelenek", "Ekonomik zenginlik"],
        "cevap": 2,
    },
    {
        "id": 59, "kategori": "Siyaset Felsefesi",
        "soru": "Marx'ın tarih felsefesinde \"tarihsel materyalizm\" temel olarak neyi savunur?",
        "secenekler": ["Fikirlerin tarihi belirlediğini", "Tanrısal takdirin tarihi yönlendirdiğini", "Bireysel kahramanların tarihi tek başına yaptığını", "Üretim ilişkileri ve sınıf mücadelesinin tarihsel değişimin temel itici gücü olduğunu", "Tarihin hiç değişmeden döngüsel tekrarlandığını"],
        "cevap": 3,
    },
    {
        "id": 60, "kategori": "Siyaset Felsefesi",
        "soru": "Hobbes'un \"Leviathan\" eserinde devlet (egemen güç) neden gereklidir?",
        "secenekler": ["Sanatı desteklemek için", "Sadece vergi toplamak için", "Dini törenleri yönetmek için", "Ticareti tamamen yasaklamak için", "Doğa durumundaki güvensizliği ve kaosu sona erdirip düzeni sağlamak için"],
        "cevap": 4,
    },
    {
        "id": 61, "kategori": "Siyaset Felsefesi",
        "soru": "John Rawls'un \"Bir Adalet Teorisi\" eserinde önerdiği \"bilgisizlik peçesi\" (veil of ignorance) düşünce deneyi neye hizmet eder?",
        "secenekler": ["Adil ilkelerin, kişilerin toplumdaki konumlarını bilmeden tarafsızca seçilmesini sağlamaya", "İnsanların geçmişini unutturarak cezalandırmaya", "Devletin tüm bilgiyi gizlemesine", "Sınıf ayrımını pekiştirmeye", "Dini otoriteyi güçlendirmeye"],
        "cevap": 0,
    },
    {
        "id": 62, "kategori": "Siyaset Felsefesi",
        "soru": "Anarşizmin temel siyasi iddiası nedir?",
        "secenekler": ["Güçlü bir merkezi devletin gerekliliği", "Devlet otoritesinin (özellikle zorlayıcı/hiyerarşik biçimlerinin) meşru olmadığı, gönüllü örgütlenmenin savunulması", "Monarşinin en iyi yönetim biçimi olduğu", "Dinin devleti yönetmesi gerektiği", "Sınıfsız toplumun imkânsız olduğu"],
        "cevap": 1,
    },
    {
        "id": 63, "kategori": "Siyaset Felsefesi",
        "soru": "Aristoteles'e göre \"insan doğası gereği zoon politikon'dur\" (siyasal/toplumsal bir hayvandır) sözü neyi ifade eder?",
        "secenekler": ["İnsanın yalnız yaşamaya elverişli olduğunu", "Hayvanların da siyaset yaptığını", "İnsanın ancak polis (şehir devlet/toplum) içinde tam anlamıyla kendini gerçekleştirebileceğini", "Devletin gereksiz olduğunu", "İnsan doğasının kötü olduğunu"],
        "cevap": 2,
    },
    {
        "id": 64, "kategori": "Siyaset Felsefesi",
        "soru": "Mill'in \"Özgürlük Üzerine\" eserindeki \"zarar ilkesi\" (harm principle) neyi savunur?",
        "secenekler": ["Devletin her konuda bireyi sınırlayabileceğini", "Özgürlüğün hiçbir koşulda sınırlanamayacağını", "Sadece çoğunluğun özgür olabileceğini", "Bireyin özgürlüğünün ancak başkasına zarar verdiği durumlarda sınırlanabileceğini", "Ekonomik özgürlüğün önemsiz olduğunu"],
        "cevap": 3,
    },
    # --- Estetik ---
    {
        "id": 65, "kategori": "Estetik",
        "soru": "Aristoteles'in \"Poetika\" eserinde tragedyanın izleyicide korku ve acıma yoluyla duyguların arınmasını ifade eden kavram nedir?",
        "secenekler": ["Mimesis", "Hybris", "Logos", "Ethos", "Katharsis"],
        "cevap": 4,
    },
    {
        "id": 66, "kategori": "Estetik",
        "soru": "Platon'un sanata (özellikle şiir ve resme) bakışı temel olarak nasıldır?",
        "secenekler": ["Sanatı, gerçek olan İdealar'ın taklidinin de taklidi (mimesisin mimesisi) olduğu için ikinci dereceden/şüpheli görür", "Sanatı gerçekliğin en yüksek ifadesi olarak yüceltir", "Sanatın hiçbir işlevi olmadığını savunur", "Sadece müziği önemser", "Sanatın devletten tamamen bağımsız olması gerektiğini savunur"],
        "cevap": 0,
    },
    {
        "id": 67, "kategori": "Estetik",
        "soru": "Kant'ın \"Yargı Gücünün Eleştirisi\" eserinde estetik yargının özelliği olarak tanımladığı \"amaçsız amaçlılık\" ne ifade eder?",
        "secenekler": ["Güzel nesnenin belirli bir pratik yarara hizmet etmesi gerektiğini", "Güzelin, herhangi bir kavramsal/pratik amaca bağlı olmaksızın haz uyandırmasını", "Sanatın her zaman dini bir amacı olması gerektiğini", "Estetik yargının tamamen keyfi ve tartışılamaz olduğunu", "Güzelliğin sadece doğada bulunabileceğini"],
        "cevap": 1,
    },
    {
        "id": 68, "kategori": "Estetik",
        "soru": "\"Yüce\" (sublime) kavramı estetikte genellikle neyi ifade eder?",
        "secenekler": ["Küçük ve zarif olanı", "Sadece geometrik simetriyi", "Akıl ve hayal gücünün kavramakta zorlandığı, hem hayranlık hem ürperti uyandıran büyüklük/güç karşısındaki duyguyu", "Günlük sıradan güzelliği", "Sanatsal teknik kusursuzluğu"],
        "cevap": 2,
    },
    {
        "id": 69, "kategori": "Estetik",
        "soru": "Hegel'in sanat felsefesinde sanatın tarihsel gelişiminde en yüksek/en soyut evre olarak gördüğü sanat dalı hangisidir?",
        "secenekler": ["Mimari", "Heykel", "Resim", "Şiir (dilin/düşüncenin sanatı)", "Dans"],
        "cevap": 3,
    },
    {
        "id": 70, "kategori": "Estetik",
        "soru": "\"Sanat sanat içindir\" (l'art pour l'art) anlayışı neyi savunur?",
        "secenekler": ["Sanatın ahlaki veya toplumsal bir amaca hizmet etmesi gerektiğini", "Sanatın yasaklanması gerektiğini", "Sanatın yalnızca dini amaçlara hizmet etmesi gerektiğini", "Sanatçının toplumsal sorumluluğunun her şeyden önemli olduğunu", "Sanatın kendi başına, dışsal (ahlaki, siyasi, dini) bir amaca bağlı olmaksızın değerli olduğunu"],
        "cevap": 4,
    },
    {
        "id": 71, "kategori": "Estetik",
        "soru": "Nietzsche'nin \"Trajedyanın Doğuşu\" eserinde sanatın kaynağı olarak sunduğu iki karşıt ilke hangileridir?",
        "secenekler": ["Apolloncu (düzen, biçim, akıl) ve Dionysosçu (coşku, sınırsızlık, sarhoşluk)", "Akıl ve duygu", "İyi ve kötü", "Töz ve ilinek", "Yalnızca Aristotelesçi madde ve form"],
        "cevap": 0,
    },
    {
        "id": 72, "kategori": "Estetik",
        "soru": "Estetikte \"form\" ile \"içerik\" arasındaki klasik tartışma temelde neyi konu alır?",
        "secenekler": ["Bir sanat eserinin fiyatını", "Bir sanat eserinin değerinin biçimsel/yapısal niteliklerinden mi yoksa taşıdığı anlam/mesajdan mı geldiğini", "Sanat eserinin fiziksel boyutunu", "Sanatçının milliyetini", "Eserin nerede sergileneceğini"],
        "cevap": 1,
    },
    # --- Mantık ---
    {
        "id": 73, "kategori": "Mantık",
        "soru": "\"Tüm insanlar ölümlüdür. Sokrates bir insandır. O halde Sokrates ölümlüdür.\" çıkarımı hangi akıl yürütme türüne örnektir?",
        "secenekler": ["Tümevarım", "Analoji", "Tümdengelim (dedüksiyon) / kıyas (silojizm)", "Abdüksiyon", "Diyalektik"],
        "cevap": 2,
    },
    {
        "id": 74, "kategori": "Mantık",
        "soru": "\"Şimdiye kadar gözlemlediğim yüzlerce karga siyahtı; öyleyse bütün kargalar siyahtır\" çıkarımı hangi akıl yürütme türüne örnektir?",
        "secenekler": ["Tümdengelim", "Kıyas (silojizm)", "Modus tollens", "Tümevarım (endüksiyon)", "Redüksiyo ad absurdum"],
        "cevap": 3,
    },
    {
        "id": 75, "kategori": "Mantık",
        "soru": "Klasik mantıkta \"çelişmezlik ilkesi\" neyi ifade eder?",
        "secenekler": ["Her önerme mutlaka doğrudur", "Her önerme yanlıştır", "Mantığın gerçeklikle ilgisi yoktur", "Çelişkiler her zaman kabul edilebilir", "Bir önerme aynı anda hem doğru hem yanlış olamaz"],
        "cevap": 4,
    },
    {
        "id": 76, "kategori": "Mantık",
        "soru": "\"Bu argüman geçerlidir ama sağlam (sound) değildir\" ifadesi hangi durumu tanımlar?",
        "secenekler": ["Argümanın mantıksal yapısı doğru sonucu garanti eder ama öncüllerden en az biri gerçekte yanlıştır", "Öncüller doğru, sonuç yanlıştır", "Argümanın hiçbir öncülü yoktur", "Argüman hem geçerli hem sağlamdır", "Argümanın sonucu yoktur"],
        "cevap": 0,
    },
    {
        "id": 77, "kategori": "Mantık",
        "soru": "\"Ya yağmur yağıyor ya da yağmıyor\" önermesi klasik mantıkta doğrudan hangi ilkeye örnektir?",
        "secenekler": ["Özdeşlik ilkesi", "Üçüncü halin imkânsızlığı ilkesi (tertium non datur)", "Yeter-sebep ilkesi", "Çelişmezlik ilkesi", "Tümevarım ilkesi"],
        "cevap": 1,
    },
    {
        "id": 78, "kategori": "Mantık",
        "soru": "\"P ise Q doğrudur. P yanlıştır. O halde Q yanlıştır.\" biçimindeki akıl yürütme hangi formel yanılgıya (geçersiz çıkarıma) örnektir?",
        "secenekler": ["Modus ponens (geçerli)", "Modus tollens (geçerli)", "Öncülü reddetme yanılgısı (denying the antecedent)", "Hipotetik kıyas (geçerli)", "Ayrık kıyas (geçerli)"],
        "cevap": 2,
    },
    {
        "id": 79, "kategori": "Mantık",
        "soru": "\"Herkes bunu söylüyor, o halde doğru olmalı\" biçimindeki mantık hatasına ne ad verilir?",
        "secenekler": ["Modus ponens", "Petitio principii", "Post hoc ergo propter hoc", "Argumentum ad populum (çoğunluğa/halka başvuru yanılgısı)", "Kıyas (silojizm)"],
        "cevap": 3,
    },
    {
        "id": 80, "kategori": "Mantık",
        "soru": "Bir argümanda sonucun, kanıtlanması gereken öncülde zaten varsayılmış olduğu döngüsel akıl yürütmeye ne ad verilir?",
        "secenekler": ["Modus tollens", "Tümevarım", "Ayrık kıyas", "Analoji", "Petitio principii (döngüsellik / başa dönme yanılgısı)"],
        "cevap": 4,
    },
    # --- Zihin Felsefesi ve Çağdaş Felsefe ---
    {
        "id": 81, "kategori": "Zihin Felsefesi ve Çağdaş Felsefe",
        "soru": "Zihin felsefesinde \"zihin-beden problemi\" temel olarak neyi sorgular?",
        "secenekler": ["Zihinsel (mental) durumlar ile fiziksel (bedensel/beyinsel) durumlar arasındaki ilişkinin doğasını", "Zihnin nasıl beslendiğini", "İnsanın kaç organı olduğunu", "Beynin ağırlığını", "Ruh göçünün gerçekliğini"],
        "cevap": 0,
    },
    {
        "id": 82, "kategori": "Zihin Felsefesi ve Çağdaş Felsefe",
        "soru": "Fonksiyonalizm (işlevselcilik) zihin felsefesi görüşüne göre zihinsel durumlar neyle tanımlanır?",
        "secenekler": ["Sadece beyin dokusunun kimyasal yapısıyla", "Girdi-çıktı ilişkileri içindeki nedensel/işlevsel rolleriyle, hangi fiziksel altyapıda gerçekleştiği önemli olmaksızın", "Ruhun ölümsüzlüğüyle", "Sadece dışsal davranışla, iç durum yokmuş gibi", "Tanrısal müdahaleyle"],
        "cevap": 1,
    },
    {
        "id": 83, "kategori": "Zihin Felsefesi ve Çağdaş Felsefe",
        "soru": "Searle'ün \"Çin Odası\" düşünce deneyi neyi eleştirmek için tasarlanmıştır?",
        "secenekler": ["İnsan zihninin var olmadığını", "Çince dilinin öğrenilemeyeceğini", "Güçlü yapay zekanın (sözdizimsel sembol işlemenin) gerçek anlama/bilinç ürettiği iddiasını", "Bilgisayarların hiçbir işlem yapamayacağını", "Davranışçılığı desteklemeyi"],
        "cevap": 2,
    },
    {
        "id": 84, "kategori": "Zihin Felsefesi ve Çağdaş Felsefe",
        "soru": "Sartre'ın varoluşçuluğunda \"varoluş özden önce gelir\" ilkesi neyi ifade eder?",
        "secenekler": ["İnsanın önceden belirlenmiş sabit bir doğası/özü olduğunu", "Nesnelerin de özden önce var olduğunu", "Tanrının insanı belirli bir amaçla yarattığını", "İnsanın önce var olduğunu, kendi özünü/anlamını sonradan seçimleriyle yarattığını", "Toplumun bireyin özünü tamamen belirlediğini"],
        "cevap": 3,
    },
    {
        "id": 85, "kategori": "Zihin Felsefesi ve Çağdaş Felsefe",
        "soru": "Kierkegaard, varoluşun estetik, etik ve dinsel olmak üzere kaç aşamasından/evresinden söz eder?",
        "secenekler": ["İki", "Dört", "Beş", "Altı", "Üç"],
        "cevap": 4,
    },
    {
        "id": 86, "kategori": "Zihin Felsefesi ve Çağdaş Felsefe",
        "soru": "Camus'nün \"absürt\" (saçma) kavramı neyi ifade eder?",
        "secenekler": ["İnsanın anlam arayışı ile evrenin sessiz/anlamsız kayıtsızlığı arasındaki uyuşmazlığı", "Evrenin mantıklı bir anlam sunduğunu", "Tanrının varlığının kanıtlanabilir olduğunu", "Bilimin her şeyi açıklayabileceğini", "Toplumsal düzenin mükemmelliğini"],
        "cevap": 0,
    },
    {
        "id": 87, "kategori": "Zihin Felsefesi ve Çağdaş Felsefe",
        "soru": "Wittgenstein'ın geç dönem felsefesinde (Felsefi Soruşturmalar) öne sürdüğü \"dil oyunları\" kavramı neyi vurgular?",
        "secenekler": ["Dilin sabit, evrensel bir mantıksal yapısı olduğunu", "Kelimelerin anlamının, içinde kullanıldıkları pratik/bağlamsal \"dil oyunlarındaki\" kullanımdan geldiğini", "Dilin sadece resim/tasvir işlevi gördüğünü", "Dilin düşünceyle hiç ilgisi olmadığını", "Matematiksel dilin tek geçerli dil olduğunu"],
        "cevap": 1,
    },
    {
        "id": 88, "kategori": "Zihin Felsefesi ve Çağdaş Felsefe",
        "soru": "Foucault'nun \"iktidar-bilgi\" (pouvoir-savoir) kavramı temel olarak neyi savunur?",
        "secenekler": ["Bilginin iktidardan tamamen bağımsız, tarafsız olduğunu", "İktidarın sadece devlette bulunduğunu", "Bilginin üretiminin iktidar ilişkileriyle iç içe geçtiğini, bilginin de iktidarı ürettiğini/yeniden ürettiğini", "Bilginin sadece bilimsel yöntemle üretildiğini", "İktidarın her zaman açık şiddetle uygulandığını"],
        "cevap": 2,
    },
    {
        "id": 89, "kategori": "Zihin Felsefesi ve Çağdaş Felsefe",
        "soru": "Frankfurt Okulu'nun (Adorno, Horkheimer vb.) geliştirdiği \"eleştirel teori\" temel olarak neyi amaçlar?",
        "secenekler": ["Toplumu olduğu gibi betimlemekle yetinmeyi", "Sadece ekonomik verileri analiz etmeyi", "Dini kurumları güçlendirmeyi", "Toplumsal tahakküm ve ideolojiyi eleştirerek insan özgürleşmesine katkı sunmayı", "Pozitivizmi olduğu gibi kabul etmeyi"],
        "cevap": 3,
    },
    {
        "id": 90, "kategori": "Zihin Felsefesi ve Çağdaş Felsefe",
        "soru": "Postmodern düşüncede Lyotard'ın \"büyük anlatılara (metanarratives) karşı kuşku\" olarak tanımladığı tavır neyi ifade eder?",
        "secenekler": ["Tüm bilimin reddedilmesi gerektiğini", "Sadece küçük hikayelerin anlatılmaması gerektiğini", "Modernizmin aynen sürdürülmesi gerektiğini", "Tarihin sona erdiğini kanıtladığını", "Aydınlanma, Marksizm gibi evrensel/kapsayıcı açıklayıcı sistemlere duyulan güvenin sarsılmasını"],
        "cevap": 4,
    },

   {
  "id": 91,
  "kategori": "19. Yüzyıl Felsefesi",
  "soru": "Friedrich Nietzsche'nin felsefesinde, yaşamdaki tüm olayların (hem acıların hem de sevinçlerin) aynı şekilde sonsuz kez tekrar edeceği düşüncesi üzerine kurulu olan ve bireyin bu yazgıyı büyük bir evetlemeyle kabul etmesini öngören kavram aşağıdakilerden hangisidir?",
  "secenekler": [
    "Üstinsan (Übermensch)",
    "Güç İstenci (Wille zur Macht)",
    "Ebedi Dönüş (Ewige Wiederkunft)",
    "Efendi-Köle Ahlakı",
    "Nihilizm"
  ],
  "cevap": 2
},
   {
  "id": 92,
  "kategori": "Antik Yunan Felsefesi",
  "soru": "Platon'un 'Devlet' adlı eserinde yer alan; insanların duyular dünyasındaki yanılsamaları gerçeklik sanmasını ve felsefi aydınlanma ile hakikatin (İdeaların) bilgisine ulaşma sürecini anlattığı ünlü benzetme aşağıdakilerden hangisidir?",
  "secenekler": [
    "Mağara Alegorisi",
    "Güneş Alegorisi",
    "Bölünmüş Çizgi Analojisi",
    
    "Gyges'in Yüzüğü",
    "Arabacı Mitosu"
  ],
  "cevap": 0
},
   {
  "id": 93,
  "kategori": "19. Yüzyıl Felsefesi",
  "soru": "Karl Marx'ın düşünce sisteminde; kapitalist üretim biçiminde işçinin kendi ürettiği ürüne, üretim sürecine, kendi insan doğasına (türsel varlığına) ve diğer insanlara karşı bağını yitirmesini, ürettiği nesnelerin ona hükmetmeye başlamasını ifade eden temel kavram aşağıdakilerden hangisidir?",
  "secenekler": [
    "Yabancılaşma",
    "Artı-Değer",
    "Tarihsel Materyalizm",
    "Sınıf Bilinci",
    "Altyapı ve Üstyapı"
  ],
  "cevap": 0
},
   {
  "id": 94,
  "kategori": "20. Yüzyıl Felsefesi",
  "soru": "Antonio Gramsci'nin siyaset felsefesinde, egemen sınıfın iktidarını yalnızca zor ve baskı aygıtlarıyla değil, aynı zamanda eğitim, medya, aile ve din gibi sivil toplum kurumları aracılığıyla toplumun rızasını üreterek sürdürmesini ifade eden temel kavram aşağıdakilerden hangisidir?",
  "secenekler": [
    
    "Organik Aydın",
    "Tarihsel Blok",
    "Mevzi Savaşı",
    "Devletin İdeolojik Aygıtları",
    "Kültürel Hegemonya",
  ],
  "cevap": 4
},
   {
  "id": 95,
  "kategori": "Çağdaş Felsefe",
  "soru": "Slavoj Žižek, klasik Marksist 'yanlış bilinç' (Ne yaptıklarını bilmiyorlar ama yapıyorlar) formülünü eleştirerek günümüz toplumunu analiz eder. Žižek'e göre çağdaş toplumda bireyler sistemin yalanlarının, absürtlüklerinin ve adaletsizliğinin gayet iyi farkındadırlar; ancak bu farkındalığa rağmen mevcut düzeni yeniden üretmeye ve sisteme itaat etmeye devam ederler ('Ne yaptıklarını çok iyi biliyorlar, ama yine de yapıyorlar'). Žižek'in günümüz ideolojisinin işleyiş biçimini tanımladığı bu kavrama ne ad verilir?",
  "secenekler": [
    "İnterpasivite",
    "İdeolojik Sinizm (Cynicism)",
    "Büyük Öteki (Big Other)",
    "Sembolik Şiddet",
    "Hipergerçeklik"
  ],
  "cevap": 1
},
   {
  "id": 96,
  "kategori": "19. Yüzyıl Felsefesi",
  "soru": "Friedrich Engels, 'Ailenin, Özel Mülkiyetin ve Devletin Kökeni' adlı eserinde, anaerkil düzenden ataerkil düzene geçişi 'kadın cinsinin dünya tarihsel yenilgisi' olarak nitelendirir. Engels'e göre, kadının toplumsal statüsünün düşmesine ve erkeğin egemenliğine dayalı katı tek eşli (monogam) ailenin ortaya çıkmasına neden olan temel tarihsel gelişme aşağıdakilerden hangisidir?",
  "secenekler": [
    "Özel mülkiyetin ortaya çıkması ve servetin kesin olarak bilinen meşru mirasçılara aktarılma güdüsü",
    "Din kurumunun kurumsallaşarak toplumsal ahlak kurallarını katılaştırması",
    "Devletin kurularak kadın ve erkek arasındaki ilişkileri kanunlarla düzenlemeye başlaması",
    "Sanayi Devrimi ile birlikte kadınların üretim sürecinden tamamen dışlanması",
    "Göçebe yaşamdan yerleşik hayata geçişle birlikte nüfus planlaması ihtiyacının doğması"
  ],
  "cevap": 0
},
   {
  "id": 97,
  "kategori": "19. Yüzyıl Felsefesi",
  "soru": "Friedrich Engels'in 'Ailenin, Özel Mülkiyetin ve Devletin Kökeni' adlı eserine göre, özel mülkiyetin ve sınıfların henüz ortaya çıkmadığı ilkel komünal toplumlarda hakim olan toplumsal düzen aşağıdakilerden hangisidir?",
  "secenekler": [
    "Ataerkil (Patriyarkal) düzen",
    "Tek eşli (Monogam) aile düzeni",
    "Kast sistemi",
    "Anaerkil (Matriyarkal) düzen",
    "Feodal düzen"
  ],
  "cevap": 3
},
   {
  "id": 98,
  "kategori": "19. Yüzyıl Felsefesi",
  "soru": "Karl Marx'ın kapitalist üretim biçimini analiz ettiği teorisinde; işçinin emeğiyle yarattığı toplam değer ile ona ücret olarak ödenen değer arasındaki farkı ifade eden ve kapitalistin elde ettiği kârın asıl kaynağını oluşturan temel kavram aşağıdakilerden hangisidir?",
  "secenekler": [
    "Kullanım Değeri",
    "Artık-Değer",
    "Mübadele Değeri",
    "Meta Fetişizmi",
    "Emek-Değer Teorisi"
  ],
  "cevap": 1
},
   
   {
  "id": 99,
  "kategori": "19. Yüzyıl Felsefesi",
  "soru": "Karl Marx ile anarşist düşünür Pierre-Joseph Proudhon arasındaki temel ayrılık devrim stratejisi ve devlet otoritesi üzerineydi. Proudhon devletin hemen ortadan kaldırılmasını ve barışçıl kooperatifler kurulmasını savunurken; Marx, kapitalizmden sınıfsız topluma geçiş sürecinde işçi sınıfının devlet iktidarını ele geçirip burjuvaziyi bastırmak için siyasi egemenliğini kurması zorunluluğunu savunmuştur. Marx'ın devleti geçici bir aygıt olarak kullanmayı öngördüğü bu geçiş aşamasına ne ad verilir?",
  "secenekler": [
    "Proletarya Diktatörlüğü",
    "Sürekli Devrim",
    "Demokratik Merkeziyetçilik",
    "Öncü Parti Teorisi",
    "Sınıf Uzlaşması"
  ],
  "cevap": 0
},
   
   {
  "id": 100,
  "kategori": "19. Yüzyıl Felsefesi",
  "soru": "Kendisini siyasi anlamda 'anarşist' olarak tanımlayan ilk düşünür olan Pierre-Joseph Proudhon'un savunduğu; devletin, otoritenin ve kapitalist sömürünün olmadığı, üreticilerin serbest sözleşmelere, kar amacı gütmeyen adil kredi sistemlerine ve gönüllü işçi kooperatiflerine dayanarak toplumu örgütlediği sosyo-ekonomik modele ne ad verilir?",
  "secenekler": [
    "Mutualizm (Karşılıkçılık)",
    "Anarko-Sendikalizm",
    "Nihilizm (Hiççilik)",
    "Ütopik Sosyalizm",
    "Faydacılık (Utilitarizm)"
  ],
  "cevap": 0
}
   
]

KATEGORILER = sorted(set(s["kategori"] for s in SORULAR))

if __name__ == "__main__":
    print(f"Toplam soru: {len(SORULAR)}")
    for k in KATEGORILER:
        n = sum(1 for s in SORULAR if s["kategori"] == k)
        print(f"  {k}: {n}")
    for s in SORULAR:
        assert len(s["secenekler"]) == 5, s["id"]
        assert 0 <= s["cevap"] <= 4, s["id"]
    ids = [s["id"] for s in SORULAR]
    assert ids == list(range(1, len(SORULAR) + 1)), f"id 1..{len(SORULAR)} sirali degil"
    from collections import Counter
    dagilim = Counter(s["cevap"] for s in SORULAR)
    print("Cevap harfi dagilimi:", {["A","B","C","D","E"][k]: v for k, v in sorted(dagilim.items())})
    print(f"Dogrulama basarili: {len(SORULAR)} soru, her biri 5 secenek + gecerli cevap indeksi.")
