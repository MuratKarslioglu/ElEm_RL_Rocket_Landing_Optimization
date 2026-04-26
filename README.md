# RL Rocket Optimization Assignment

## Görev Tanımı

Bu çalışma kapsamında öğrencilerden, kendilerine verilen simülasyon ortamında yer alan roketin iki bayrak arasına güvenli, dengeli ve kontrollü bir şekilde iniş yapmasını sağlayan bir Reinforcement Learning (RL) ajanı geliştirmeleri beklenmektedir.

Öğrenciler, bu görev için kullanacakları RL modelini kendileri seçecektir. Seçilen modelin göreve uygunluğu, uygulama başarısı ve raporda yapılan teknik açıklamalar değerlendirme sürecinde dikkate alınacaktır.

## Temel Amaç

Görevin temel amacı, roketin simülasyon ortamında güvenli iniş alanı olarak tanımlanan iki bayrak arasına başarılı şekilde inmesini sağlamaktır.

Başarı kriteri şu şekildedir:

> Roket, üst üste 5 denemede güvenli ve doğru şekilde iniş yapmalıdır.

Simülasyon ortamında yer alan değerlendirme metrikleri, iniş başarısını belirlemek için kullanılacaktır. Öğrencilerin ortamın temel değerlendirme fonksiyonlarını değiştirmemesi, yalnızca ajan karar mekanizmasını geliştirmesi beklenmektedir.

## Öğrencilerden Beklenenler

Öğrenciler aşağıdaki gereklilikleri yerine getirmelidir:

1. Verilen simülasyon ortamını çalıştırmak.
2. Roketin güvenli iniş yapmasını sağlayacak bir RL modeli seçmek.
3. Seçilen RL modelini simülasyon ortamına entegre etmek.
4. Ajanın üst üste 5 başarılı iniş yapmasını hedeflemek.
5. Yapılan geliştirmeleri ve sonuçları raporlamak.

## Rapor İçeriği

Hazırlanacak raporda aşağıdaki başlıklara yer verilmelidir:

### 1. Model Seçimi

Kullanılan RL modelinin adı belirtilmelidir. Öğrenciler, seçtikleri modeli neden tercih ettiklerini teknik gerekçelerle açıklamalıdır.

Örneğin:

- DQN
- PPO
- A2C
- SARSA
- Q-Learning
- Policy Gradient
- Diğer uygun RL yöntemleri

### 2. Kod Analizi

Raporda verilen başlangıç kodunun genel olarak ne yaptığı açıklanmalıdır. Özellikle aşağıdaki noktalar ele alınmalıdır:

- Simülasyon ortamının nasıl başlatıldığı
- Ajanın aksiyonları nasıl seçtiği
- Gözlem değerlerinin ne anlama geldiği
- Reward, Task Score, Physical Score ve Correct Landing metriklerinin nasıl kullanıldığı
- Kodda öğrencinin değiştirmesi gereken bölümün neresi olduğu

### 3. Yapılan Değişiklikler

Öğrenciler, simülasyon ortamında veya ajan yapısında yaptıkları değişiklikleri açıkça belirtmelidir.

Özellikle şu sorulara cevap verilmelidir:

- Hangi fonksiyonlar değiştirildi?
- Aksiyon seçme mekanizması nasıl geliştirildi?
- Ajan hangi gözlem değerlerini kullandı?
- Başarılı iniş için nasıl bir karar stratejisi uygulandı?

### 4. Eğitim ve Optimizasyon Süreci

Eğer ajan eğitildiyse veya hiperparametre optimizasyonu yapıldıysa, bu süreç ayrıntılı olarak açıklanmalıdır.

Raporda aşağıdaki bilgiler yer almalıdır:

- Kullanılan öğrenme oranı
- Episode sayısı
- Discount factor değeri
- Exploration stratejisi
- Batch size, buffer size veya network yapısı gibi model özelindeki parametreler
- Hiperparametrelerin neden bu şekilde seçildiği
- Yapılan denemeler sonucunda elde edilen performans farkları

### 5. Sonuçlar

Öğrenciler, geliştirdikleri ajanın performansını raporlamalıdır.

Bu bölümde şu bilgiler verilmelidir:

- Toplam kaç deneme yapıldığı
- Üst üste 5 başarılı iniş elde edilip edilmediği
- Ortalama reward değeri
- Task Score ve Physical Score sonuçları
- Başarılı ve başarısız iniş örneklerinin kısa analizi

## Teslim Kuralları

- Kod dosyaları GitHub üzerinden paylaşılabilir.
- Rapor dosyası GitHub üzerinden değil, WhatsApp üzerinden teslim edilmelidir.
- Rapor açık, düzenli ve teknik olarak anlaşılır bir şekilde hazırlanmalıdır.
- Öğrenciler, kendi geliştirdikleri çözümün nasıl çalıştığını açıklayabilmelidir.

## Değerlendirme Kriterleri

Çalışmalar aşağıdaki ölçütlere göre değerlendirilecektir:

| Kriter | Açıklama |
|---|---|
| Görev Başarısı | Roketin üst üste 5 başarılı iniş yapabilmesi |
| Model Seçimi | Seçilen RL modelinin göreve uygunluğu |
| Kod Analizi | Başlangıç kodunun ve yapılan değişikliklerin doğru açıklanması |
| Optimizasyon | Hiperparametre ayarlamalarının teknik gerekçelerle sunulması |
| Raporlama | Raporun açık, düzenli ve akademik formatta hazırlanması |
| Uygulama Kalitesi | Kodun çalışabilir, anlaşılır ve düzenli olması |

## Not

Öğrenciler, simülasyon ortamındaki değerlendirme fonksiyonlarını ve metrik hesaplama bölümlerini değiştirmemelidir. Geliştirme yapılması beklenen ana bölüm, ajanın aksiyon seçme mekanizmasıdır.
