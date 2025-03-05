# 🤖 Nyxie: Protogen Telegram Chatbot 🌟

## 📝 Proje Açıklaması
Nyxie, Google'ın Gemini AI teknolojisini kullanan gelişmiş bir Telegram sohbet botudur. Çoklu dil desteği, görüntü işleme ve derin web arama özellikleriyle donatılmış, kullanıcı dostu bir asistan olarak tasarlanmıştır.

### 🤔 Nyxie Nedir?

Nyxie, sadece bir chatbot değil, aynı zamanda:
- 🧠 Gelişmiş yapay zeka teknolojisi ile çalışan bir dijital arkadaş
- 🌍 Çoklu dil desteği olan bir iletişim asistanı
- 🕰️ Zamansal ve mekânsal farkındalığa sahip bir AI
- 🌈 Dinamik kişilik profili ile etkileşime giren bir asistan
- 🤖 Protogen kimliğine sahip, duygusal ve yaratıcı bir AI

**Nyxie'nin Benzersiz Özellikleri:**
- Her etkileşimi benzersiz ve kişiselleştirilmiş kılan dinamik zaman bağlamı
- Günün saatine, mevsimine ve kullanıcının yerel bağlamına göre değişen kişilik
- Kullanıcının dilini ve tercihlerini anlayan ve uyarlayan akıllı bir sistem
- Semantik bellek ile ilgili konuşmaları hatırlama ve bağlamsal cevaplar üretme yeteneği

## 🚀 Özellikler

### 1. 💬 Gelişmiş Konuşma Yeteneği
- Gemini AI ile dinamik ve bağlamsal yanıtlar
- Kullanıcı tercihlerini öğrenme ve hatırlama
- Çoklu dil desteği (Türkçe, İngilizce ve diğer diller)
- Doğal dil işleme ile otomatik dil tespiti
- Otomatik emoji ekleme ve yanıt zenginleştirme
- Kullanıcı dilini ve tercihlerini otomatik algılama
- Zaman ve bağlam duyarlı kişilik

### 2. 🕒 Zamansal Kişilik Uyarlaması
- Günün saatine göre dinamik kişilik profili
- Mevsim, hafta günü ve günün periyoduna göre yanıt uyarlama
- Kullanıcının yerel saat dilimini ve zamanını otomatik algılama
- Günün saatine, mevsimine ve özel günlere göre kişilik değişimi
- Hafta içi/hafta sonu ve tatil günlerinde farklı davranış modları
- TimezoneFinder ve Geopy ile hassas zaman dilimi tespiti

### 3. 🔍 Derin Web Arama
- `/derinarama` komutu ile gelişmiş web araması
- DuckDuckGo arama motoru entegrasyonu
- Fallback olarak Google arama desteği
- İteratif ve derinlemesine arama yeteneği
- Arama sonuçlarını akıllıca analiz etme
- Çoklu kaynaklardan bilgi toplama ve özetleme

### 4. 🖼️ Görüntü ve Video İşleme
- Gönderilen görüntüleri ve videoları analiz etme
- Google Cloud Vision API ile görüntü tanıma
- Multimedya içeriği hakkında detaylı açıklama üretme
- Görsel içerik bağlamını anlama ve yorumlama

### 5. 🧠 Gelişmiş Semantik Bellek Sistemi
- Semantik benzerlik tabanlı konuşma hatırlama
- Kullanıcı sorguları ile ilgili geçmiş konuşmaları akıllıca bulma
- SentenceTransformer ile gelişmiş vektör temsili
- Konu ve anlam tabanlı bellek indeksleme
- Konuşma segmentlerini otomatik tanımlama
- Ebbinghaus unutma eğrisine dayalı zamansal bellek modeli
- İnsan belleği modelini taklit eden çoklu faktör tabanlı hatırlama
- Otomatik bellek optimizasyonu ve akıllı bağlam yönetimi
- Anlamlı konuları otomatik çıkarma ve ilişkilendirme

### 6. 📋 Akıllı Bellek Yönetimi
- Token limitini aşmadan maksimum bağlam koruma
- Semantik olarak önemli mesajları koruyarak bellek optimizasyonu
- Otomatik bağlam budama algoritması
- En alakalı konuşma parçalarını akıllıca seçme
- Bellek sınırlamalarını aşarken bile tutarlı cevaplar sağlama
- Kullanıcı tercihlerini ve geçmiş etkileşimlerini akıllıca saklama
- Her kullanıcı için ayrı JSON hafıza dosyaları
- Güvenli ve şifrelenmiş kullanıcı verileri

### 7. 🌐 Akıllı Web Arama
- Gemini AI ile dinamik web arama
- Kullanıcı sorgularını akıllıca yorumlama
- Akıllı web arama sonuçlarını analiz etme ve özetleme
- Çoklu kaynaklardan bilgi toplama
- Arama sonuçlarını kullanıcı diline çevirme
- Güvenilir ve güncel bilgi sağlama
- Web arama gereksinimini otomatik değerlendirme

## 🛠️ Gereksinimler

### Yazılım Gereksinimleri
- Python 3.8+
- pip paket yöneticisi

### Gerekli Kütüphaneler
- python-telegram-bot
- google-generativeai
- python-dotenv
- requests
- geopy
- timezonefinder
- emoji
- langdetect
- Pillow
- httpx
- google-cloud-vision
- sentence-transformers
- scikit-learn
- numpy
- torch
- duckduckgo_search
- beautifulsoup4

## 🔧 Kurulum

### 1. Depoyu Klonlama
```bash
git clone https://github.com/stixyie/Nyxie-Protogen-Chatbot-Telegram-v9-main.git
cd Nyxie-Protogen-Chatbot-Telegram-v4-main
```

### 2. Sanal Ortam Oluşturma
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Bağımlılıkları Yükleme
```bash
pip install -r requirements.txt
```

## 🔐 Konfigürasyon

### Gerekli API Anahtarları
`.env` dosyasında aşağıdaki API anahtarlarını yapılandırın:
- `TELEGRAM_TOKEN`: Telegram Bot Token you need to get this token from here: https://t.me/BotFather
- `GEMINI_API_KEY`: Google Ai Studio API Key you need to get this key from here: https://aistudio.google.com/apikey

### Örnek `.env` Dosyası
```
TELEGRAM_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_gemini_api_key
```

## 🚀 Kullanım

### Bot'u Başlatma
```bash
python bot.py
```

### Telegram'da Kullanım
1. Bot'a `/start` komutu ile başlayın
2. Mesaj, görüntü veya video gönderin
3. Sohbet için bot ile etkileşime geçin

## 🛡️ Güvenlik

- Kullanıcı verileri şifrelenmiş JSON dosyalarında saklanır
- Maksimum token sınırlaması ile bellek yönetimi
- Hassas bilgilerin loglanmaması

## 🤝 Destek

### Sorun Bildirim
- GitHub Issues: [Proje Sayfası](https://github.com/stixyie/Nyxie-Protogen-Chatbot-Telegram-v8-main/issues)

### Katkıda Bulunma
1. Projeyi forklayın
2. Yeni bir branch oluşturun
3. Değişikliklerinizi yapın
4. Pull Request açın

## 📄 Lisans

Bu proje GPL-3.0 Lisansı altında yayınlanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 🌟 Teşekkür

- **Stixyie**: Proje yaratıcısı ve baş geliştirici
- **Google**: Gemini ve Cloud Vision API'ları

---

**Not**: Nyxie, sürekli gelişen bir AI projesidir. Geri bildirimleriniz ve katkılarınız çok değerlidir! 🚀
