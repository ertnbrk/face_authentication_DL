
# Yüz Tanıma Siamese Ağı Projesi
Bu proje, yüz tanıma için siyam ağları kullanarak bir model geliştirmeyi amaçlar. Proje, veri toplama, veri ön işleme, model oluşturma, eğitim, değerlendirme ve gerçek zamanlı test adımlarını içerir.

## Kullanım
**Veri Toplama:** Veri toplama işlemi için collect_data fonksiyonunu kullanabilirsiniz. Bu fonksiyon, kameradan alınan görüntüleri pozitif, negatif ve referans (anchor) olarak sınıflandırır ve ilgili klasörlere kaydeder. Tuş atamaları şu şekildedir:

>- 'a' tuşu: Referans (anchor) görüntüyü kaydetmek için kullanılır.
>- 'p' tuşu: Pozitif (yüz içeren) görüntüyü kaydetmek için kullanılır.
>- 'q' tuşu: Veri toplama işlemini sonlandırır.

 ------------

**Veri Ön İşleme:** preprocess_images fonksiyonu, veri setindeki görüntüleri okur, yeniden boyutlandırır ve normalleştirme işlemi uygular.

**Ağ Oluşturma:** Siyam ağı modeli, make_embedding ve make_siamese_model fonksiyonlarıyla oluşturulur. Bu model, iki görüntü arasındaki benzerliği değerlendirmek için kullanılır.

**Eğitim:** Model, train_step ve train fonksiyonlarıyla eğitilir. train_data ve test_data setleri, eğitim ve test için hazırlanır. Eğitim sırasında, belirli bir sayıda dönemde model kaydedilir.

**Değerlendirme:** Eğitim sonrasında modelin performansı test_data seti üzerinde değerlendirilir. Hassasiyet, ölçüm için kullanılan bir metriktir.

**Gerçek Zamanlı Test:** verify fonksiyonu, gerçek zamanlı yüz doğrulama işlemi için kullanılır. Bu fonksiyon, kameradan alınan görüntüyü ve referans (anchor) olarak kaydedilmiş görüntüleri kullanarak yüz doğrulama işlemini gerçekleştirir.

## Gereksinimler
> - Python 3.x
> - TensorFlow
> - OpenCV
> - NumPy
> - Matplotlib
