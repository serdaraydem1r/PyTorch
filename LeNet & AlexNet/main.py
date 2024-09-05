#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.datasets as datasets
import torch.nn.functional as F
#%%
transform =transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train = datasets.MNIST('../data', train=True, transform=transform, download=True)
test = datasets.MNIST('../data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=64,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test,batch_size=64,shuffle=False)
#%%
class LeNet(nn.Module):

    '''
        LeNet Modelinin Özellikleri ve Katkıları
	        •	Basit ve Etkili: LeNet, görüntü sınıflandırma problemlerinde derin öğrenme uygulamalarının basit bir örneğini sunar ve temel CNN yapı taşlarını gösterir.
	        •	Düşük Hesaplama Maliyeti: Modern modellere kıyasla nispeten küçük boyutlu olduğu için düşük hesaplama gücüyle çalıştırılabilir.
	        •	Öncü Model: Günümüzde yaygın olarak kullanılan derin öğrenme mimarilerinin (örneğin, AlexNet, VGG, ResNet) gelişimine ilham kaynağı olmuştur.
    '''
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5,1,2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5,1,2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

'''
__init__ metodu : 
    self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
	•	Conv2d(1, 6, 5, 1, 2): Bir evrişim katmanı (Convolutional layer) tanımlar.
	•	1: Giriş kanalı sayısı. MNIST siyah beyaz bir veri seti olduğu için giriş kanalı 1’dir (gri tonlamalı).
	•	6: Çıktı kanalı sayısı. Bu evrişim katmanı, 6 tane özellik haritası (feature map) üretir.
	•	5: Çekirdek (filtre) boyutu 5x5’tir.
	•	1: Kayma (stride) değeri 1’dir, yani filtre her seferinde bir piksel kayar.
	•	2: Dolgu (padding) değeri 2’dir, bu da giriş görüntüsünün kenarlarına 2 piksel dolgu ekleyerek boyutunu korur.
	
	self.relu = nn.ReLU()
	•ReLU(): Düzeltme doğrusal birimi (Rectified Linear Unit) aktivasyon fonksiyonu.
     Negatif değerleri sıfır yapar ve pozitif değerleri olduğu gibi bırakır. 
	 Modelin doğrusal olmayan özellikleri öğrenmesine yardımcı olur.
		
	self.pool = nn.MaxPool2d(2, 2)
	•MaxPool2d(2, 2): Maksimum havuzlama katmanı. 2x2 boyutunda bir pencere kullanarak özellik haritasını küçültür.
	 Özellik haritasının genişliğini ve yüksekliğini yarıya indirir (boyut azaltma).
		 
	self.conv2 = nn.Conv2d(6, 16, 5, 1, 2)
	•	Conv2d(6, 16, 5, 1, 2): İkinci evrişim katmanı.
	•	6: Giriş kanalı sayısı. İlk evrişim katmanının çıktısı 6 kanaldı.
	•	16: Çıktı kanalı sayısı. Bu katman, 16 özellik haritası üretir.
	•	5: Çekirdek boyutu 5x5.
	•	1: Kayma değeri 1.
	•	2: Dolgu değeri 2.
	
	self.fc1 = nn.Linear(16*5*5, 120)
	•	Linear(16*5*5, 120): Tam bağlı (fully connected) katman. Evrişim katmanlarından gelen özellik haritalarını bir vektöre çevirir.
	•	16*5*5: Giriş birimi sayısı. Evrişim ve havuzlama işlemlerinden sonra kalan özellik haritalarının boyutudur.
	•	120: Çıkış birimi sayısı. 120 nöron içerir.
	
	self.fc2 = nn.Linear(120, 84)
	•	Linear(120, 84): İkinci tam bağlı katman.
	•	120: Giriş birimi sayısı.
	•	84: Çıkış birimi sayısı. 84 nöron içerir.
	
	self.fc3 = nn.Linear(84, 10)
	•	Linear(84, 10): Çıkış katmanı.
	•	84: Giriş birimi sayısı.
	•	10: Çıkış birimi sayısı. MNIST verisetinde 10 sınıf olduğu için 10 nöron içerir (0-9 arası rakamlar).
	
forward metodu:
    x = self.pool(self.relu(self.conv1(x)))
    •	Giriş verisi önce birinci evrişim katmanından (conv1) geçer.
	•	Daha sonra ReLU aktivasyonu (relu) uygulanır.
	•	Max havuzlama (pool) katmanından geçirilerek boyutlar küçültülür.
	
	x = self.pool(self.relu(self.conv2(x)))
	•	İkinci evrişim katmanı (conv2) uygulanır.
	•	ReLU aktivasyonu ve max havuzlama uygulanarak özellik haritaları küçültülür.
	
	x = x.view(-1, 16 * 5 * 5)
	•Özellik haritaları düzleştirilir (flatten). 
	 Burada -1 batch boyutunu otomatik hesaplamak için kullanılır.
		
	x = self.relu(self.fc1(x))
	•İkinci tam bağlı katman (fc2) ve ReLU aktivasyonu uygulanır.
	
	x = self.fc3(x)
	•Çıkış katmanı (fc3) uygulanır.
	
	x = F.softmax(x, dim=1)
	•F.softmax(x, dim=1): Sonuçlar, softmax fonksiyonu ile olasılıklara dönüştürülür. 
	Bu, her sınıf için tahmin edilen olasılıkları verir. 
	dim=1 parametresi, her bir örnek için (batch boyutu üzerinden) olasılıkların hesaplanacağını belirtir.
'''

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    '''
    __init__ metodu:
   * nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2):
	•	Girdi Kanalı (1): Gri tonlamalı görüntüler kullanıldığı için 1 girdi kanalı.
	•	Çıktı Kanalı (64): İlk evrişim katmanından 64 özellik haritası çıkarır.
	•	Çekirdek Boyutu (11x11): 11x11 boyutunda filtreler kullanır.
	•	Stride (4): Filtrelerin 4’er piksel kaymasını sağlar.
	•	Padding (2): Kenarlara 2 piksel dolgu ekler.
	
	* nn.ReLU():
	ReLU aktivasyon fonksiyonu. Doğrusal olmayan özellikleri öğrenmeye yardımcı olur.
    
    * nn.MaxPool2d(kernel_size=3, stride=2): Maksimum havuzlama katmanı.
	•	Kernel Size (3x3): 3x3 boyutunda pencere.
	•	Stride (2): Pencere 2 piksel kayarak ilerler.
	•	Devam Eden Evrişim Katmanları:
	•	İkinci evrişim katmanı: nn.Conv2d(64, 192, kernel_size=5, padding=2)
	•	64 giriş kanalı ve 192 çıkış kanalı.
	•	5x5 çekirdek boyutu ve 2 piksel dolgu.
	•	Üçüncü evrişim katmanı: nn.Conv2d(192, 384, kernel_size=3, padding=1)
	•	192 giriş kanalı ve 384 çıkış kanalı.
	•	3x3 çekirdek boyutu ve 1 piksel dolgu.
	•	Dördüncü evrişim katmanı: nn.Conv2d(384, 256, kernel_size=3, padding=1)
	•	384 giriş kanalı ve 256 çıkış kanalı.
	•	3x3 çekirdek boyutu ve 1 piksel dolgu.
	•	Beşinci evrişim katmanı: nn.Conv2d(256, 256, kernel_size=3, padding=1)
	•	256 giriş ve çıkış kanalı.
	•	3x3 çekirdek boyutu ve 1 piksel dolgu.
	•	Son Maksimum Havuzlama Katmanı:
	•	Üçüncü maksimum havuzlama katmanı: nn.MaxPool2d(kernel_size=3, stride=2).
   
   self.classifier 
   
   	•	nn.Dropout(): Aşırı öğrenmeyi (overfitting) önlemek için dropout katmanı. Eğitim sırasında rastgele nöronları devre dışı bırakır.
	•	nn.Linear(256 * 6 * 6, 4096): Tam bağlı katman.
	•	Giriş Birimi (256 * 6 * 6): Özellik haritalarının boyutları düzleştirilerek girdi yapılır.
	•	Çıkış Birimi (4096): 4096 nöron içerir.
	•	nn.ReLU(): ReLU aktivasyon fonksiyonu.
	•	nn.Linear(4096, 4096): İkinci tam bağlı katman. 4096 giriş ve 4096 çıkış nöronu.
	•	nn.Linear(4096, 10): Çıkış katmanı.
	•	Giriş Birimi (4096): Önceki tam bağlı katmandan gelen giriş.
	•	Çıkış Birimi (10): 10 sınıfa sahip MNIST veri seti için çıkış nöronları.
	
	
	forward metodu 
	
	•	self.features(x): Giriş verisi (örneğin bir görüntü), sırasıyla evrişim, ReLU, havuzlama katmanları gibi katmanlardan geçirilir.
	•	x.view(x.size(0), 256 * 6 * 6): Özellik haritaları düzleştirilir. Bu işlem, veriyi tam bağlı katmanlara girdi olarak hazırlamak için gereklidir. x.size(0) batch boyutunu korur.
	•	self.classifier(x): Düzleştirilmiş veriler tam bağlı katmanlardan geçirilir. Dropout ve ReLU aktivasyonları bu aşamada uygulanır ve son olarak 10 sınıf için tahminler yapılır.
   
    '''

