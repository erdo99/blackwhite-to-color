# BlackWhitetoColor

LAB uzayında çalışan **U-Net** ile görüntü renklendirme: **otomatik** (yalnızca L → a\*/b\*) veya isteğe bağlı **ipuculu** (4 kanal) mod. Kayıp: L1 (kroma ağırlıklı seçenek) + VGG perceptual; isteğe bağlı GAN.

## Gereksinimler

- Python 3.10+ (3.13 ile de denendi)
- PyTorch + torchvision (**CUDA** önerilir)
- [requirements-full.txt](requirements-full.txt) ile tam kurulum

```powershell
cd BlackWhitetoColor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-full.txt
```

Hafif kurulum: [requirements.txt](requirements.txt) (torch zaten kuruluysa).

## Veri klasörü

Aşağıyı **kendin oluştur** (Git’e girmez, `.gitignore`’da):

```text
data/train/   # RGB eğitim görselleri
data/val/     # RGB doğrulama görselleri
```

Alt klasörler desteklenir. Uzantılar: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`

## Yapılandırma

Tüm ayarlar [config.yaml](config.yaml) içinde:

- `data.use_hints` + `model.in_channels` (1 = otomatik, 4 = ipuculu)
- `train.lambda_l1`, `lambda_perceptual`, `chroma_l1_scale`
- `data.max_train_samples`, `train_sequential_chunks`

## Eğitim

```powershell
python train.py --config config.yaml
```

Checkpoint: `checkpoints/best.pt` (val combined metrikine göre). TensorBoard: `tensorboard --logdir ./runs`

## Çıkarım ve arayüz

```powershell
python infer.py --image "yol\foto.jpg" --checkpoint checkpoints\best.pt
python app.py
```

Gradio varsayılan: `http://127.0.0.1:7860`

## Google Colab

`colab.ipynb` (çoğu zaman `.gitignore`’da; dosyayı Colab’a **sen yükle**). Not defteri, her oturumda **`git clone`** ile GitHub’dan `train.py` + `src` indirir; **görseller** ve **`best.pt` çıktısı** Drive’da kalır (Colab kapanınca `/content` silinir). Ayrıntılar not defterinin ilk bölümünde.

## GitHub’a yükleme

1. [Git for Windows](https://git-scm.com/download/win) kurulu olsun.
2. Bu klasörde (henüz repo değilse):

```powershell
cd C:\Users\erdem\BlackWhitetoColor
git init
git add .
git commit -m "Initial commit: LAB colorization U-Net"
```

3. GitHub’da **yeni boş repo** oluştur (README ekleme seçeneğini kapatabilirsin).
4. Uzak adresi ekle ve gönder:

```powershell
git remote add origin https://github.com/KULLANICI_ADIN/REPO_ADI.git
git branch -M main
git push -u origin main
```

**Not:** `checkpoints/*.pt`, `runs/`, `data/train`, `data/val` `.gitignore` ile **dışarıda** kalır; büyük dosya ve veri seti GitHub limitine takılmaz. Modeli paylaşmak için **Releases**’e `.pt` ekleme, Drive veya başka depo kullanma daha uygun olur.

## Lisans

Projeye uygun bir `LICENSE` dosyası eklemediysen, dağıtım için bir lisans seçmen iyi olur.

## Daha fazla ayrıntı

Türkçe özet: [PROJE-OZET.md](PROJE-OZET.md), [EGITIM.md](EGITIM.md)
