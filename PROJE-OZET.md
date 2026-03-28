# Proje özeti — BlackWhitetoColor

## Nasıl çalışıyor?

1. **Eğitim (varsayılan — otomatik):** RGB görseller **LAB**’a çevrilir. Girdi **yalnızca L** (1 kanal); **U-Net** **a\*, b\*** tahmin eder (`model.in_channels: 1`, `data.use_hints: false`). Kayıp: **kroma ağırlıklı L1(ab)** (`chroma_l1_scale`, 0 ise düz L1) + **VGG perceptual**; isteğe bağlı **GAN** (ayırıcıda **spectral norm**).

2. **Eğitim (ipuculu):** `use_hints: true` ve `in_channels: 4` iken L + maske + ipucu **ab** birleştirilir; eğitimde seyrek ipucu simülasyonu kullanılır.

3. **Çıkarım:** **Otomatik** modda gri/renkli yükleme → L’den renk. **Ipuculu** modda arka plan + fırça; ipucu tensörleri çıkarılır. İsteğe bağlı **tam çözünürlük**: küçük boyutta **ab** tahmini, tam boy **L** ile birleştirme (`refine`).

---

## Dosyalar (kısa)

| Dosya / klasör | İşi |
|----------------|-----|
| `config.yaml` | Veri yolları, model boyutu, eğitim hiperparametreleri, checkpoint yolu |
| `requirements.txt` / `requirements-full.txt` | Hafif veya tam bağımlılık listesi |
| `train.py` | Eğitim döngüsü, checkpoint kaydı |
| `infer.py` | Checkpoint yükleme, tek görüntü renklendirme, `colorize` / `colorize_variants` |
| `app.py` | Gradio: otomatik veya ipuculu mod (`use_hints`) |
| `src/color_space.py` | RGB ↔ LAB dönüşümü (PyTorch) |
| `src/hints.py` | Eğitimde rastgele ipucu üretimi; çıkarımda ipucu çıkarma |
| `src/dataset.py` | Klasörden görüntü okuma, augmentasyon; 1 veya 4 kanal model girdisi |
| `src/losses.py` | `chroma_weighted_l1`, VGG perceptual, hinge GAN kayıpları |
| `src/models/unet.py` | U-Net: 1→2 (otomatik) veya 4→2 (ipuculu) |
| `src/models/discriminator.py` | İsteğe bağlı PatchGAN (Conv’larda spectral norm) |
| `colab.ipynb` | Google Colab notu (`.gitignore`’da olabilir; repoya dahil edilmeyebilir) |
| `src/refine.py` | AB kanallarını tam çözünürlüğe yükseltme + L birleştirme |
| `src/segment_hints.py` | Etiket haritasından toplu renk ipuçları (isteğe bağlı) |
| `data/train/`, `data/val/` | Eğitim/doğrulama için RGB görüntü klasörleri |

---

## Kütüphaneler hazır mı?

**Hayır — otomatik kurulmaz.** Önerilen tam kurulum:

```powershell
cd c:\Users\erdem\BlackWhitetoColor
pip install -r requirements-full.txt
```

Torch zaten kuruluysa yalnızca eksikler için `requirements.txt` yeterli olabilir.

---

## Sanal ortam (venv) gerekli mi?

**Zorunlu değil**, ama **önerilir.** Projeyi sistem Python’undan ayırır; sürüm çakışması yaşamazsın.

```powershell
cd c:\Users\erdem\BlackWhitetoColor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-full.txt
```

---

## Nasıl test edebilirsin?

### 1) Kurulum ve import testi

```powershell
.\.venv\Scripts\Activate.ps1   # venv kullanıyorsan
pip install -r requirements-full.txt
python -c "from src.models.unet import HintGuidedUNet; import torch; m=HintGuidedUNet(in_channels=1); x=torch.randn(1,1,256,256); print(m(x).shape)"
```

Beklenen çıktı: `torch.Size([1, 2, 256, 256])` — ipuculu mimari için `in_channels=4` ve `x` boyutu `(1,4,256,256)` dene.

### 2) LAB dönüşümü

```powershell
python -c "import torch; from src.color_space import rgb_to_lab, lab_to_rgb; r=torch.rand(1,3,64,64); l=rgb_to_lab(r); r2=lab_to_rgb(l); print(r2.shape)"
```

Beklenen: `torch.Size([1, 3, 64, 64])`

### 3) Eğitim (veri varsa)

`data/train` içine birkaç `.jpg` / `.png` koy, sonra:

```powershell
python train.py --config config.yaml
```

### 4) Arayüz (checkpoint varken)

`checkpoints/best.pt` oluştuktan sonra:

```powershell
python app.py
```

Tarayıcıda: **otomatik** modda görüntü yükle → **Renklendir**; **ipuculu** modda ImageEditor ile ipucu ver.

### 5) Checkpoint yokken arayüz

Uygulama açılır ama renklendirme için önce eğitim gerekir; aksi halde checkpoint bulunamadı uyarısı alırsın.

---

## Özet

| Soru | Cevap |
|------|--------|
| Kütüphaneler projede “hazır” mı? | `requirements-full.txt` ile `pip install`; hafif liste `requirements.txt`. |
| venv şart mı? | Hayır; kullanman iyi pratik. |
| En hızlı test | Yukarıdaki `HintGuidedUNet` + `torch.randn` tek satırlık komut. |
