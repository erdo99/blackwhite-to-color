# Proje özeti — BlackWhitetoColor

## Nasıl çalışıyor?

1. **Eğitim:** Renkli (RGB) görseller alınır, **LAB** uzayına çevrilir. **L\*** (parlaklık) gri görüntü gibi kullanılır; rastgele nokta/bölge seçilerek **kullanıcı ipucu** taklit edilir (maske + doğru **a\*, b\*** değerleri). **U-Net** bu 4 kanallı girdiden tüm görüntü için **a\*, b\*** tahmin eder. Kayıp: **L1** + **VGG perceptual** (isteğe bağlı **GAN**).

2. **Çıkarım:** Kullanıcı gri görüntü yükler, üzerine fırça ile renk çizer. Arka plan ile boyalı görüntü karşılaştırılarak **ipucu maskesi** ve **hint renkleri** çıkarılır; model renkli çıktı üretir. İsteğe bağlı **yüksek çözünürlük**: a\*/b\* küçük boyutta tahmin edilip tam boy **L\*** ile birleştirilir.

---

## Dosyalar (kısa)

| Dosya / klasör | İşi |
|----------------|-----|
| `config.yaml` | Veri yolları, model boyutu, eğitim hiperparametreleri, checkpoint yolu |
| `requirements.txt` | Python paket listesi (`pip` ile kurulacak) |
| `train.py` | Eğitim döngüsü, checkpoint kaydı |
| `infer.py` | Checkpoint yükleme, tek görüntü renklendirme, `colorize` / `colorize_variants` |
| `app.py` | Gradio arayüzü (görüntü + fırça, çalıştır) |
| `src/color_space.py` | RGB ↔ LAB dönüşümü (PyTorch) |
| `src/hints.py` | Eğitimde rastgele ipucu üretimi; çıkarımda ipucu çıkarma |
| `src/dataset.py` | Klasörden görüntü okuma, augmentasyon, 4 kanallı tensör |
| `src/losses.py` | L1 (train içinde), VGG perceptual, hinge GAN kayıpları |
| `src/models/unet.py` | Hint-guided U-Net (4→2 kanal) |
| `src/models/discriminator.py` | İsteğe bağlı PatchGAN ayırıcı |
| `src/refine.py` | AB kanallarını tam çözünürlüğe yükseltme + L birleştirme |
| `src/segment_hints.py` | Etiket haritasından toplu renk ipuçları (isteğe bağlı) |
| `data/train/`, `data/val/` | Eğitim/doğrulama için RGB görüntü klasörleri |

---

## Kütüphaneler hazır mı?

**Hayır — otomatik kurulmaz.** `requirements.txt` sadece **hangi paketlerin gerekli olduğunu** listeler. Bilgisayarda Python varsa şunu bir kez çalıştırman gerekir:

```powershell
cd c:\Users\erdem\BlackWhitetoColor
pip install -r requirements.txt
```

Gerekli paketler: `torch`, `torchvision`, `numpy`, `Pillow`, `PyYAML`, `tqdm`, `gradio`.

---

## Sanal ortam (venv) gerekli mi?

**Zorunlu değil**, ama **önerilir.** Projeyi sistem Python’undan ayırır; sürüm çakışması yaşamazsın.

```powershell
cd c:\Users\erdem\BlackWhitetoColor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Nasıl test edebilirsin?

### 1) Kurulum ve import testi

```powershell
.\.venv\Scripts\Activate.ps1   # venv kullanıyorsan
pip install -r requirements.txt
python -c "from src.models.unet import HintGuidedUNet; import torch; m=HintGuidedUNet(); x=torch.randn(1,4,256,256); print(m(x).shape)"
```

Beklenen çıktı: `torch.Size([1, 2, 256, 256])`

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

Tarayıcıda açılan adreste görüntü yükle, ipucu çiz, **Run** de.

### 5) Checkpoint yokken arayüz

Uygulama açılır ama renklendirme için önce eğitim gerekir; aksi halde checkpoint bulunamadı uyarısı alırsın.

---

## Özet

| Soru | Cevap |
|------|--------|
| Kütüphaneler projede “hazır” mı? | Liste `requirements.txt` içinde; `pip install` ile kurulmalı. |
| venv şart mı? | Hayır; kullanman iyi pratik. |
| En hızlı test | Yukarıdaki `HintGuidedUNet` + `torch.randn` tek satırlık komut. |
