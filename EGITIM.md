# Model eğitimi — adım adım

Bu dosya, `train.py` ile LAB renklendirme modelini nasıl eğiteceğini özetler. Varsayılan (`config.yaml`): **otomatik** mod — `data.use_hints: false`, `model.in_channels: 1` (yalnızca L → a\*/b\*). İsteğe bağlı: **ipuculu** mod (`use_hints: true`, `in_channels: 4`).

---

## 1. Ön koşullar

- Python kurulu (projede **3.13** kullanıldığı görülmüş; başka 3.x de olur).
- PyTorch + torchvision kurulu (tercihen **CUDA**’lı sürüm — eğitim hızlanır).
- Proje bağımlılıkları: tam kurulum için `requirements-full.txt`, sadece eksikler için `requirements.txt` (bkz. `PROJE-OZET.md`).

Kontrol:

```powershell
cd C:\Users\erdem\BlackWhitetoColor
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 2. Eğitim verisini hazırlama

- **Renkli** görseller kullanılır (RGB). Model bunları LAB’a çevirir; **otomatik** modda girdi yalnızca **L** (parlaklık), hedef **a\*/b\***. **Ipuculu** modda eğitimde seyrek ipucu simülasyonu da eklenir (`hints` bölümü).
- Varsayılan klasörler (`config.yaml` içindeki `data` bölümü):

| Klasör | Amaç |
|--------|------|
| `data/train/` | Eğitim görselleri (çoğu dosya burada) |
| `data/val/` | Doğrulama; birkaç görüntü koyman iyi olur. Klasör **boşsa** doğrulama seti 0 örnek olur, `best.pt` seçimi anlamlı olmaz — o zaman geçici olarak `config.yaml` içinde `val_dir: "./data/train"` yapabilirsin (küçük denemeler için; idealde train/val ayrıdır). |

**Desteklenen uzantılar:** `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`

Alt klasörlere koyabilirsin; kod recursive tarar.

**Örnek:**

```text
BlackWhitetoColor/
  data/
    train/
      foto1.jpg
      foto2.png
      ...
    val/          (isteğe bağlı)
      test1.jpg
```

Yeterli veri: pratikte **yüzlerce / binlerce** görüntü daha iyi sonuç verir; az sayıda görüntüyle de çalışır ama genelleme sınırlı kalır.

---

## 3. Ayarları kontrol et (`config.yaml`)

Önemli alanlar:

- **`data.use_hints`** / **`model.in_channels`**: `false` + **1** = otomatik; `true` + **4** = ipuculu (uyumsuz kombinasyon `train.py` reddeder).
- **`data.train_dir`**, **`data.val_dir`**: Veri yolları.
- **`data.max_train_samples`**: Tüm train yerine en fazla bu kadar görüntü (alfabetik ilk N). `train_sequential_chunks: true` ise her epoch sıradaki N blok.
- **`data.image_size`**: Varsayılan **256** (VRAM’e göre 128’e düşürebilirsin).
- **`data.num_workers`**: Windows’ta takılma olursa **0** dene.
- **`train.batch_size`**: Varsayılan **8**; GPU belleği yetmezse **4** veya **2** yap.
- **`train.epochs`**: Varsayılan **100** (erken durdurmak için sayıyı azaltabilirsin).
- **`train.lambda_l1`**, **`lambda_perceptual`**, **`chroma_l1_scale`**: Kayıp dengesi; `chroma_l1_scale: 0` düz L1(ab).
- **`train.use_gan`**: Varsayılan **false** (L1 + perceptual). **true** ise GAN devreye girer; ayırıcıda **spectral normalization** kullanılır.
- **`train.checkpoint_dir`**: Varsayılan **`./checkpoints`** — en iyi model **`best.pt`** olarak buraya yazılır.

İlk denemede `train_dir` / `val_dir` ve gerekirse `batch_size` / `image_size` ile oynaman yeterli.

---

## 4. Eğitimi başlatma

Proje kökünde:

```powershell
cd C:\Users\erdem\BlackWhitetoColor
python train.py --config config.yaml
```

Başka bir config dosyası kullanacaksan:

```powershell
python train.py --config benim_ayarlarim.yaml
```

Eğitim sırasında:

- Konsolda **epoch** ve **L1 (kroma ağırlıklı)** / **perceptual** benzeri loglar görünür.
- **`checkpoints/best.pt`**: Doğrulama metriğine göre seçilen en iyi üretici (generator) ağırlıkları.
- **`checkpoints/epoch_N.pt`**: Her `save_every` epoch’ta ek kayıtlar (varsayılan her epoch).

---

## 5. Eğitim bittikten sonra

- Arayüz: `python app.py` — `infer.checkpoint` genelde `./checkpoints/best.pt`. **Otomatik** modda tek görüntü yüklenir; **ipuculu** modda ImageEditor ile fırça kullanılır.
- Komut satırı: `python infer.py --image yol\gorsel.jpg` — 1 kanallı checkpoint’te otomatik renklendirme; 4 kanallıda ipucusuz sıfır mask ile de çalışır ama asıl amaç UI ile ipucu.

---

## 6. Sık karşılaşılan durumlar

| Sorun | Ne yapmalı? |
|--------|-------------|
| `No images in .../train` | `data/train` içine en az bir uygun formatta görüntü koy. |
| CUDA out of memory | `config.yaml` → `train.batch_size` küçült veya `data.image_size` düşür. |
| Çok yavaş | CUDA kurulu ve `torch.cuda.is_available()` True mu kontrol et; CPU ile de çalışır ama uzun sürer. Windows’ta DataLoader takılırsa `num_workers: 0`. |
| İlk kez perceptual loss | VGG16 ağırlıkları internetten indirilebilir; firewall/proxy engelini kontrol et. |

---

## 7. Kısa özet komutlar

```powershell
cd C:\Users\erdem\BlackWhitetoColor
# Veriyi koy: data\train\*.jpg ...
python train.py --config config.yaml
# Sonra:
python app.py
```

Bu kadar: veri → `train.py` → `checkpoints/best.pt` → `app.py` veya `infer.py`.
