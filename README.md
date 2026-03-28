# CEG3004 Mini-Project — Environmental Sound Classification
**Group:** Pr_4

---

## Project Overview

This project focuses on building a **robust Environmental Sound Classification (ESC) system** using **Digital Signal Processing (DSP)** and **Machine Learning** techniques.

The goal is to accurately classify environmental sounds into **50 classes**, while maintaining strong performance under:
- Clean audio
- Noisy audio
- Band-limited audio

---

## Project Aim

To design a **robust audio classification pipeline** that performs well under both clean and distorted conditions.

---

## Objectives
- Train on labeled environmental sound data
- Extract meaningful DSP features
- Build a machine learning classifier
- Ensure robustness to noise and distortions

---

## 📂 Dataset
- Based on the **ESC-50 dataset**
- 2,000 audio clips (5 seconds each, mono)
- 50 sound classes (40 clips per class)

### Dataset Structure

data/
│── train/
│ ├── audio/
│ ├── labels.csv
│
│── submission/
│ ├── audio/
│ ├── metadata.csv

---

### Submission Set Variants
Each audio clip has:
- Clean version
- Noisy version
- Band-limited version

This tests robustness of the model.

---

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Make sure `GROUP_ID` at the top is set to `"Pr_4"`
3. Run all cells from top to bottom (Cell 1 through Cell 10)
4. Two files will be automatically downloaded at the end:
   - `Pr_4_model.joblib`
   - `Pr_4_predictions.csv`

> The dataset is downloaded automatically in Cell 2 from Google Drive. If that fails, there is a manual download link inside the cell.

---

## What We Improved

### Preprocessing (Cell 5)
- Trim silence from the start and end of each clip
- Apply a pre-emphasis filter to boost high frequencies
- Pad or cut every clip to exactly 5 seconds
- Normalise the volume using RMS normalisation (more stable than peak normalisation when there is background noise)

### Feature Extraction (Cell 6)
We extract several types of features from each audio clip:

| Feature | What it captures |
|---|---|
| MFCC + deltas | Tone and timbre of the sound |
| Log-mel spectrogram + CMVN | Overall frequency shape, normalised for noise robustness |
| Spectral features | Brightness, spread, and energy of the sound |
| Log-mel deltas | How the frequency shape changes over time |
| Zero-crossing rate | Whether the sound is more percussive or tonal |

### Model (Cell 8)
We use a soft-voting ensemble of two classifiers:
- **SVM** with an RBF kernel (good at finding boundaries between similar-sounding classes)
- **Random Forest** with 300 trees (good at handling varied features)

Both classifiers use `class_weight='balanced'` to handle the fact that all 50 classes have the same number of clips.

We also print a per-class report after training so we can see which sound classes the model struggles with.

### Robustness / Augmentation (Optional cell)
During training, we randomly apply one or more of the following to each clip:
- Add a small amount of background noise
- Randomly adjust the volume (±6 dB)
- Apply a bandpass filter (simulates low-quality recordings)
- Shift the audio slightly in time

This is only applied during training — submission clips are never modified.

---

## Dependencies

All dependencies are installed automatically in Cell 1:

```
numpy
scipy
pandas
scikit-learn
librosa
soundfile
tqdm
```

---

## Submission Files

| File | Description |
|---|---|
| `Pr_4_model.joblib` | Trained ensemble model |
| `Pr_4_predictions.csv` | Predicted labels for all submission clips |

The predictions CSV has two columns: `clip_id` and `predicted_label`. The `clip_id` values are taken directly from the submission metadata and are not modified.
