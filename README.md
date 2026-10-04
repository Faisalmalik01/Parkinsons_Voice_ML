# Parkinson's Disease Classification from Voice Recordings

A beginner-friendly, end-to-end machine learning project that classifies
voice recordings as **Parkinson's** or **healthy**, built in Google Colab following a
9-step ML workflow.

## Dataset
Kaggle: [Parkinson's Voice Dataset](https://www.kaggle.com/datasets/ucimachinelearning/parkinsons-voice-dataset)
(not included in this repo, download it from Kaggle).
- 1,134 `.wav` files (8 kHz, mono) → **567 unique** recordings after duplicate removal
  (287 healthy, 280 Parkinson's)
- No speaker IDs are provided (see Limitations)

## Pipeline
1. **Problem definition:** binary classification from voice audio
2. **Data acquisition:** unzip and inspect the files
3. **Cleaning:** found that every file appeared twice (567 exact duplicates by MD5 hash); removed them to prevent train/test leakage
4. **Preprocessing:** fixed sample rate, silence trimming, volume normalization
5. **Splitting:** 70/15/15 stratified (396 / 85 / 86), zero file overlap
6. **Feature extraction:** 34 features (13 MFCC mean+std, pitch, energy, spectral); scaler fitted on train only
7. **Models:** Logistic Regression, SVM (RBF), Random Forest
8. **Evaluation:** test set opened once, at the end
9. **Deployment:** Gradio app (upload or record a voice clip)

## Results
| Model | CV accuracy (train) | Validation accuracy |
|---|---|---|
| Logistic Regression | 0.768 | 0.824 |
| SVM (RBF) | 0.889 | 0.918 |
| **Random Forest** | **0.912** | **0.918** |

**Test set (Random Forest):** accuracy 0.942, ROC-AUC 0.991,
Parkinson's recall 0.90, precision 0.97 (86 files, so the realistic range is about 91-94%).

## Key findings
- 9 of the top 10 important features are variability features (`_std`): unsteadiness over time matters more than average values.
- Shortcut checks passed: a duration-only model scores 0.49 (chance), and removing loudness (RMS) features doesn't hurt accuracy (0.914 CV).

## Limitations
- No speaker IDs, so the same person may appear in both train and test, which can inflate scores
- Small dataset and small test set (86 files)
- Domain shift: recordings from other microphones, rooms, or speech (instead of sustained sounds) can give unreliable results
- Feature importance shows association, not causation
- **Educational demo only, not a medical diagnosis**

## Run it
1. Open `parkinsons_voice_classification.ipynb` in Google Colab
2. Download the dataset from Kaggle and upload the zip to Google Drive
3. Update the file paths in the notebook, then *Runtime → Run all*

## Tech
Python, librosa, scikit-learn, pandas, NumPy, matplotlib, Gradio

## Author
Faisal Malik 
