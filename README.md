# Parkinson's Disease Classification from Voice Recordings

A beginner-friendly, end-to-end machine learning project that classifies
voice recordings as **Parkinson's** or **healthy**. Built in Google Colab following a
9-step ML workflow, with a small web demo.

> **Educational project only. Not a medical diagnosis tool.**

## Notebook
[`Parkinsons_Voice_ML.ipynb`](Parkinsons_Voice_ML.ipynb): every step has a text explanation, the code, and the saved outputs.

## Dataset
Kaggle: *Parkinson's Voice Dataset* (not included in this repo, download it from Kaggle).

- 1,134 `.wav` files (8 kHz, mono) → **567 unique** recordings after duplicate removal (287 healthy, 280 Parkinson's)
- **The dataset page gives no documentation** of the source, participants, or recording protocol, so the labels are taken as given and are unverified
- File names are numbers only, so there are **no speaker IDs**

## Pipeline
1. **Problem definition:** voice audio in, Parkinson's or healthy out (binary classification)
2. **Data acquisition:** mount Google Drive, unzip, inspect the files
3. **Cleaning:** checked corrupt files, sample rates, durations, silence. Found that **every file appeared twice** (567 exact duplicates, found by MD5 fingerprint) and removed them to prevent train/test leakage
4. **Preprocessing:** fixed sample rate, silence trimming, volume normalisation (the data turned out to be already trimmed and normalised)
5. **Splitting:** 70/15/15 stratified (396 / 85 / 86 files), zero overlap between sets
6. **Feature extraction:** 34 features (13 MFCC mean and std, pitch, energy, spectral); scaler fitted on the training set only
7. **Models:** Logistic Regression, SVM (RBF), Random Forest, compared with 5-fold cross-validation
8. **Evaluation:** test set opened once, at the end
9. **Deployment:** Gradio web app (upload or record a clip)

## Results
| Model | Train | CV accuracy | Validation |
|---|---|---|---|
| Logistic Regression | 0.808 | 0.768 | 0.824 |
| SVM (RBF) | 0.957 | 0.889 | 0.918 |
| **Random Forest** | 1.000 | **0.912** | 0.918 |

**Test set (Random Forest, chosen before looking at the test set):** accuracy 0.942, ROC-AUC 0.991,
Parkinson's recall 0.90, precision 0.97 (86 files, so the realistic range is about 91 to 94%).
All 5 mistakes had a probability between 0.39 and 0.56, close to the 0.5 decision line.

## Key findings
- 9 of the top 10 important features measure **variability** (`_std`): how much loudness, pitch and spectrum change over the clip matters more than their averages.
- Shortcut checks passed: a duration-only model scores 0.49 (chance level), and removing the loudness (RMS) features does not hurt accuracy (CV 0.914).
- Step 8b of the notebook contains two extra checks (near-duplicates and recording signature) and a corrected cross-validation.

## Limitations
- **Undocumented dataset:** the results show the model separates the two folders; they cannot be claimed to detect Parkinson's in general
- **No speaker IDs:** the same person may appear in both train and test, which can inflate scores
- Small dataset and small test set (86 files, one file is worth 1.2%)
- **Domain shift:** on my own voice recorded with a laptop microphone (speech with pauses), the app gave 70% and 93% "Parkinson's" for the same healthy person. The model is sensitive to pauses and to recording conditions it never saw
- Internal pauses in a clip are not trimmed
- Feature importance shows association, not causation

## Run it
1. Download the dataset from Kaggle and upload the zip to the root of your Google Drive as `archive.zip`
2. Open `Parkinsons_Voice_ML.ipynb` in Google Colab
3. **Runtime, then Run all.** Features and the trained model are saved to `MyDrive/ml_project/`

## Future work
- Use a documented dataset with speaker IDs and split by speaker
- Test on an external dataset from a different source
- Remove internal silence and reduce noise before feature extraction
- Data augmentation on the training set only
- Calibrated probabilities and an input-quality check in the app

## Tech
Python, librosa, scikit-learn, pandas, NumPy, matplotlib, Gradio
