# 🛡️ SentinelSpam: Email & Message Spam Detection System

An enterprise-ready, high-accuracy Machine Learning and Natural Language Processing (NLP) pipeline for detecting unsolicited, fraudulent, and malicious communications. Built with **scikit-learn**, **TF-IDF n-gram vectorization**, and a calibrated **Multinomial Naive Bayes** classifier, paired with a cybersecurity-themed **Streamlit** user interface.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Technology Stack](#-technology-stack)
4. [Dataset & Data Transparency](#-dataset--data-transparency)
5. [Architecture & Workflow](#-architecture--workflow)
6. [Evaluation Methodology & Actual Results](#-evaluation-methodology--actual-results)
7. [Project Structure](#-project-structure)
8. [Installation & Windows Quick Start](#-installation--windows-quick-start)
9. [Running Automated Tests](#-running-automated-tests)
10. [Known Limitations & Future Roadmap](#-known-limitations--future-roadmap)
11. [License & Ethics](#-license--ethics)

---

## 🎯 Project Overview

Email and messaging spam pose severe cybersecurity and productivity risks, spanning commercial unsolicited advertisements to deceptive phishing attacks and financial scams. 

The primary objective of this project is to build an accurate, transparent, and reproducible spam classification system that:
- Normalizes and tokenizes text while preserving crucial spam signals (URLs, currency symbols, numbers, and high-frequency punctuation).
- Extracts robust feature representations using **Term Frequency-Inverse Document Frequency (TF-IDF)** with unigram and bigram coverage.
- Employs a **Multinomial Naive Bayes** probabilistic classifier optimized for high spam precision to minimize harmful false positives on legitimate emails.
- Delivers an intuitive, real-time SaaS security dashboard via **Streamlit** to inspect live classifications, probability distributions, token-level signals, and test telemetry.

---

## ✨ Key Features

- **Spam-Signal Preserving Preprocessor:** Converts URLs into `http_token`, currency markers into `currency_token`, phone/account numbers into `number_token`, and excessive exclamations into `excl_token`, ensuring critical indicators are never stripped away.
- **Data-Leakage Free Pipeline:** All vectorizer transformations and vocabulary fittings occur strictly on the training partition.
- **5-Fold Stratified Cross-Validation:** Compared candidate architectures (Multinomial Naive Bayes vs. Balanced Logistic Regression) prior to final evaluation.
- **Live Token Signal Breakdown:** Highlights high-weight keywords and tokens driving the classifier's prediction for full model interpretability.
- **Cybersecurity SaaS UI:** Modern dark-mode interface with pre-loaded testing templates, responsive meters, and confusion matrix visualization.
- **Thoroughly Tested:** Comprehensive automated test suite (`pytest`) covering data hygiene, text normalization, edge cases, and model persistence.

---

## 💻 Technology Stack

| Domain | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.12 (64-bit) | Core runtime environment |
| **Data Manipulation** | `pandas` (>=2.2), `numpy` (>=1.26) | Dataset loading, validation, and schema checks |
| **Machine Learning** | `scikit-learn` (>=1.4) | TF-IDF extraction, Multinomial Naive Bayes, Logistic Regression, pipeline construction, metrics |
| **Model Persistence** | `joblib` (>=1.3) | Atomic pipeline serialization and fast inference caching |
| **Web Interface** | `streamlit` (>=1.35) | Modern interactive cybersecurity SaaS dashboard |
| **Visualization** | `matplotlib` (>=3.8), `seaborn` (>=0.13) | Confusion matrix heatmaps and diagnostic plots |
| **Testing** | `pytest` (>=8.0) | Automated test suite for data, inference, and preprocessing |

---

## 📂 Dataset & Data Transparency

- **Benchmark Dataset:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (Tiago A. Almeida & José María Gómez Hidalgo).
- **Public Domain / Attribution:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Raw Total Records:** 5,572 records.
- **Deduplication:** 414 duplicate records removed to ensure zero data leakage across train and test partitions.
- **Cleaned Total Records:** 5,158 unique records.
- **Class Distribution:**
  - **Legitimate (Ham):** 4,516 records (87.55%)
  - **Spam:** 642 records (12.45%)
- **Data Disclosure:** Although sourced from messaging and SMS communications, the textual lexical patterns (urgency, monetary rewards, shortened phishing URLs, lottery announcements) directly reflect email spam and phishing mechanics.

---

## 🧠 Architecture & Workflow

```text
Input Message
    │
    ▼
[ Text Preprocessing ]
    ├── Lowercasing & whitespace normalization
    ├── Regex tokenization: URLs -> 'http_token'
    ├── Currency symbols -> 'currency_token'
    ├── Digits & numbers -> 'number_token'
    └── Exclamation bursts -> 'excl_token'
    │
    ▼
[ TF-IDF Vectorizer ]
    ├── Unigrams & Bigrams (1, 2)
    ├── Sublinear term-frequency scaling
    └── English stop-words filtering
    │
    ▼
[ Multinomial Naive Bayes (alpha=0.2) ]
    ├── Log-likelihood class calculation
    └── Calibrated posterior probability output
    │
    ▼
[ Inference Output ]
    ├── Classification: SPAM or NOT SPAM
    ├── Confidence % & Probability distribution
    └── Key detected lexical signals
```

---

## 📈 Evaluation Methodology & Actual Results

Evaluation was executed on a strictly held-out **20% stratified test split** (1,032 messages: 904 legitimate, 128 spam) that was never seen during vocabulary fitting or parameter tuning.

### Actual Held-out Test Metrics

| Metric | Result | Target / Industry Benchmark |
| :--- | :--- | :--- |
| **Overall Accuracy** | **98.55%** | > 95% |
| **Spam Precision** | **98.29%** | > 95% (Minimizes false spam alerts) |
| **Spam Recall** | **89.84%** | > 85% (Catches vast majority of threats) |
| **Spam F1-Score** | **93.88%** | > 90% |
| **ROC-AUC Score** | **0.9931** | > 0.95 |

### Confusion Matrix Breakdown (Test Set: N = 1,032)

| | Predicted Legitimate (Ham) | Predicted Spam |
| :--- | :---: | :---: |
| **Actual Legitimate (Ham)** | **902** *(True Negatives)* | **2** *(False Positives)* |
| **Actual Spam** | **13** *(False Negatives)* | **115** *(True Positives)* |

### Error Analysis & Insights
- **False Positives (2 cases):** Short colloquial chat messages (e.g. *"sms chat with me"*) received borderline spam probability (~51-54%) due to historical association with premium SMS chat spam.
- **False Negatives (13 cases):** Conversational camouflage without explicit marketing keywords (e.g. *"Sorry I missed your call let's talk when you have the time"* followed by an unrecognized phone number).

---

## 📁 Project Structure

```text
EMAIL SPAM DETECTION/
├── .gitignore                     # Git exclusions (venv, caches, temp files)
├── requirements.txt               # Pinned project dependencies
├── README.md                      # Comprehensive project documentation
├── app.py                         # Streamlit SaaS web application
├── data/
│   ├── raw/
│   │   └── sms_spam_raw.tsv       # Raw benchmark dataset
│   └── processed/
│       └── clean_spam.csv         # Cleaned, validated, deduplicated data
├── models/
│   └── spam_pipeline.joblib       # Serialized TF-IDF + MultinomialNB pipeline
├── reports/
│   ├── dataset_summary.json       # Schema, null, and label distribution stats
│   ├── metrics.json               # Actual held-out test evaluation report
│   ├── error_analysis.json        # False positive & false negative case logs
│   └── confusion_matrix.png       # Confusion matrix visualization
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Dataset downloader and schema validator
│   ├── preprocessing.py          # Tokenizer and regex spam signal extractor
│   ├── train_model.py             # 5-fold CV, pipeline training, and evaluation
│   └── predict.py                 # Safe inference engine with caching
└── tests/
    ├── __init__.py
    ├── test_dataset.py            # Automated tests for dataset schema
    ├── test_preprocessing.py      # Automated tests for text normalization
    └── test_prediction.py         # Automated tests for inference & edge cases
```

---

## 🚀 Installation & Windows Quick Start

### 1. Clone or Open Workspace
Open Windows PowerShell and navigate to the project directory:
```powershell
cd "d:\EMAIL SPAM DETECTION"
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Acquire Dataset & Train Model
To download the verified benchmark dataset and train the full ML pipeline:
```powershell
python -m src.data_loader
python -m src.train_model
```
*Outputs will be generated in `models/spam_pipeline.joblib` and `reports/`.*

### 5. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
*The interactive dashboard will open automatically in your browser at `http://localhost:8501`.*

---

## 🧪 Running Automated Tests

Run the complete test suite using `pytest`:
```powershell
pytest -v
```
All **17 automated unit and integration tests** verify:
- Non-null and valid data formatting.
- Preservation of URL tokens, currency markers, and numbers.
- Safe handling of empty, whitespace, and null inputs.
- Accurate prediction classification and probability calculation.
- Model artifact availability and deserialization integrity.

---

## ⚠️ Known Limitations & Future Roadmap

- **Adversarial Obfuscation:** The model analyzes text tokens; messages containing zero-width spaces, character homoglyphs (e.g. Cyrillic letters replacing Latin), or text embedded entirely inside images require an optical character recognition (OCR) or multimodal preprocessing step.
- **Domain Shift:** While general spam cues overlap, formal corporate spear-phishing messages may require training on enterprise email headers (SPF, DKIM, DMARC) in addition to body text.
- **Future Improvements:**
  - [ ] Add header-level metadata analysis (Sender domain reputation, SPF/DKIM verification).
  - [ ] Experiment with lightweight Transformer models (e.g., DistilBERT) for nuanced semantic context.
  - [ ] Add batch CSV/EML upload and inbox scanning simulation.

---

## 📄 License & Attribution

- **Dataset:** UCI Machine Learning Repository / Tiago A. Almeida & José María Gómez Hidalgo.
- **Code:** Distributed under the MIT License for educational and portfolio demonstration.
