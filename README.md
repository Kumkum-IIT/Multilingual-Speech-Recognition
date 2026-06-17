# Multilingual Speech Recognition System

## Overview

This project implements a multilingual Automatic Speech Recognition (ASR) pipeline for English, Hindi, Urdu, and Tamil speech. It combines the strengths of OpenAI Whisper for language detection and English transcription with the AI4Bharat Indic Conformer model for Indian language transcription.

The solution was developed for a multilingual speech recognition competition and includes data preprocessing, language identification, transcription, validation, and submission generation.

---

## Features

* Automatic language detection using Whisper
* English speech transcription using Whisper Medium
* Hindi, Urdu, and Tamil transcription using AI4Bharat Indic Conformer
* Text normalization and cleaning
* Number-to-word conversion for English transcripts
* Validation using Word Error Rate (WER)
* Competition submission file generation

---

## Project Workflow

### 1. Environment Setup

Required libraries:

* faster-whisper
* transformers
* torchaudio
* torch
* jiwer
* librosa
* soundfile
* tqdm
* num2words
* onnxruntime

Install dependencies:

```bash
pip install faster-whisper jiwer librosa soundfile tqdm num2words
pip install transformers torchaudio onnxruntime onnx
```

---

### 2. Dataset Loading

The project loads:

* `train.csv`
* `test.csv`

and constructs absolute paths to audio files.

```text
competition_data/
├── train/
├── test/
├── train.csv
└── test.csv
```

---

### 3. Language Detection

Language labels are inferred from transcript scripts:

| Language | Script Detection |
| -------- | ---------------- |
| English  | Latin characters |
| Hindi    | Devanagari       |
| Tamil    | Tamil Unicode    |
| Mixed    | Multiple scripts |

For inference, Whisper is used to detect the spoken language directly from audio.

---

### 4. Text Preprocessing

The pipeline performs:

* Lowercasing
* Special character removal
* Whitespace normalization
* English number-to-word conversion
* Output formatting

Example:

```text
Input:
I have 12 books

Output:
i have twelve books.
```

---

### 5. Models Used

#### Whisper Medium

Used for:

* Language detection
* English transcription
* Fallback transcription

Model:

```python
WhisperModel("medium")
```

#### AI4Bharat Indic Conformer

Used for:

* Hindi transcription
* Urdu transcription
* Tamil transcription

Model:

```python
ai4bharat/indic-conformer-600m-multilingual
```

---

### 6. Transcription Strategy

| Detected Language | Model Used                      |
| ----------------- | ------------------------------- |
| English           | Whisper                         |
| Hindi             | Indic Conformer                 |
| Urdu              | Indic Conformer (Hindi decoder) |
| Tamil             | Indic Conformer                 |
| Others            | Whisper                         |

Pipeline:

```text
Audio
 ↓
Language Detection
 ↓
Model Selection
 ↓
Transcription
 ↓
Text Normalization
 ↓
Final Prediction
```

---

### 7. Validation

The training data is split using stratified sampling:

```python
train_test_split(
    test_size=0.2,
    stratify=train_df["language"]
)
```

Performance is evaluated using:

```python
jiwer.wer()
```

Metric:

* Word Error Rate (WER)

Lower WER indicates better transcription quality.

---

### 8. Submission Generation

The system processes every audio file in the test set and generates:

```csv
id,text
0,predicted transcription
1,predicted transcription
...
```

Submission file:

```text
submission.csv
```

---

## Future Improvements

* Fine-tune Whisper on competition data
* Add language-specific post-processing
* Ensemble multiple ASR models
* Improve mixed-language transcription
* Add punctuation restoration
* Use larger Whisper models for higher accuracy

---

## Authors

Developed as a multilingual speech recognition solution leveraging:

* OpenAI Whisper
* AI4Bharat Indic Conformer
* PyTorch
* Hugging Face Transformers
