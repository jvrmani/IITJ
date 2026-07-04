# Sanskrit → English NMT — Custom BiLSTM + Bahdanau Attention Seq2Seq (Unsubmitted Variant)

**Assignment 2 — Natural Language Understanding**

This repository contains an earlier, **from-scratch** sequence-to-sequence model built for the Sanskrit-to-English translation assignment. It is kept here for reference/comparison only — **this was not the final submitted model**, since its BLEU/BERTScore came out lower than the fine-tuned pre-trained model used for the actual submission (see that repo for the submitted version).

## Why this wasn't submitted

| Metric | This model (custom BiLSTM+Attention) | Submitted model (fine-tuned IndicTrans2) |
|---|---|---|
| Parameters | 48.8M (trained from scratch) | 211.8M (fine-tuned, pre-trained) |
| BLEU | 0.1388 (HF `evaluate` library) | 0.2073 (NLTK, per assignment spec) |
| BERTScore F1 (rescaled) | 0.3436 | 0.5613 |
| Inference time (test set) | 72.98 sec | 111.85 sec |

Training a sequence-to-sequence model entirely from scratch on this assignment's dataset size struggles against Sanskrit's rich morphology (compounding/sandhi, free word order) — validation loss plateaued around epoch 10–13 (see Training Dynamics below) while train loss kept falling, and BLEU/BERTScore never reached a competitive level. **Note on the BLEU number above:** this run used Hugging Face's `evaluate.load("bleu")`, not the NLTK BLEU the assignment rubric specifies — the two use different tokenization/normalization and are not directly comparable to the submitted model's NLTK BLEU figure. This is flagged here rather than presented as an apples-to-apples number.

## Architecture

- **Encoder:** 3-layer stacked bidirectional LSTM (`ParaDeepFeatureEncoder`), 512-dim embeddings, 512-dim hidden state per direction, with a linear projection collapsing each layer's forward/backward final states down to the decoder's hidden size.
- **Attention:** Bahdanau-style additive attention (`AdvancedBahdanauAlignment`) — encoder and decoder states are projected into a shared space, combined additively, passed through `tanh`, and scored to produce a softmax attention distribution over encoder positions, masked to ignore padding.
- **Decoder:** 3-layer LSTM (`ParaDeepFeatureDecoder`) that consumes the previous target token embedding concatenated with the attention context vector at each step, with a fully-connected output layer combining the LSTM output, context vector, and token embedding to predict the next token.
- **Decoding:** custom beam search (`execute_safe_beam_decoding`, beam width 5) with a length cap of 50 tokens.
- **Vocabulary:** word-level tokenization (whitespace split) for both Sanskrit and English, with a minimum token-frequency floor of 2 to exclude rare/noisy tokens.

## Training setup

| Hyperparameter | Value |
|---|---|
| Embedding size | 512 |
| Hidden size | 512 |
| Encoder/decoder depth | 3 layers |
| Dropout | 0.4 |
| Batch size | 32 |
| Optimizer | Adam, LR 3e-4, weight decay 1e-4 |
| LR schedule | `ReduceLROnPlateau` (factor 0.5, patience 2, on validation loss) |
| Teacher forcing ratio | 0.5 |
| Gradient clipping | max norm 1.0 |
| Max epochs / early-stop patience | 30 / 6 (on validation loss) |

Training ran the full 30 epochs before the early-stopping window triggered. Train loss decreased steadily throughout (6.15 → 3.40), but validation loss plateaued from roughly epoch 10 onward (~5.30–5.37) with no further meaningful improvement — a sign that the model had extracted what it could from the available training data and further epochs mainly increased the train/validation loss gap (overfitting) rather than improving generalization.

## Data

Only the dataset provided with the assignment was used: `train_sa_10000.csv`/`train_en_10000.csv`, `dev_sa_1000.csv`/`dev_en_1000.csv`, `test_sa_1000.csv`/`test_en_1000.csv`, merged on `Source_id` with missing/duplicate rows dropped. No external data or APIs were used — this is a fully from-scratch model with no pre-trained components.

## Repository contents

| File | Description |
|---|---|
| `NLU_Assignment_2_Option_5_seq2seq_Model_with_90M_Parameters.ipynb` | Full notebook: data loading/vocabulary construction, model definition, training loop, beam-search decoding, and evaluation (BLEU via `evaluate`, BERTScore via `bert-score`). |
| `submission.csv` | Test-set predictions from this model (not the one actually submitted for grading). |
| `loss_history.png` | Train/validation loss curve. |

## How to run

1. Open in Google Colab (GPU runtime), with `train/dev/test_{sa,en}_*.csv` available in the configured `STORAGE_PATH` on Google Drive.
2. Run all cells top to bottom — data loading/vocab building, training (up to 30 epochs with early stopping), then beam-search inference and evaluation on the test set.

## Disclosures

- No pre-trained models or external APIs were used anywhere in this notebook — encoder, decoder, attention, and vocabulary are all trained/built from scratch on the provided data only.
- Kept in this repository for transparency/comparison against the submitted, fine-tuned model, not as the graded submission.
