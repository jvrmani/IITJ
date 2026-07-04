# Sanskrit → English Neural Machine Translation
**Assignment 2 — Natural Language Understanding**

Fine-tuning [AI4Bharat IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) (Indic→English, distilled, ~211.8M parameters) to translate Sanskrit sentences into English, evaluated with BLEU, BERTScore, and inference efficiency.

## Overview

Sanskrit is a low-resource, morphologically rich language (extensive compounding/sandhi, free word order), which makes training a sequence-to-sequence model from scratch difficult on a dataset of this size. This project **fully fine-tunes** the pre-trained multilingual checkpoint `ai4bharat/indictrans2-indic-en-dist-200M` — which already covers Sanskrit (`san_Deva`) among its 22 supported scheduled Indian languages — on the assignment's provided parallel Sanskrit–English data. Use of this pre-trained checkpoint was confirmed as acceptable by the course TA and is disclosed throughout the report.

## Results

| Metric | From-scratch BiLSTM + Attention (48.8M params) | Fine-tuned IndicTrans2 (211.8M params) |
|---|---|---|
| BLEU (NLTK, default weights) | 0.1009 | **0.2073** |
| BERTScore F1 (rescaled) | 0.2049 | **0.5613** |
| Inference time (1,000-sentence test set) | 149.80 sec | **111.85 sec** |
| Total / trainable parameters | 48.8M | 211.8M (100% trainable) |

## Repository contents

| File | Description |
|---|---|
| `NLU_Assignment2_IndicTrans2_Modular.ipynb` | Main notebook: 10 independent modules (setup, data loading, model loading, dataset/dataloader, training, curves, inference, metrics, submission, report examples). Each module auto-recovers its inputs from disk, so it can be re-run standalone without repeating earlier steps. |
| `submission.csv` | Required test-set predictions (`Source_id`, `Sentence_en`). |
| `NLU_Assignment2_Report.pdf` / `.docx` | Assignment report: architecture, training procedure, results, translation examples with error analysis, experiments, discussion, references. |

## Method summary

- **Model:** `ai4bharat/indictrans2-indic-en-dist-200M` — an NLLB-style encoder–decoder Transformer, used with its own tokenizer and `IndicProcessor` (script normalization + number/entity placeholder masking) on the Sanskrit source side.
- **Fine-tuning:** full fine-tune (every parameter trainable, no LoRA/adapters/frozen layers). AdamW, LR 1.5e-5, linear warmup + decay, label smoothing 0.1, gradient accumulation (effective batch size 32), FP16 mixed precision, gradient clipping.
- **Model selection:** early stopping on **dev-set BLEU** (patience 2), not training loss — loss kept falling after epoch 3 while dev BLEU peaked at epoch 3 and regressed, so the epoch-3 checkpoint was restored.
- **Decoding:** beam search (`num_beams=5`) at inference.
- **Evaluation:** NLTK corpus BLEU (default weights) and BERTScore F1 with `rescale_with_baseline=True`, plus wall-clock inference time and parameter counts, matching the assignment's grading rubric.

## Data

Only the dataset provided with the assignment was used (`train/dev/test_sa.csv` + matching `_en.csv`, merged on `Source_id`; duplicates/missing rows dropped). No external parallel corpora or translation APIs were used at any point — the only external dependency is a one-time download of the pre-trained model weights from the Hugging Face Hub.

## How to run

1. Open the notebook in Google Colab (GPU runtime).
2. Upload all Dataset files in Google Drive in folder "MyDrive/Sanskrit_NMT_Project"
3. Set an `HF_TOKEN` Colab secret (Hugging Face account with the model's usage terms accepted) before running Module 2, since the checkpoint is a gated repository.
4. Run modules in order (0 → 9) for a full pipeline run, or jump to any later module directly if its prerequisite artifacts already exist on Drive (each module lists its dependencies in its header).


## Disclosures

- **Pre-trained model used:** `ai4bharat/indictrans2-indic-en-dist-200M` (AI4Bharat, MIT license), fully fine-tuned on the assignment's training data only. Confirmed acceptable by the course TA.
- **No external APIs:** all translation and evaluation runs locally after the one-time model-weights download.
- Full method details, hyperparameters, and limitations are in the report.
