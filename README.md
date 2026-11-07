# Beyond Lexical Shortcuts: A Statistical and Robustness Assessment of Transformer Encoders for Depression Detection in Social Media

**Amir Seyed Danesh** (corresponding author, seyeddanesh@guilan.ac.ir) and **Amin Rezanejad** (aminrezanejad@msc.guilan.ac.ir)
University of Guilan, Department of Computer Engineering

## Summary
We evaluate depression-related text classification on a public Reddit dataset, comparing TF-IDF models, frozen all-MiniLM-L6-v2 embeddings and a fully fine-tuned DistilBERT, using bootstrap CIs, paired tests, stratified cross-validation, keyword ablation and interpretability analyses. In-domain scores are very high (F1 up to 0.98), but they rely heavily on lexical cues and dataset artefacts, so they should not be read as evidence of clinical detection capability.

## Contents
- `Depression_Detection_Merged.ipynb` – full analysis with stored outputs
  - **Part A** – TF-IDF baselines, frozen MiniLM-L6 (ONNX Runtime) + LR/MLP, statistical tests, LIME/occlusion, keyword ablation (CPU run; needs a local `assets/` folder with `data.csv`, `minilm.onnx`, `minilm_tok.json`)
  - **Part B** – fine-tuned DistilBERT on original vs. ablated data (Colab T4; loads the data from Hugging Face)

Outputs are those of the original runs. Part A and Part B were run in different environments.

## Data
[`hugginglearners/reddit-depression-cleaned`](https://huggingface.co/datasets/hugginglearners/reddit-depression-cleaned). Labels are subreddit-based, not clinical. Seed 42, stratified 70/10/20 split (5,355 / 765 / 1,530 posts after de-duplication).

## Disclaimer
Research code only. Not for diagnosis or clinical use.
