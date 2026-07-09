# SenFlow: Inter-Sentence Flow Modeling for AI-Generated Text Detection in Hybrid Documents

This repository contains the official codebase and a license-aware sample of the MOSAIC benchmark for the paper "SenFlow: Inter-Sentence Flow Modeling for AI-Generated Text Detection in Hybrid Documents".

## 1. Environment Setup

```bash
conda create -n senflow python=3.10
conda activate senflow
pip install -r requirements.txt
```

## 2. Pipeline Usage

The SenFlow pipeline consists of three main stages: dataset generation, feature extraction, and model training/evaluation.

### Stage 1: Hybrid Document Generation (MOSAIC)

To generate the hybrid documents using the multi-block masking strategy:

```
# Example: Generate PubMed documents using DeepSeek
python generate_mosaic.py \
    --domain pubmed \
    --generator deepseek \
    --target 4000
```

*(Note: Requires `DEEPSEEK_API_KEY` or `MOONSHOT_API_KEY` environment variables).*

### Stage 2: Feature Extraction

After reconstructing the full MOSAIC files locally from the upstream datasets, extract token probabilities, entropies, and proxy-model hidden states using the LoRA-aligned Llama-3.1-8B-Instruct:

```
python prepare_features.py \
    --input data/pubmed_deepseek_4000.json \
    --output features/pubmed_deepseek_features.pt
```

### Stage 3: Training & Evaluation

To train and evaluate the SenFlow network with the Hybrid Adjacency GCN and CRF decoder:

```
# Unified Protocol Evaluation
python run_senflow_pipeline.py \
    --train features/pubmed_deepseek_features.pt \
    --use_crf \
    --config X \
    --save_dir checkpoints/
```

## Data Availability

For double-blind review purposes, this repository includes a small license-aware format sample at `MOSAIC/MOSAIC_sample_license_aware.json`. The sample contains metadata, document references, sentence labels/spans, and AI-generated replacement sentences, but does not redistribute full source documents from the upstream PubMed/XSum corpora.

The full MOSAIC benchmark is intended to be reconstructed locally from the upstream datasets using the released generation scripts, prompts, seeds, and configuration, subject to the upstream datasets' license and redistribution terms. See `DATA_RELEASE.md` for the release policy.
