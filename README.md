# NLP Question Answering Model Comparison

## Overview

This project explores **question answering (QA)** using transformer-based language models. It compares fine-tuned **DistilBERT** and **T5** models with **ChatGPT** and **LLaMA** on questions from the **SQuAD v1.1** dataset.

The project covers dataset preparation, transformer fine-tuning, checkpoint evaluation, answer generation, quantitative model comparison, and visualization of the results.

This project was originally developed as part of the **Natural Language Processing** module at Ostbayerische Technische Hochschule Amberg-Weiden (OTH Amberg-Weiden).

## Models

Four different approaches to question answering are evaluated:

- **DistilBERT** – fine-tuned for extractive question answering
- **T5** – fine-tuned for generative question answering
- **ChatGPT** – evaluated using generated answers to the same questions
- **LLaMA** – evaluated using generated answers to the same questions

This allows both traditional fine-tuned transformer models and large language models to be compared on the same QA task.

## Dataset

The project uses **SQuAD v1.1 (Stanford Question Answering Dataset)**.

`prepare_datasets.py` downloads and processes the official SQuAD data and creates local datasets for training and evaluation.

A fixed subset of **100 questions** is also generated for comparing the different models under the same conditions.

## Training

The project supports fine-tuning two transformer architectures:

### DistilBERT

DistilBERT is trained as an **extractive QA model**. Given a question and context, the model predicts the start and end positions of the answer within the context.

### T5

T5 is trained as a **generative QA model**. The question and context are provided as input and the model generates the answer as text.

Model checkpoints are saved after each training epoch so that different stages of training can be evaluated and compared.

## Model Comparison

The project includes scripts for:

- Comparing DistilBERT checkpoints
- Comparing T5 checkpoints
- Collecting ChatGPT answers
- Collecting LLaMA answers
- Comparing all four models
- Exporting prompts
- Visualizing evaluation results

The models are evaluated using question-answering metrics including **Exact Match (EM)** and **F1 score**.

## Technologies

The project was developed in Python using:

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- DistilBERT
- T5
- LLaMA
- ChatGPT
- Jupyter Notebook

## Project Structure

```text
├── prepare_datasets.py
├── train_model.py
├── compare_checkpoints_distilbert.py
├── compare_checkpoints_t5.py
├── compare_models.py
├── collect_chatgpt_answers.py
├── collect_llama_answers.py
├── export_prompts.py
├── requirements.txt
├── visualizations.ipynb
├── visualizations.pdf
├── Bericht/
│   └── Bericht.pdf
└── results/
    ├── prompts.txt
    ├── tdb_chatgpt_outputs.json
    ├── tdb_llama_outputs.json
    └── tdb_compare_4models.json
```

## Installation

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

The main dependencies are PyTorch, Transformers, Datasets, Accelerate, tqdm, and TensorFlow/Keras compatibility packages.

## Preparing the Dataset

Run:

```bash
python prepare_datasets.py
```

This downloads SQuAD v1.1 and creates the local training and evaluation datasets.

## Training

The training script supports both DistilBERT and T5.

For example:

```bash
python train_model.py --model distilbert --epochs 5 --out_dir checkpoints/distilbert_squad_ft
```

or:

```bash
python train_model.py --model t5 --epochs 5 --out_dir checkpoints/t5_squad_ft
```

Training checkpoints are saved after each epoch.

## Results and Visualization

Model outputs and comparison results are stored in the `results/` directory.

The analysis and visualizations can be viewed in:

```text
visualizations.ipynb
```

A PDF export is also included for convenient viewing:

```text
visualizations.pdf
```

The complete project report can be found in:

```text
Bericht/Bericht.pdf
```

## About

This project demonstrates practical NLP and transformer-based question answering, including dataset preparation, fine-tuning pretrained language models, extractive and generative QA, checkpoint selection, LLM evaluation, quantitative model comparison, and result visualization.
