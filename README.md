# Fine-Tuning XLM-RoBERTa for Persian Question Answering (PQuAD)

## Purpose

This project fine-tunes the `xlm-roberta-base` model for the task of question answering in Persian using the PQuAD dataset.

## Overview of Features

- **Data Preprocessing**: The notebook preprocesses the PQuAD dataset for compatibility with the Hugging Face `Transformers` library.
- **Model Fine-Tuning**: The `xlm-roberta-base` model is fine-tuned on the PQuAD dataset for effective Persian language question answering.
- **Evaluation**: The model's performance is evaluated using standard metrics like F1 Score and Exact Match (EM).

## Original Work

This project is based on the [Hugging Face Question Answering Notebook](https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb), which demonstrates how to fine-tune models for the SQuAD dataset.

## Modifications

- **Dataset**: Replaced the SQuAD dataset with the PQuAD dataset, which is a Persian version of the SQuAD dataset tailored for question answering in the Persian language.
- **Model**: Used `xlm-roberta-base` to fine-tune the model for Persian QA tasks.

## PQuAD Dataset

PQuAD is a Persian Question Answering dataset that mirrors the structure and format of the SQuAD dataset, adapted for the Persian language.


## Model Performance Metrics

The fine-tuned model was evaluated on the PQuAD validation dataset. Below are the key performance metrics:

| Metric            | Score  |
|-------------------|--------|
| Exact Match (Overall)    | 71.98% |
| F1 Score (Overall)       | 85.56% |
| Total Questions          | 7,976  |
| Has Answer - Exact Match | 65.49% |
| Has Answer - F1 Score    | 83.55% |
| Total 'Has Answer'       | 5,995  |
| No Answer - Exact Match  | 91.62% |
| No Answer - F1 Score     | 91.62% |
| Total 'No Answer'        | 1,981  |
| Best Exact Match         | 71.98% |
| Best F1 Score            | 85.56% |
| Best Threshold (Exact)   | 0.0    |
| Best Threshold (F1)      | 0.0    |


## Model Link

You can access and use the fine-tuned model on Hugging Face here: [kianshokraneh/xlm-roberta-base-finetuned-pquad](https://huggingface.co/kianshokraneh/xlm-roberta-base-finetuned-pquad).
