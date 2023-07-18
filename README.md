# Tibetan PLMs Fine-tuning

## Introduction

This repo is the fine-tuning part in the paper below.

***[Pay Attention to the Robustness of Chinese Minority Language Models! Syllable-level Textual Adversarial Attack on Tibetan Script](https://trustnlpworkshop.github.io/papers/6.pdf) (Cao et al., ACL 2023 Workshop - TrustNLP)***

## Environment

Python 3.9.14

torch==1.12.1

transformers==4.22.2

datasets==2.5.1

evaluate==0.2.2

sentencepiece==0.1.97

sklearn==0.0

You can install the above packages with the command below.

```
python install -r requirements.txt
```

## Datasets

dir：dataset

- TNCC ([End-to-End Neural Text Classification for Tibetan (Qun et al., CCL 2017)](http://www.cips-cl.org/static/anthology/CCL-2017/CCL-17-104.pdf))
    - TNCC-document
    - TNCC-title
- TUSA ([Sentiment Analysis of Tibetan Short Texts Based on Graphical Neural Networks and Pre-training Models (Zhu et al., Journal of Chinese Information Processing 2023)](http://jcip.cipsc.org.cn/CN/Y2023/V37/I2/71))

TNCC related GitHub repo：[https://github.com/FudanNLP/Tibetan-Classification](https://github.com/FudanNLP/Tibetan-Classification)

TUSA related GitHub repo：[https://github.com/UTibetNLP/TU_SA](https://github.com/UTibetNLP/TU_SA)

Each dataset is split into a training set, a validation set, and a test set according to a ratio of 8:1:1.

|     dataset     |   train    | validation |    test    |
|:---------------:|:----------:|:----------:|:----------:|
|  TNCC-document  |    7364    |    920     |    920     |
|   TNCC-title    |    7422    |    927     |    927     |
|      TUSA       |    8000    |    1000    |    1000    |

## Dataset Loader Scripts

dir：dataset_loader

tncc-document.py, tncc-title.py, tusa.py is the loader script of TNCC-document, TNCC-title, TUSA.

tncc-document_v2.py and tncc-title_v2.py remove the space between two syllables in TNCC.

tncc-document_v3.py and tncc-title_v3.py replace the space between two syllables in TNCC with the tsheg.

## PLMs

dir：model

- CINO ([CINO: A Chinese Minority Pre-trained Language Model (Yang et al., COLING 2022)](https://aclanthology.org/2022.coling-1.346.pdf))
    - CINO-large-v2
    - CINO-base-v2

CINO-large-v2 related Hugging Face repo：[https://huggingface.co/hfl/cino-large-v2](https://huggingface.co/hfl/cino-large-v2)

CINO-base-v2 related Hugging Face repo：[https://huggingface.co/hfl/cino-base-v2](https://huggingface.co/hfl/cino-base-v2)

## Fine-tuning Scripts

dir：fine-tuning

## Fine-tuned LMs

dir：saved_model

You can also find all the fine-tuned LMs in our paper on Hugging Face.

|                LM                 |                                                                    URL                                                                    |
|:---------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------:|
|   cino-base-v2_TNCC-title_tsheg   |     [https://huggingface.co/UTibetNLP/cino-base-v2_TNCC-title_tsheg](https://huggingface.co/UTibetNLP/cino-base-v2_TNCC-title_tsheg)      |
| cino-base-v2_TNCC-document_tsheg  |  [https://huggingface.co/UTibetNLP/cino-base-v2_TNCC-document_tsheg](https://huggingface.co/UTibetNLP/cino-base-v2_TNCC-document_tsheg)   |
|         cino-base-v2_TUSA         |                 [https://huggingface.co/UTibetNLP/cino-base-v2_TUSA](https://huggingface.co/UTibetNLP/cino-base-v2_TUSA)                  |
|  cino-large-v2_TNCC-title_tsheg   |    [https://huggingface.co/UTibetNLP/cino-large-v2_TNCC-title_tsheg](https://huggingface.co/UTibetNLP/cino-large-v2_TNCC-title_tsheg)     |
| cino-large-v2_TNCC-document_tsheg | [https://huggingface.co/UTibetNLP/cino-large-v2_TNCC-document_tsheg](https://huggingface.co/UTibetNLP/cino-large-v2_TNCC-document_tsheg)  |
|        cino-large-v2_TUSA         |                [https://huggingface.co/UTibetNLP/cino-large-v2_TUSA](https://huggingface.co/UTibetNLP/cino-large-v2_TUSA)                 |