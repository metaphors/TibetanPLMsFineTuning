# Tibetan PLMs Fine-tuning

## Introduction

This repo is the fine-tuning part in the paper below.

***Multi-Granularity Tibetan Textual Adversarial Attack Method Based on Masked Language Model (Cao et al., WWW 2024 Workshop - SocialNLP)***

⬆️ commit id: HEAD

***[Pay Attention to the Robustness of Chinese Minority Language Models! Syllable-level Textual Adversarial Attack on Tibetan Script](https://aclanthology.org/2023.trustnlp-1.4) (Cao et al., ACL 2023 Workshop - TrustNLP)***

⬆️ commit id: fc2041350c8c7e51bbf74536579eb72a0a1e7bd5

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

|    dataset    | train | validation | test |
|:-------------:|:-----:|:----------:|:----:|
| TNCC-document | 7364  |    920     | 920  |
|  TNCC-title   | 7422  |    927     | 927  |
|     TUSA      | 8000  |    1000    | 1000 |

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
    - CINO-small-v2
- Tibetan-BERT ([Research and Application of Tibetan Pre-training Language Model Based on BERT (Zhang et al., ICCIR 2022)](https://dl.acm.org/doi/10.1145/3548608.3559255))

CINO-large-v2 related Hugging Face repo：[https://huggingface.co/hfl/cino-large-v2](https://huggingface.co/hfl/cino-large-v2)

CINO-base-v2 related Hugging Face repo：[https://huggingface.co/hfl/cino-base-v2](https://huggingface.co/hfl/cino-base-v2)

CINO-small-v2 related Hugging Face repo：[https://huggingface.co/hfl/cino-small-v2](https://huggingface.co/hfl/cino-small-v2)

Tibetan-BERT related Hugging Face repo：[https://huggingface.co/UTibetNLP/tibetan_bert](https://huggingface.co/UTibetNLP/tibetan_bert)

## Fine-tuning Scripts

dir：fine-tuning

## Fine-tuned LMs

dir：saved_model

You can also find all the fine-tuned LMs in our paper on Hugging Face.

|                LM                 |                                                                    URL                                                                    |
|:---------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------:|
|  cino-small-v2_TNCC-title_tsheg   |    [https://huggingface.co/UTibetNLP/cino-small-v2_tncc-title_tsheg](https://huggingface.co/UTibetNLP/cino-small-v2_tncc-title_tsheg)     |
|        cino-small-v2_TUSA         |                [https://huggingface.co/UTibetNLP/cino-small-v2_tusa](https://huggingface.co/UTibetNLP/cino-small-v2_tusa)                 |
|   cino-base-v2_TNCC-title_tsheg   |     [https://huggingface.co/UTibetNLP/cino-base-v2_TNCC-title_tsheg](https://huggingface.co/UTibetNLP/cino-base-v2_TNCC-title_tsheg)      |
| cino-base-v2_TNCC-document_tsheg  |  [https://huggingface.co/UTibetNLP/cino-base-v2_TNCC-document_tsheg](https://huggingface.co/UTibetNLP/cino-base-v2_TNCC-document_tsheg)   |
|         cino-base-v2_TUSA         |                 [https://huggingface.co/UTibetNLP/cino-base-v2_TUSA](https://huggingface.co/UTibetNLP/cino-base-v2_TUSA)                  |
|  cino-large-v2_TNCC-title_tsheg   |    [https://huggingface.co/UTibetNLP/cino-large-v2_TNCC-title_tsheg](https://huggingface.co/UTibetNLP/cino-large-v2_TNCC-title_tsheg)     |
| cino-large-v2_TNCC-document_tsheg | [https://huggingface.co/UTibetNLP/cino-large-v2_TNCC-document_tsheg](https://huggingface.co/UTibetNLP/cino-large-v2_TNCC-document_tsheg)  |
|        cino-large-v2_TUSA         |                [https://huggingface.co/UTibetNLP/cino-large-v2_TUSA](https://huggingface.co/UTibetNLP/cino-large-v2_TUSA)                 |
|   Tibetan-BERT_TNCC-title_tsheg   |     [https://huggingface.co/UTibetNLP/tibetan-bert_tncc-title_tsheg](https://huggingface.co/UTibetNLP/tibetan-bert_tncc-title_tsheg)      |
|          Tibetan-BERT_TUSA        |                 [https://huggingface.co/UTibetNLP/tibetan-bert_tusa](https://huggingface.co/UTibetNLP/tibetan-bert_tusa)                  |

## Citation

If you think our work useful, please kindly cite our paper.

```
@inproceedings{cao-etal-2023-pay-attention,
    title = "Pay Attention to the Robustness of {C}hinese Minority Language Models! Syllable-level Textual Adversarial Attack on {T}ibetan Script",
    author = "Cao, Xi  and
      Dawa, Dolma  and
      Qun, Nuo  and
      Nyima, Trashi",
    booktitle = "Proceedings of the 3rd Workshop on Trustworthy Natural Language Processing (TrustNLP 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.trustnlp-1.4",
    pages = "35--46"
}
```
