# Tibetan PLMs Fine-tuning

## Introduction

This repo is the fine-tuning part in the paper below.

***[Human-in-the-Loop Generation of Adversarial Texts: A Case Study on Tibetan Script (Cao et al., IJCNLP-AACL 2025 Demo)](https://aclanthology.org/2025.ijcnlp-demo.2/)***

***[TSCheater: Generating High-Quality Tibetan Adversarial Texts via Visual Similarity (Cao et al., ICASSP 2025)](https://ieeexplore.ieee.org/document/10889732)***

***[Multi-Granularity Tibetan Textual Adversarial Attack Method Based on Masked Language Model (Cao et al., WWW 2024 Workshop - SocialNLP)](https://dl.acm.org/doi/10.1145/3589335.3652503)***

⬆️ commit id: HEAD

***[Pay Attention to the Robustness of Chinese Minority Language Models! Syllable-level Textual Adversarial Attack on Tibetan Script (Cao et al., ACL 2023 Workshop - TrustNLP)](https://aclanthology.org/2023.trustnlp-1.4)***

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
@inproceedings{cao-etal-2025-human,
    title = "Human-in-the-Loop Generation of Adversarial Texts: A Case Study on {T}ibetan Script",
    author = "Cao, Xi  and
      Sun, Yuan  and
      Li, Jiajun  and
      Gesang, Quzong  and
      Qun, Nuo  and
      Tashi, Nyima",
    editor = "Liu, Xuebo  and
      Purwarianti, Ayu",
    booktitle = "Proceedings of The 14th International Joint Conference on Natural Language Processing and The 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics: System Demonstrations",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.ijcnlp-demo.2/",
    pages = "9--16",
    ISBN = "979-8-89176-301-2"
}
```

```
@INPROCEEDINGS{10889732,
  author={Cao, Xi and Gesang, Quzong and Sun, Yuan and Qun, Nuo and Nyima, Tashi},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={TSCheater: Generating High-Quality Tibetan Adversarial Texts via Visual Similarity}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10889732}}
```

```
@inproceedings{10.1145/3589335.3652503,
    author = {Cao, Xi and Qun, Nuo and Gesang, Quzong and Zhu, Yulei and Nyima, Trashi},
    title = {Multi-Granularity Tibetan Textual Adversarial Attack Method Based on Masked Language Model},
    year = {2024},
    isbn = {9798400701726},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3589335.3652503},
    doi = {10.1145/3589335.3652503},
    booktitle = {Companion Proceedings of the ACM on Web Conference 2024},
    pages = {1672–1680},
    numpages = {9},
    location = {Singapore, Singapore},
    series = {WWW '24}
}
```

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
