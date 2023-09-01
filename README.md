# KorAttack

## Setups
[![Python](https://img.shields.io/badge/python-3.7.10-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-385/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.5.0-red?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)


## Datasets
The dataset name must be specified in the "--data_path" argument
- [nsmc](https://github.com/e9t/nsmc)
- [korean-hate-speech](https://github.com/kocohub/korean-hate-speech)

## Train and Test
```
# 1. fine-tune KoBERT
python train.py

# 2. attack fine-tuned KoBERT
python attack_importance.py
```
## Citation
Please cite our paper if you use our code:
```
Hyojun Kim, Jee-Hyong Lee.(2023).
KorAttack : Hueristic text adversarial attack on Korean dataset
Proceedings of KIIS Spring Conference 2023 Vol. 33, No. 1. 176-178
```