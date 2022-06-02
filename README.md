# SAMIL
Official implementation of Paper：SHUFFLE ATTENTION MULTIPLE INSTANCES LEARNING FOR BREAST CANCER WHOLE SLIDE IMAGE CLASSIFICATION (submitted to ICIP 2022)

![image](https://github.com/CunqiaoHou/SAMIL/blob/main/img/SAMIL.jpg)

# Introduction
This paper presents a novel two-stage shuffle attention MIL (SAMIL) model for breast cancer WSI classification. SAMIL first introduces shuffle attention to extract important features from both spatial and channel dimensions, which well includes pixel-level pairwise relationships and channel dependencies, thus helping select more discriminant breast cancer instances for bag-level prediction. Additionally, it stacks multi-head attention with long short-term memory (LSTM) to construct an aggregator, and this adaptively high-lights the most distinctive instance features while exploring the correlation between selected breast cancer instances more
effectively. 

# Requirements:
```
Python: >=3.6   
PyTorch: >=1.6
OpenSlide: 3.4.1
```

# How to run the code?
```
To train a model, use script MIL_train.py.
To run a model on a test set, use script MIL_test.py.

To train the LSTM aggregator model, use script LSTM_train.py.
To run a model on a test set, use script LSTM_test.py.
```
