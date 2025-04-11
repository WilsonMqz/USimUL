# USimUL

This repository is the official implementation of the paper "Learning from Uncertain Similarity and Unlabeled Data" and technical details of this approach can be found in the paper.


## Requirements:
- Python 3.6.13
- numpy 1.19.2
- Pytorch 1.7.1
- torchvision 0.8.2
- pandas 1.1.5
- scipy 1.5.4


## Arguments:
- mo: model
- ds: data set
- uci: uci dataset or not
- lr: learning rate
- wd: weight decay
- gpu: the gpu index
- ep: training epoch number
- bs: training batch size
- me: method name
- prior: class prior probability
- n: number of unlabeled data pairs
- run_times: random running times

## Demo:
```
python main.py -mo mlp -ds mnist -uci 0 -lr 1e-4 -wd 1 -gpu 0 -ep 100 -seed 1 -bs 256 -me PrivacySimilarity -prior 0.4 -n 12000 -run_times 5
```
