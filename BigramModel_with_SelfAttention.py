import torch
import torch.nn as nn
import torch.nn.functional as F

# Let's download and read in the tiny shakespeare dataset
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as tinyshkspr:
    book = tinyshkspr.read()


