import torch
import torch.nn as nn
import torch.nn.functional as F

# Let's download and read in the tiny shakespeare dataset
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as tinyshkspr:
    book = tinyshkspr.read()

r = sorted(set(book))
chars = ''.join(r)
# Number of unique characters in the model (charsize = 65)
charsize = len(chars)

## I used rfind() function to create my encoder and deccoder instead of
## using dictionary which was used in the Original code
encoder = lambda c: [chars.rfind(c[i]) for i in range(len(c))]
decoder = lambda c: "".join([chars[i] for i in c])

# Encode the Book (Type = Python List)
book_digits_List=encoder(book)
# Convert Encoded Book from python List to PyTorch Tensor
book_digits = torch.tensor(book_digits_List)

