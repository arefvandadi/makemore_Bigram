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

# Creating Training and Evaluation Data
n=len(book_digits)*9//10
Train_data = book_digits[:n]
Val_data = book_digits[n:]

# Define a function to grab random batches from either Training or Evaluation data
def get_batch(x):
  data = Train_data if x == 'train' else Val_data
  Batch_start = torch.randint(0, len(data)-context,(batch_size,))
  xb = torch.stack([data[Batch_start[i]:Batch_start[i]+context] for i in range(batch_size)])
  yb = torch.stack([data[Batch_start[i]+1:Batch_start[i]+context+1] for i in range(batch_size)])
  return xb, yb



