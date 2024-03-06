import torch
import torch.nn as nn
import torch.nn.functional as F

################ Initial Hyperparameters ###########
## Hyperparameter ##
#number of embedding for each character --> i.e. C channels entering Self-Attention Head
n_embed = 32
# number of batches for each iteration
batch_size = 4
# How many characters are used to predict the next? Context = 1 --> Bigram model, Context > 1 --> Ngram model
context = 8
# number of Self-Attention heads
num_heads = 4
# size of Self-Attention head
head_size = n_embed//num_heads
# number of Transformer (Multi-head Self-Attention + Feed Forward) layers
n_layer = 3
# dropout
dropout = 0.2
# Max Number of Iteration to Train the Model
max_iters = 5000
# Loss Evaluation Interval
eval_interval = 100
# Number of batches to Mean over for Loss evaluation at every interval
eval_iters = 200
################ Initial Hyperparameters ###########

################ Initial Hyperparameters Multiplied by 4 ###########
# ## Hyperparameter ##
# #number of embedding for each character --> i.e. C channels entering Self-Attention Head
# n_embed = 128
# # number of batches for each iteration
# batch_size = 16
# # How many characters are used to predict the next? Context = 1 --> Bigram model, Context > 1 --> Ngram model
# context = 32
# # number of Self-Attention heads
# num_heads = 16
# # size of Self-Attention head
# head_size = n_embed//num_heads
# # number of Transformer (Multi-head Self-Attention + Feed Forward) layers
# n_layer = 12
# # dropout
# dropout = 0.2
# # Max Number of Iteration to Train the Model
# max_iters = 5000
# # Loss Evaluation Interval
# eval_interval = 100
# # Number of batches to Mean over for Loss evaluation at every interval
# eval_iters = 200
################ Initial Hyperparameters Multiplied by 2 ###########

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



