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

############# Creating a Transformer Block as Seen in Attention is All You Need paper for a Bigram model ###############

# A Class definition for Each Self-Attetion Block
class Head(nn.Module):

  def __init__(self):
    super().__init__()
    self.key = nn.Linear(n_embed,head_size, bias=False)
    self.query = nn.Linear(n_embed,head_size, bias=False)
    self.value = nn.Linear(n_embed,head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(context, context)))
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    B,T,C = x.shape
    k = self.key(x) # Batch (B) X (T)_Entry X head_size (C) --- since This will be applied to the output of token_emb + pos_emb which is Batch (B) X (T)_Entry X n_embed (C)
    q = self.query(x) # Batch (B) X (T)_Entry X head_size (C) --- since This will be applied to the output of token_emb + pos_emb which is Batch (B) X (T)_Entry X n_embed (C)

    qk = q @ k.transpose(-2,-1) * C**0.5 ## (B ,T ,C) X (B, C, T) = (B, T, T)
    wei = torch.ones(T,T)
    wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
    wei = F.softmax(wei, dim=-1) # (B, T, T)
    wei = self.dropout(wei) # dropout some of the information from previous nodes (I don't know why!!)
    # perform the weighted aggregation of the values
    v = self.value(x) # Batch (B) X (T)_Entry X head_size (C) --- since This will be applied to the output of token_emb + pos_emb which is Batch (B) X (T)_Entry X n_embed (C)
    out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
    return out # each head takes n_embed and outputs head_size (C) --> Batch (B) X (T)_Entry X head_size (C)

# A Class definition to Bring Multiple Attention blocks together and create a Multihead Attention Block
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed,n_embed) ## Projection Layer back to the residual path: To me it only helps to match the dimensions to the residual path if different
        self.dropout = nn.Dropout(dropout) # introducing dropout rigt before it is added back to the residual path

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) ## Projection Layer back to the residual path: To me it only helps to match the dimensions to the residual path if different
        return out # outputs n_embed ---> by concatenating all the head_size self-attention Heads outputs

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed), # In Attention is All You Need paper the hidden layers of the Feed Forward are 4 times the input and output (Page 5 of paper)
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed), ## Projection Layer back to the residual path: To me it only helps to match the dimensions to the residual path if different
            nn.Dropout(dropout) # introducing dropout rigt before it is added back to the residual path
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication block (Multi-Head Self Attention) followed by computation blcok (FeedForward) """

    def __init__(self):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedFoward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) ## the addition of x in this argument is Resudial Connection. ## Layer norm is applied before Self Attention
        x = x + self.ffwd(self.ln2(x)) ## the addition of x in this argument is Resudial Connection ## Layer norm is applied before Feed Forward
        return x
#
#
#
############## Creating a Bigram Model (It is named Ngram here though but it is Bigram !!!!!) ###################

class NgramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.channels = nn.Embedding(charsize,n_embed)
    self.pos_embedding = nn.Embedding(context,n_embed) # positional embedding which will be the same for all entries in the batch i.e. for each context
    #self.SelfAttnHead = Head()
    self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
    self.ln_final = nn.LayerNorm(n_embed) # Another Layer norm right before the last nn.Linear layer
    #self.SelfAttnHead = MultiHeadAttention() # takes n_embd and outputs n_embd
    #self.ffw = FeedFoward() # takes n_embd and outputs n_embd --> Allows the NN to think about the communication between tokens coming from Self-Sttention before generating Logits
    self.lm_head = nn.Linear(n_embed,charsize)

  def forward(self,random_training_batch):
    # random_training_batch argument is generated by getbatch function (what we are calling xb in this code)
    B,T = random_training_batch.shape
    # looks up in the Embedding table created in the constructor to assign weights to each character coming in
    token_emb = self.channels(random_training_batch) # Batch (B) X (T)_Entry X n_embed (C)
    pos_emb = self.pos_embedding(torch.arange(T)) # (T)_Entry X n_embed (C)
    x = token_emb + pos_emb # pos_emb is getting broadcasted across all Batches or first dimension ---> Batch (B) X (T)_Entry X n_embed (C)
    x = self.blocks(x) # applies blocks of Multi-Head self-attention and feedforward
    x = self.ln_final(x)
    logits = self.lm_head(x) # Batch (B) X (T)_Entry X charsize (C=65)
    return logits

  def LossFunction(self,logits,random_training_batch_nextChar):
    logits = logits.view(-1,charsize) # we are doing this since Pytorch functinal.cross_entropy function needs Channels to be assigned to the second dimension
    Target = random_training_batch_nextChar.view(-1)
    Loss = FF.cross_entropy(logits,Target)
    return Loss

  def generate(self, initiator_token, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        #crop the input to the generator so it is not larger than context size since positional embedding defined above only accpepts values up to context (T)
        initiator_token_cropped = initiator_token[:,-context:]
        # get the predictions
        logits = self.forward(initiator_token_cropped)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        initiator_token_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        initiator_token = torch.cat((initiator_token, initiator_token_next), dim=1) # (B, T+1)
    return initiator_token

############## Creating a Bigram Model (It is named Ngram here though but it is Bigram !!!!!) ###################



