import os
import torch # PyTorch
import torch.nn as nn
from torch.nn import functional as F

# character-based lm transformer will look and predict which character comes next in sequence
# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# read it in
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print('length of dataset in characters: ', len(text))

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
alpha_size = len(chars)
print(''.join(chars))
print(alpha_size)

# tokenize input text (convert raw text as a string to some sequence of integers according to some vocabulary)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))

# let's now encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data[:1000]) # chars look like this to GPT

# Split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
# helps us understand model overfitting (trained memorization to the specific data too much)
train_data = data[:n]
val_data = data[n:]

# Now preprocess for plug in text to transformer for training, can't feed whole text, train on random sample batches of chunks of dataset
block_size = 8 # max length of chunk
print(train_data[:block_size+1]) # first 9 characters of training set, 8 individual examples
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size): # 8 examples
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")
# train on 8 examples with context between 1 and block_size so transformer network is used to seeing contexts from as little as one all the way up the sequence
# useful for inference while sampling, sampling generation can start with 1 character of context and predict up till block_size, then need truncating
# time dimension and batch dimension of tensors to feed
# while sampling chunks, have batches of chunks in a single tensor

torch.manual_seed(1337) # random samples, seed in random number generator
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
# 4 batches of chunks of 8 chars/tokens

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # list of 4 numbers randomly generated - random offsets into training set
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# 4x8 input and target tensors:
xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")
# prints 4 batches, each with 8 example inputs and the target

# input to transformer tensor
print(xb)

# simplest possible LM NN model:
class BigramLanguageModel(nn.Module):

    def __init__(self, alpha_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(alpha_size, alpha_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) 4x8x65 - logits are scores for next character in sequence
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # cross entropy loss is measuring quality of logits w/respect to targets

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(alpha_size)
logits, loss = m(xb, yb) # pass in inputs and targets
print(logits.shape)
print(loss)

# completely random:
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# TRAINING
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3) # takes the gradients and updates parameters using the gradients

batch_size = 32
for steps in range(100): # increase number of steps for good results... 
    
    # sample a new batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True) # zero out all the gradients from previous step (epoch)
    loss.backward() # getting/calculating gradients for all parameters
    optimizer.step() # using gradients to update/adjust params

print(loss.item())
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))

# Need tokens to start talking to each other to recognize context and make better predictions
# Self-attention math trick
# toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)
'''
Mat Mult with a tril:
a (wei)=
tensor([[1.0000, 0.0000, 0.0000], a 1st row dot product with b 1st column produces 2 in c, 2nd row dot first column produces 4 in c...
        [0.5000, 0.5000, 0.0000],
        [0.3333, 0.3333, 0.3333]])
--
b (x)=
tensor([[2., 7.],
        [6., 4.],
        [6., 5.]])
--
c=
tensor([[2.0000, 7.0000],
        [4.0000, 5.5000],
        [4.6667, 5.3333]]) bottom is average of 3 b rows
'''

# consider the following toy example:
torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels (info at each point in the sequence), up to 8 tokens need to communicate, should not with tokens after itself
# best way to communicate with past is to average preceding elements, 5th token takes channels from its step and averaged channels from past steps - feature vector summarizes its context
x = torch.randn(B,T,C)
x.shape

# We want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B,T,C)) # bag of words
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t,C)
        xbow[b,t] = torch.mean(xprev, 0) # store averages in bag of words

# version 2: using matrix multiply for a weighted aggregation
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B, T, T) @ (B, T, C) ----> (B, T, C)
torch.allclose(xbow, xbow2)

# version 3: use Softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T)) # weights=interaction strength/affinities
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x # aggregation
torch.allclose(xbow, xbow3)
# Can do weighted aggregation of past elements by using mat mult of a lower triangular fashion, and elements in the lower triangular part tell how much of each element fuses

# version 4: self-attention!
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# every single token will emit two vectors: query and key, query is what I'm looking for, key is what do I contain, way to get affinities between them is dot product (wei) between the keys and queries

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
wei =  q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
#out = wei @ x

out.shape

'''
wei[0]:
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1574, 0.8426, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2088, 0.1646, 0.6266, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5792, 0.1187, 0.1889, 0.1131, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0294, 0.1052, 0.0469, 0.0276, 0.7909, 0.0000, 0.0000, 0.0000],
        [0.0176, 0.2689, 0.0215, 0.0089, 0.6812, 0.0019, 0.0000, 0.0000],
        [0.1691, 0.4066, 0.0438, 0.0416, 0.1048, 0.2012, 0.0329, 0.0000],
        [0.0210, 0.0843, 0.0555, 0.2297, 0.0573, 0.0709, 0.2423, 0.2391]], # 8th token 0.2391 knows content and position, has strong affinity for 4th token 0.2297
       grad_fn=<SelectBackward0>)
'''

'''
Notes:
It calculates weighted representations of each input token by attending to all other tokens in the sequence.
In a single head of self-attention, the input sequence is transformed into three different linear projections: the query, key, and value. These projections are used to compute the attention scores between each pair of tokens in the sequence. The attention scores determine how much each token should attend to other tokens during the computation of the weighted representations.
Once the attention scores are calculated, they are used to weight the value projections. The weighted values are then summed to obtain the final representation for each token in the sequence. This process allows the model to assign importance to different tokens based on their relevance to other tokens in the sequence.

Attention is a communication mechanism. Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.
There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.
Each example across batch dimension is of course processed completely independently and never "talk" to each other
In an "encoder" attention block just delete the single line that does masking with tril, allowing all tokens to communicate. This block here is called a "decoder" attention block because it has triangular masking, and is usually used in autoregressive settings, like language modeling.
"self-attention" just means that the keys and values are produced from the same source as queries. In "cross-attention", the queries still get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)
"Scaled" attention additional divides wei by 1/sqrt(head_size). This makes it so when input Q,K are unit variance, wei will be unit variance too and Softmax will stay diffuse and not saturate too much. Illustration below

In a Decoder Transformer, the layers are responsible for processing the input and generating the output sequence. The layers in a Decoder Transformer typically consist of self-attention and feed-forward sub-layers, and they are stacked on top of each other to form a deep network. 
'''

k = torch.randn(B,T,head_size)
q = torch.randn(B,T,head_size)
wei = q @ k.transpose(-2, -1) * head_size**-0.5
k.var()
q.var()
wei.var()
torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]), dim=-1)
torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])*8, dim=-1) # gets too peaky, converges to one-hot

class LayerNorm1d: # (used to be BatchNorm1d)
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
  
  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]

torch.manual_seed(1337)
module = LayerNorm1d(100)
x = torch.randn(32, 100) # batch size 32 of 100-dimensional vectors
x = module(x)
x.shape

x[:,0].mean(), x[:,0].std() # mean,std of one feature across all batch inputs
x[0,:].mean(), x[0,:].std() # mean,std of a single input from the batch, of its features
# French to English translation example:

# <--------- ENCODE ------------------><--------------- DECODE ----------------->
# les réseaux de neurones sont géniaux! <START> neural networks are awesome!<END>

