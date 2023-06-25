# Final Transformer Script
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 16 # number of sequences/chunks B
block_size = 32  # 32 characters, 32 contexts, 32 targets - time dimension T
n_embd = 64     # C
learning_rate = 1e-3
n_head = 4 # head_size will be n_embd // 4 = 16 dimensional
n_layer = 4
dropout = 0.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

# read in
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 65 unique chars/tokens/iints
chars = sorted(list(set(text)))
vocab_size = len(chars)

# tokenize string to integer according to vocabulary, store in torch.tensor
stoi = { ch:i for i,ch in enumerate(chars) } # mapping from unique chars to ints
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: stoi
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: itos
data = torch.tensor(encode(text), dtype=torch.long)
# tensor is multi-dim mat/array [], dimensions such as batch size, sequence length (time dimension), embedding size

# split data into train (90%) and validation (10%) sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# plug in training data into transformer - train on random batches of chunks/blocks of train_data
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # generate 4 random chunk positions
    x = torch.stack([data[i:i+block_size] for i in ix]) # contexts
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # targets
    x, y = x.to(device), y.to(device)
    return x, y # returns 4 batches of chunks of 8 tokens/chars for inputs and targets

# xb input tensor is fed into transformer, transformer processes contexts, looks up correct target to predict in yb

class Head(nn.Module):
    ''' one head of self-attention '''

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # project n_embd channels of identity & position into head_size channels/dimensions (random)
        self.query = nn.Linear(n_embd, head_size, bias=False) 
        self.value = nn.Linear(n_embd, head_size, bias=False) 
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape   # C is n_embd
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenates back to n_embd C
        out = self.dropout(self.proj(out)) # back into residual pathway
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer going back into the residual pathway
            nn.Dropout(dropout), # prevent overfitting
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    # constructor
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, h_head: number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head # 4 16-dimensional self-attention heads
        self.sa = MultiHeadAttention(n_head, head_size) # sa and concatenates back to n_embd C
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        # residual connections - fork off and come back:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Feed into neural network: bigram language model
# inputs (context indices/characters):
# tensor([[57, 43, 60, 43, 52,  1, 63, 43],
#        [60, 43, 42,  8,  0, 25, 63,  1],
#        [56, 42,  5, 57,  1, 57, 39, 49],
#        [43, 57, 58, 63,  6,  1, 58, 46]])

class BigramLanguageModel(nn.Module):
# Bigram model predicts likelihood of a next token/word in a sequence based on previous word/token - depends only on the immediate previous one

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # ex: 57 will pluck out 57th row (65-dimensional embedding vector)
        # each int token/char is represented by a 65-dimensional embedding vector [-0.2442, 0.1703, ...] where each channel/dimension of the vector represents the score for the next token - hence why we need 65 possible channels for 65 possible next tokens
        # the embedding vector is not the semantic meaning of the char in this case, but is rather the scores/predictions/logits of all possible next chars - these scores can be converted to a prob distr which is the predictions assigned to each label/class/token, the target is the ground truth label/class/token/index
        # self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # 65x65 lookup table -- 65 unique tokens x 65 embedding channels/dimensions/next_token_scores
    
        # SELF-ATTENTION VERSION:
        # for self-attention implementation, need a level of indirection, the embedding vector is not directly the scores/logits of next chars, is the semantic meaning
        # n_embd is number of embedding dimensions
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #65xn_embd lookup table, will be updated during training
        # to go from token embeddings to logits, need a linear layer
        self.lm_head = nn.Linear(n_embd, vocab_size) # converts 16x32x64 token embeddings to 16x32x65 logits
        # don't encode just the identities of idx tokens, but also position
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # 32x64 position table, updated during training
        #self.sa_head = Head(n_embd)
        #self.sa_heads = MultiHeadAttention(4, n_embd//4) # i.e. 4 16-dimensional self-attention heads concatenates back to n_embd C
        #self.ffwd = FeedForward(n_embd)
        #self.blocks = nn.Sequential(
        #    Block(n_embd, n_head=4),
        #    Block(n_embd, n_head=4),
        #    Block(n_embd, n_head=4),
        #    nn.LayerNorm(n_embd),
        #)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers

        #logits = self.token_embedding_table(idx) # (B,T,C)

        # SELF-ATTENTION VERSION:
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd C) 16x32x64 no longer logits, but token embeddings - each token has embedding vector of n_embd channels
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C) x holds token identities and positions where they occur
        #x = self.sa_heads(x) # apply heads of self-attention (B,T,head_size)
        #x = self.ffwd(x) # (B,T,C) layer for thinking on self-attended data
        x = self.blocks(x) # (B,T,n_embd C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size) converts the token embedding vectors and positions to scores/logits

        # evaluate the loss between the predicted labels (probability distribution) and the target labels (ground truth)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss # scores for next character in the sequence

    # generate text up to max_new_tokens, continues generation of new tokens (8+1+2+3...max_new_tokens) in time dimension in all 4 batch dimensions:
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context (current batch of inputs)
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond) # don't pass targets and no need to evaluate loss here in generate(), since loss is calculated in the forward function DURING TRAINING
            # focus only on the last time step (last element in T dimension)
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities (convert logits to prob distr)
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # predicts one idx_next sample index for each T dimension (B, 1)
            # append sampled index to the running sequence, whatever is predicted is concatenated on top of the previous idx along the time dimension
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
        
model = BigramLanguageModel() # construct lm
# Ridiculous old version: right now we are feeding in the entire sequence, but since this is a bigram model, we only need the immediate previous token to make a prediction
#   later, characters will look further back in the history sequence: self attention

# Instead of printing loss for each batch, estimate average loss
eval_iters = 200
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

### Train the model:
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
# create a PyTorch optimizer - takes the gradients and updates the parameters (weights/channels/scores of the embedding vectors in the embedding table) using the gradients
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # stochastic gradient descent is simplest optimizer, but let's use AdamW, set learning rate to 1e-3
max_iters = 5000
eval_interval = 100
for iter in range(max_iters): # increase number of steps for good results... 
    
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"epoch {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb) # loss given the targets
    optimizer.zero_grad(set_to_none=True) # clear previous iteration gradients
    loss.backward() # loss is backpropagated to compute the gradients of the parameters w/respect to the loss
    optimizer.step() # update the parameters

# print(loss.item()) --each individual batch loss is noisy, so we need to average up losses using estimate_loss to get a better final value for loss

### generate text from the model:
idx = torch.zeros((1, 1), dtype=torch.long) # 0 (newline character) is how we kick of the generation B=1, T=1
print(decode(m.generate(idx, max_new_tokens=500)[0].tolist())) # need to index into [0] row to pluck the single batch dimension (array of indices) generated