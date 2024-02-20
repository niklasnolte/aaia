# %%
import os
import io
import functools
import torch
import tqdm
import math
from utils import make_alibias, basis_emb
import numpy as np

# %%
class SelfAttention(torch.nn.Module):

    def __init__(self, config):


        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = torch.nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = torch.nn.Dropout(config.dropout)
        self.resid_dropout = torch.nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        self.register_buffer("alibias", make_alibias(config.block_size, ms=torch.arange(1,config.n_head+1)*.05))


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att += self.alibias[:, :, :T, :T]
        # att += torch.arange(T, device=x.device)[None, None, None, :]


        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = torch.nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = torch.nn.GELU()
        self.c_proj  = torch.nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Model(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = torch.nn.ModuleDict(dict(
            wte = torch.nn.Embedding(config.vocab_size, config.n_embd),
            drop = torch.nn.Dropout(config.dropout),
            h = torch.nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        # self.head = torch.nn.ModuleList([torch.nn.Linear(config.n_embd, config.vocab_size, bias=False) for _ in range(config.output_dim)])
            

        self.head = torch.nn.ModuleList([torch.nn.Sequential(
          torch.nn.Linear(config.n_embd, config.vocab_size, bias=True)
        ) for _ in range(config.output_dim)])

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)

        x = x.amax(dim=1)
        out = []
        for head in self.head:
            logits = head(x)
            out.append(logits)
        out = torch.stack(out, dim=1)
        return out

    

from dataclasses import dataclass
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    output_dim: int = 1


# %%
def making_up_target(X):
  # minimum distance between ones
  arange = torch.arange(len(X[0]))[None].expand(len(X), -1) 
  mask1 = (X == 1).long()
  return (arange * mask1).amax(-1, keepdim=True)

Xall = (torch.rand((10000, 12)) < .2).long()
Yall = making_up_target(Xall) + Xall.max() + 1

X = Xall
Y = Yall

torch.manual_seed(10)
perm = torch.randperm(len(X))
X = X[perm]
Y = Y[perm]
# max_rotations = (X == 0).sum(1)
# n_rotations = torch.randint(max_rotations.amax(), (len(max_rotations),))
# X = X.clone()
# for i in tqdm.trange(len(X)):
#   rots = (n_rotations[i]%max_rotations[i]).item()
#   X[i] = torch.roll(X[i], rots, 0).long()

split = int(0.8*len(X))
X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]

DEVICE = "cuda:1"

X_train = X_train.to(DEVICE)
Y_train = Y_train.to(DEVICE)
X_val = X_val.to(DEVICE)
Y_val = Y_val.to(DEVICE)
X_test = X_val
Y_test = Y_val
# %%
# define the model architecture
D_MODEL = 32
config = GPTConfig(block_size=X.shape[1], vocab_size=Yall.max().item() + 1, n_layer=2, n_head=D_MODEL//32, n_embd=D_MODEL, output_dim=Yall.shape[-1])

torch.manual_seed(100)
np.random.seed(100)

model = Model(config).to(DEVICE)

# define the training loop
EPOCHS = 1000
BATCH_SIZE = 256
EVAL_FREQ = 1

# define the optimizer and the loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 1e-2, total_iters = math.ceil(len(X_train) / BATCH_SIZE) * EPOCHS)
criterion = lambda pred, true: torch.nn.functional.cross_entropy(pred.view(-1, pred.shape[-1]), true.view(-1))


try:
  for epoch in (bar:=tqdm.trange(EPOCHS)):
    if epoch % EVAL_FREQ == 0:
      with torch.inference_mode():
        val_loss = 0
        val_acc = 0
        for i in range(0, len(X_val), BATCH_SIZE):
          x_batch = X_val[i:i+BATCH_SIZE]
          y_batch = Y_val[i:i+BATCH_SIZE]
          y_pred = model(x_batch)
          val_loss += criterion(y_pred, y_batch).item()
          val_acc += (y_pred.argmax(-1) == y_batch).all(dim=1).float().mean().item()
        val_loss /= math.ceil(len(X_val) / BATCH_SIZE)
        val_acc /= math.ceil(len(X_val) / BATCH_SIZE)
        
        test_loss = 0
        test_acc = 0
        for i in range(0, len(X_test), BATCH_SIZE):
          x_batch = X_test[i:i+BATCH_SIZE]
          y_batch = Y_test[i:i+BATCH_SIZE]
          y_pred = model(x_batch)
          test_loss += criterion(y_pred, y_batch).item()
          test_acc += (y_pred.argmax(-1) == y_batch).all(dim=1).float().mean().item()
        test_loss /= math.ceil(len(X_test) / BATCH_SIZE)
        test_acc /= math.ceil(len(X_test) / BATCH_SIZE)
    for i in range(0, len(X_train), BATCH_SIZE):
      x_batch = X_train[i:i+BATCH_SIZE]
      y_batch = Y_train[i:i+BATCH_SIZE]
      optimizer.zero_grad()
      y_pred = model(x_batch)
      loss = criterion(y_pred, y_batch)
      loss.backward()
      # clip grad
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
      optimizer.step()
      scheduler.step()
      bar.set_description(f"Loss: {loss.item():.2e}, val loss: {val_loss:.2e}, ood loss: {test_loss:.2e}, val acc: {val_acc:.2f}, ood acc: {test_acc:.2f}")
except KeyboardInterrupt:
  pass
else:
  torch.save(model, "model.pt")

# %%
