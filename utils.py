# %%
import torch
import math
import numpy as np

def make_alibias(T,ms):
  alibias = torch.empty(T, T)
  row = torch.concat((torch.arange(-T+1, 0, 1), torch.arange(0, -T, -1)))
  for i in range(T):
    alibias[-i-1] = row[i:i+T]
  return alibias[None,None].repeat(1, len(ms), 1,1) * ms[None,:,None,None]
# %%
def angular_emb(numbers, max_number, base):
  emb_dim = math.ceil(np.log(max_number) / np.log(base)) + 1 # sign
  emb = torch.zeros(numbers.shape[0], emb_dim, 2)
  emb[:,0] = numbers.sign()[:,None].repeat(1, 2)
  for i in range(1, emb_dim):
    # represent number in base
    # arg = numbers % base
    # numbers = numbers // base
    # arg = 2 * math.pi * arg / base
    arg = numbers / base ** (i)
    arg = arg - arg.floor()
    emb[:,i,0] = torch.cos(2*math.pi*arg)
    emb[:,i,1] = torch.sin(2*math.pi*arg)
  return emb

def basis_emb(numbers, max_number, base):
  emb_dim = math.ceil(np.log(max_number) / np.log(base)) + 1 # sign
  emb = torch.zeros(numbers.shape[0], emb_dim).long()
  emb[:,0] = (numbers > 0).long()
  numbers = numbers.abs()
  for i in range(1, emb_dim):
    emb[:,i] = numbers.long() % base + 2
    numbers = numbers // base
  return emb

if __name__ == "__main__":
    data = basis_emb(torch.arange(6, 15, 1), 150, 10)
    print(data)
# %%
