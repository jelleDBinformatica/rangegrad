import torch
adaptive_cuda = lambda x: x.cuda() if torch.cuda.is_available() else x