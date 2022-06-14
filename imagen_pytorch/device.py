import torch as th

def get_default_device_backend():
  has_cuda = th.cuda.is_available()
  has_mps = th.backends.mps.is_available()
  return ('cuda' if has_cuda else
  'mps' if has_mps else
  'cpu')