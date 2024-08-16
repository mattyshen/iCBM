import torch

# General
BASE_DIR = '/home/mattyshen/iCBM/CUB'
N_ATTRIBUTES = 312
N_CLASSES = 200

# Training
UPWEIGHT_RATIO = 9.0
MIN_LR = 0.0001
LR_DECAY_SIZE = 0.1
DEVICE = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')

def set_device(gpu_id):
    global DEVICE
    DEVICE = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
def get_device():
    return DEVICE

