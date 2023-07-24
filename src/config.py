import torch

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'word_threshold': 32,
    'vocab_size': 40000,
    'batch_size': 128,
    'special_tokens': ['<unk>', '<bos>', '<eos>', '<pad>'],
}
