import torch


def get_device():
    """
    使用している環境に応じて、GPUを使用できるように、デバイスを選択
    優先順位は
    ① GPU
    ② MPS (Mac GPU)
    ③ CPU
    Args:
        None
    Returns:
        device[str]: 'cuda', 'mps' or 'cpu'
    """
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device
