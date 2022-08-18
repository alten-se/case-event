import os
def force_CPU() -> None:
    """
    hides GPU from tensorflow, to force it to use CPU
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
