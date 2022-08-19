import os
from types import ModuleType

def force_CPU() -> None:
    """
    hides GPU from tensorflow, to force it to use CPU
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

def silence_tf() -> ModuleType:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    return tf