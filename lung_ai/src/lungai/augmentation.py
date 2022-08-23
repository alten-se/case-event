import numpy as np
import librosa
from functools import partial


def add_noise(data, x):
    noise = np.random.randn(len(data))
    data_noise = data + x * noise
    return data_noise

aug_dict = {
    "no_mod": lambda x: x,
    "noise": partial(add_noise, x=0.005),
    "shift": partial(np.roll, shift=1600),
    "speed_up": partial(librosa.effects.time_stretch, rate=1.2),
    "slow_down": partial(librosa.effects.time_stretch, rate=0.8)
}