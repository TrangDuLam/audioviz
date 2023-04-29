import numpy as np
import scipy

from IPython.display import Audio

import numpy.typing as npt

def audio_cutter(y: npt.ArrayLike, sr: int, start_time: float, end_time: float) -> npt.ArrayLike :
    '''
    To cut the input signal with start and end time

    y: input signal
    sr: sampling rate (22050 Hz by default)
    start_time: start time of the cut
    end_time: end time of the cut
    '''
    start_index = int(start_time * sr)
    end_index = int(end_time * sr)
    return y[start_index:end_index]

def audio_player(y: npt.ArrayLike, sr: int) -> None :
    '''
    To play the input signal

    y: input signal
    sr: sampling rate (22050 Hz by default)
    '''
    return Audio(data=y, rate=sr)