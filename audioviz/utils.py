import numpy as np
import scipy
import librosa

import IPython.display as ipd
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

def load_example_audio(filename: str) -> None :
    '''
    To load the example audio

    '''
    if filename == 'Mozart' :
        y, sr = librosa.load('./example/Mozart_Turkish_March.wav', duration=30)
    if filename == 'Brahms' :
        y, sr = librosa.load('./exampl/Hungarian_Dance_No_5.wav', duration=30)
        
    return y, sr