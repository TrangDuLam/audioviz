import librosa

import numpy as np
import plotly.graph_objects as go

from numpy import typing as npt

from .colabInterface import *


def plot_waveform(y: npt.ArrayLike, sr: int, start_time: float = 0.0, end_time: float = None) -> None :
    
    '''
    To show the waveform in time domain of a sound file

    y: input signal
    sr: sampling rate (22050 Hz by default)
    '''

    startIdx = int(start_time * sr)
    fig = go.Figure()
    config = {'scrollZoom': True}
    
    if not end_time :
        times = 1 / sr * np.arange(startIdx, len(y))
        fig.add_trace(go.Scatter(x = times, y = y[startIdx:],
                        mode = 'lines'))
        fig.update_layout(title='Waveform',
                    xaxis_title='Time (sec)',
                    yaxis_title='Amplitude')
        fig.show(config=config)
    
    else :
        endIdx = int(end_time * sr)
        times = 1 / sr * np.arange(startIdx, endIdx)
        fig.add_trace(go.Scatter(x = times, y = y[startIdx:endIdx-1],
                        mode = 'lines'))
        fig.update_layout(title='Waveform',
                    xaxis_title='Time (sec)',
                    yaxis_title='Amplitude')
        fig.show(config=config)


def signal_RMS_analysis(y: npt.ArrayLike) :

    '''
    To show the RMS amplitude in time domain of a sound file

    y: input signal
    sr: sampling rate (22050 Hz by default)
    save_to_csv: to store the data to .csv (not to store by default)
    '''

    rms = librosa.feature.rms(y = y)
    times = librosa.times_like(rms)

    
        
    fig = go.Figure()
    config = {'scrollZoom': True}
    fig.add_trace(go.Scatter(x = times, y = rms[0],
                    mode = 'lines'))
    fig.update_layout(title='RMS Chart',
                xaxis_title='Time (sec)',
                yaxis_title='RMS')
    fig.show(config=config)

    result = np.round(np.vstack((times, rms[0])).T, 3)
    save_and_downloader('rms.csv', np.array(['Time', 'RMS']), result)


def plot_spectrogram(y: npt.ArrayLike, sr:int, scale : str = 'STFT') :

    '''
    To show the pitch esitmation result with spectrograms (STFT and Mel)

    y: input signal
    sr: sampling rate (22050 Hz by default)
    '''

    if scale == 'STFT' :

        S = librosa.stft(y, n_fft=2048)
        S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        
        times = librosa.times_like(S)
        freqs = librosa.fft_frequencies(sr = sr, n_fft = 2048)

        figTitle = 'STFT Spectrogram'

    elif scale == 'Mel' :

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels = 128)
        S_dB = librosa.power_to_db(S, ref=np.max)
    
        times = librosa.times_like(S)
        freqs = librosa.mel_frequencies(128)

        figTitle = 'Mel Spectrogram' 

    # Only provide STFT and Mel-scale
    else :
        raise ValueError('Only provide STFT and Mel') 

    fig = go.Figure(data=go.Heatmap(
        z=S_dB,
        x=times,
        y=freqs,
        colorbar=dict(title='dB'),
        colorscale='Viridis'))
    
    fig.update_layout(title=figTitle,
                    xaxis_title='Time (sec)',
                    yaxis_title='Frequency (Hz)')

    config = {'scrollZoom': True}
    fig.show(config=config)