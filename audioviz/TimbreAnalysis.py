import librosa

import numpy as np
from numpy import typing as npt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .colabInterface import *

def spectral_centroid_analysis(y: npt.ArrayLike, sr: int, nfft: int = 2048) -> None :

    '''
    To show spectral centroid of the input signal with time-axis

    y: input signal
    sr: sampling rate (22050 Hz by default)
    save_to_csv: to save the result as .csv
    '''

    S = librosa.stft(y, n_fft=nfft) 
    S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    cent = librosa.feature.spectral_centroid(S=np.abs(S), n_fft = nfft)

    times = librosa.times_like(cent)
    freqs = librosa.fft_frequencies(sr = sr, n_fft = nfft)

    hmap = go.Heatmap(z=S_dB, x=times, y=freqs,
                    colorbar=dict(title='dB'), colorscale='Viridis')
    
    # To beware the return shape
    centLine = go.Scatter(x=times, y=cent[0], name='Spectral Centroid',
                       mode='lines', line_color='red', line_width = 3.6)
    
    fig = go.Figure(data = [hmap, centLine])
    # Visualization issue
    # fig.update_yaxes(type="log", range=[-3, 3.7])
    fig.update_layout(title='Spectral Centroid',
                   xaxis_title='Time (sec)',
                   yaxis_title='Frequency (Hz)')
    config = {'scrollZoom': True}
    fig.show(config=config)

    result = np.round(np.vstack((times, cent)).T, 3)
    save_and_downloader('centroid.csv', np.array(['Time', 'Centroid']), result)

def rolloff_frequency_analysis(y: npt.ArrayLike, sr: int, save_to_csv: bool = True, nfft: int = 2048) -> None :

    '''
    To show roll-off frequency (0.99&0.01) of the input signal with time-axis

    y: input signal
    sr: sampling rate (22050 Hz by default)
    save_to_csv: to save the result as .csv
    '''

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft = nfft, roll_percent=0.99)
    rolloff_min = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft = nfft, roll_percent=0.01)
    S, _ = librosa.magphase(librosa.stft(y = y, n_fft=nfft))
    S_dB = librosa.amplitude_to_db(S, ref=np.max)

    times = librosa.times_like(rolloff)
    freqs = librosa.fft_frequencies(sr = sr, n_fft = nfft)
    
    hmap = go.Heatmap(z=S_dB, x=times, y=freqs,
                    colorbar=dict(title='dB'), colorscale='Viridis')
    roll099 = go.Scatter(x=times, y=rolloff[0], name='Roll-off Frequency (0.99)',
                       mode='lines', line_color='red', line_width=2)
    roll001 = go.Scatter(x=times, y=rolloff_min[0], name='Roll-off Frequency (0.01)',
                       mode='lines', line_color='white', line_width=2) 

    fig = go.Figure(data=[hmap, roll099, roll001])
    # fig.update_yaxes(type="log", range=[0.3,4])
    fig.update_layout(title='Roll-off Frequencies',
                   xaxis_title='Time (sec)',
                   yaxis_title='Frequency (Hz)')
    config = {'scrollZoom': True}
    fig.show(config=config)    

    result = np.round(np.vstack((times, rolloff, rolloff_min)).T, 3)
    save_and_downloader('rollOffFreq.csv', np.array(['Times', 'Roll-off Max', 'Roll-off Min']), result)    

def spectral_bandwidth_analysis(y: npt.ArrayLike, sr: int, save_to_csv: bool = True, nfft: int = 2048) -> None :

    '''
    To show spectral bandwidth of the input signal with time-axis

    y: input signal
    sr: sampling rate (22050 Hz by default)
    save_to_csv: to save the result as .csv
    '''
    
    S, _ = librosa.magphase(librosa.stft(y = y, n_fft = nfft))
    spec_bw = librosa.feature.spectral_bandwidth(S = S, n_fft = nfft)
    S_dB = librosa.amplitude_to_db(S, ref=np.max)
    centroid = librosa.feature.spectral_centroid(S=S)

    times = librosa.times_like(spec_bw)
    freqs = librosa.fft_frequencies(sr, n_fft = nfft)

    hmap = go.Heatmap(z=S_dB, x=times, y=freqs,
                    colorbar=dict(title='dB'), colorscale='Viridis')
    # centLine = go.Scatter(x=times, y=centroid[0], name='Spectral Centroid', mode = 'lines', color='red')
    upper = go.Scatter(x=times, y=np.maximum(0, centroid[0] - spec_bw[0]), 
                       mode = 'lines',  line_color='white')
    lower = go.Scatter(x=times, y=np.minimum(centroid[0] + spec_bw[0], sr/2), 
                       mode = 'lines', fill='tonexty', line_color='white')
    
    # https://stackoverflow.com/questions/59490141/how-to-create-properly-filled-lines-in-plotly-when-there-are-data-gaps
    
    fig = go.Figure(data=[hmap, upper, lower])
    fig.update_layout(title='Spectral Bandwidth',
                   xaxis_title='Time (sec)',
                   yaxis_title='Frequency (Hz)')
    config = {'scrollZoom': True}
    fig.show(config=config)   

    result = np.round(np.vstack((times, spec_bw)).T, 3)
    save_and_downloader('Bandwidth.csv', np.array(['Time', 'Bandwidth']), result)


def harmonic_percussive_source_separation(y: npt.ArrayLike, sr: int, nfft: int = 2048) -> None :

    '''
    To show spectrums after harmonoc percussive source seperation

    y: input signal
    sr: sampling rate (22050 Hz by default)
    '''
    # https://plotly.com/python/subplots/
    # https://community.plotly.com/t/subplots-of-two-heatmaps-overlapping-text-colourbar/38587
    
    D = librosa.stft(y)
    H, P = librosa.decompose.hpss(D)
    rp = np.max(np.abs(D))

    times = librosa.times_like(D)
    freqs = librosa.fft_frequencies(sr = sr, n_fft = nfft)

    fig = make_subplots(rows = 3, cols = 1, shared_xaxes=True, 
                        subplot_titles=('Full power spectrogram', 'Harmonic power spectrogram', 'Percussive power spectrogram'))
    fig.append_trace(
        go.Heatmap(z = librosa.amplitude_to_db(np.abs(D), ref=np.max), 
                   x=times, y=freqs,
                   colorbar=dict(title='dB'), colorscale='Viridis'), 
                   row = 1, col = 1
    )

    fig.append_trace(
        go.Heatmap(z = librosa.amplitude_to_db(np.abs(H), ref=rp), 
                   x=times, y=freqs,
                   colorbar=dict(title='dB'), colorscale='Viridis'),
                   row = 2, col = 1
    )

    fig.append_trace(
        go.Heatmap(z = librosa.amplitude_to_db(np.abs(P), ref=rp), 
                   x=times, y=freqs,
                   colorbar=dict(title='dB'), colorscale='Viridis'),
                   row = 3, col = 1
    )

    for i in range(1, 4) :
        fig.update_xaxes(title_text="Time (sec)", row=i, col=1)
        fig.update_yaxes(title_text="Frequency (Hz)", row=i, col=1)

    config = {'scrollZoom': True}
    fig.show(config=config) 


