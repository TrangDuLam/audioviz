import librosa
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy

from numpy import typing as npt

from .colabInterface import *

def onsets_detection(y: npt.ArrayLike, sr: int, write_to_wav: bool = False) -> None :

    '''
    To show onsets of the input signal with time axis and modify the .wav with onset click sounds

    y: input signal
    sr: sampling rate (22050 Hz by default)
    save_to_wav: to add onset click sounds (not to store by default)
    '''

    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(o_env, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    config = {'scrollZoom': True}

    fig = px.line(x=times, y=o_env, labels={'x':'time (sec)', 'y' : 'Onset Strength'}, title='Onset Detection')
    # Plotting vlines would need a loop if adapting Plotly
    for vLinePosistion in times[onset_frames] :
        fig.add_vline(x=vLinePosistion, line_width = 0.9, line_dash="dash", line_color="green")
    fig.show(config=config)

    # Save onset-to-time info
    result = np.round(times[onset_frames].T, 3)
    save_and_downloader('beats.csv', np.array(['Time']), result)

    # Save the processed audio
    y_onset_clicks = librosa.clicks(frames=onset_frames, sr=sr, length=len(y))
    audio_save_and_downloader('onset_click.wav', y_onset_clicks, sr)


def plot_onset_strength(y: npt.ArrayLike, sr:int, standard: bool = True, custom_mel: bool = False, cqt: bool = False) :

    '''
    To plot the onset strength of the input signal with time-axis

    y: input signal
    sr: sampling rate (22050 Hz by default)
    standard: using STFT to calculate
    custom_mel: using Mel-scale to calculate
    cqt: using CQT to calculate
    '''
    
    D = np.abs(librosa.stft(y))
    times = librosa.times_like(D)
    fig = make_subplots(rows = 3, cols = 1, shared_xaxes=True, 
                        subplot_titles=('Standard', 'Mel', 'CQT'))
    config = {'scrollZoom': True}
    
    # Standard Onset Fuction 

    
    onset_env_standard = librosa.onset.onset_strength(y=y, sr=sr)
    fig.append_trace(go.Scatter(x = times, y = onset_env_standard / onset_env_standard.max(), 
                            mode = 'lines', name='Sandard'),
                     row = 1, col = 1)
        # Shifted upward to perform better visualization
    
    onset_env_mel = librosa.onset.onset_strength(y=y, sr=sr,
                                                aggregate=np.median,
                                                fmax=8000, n_mels=256)
    fig.append_trace(go.Scatter(x = times, y = onset_env_mel / onset_env_mel.max(), 
                            mode = 'lines', name='Custom Mel'),
                     row = 2, col = 1)
        # Shifted upward to perform better visualization
        
    C = np.abs(librosa.cqt(y=y, sr=sr))
    onset_env_cqt = librosa.onset.onset_strength(sr=sr, S=librosa.amplitude_to_db(C, ref=np.max))
    fig.append_trace(go.Scatter(x = times, y = onset_env_cqt / onset_env_cqt.max(), 
                             mode = 'lines', name='CQT'),
                     row = 3, col = 1)

    for i in range(1, 4) :
        fig.update_xaxes(title_text="Time (sec)", row=i, col=1)
        fig.update_yaxes(title_text="Frequency (Hz)", row=i, col=1)
        
    fig.show(config=config)


def beat_tracking(y: npt.ArrayLike, sr:int, write_to_wav: bool = True, hop_length: int = 512) :

    '''
    To show beat of the input signal with time axis and modify the .wav with beat click sounds

    y: input signal
    sr: sampling rate (22050 Hz by default)
    save_to_wav: to add beat click sounds (not to store by default)
    hop_length: hop size between frames
    '''
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)

    config = {'scrollZoom': True}

    titleBPM = 'Beat Tracking (BPM %.2f)' % (tempo)
    fig = px.line(x=times, y=onset_env, labels={'x':'time (sec)', 'y' : 'Onset Strength'}, 
                  title=titleBPM)

    # Plotting vlines would need a loop if adapting Plotly
    for vLinePosistion in times[beats] :
        fig.add_vline(x=vLinePosistion, line_width = 0.9, line_dash="dash", line_color="green")
    fig.show(config=config)

    # Save beat-to-time info
    result = np.round(times[beats].T, 3)
    save_and_downloader('beats.csv', np.array(['Time']), result)

    # Save processed audio
    y_beats = librosa.clicks(frames=beats, sr=sr, length=len(y))
    audio_save_and_downloader('beat_click.wav', y_beats, sr)


def predominant_local_pulses(y: npt.ArrayLike, sr:int) -> None :

    '''
    To plot the predominant local pulses of the input signal with time-axis

    y: input signal
    sr: sampling rate (22050 Hz by default)
    '''

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
    times = librosa.times_like(pulse, sr=sr)

    config = {'scrollZoom': True}
    fig = px.line(x=times, y=onset_env, labels={'x':'time (sec)', 'y' : 'Onset Strength'}, title='Predominant Local Pulses')
    for vLinePosistion in times[beats_plp] :
        fig.add_vline(x = vLinePosistion, line_width = 0.9, line_dash="dash", line_color="green")

    # fig.update_layout(yaxis_range=[0,np.max(onset_env) + 1])
    fig.show(config=config)


def static_tempo_estimation(y: npt.ArrayLike, sr: int, hop_length: int = 512) -> None:
  
    '''
    To visualize the result of static tempo estimation
  
    y: input signal array
    sr: sampling rate
    hop_length: hop size between frames
    '''

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

    # Static tempo estimation
    prior = scipy.stats.uniform(30, 300)  # uniform over 30-300 BPM
    utempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, prior=prior)

    tempo = tempo.item()
    utempo = utempo.item()
    ac = librosa.autocorrelate(onset_env, max_size=2 * sr // hop_length)
    freqs = librosa.tempo_frequencies(len(ac), sr=sr,
                                    hop_length=hop_length)

    fig, ax = plt.subplots()
    ax.semilogx(freqs[1:], librosa.util.normalize(ac)[1:],
                label='Onset autocorrelation', base=2)
    ax.axvline(tempo, 0, 1, alpha=0.75, linestyle='--', color='r',
                label='Tempo (default prior): {:.2f} BPM'.format(tempo))    
    ax.axvline(utempo, 0, 1, alpha=0.75, linestyle=':', color='g',
                label='Tempo (uniform prior): {:.2f} BPM'.format(utempo)) 
    ax.set(xlabel='Tempo (BPM)', title='Static tempo estimation')
    ax.grid(True)
    ax.legend() 


def plot_tempogram(y: npt.ArrayLike, sr: int, type: str = 'autocorr', hop_length: int = 512) -> None :

    '''
    To visualize the result of dynamic tempo estimation
  
    y: input signal array
    sr: sampling rate
    type: algorithms to use
        autocoor: by auto-correlation (default)
        fourier: STFT
    hop_length: hop size between frames 
    '''
    
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)[0]

    if type == 'fourier' :
        # To determine which temp to show?
        librosa.display.specshow(np.abs(tempogram), sr=sr, hop_length=hop_length, 
                                 x_axis='time', y_axis='fourier_tempo', cmap='magma')
        plt.axhline(tempo, color='w', linestyle='--', alpha=1, label='Estimated tempo={:g}'.format(tempo))
        plt.legend(loc='upper right')
        plt.title('Fourier Tempogram')

    if type == 'autocorr' :
        ac_tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length, norm=None)
        librosa.display.specshow(ac_tempogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo', cmap='magma')
        plt.axhline(tempo, color='w', linestyle='--', alpha=1, label='Estimated tempo={:g}'.format(tempo))
        plt.legend(loc='upper right')
        plt.title('Autocorrelation Tempogram')


