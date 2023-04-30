import librosa
import scipy 
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from numpy import typing as npt

class Dashbroad() :
    
    def __init__(self, y: npt.ArrayLike, sr: int):
        self.y = y
        self.sr = sr

    def show_summary(self) :
        
        '''
        Ploting the summary of the audio file, including
        
        1. Time domain waveform
        2. Pitch contour
        3. Chord recognition heatmap 
        
        '''
        fig = make_subplots(rows = 3, cols = 1, shared_xaxes=True, 
                        subplot_titles=('Waveform', 'Pitch & Spectorgram', 'Chord Recognition'))
        
        times = (1 / self.sr) * np.arange(len(self.y))
        fig.append_trace(
            go.Scatter(x = times, y = self.y,
            mode = 'lines')
        )
        
        S_dB = librosa.amplitude_to_db(np.abs(librosa.stft(self.y, n_fft=2048)), ref=np.max)
        f0, _, _ = librosa.pyin(self.y,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'), frame_length = 2048)
        times = librosa.times_like(f0)
        freqs = librosa.fft_frequencies(sr = self.sr , n_fft = 2048)
        
        # To study
        '''
        fig.append_trace(
            go.Heatmap(z=S_dB, x=times, y=freqs, colorbar=dict(title='dB'), colorscale='Viridis')
            go.Scatter(x=times, y=f0, name='Pitch', mode='lines', line_color='red', line_width = 3.5)
        )
        '''
                    
    
    def save_summary(self):
        '''
        Saving the summary of the audio file, including
        
        1. Beat
        2. 
        
        '''
        pass