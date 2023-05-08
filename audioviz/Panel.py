import numpy as np
import librosa
import matplotlib.pyplot as plt

from .ChordAnalysis import *

class Dashboard():
    
    def __init__(self, fn) -> None:
        
        self.fn = fn
        self._y, self._sr = librosa.load(fn)
        
    def simple_chord_helper(self):

        X, Fs_X, x, Fs, x_dur = compute_chromagram_from_filename(self.fn)
        _, chord_max = chord_recognition_template(X, norm_sim='max', nonchord=True)
        
        return chord_max, x_dur
    
    def simple_pitch_helper(self):
        
        f0, _, _ = librosa.pyin(self._y,
                                fmin=librosa.note_to_hz('C2'),
                                fmax=librosa.note_to_hz('C7'))
        
        times = librosa.times_like(f0)
        
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self._y)), ref=np.max)
        
        return f0, times, D
    
    def summary(self):
        
        fig = plt.figure(figsize=(35, 25))
        # Time-domain waveform
        ax0 = fig.add_subplot(3, 1, 1)
        times = np.arange(len(self._y)) / self._sr
        ax0.plot(times, self._y)
        ax0.xaxis.set_ticklabels([])
        ax0.set_title('Waveform')
        
        # A simple version of chord recognition
        chord_map, dur = self.simple_chord_helper()
        ax1 = fig.add_subplot(3, 1, 2)
        ax1.imshow(chord_map, interpolation='none', cmap='gray_r', aspect='auto', origin='lower')
        times = np.round(np.linspace(0, dur, chord_map.shape[1]), 1)
        chord_labels = get_chord_labels(nonchord=True)
        ax1.set_xticks(np.arange(0, len(times), 323), times[::323])
        ax1.xaxis.set_ticklabels([])
        ax1.set_yticks(np.arange(0, len(chord_labels)), chord_labels)
        ax1.set_ylabel('Chord')
        ax1.set_title('Chord Recognition')
        
        # Pitch
        ax2 = fig.add_subplot(3, 1, 3)
        f0_contour, f0_times, D = self.simple_pitch_helper()
        img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax2)
        ax2.set_title('Fundamental Frequency')
        ax2.plot(f0_times, f0_contour, label='f0', color='cyan', linewidth=3)
        ax2.legend(loc='upper right')
        