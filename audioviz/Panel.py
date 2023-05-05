import numpy as np
import librosa
import matplotlib.pyplot as plt

from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor

class Dashboard():
    
    def __init__(self, fn) -> None:
        
        self.fn = fn
        self._y, self._sr = librosa.load(fn)
        
    def simple_chord_helper(self):

        dcp = DeepChromaProcessor()
        decode = DeepChromaChordRecognitionProcessor()
        chroma = dcp(self.fn)
        chrod_rec_res = decode(chroma)
        
        chord_seq = [c[2] for c in chrod_rec_res]
        time_slice = [np.round(c[0], 1) for c in chrod_rec_res]
        end_time = np.round(chrod_rec_res[-1][1], 1)
        duration = np.arange(0, end_time, 0.1)
        color_encode_list = list(set(chord_seq))
        
        chord_hm = np.ones((len(color_encode_list), len(duration)))
        for i in range(1, len(time_slice)):
            chord_hm[color_encode_list.index(chord_seq[i-1])][int(time_slice[i-1]*10):int(time_slice[i]*10)] = 0
            
        return chord_hm, color_encode_list, duration
    
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
        chord_map, enlist, dur = self.simple_chord_helper()
        ax1 = fig.add_subplot(3, 1, 2)
        ax1.imshow(chord_map, interpolation='none', cmap='spring', aspect='auto')
        ax1.set_xticks(np.arange(0, len(dur), 300), dur[::300])
        ax1.xaxis.set_ticklabels([])
        ax1.set_yticks(np.arange(0, len(enlist)), enlist)
        ax1.set_ylabel('Chord')
        ax1.set_title('Chord Recognition')
        
        # Pitch
        ax2 = fig.add_subplot(3, 1, 3)
        f0_contour, f0_times, D = self.simple_pitch_helper()
        img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax2)
        ax2.set_title('Fundamental Frequency')
        ax2.plot(f0_times, f0_contour, label='f0', color='cyan', linewidth=3)
        ax2.legend(loc='upper right')
        
        