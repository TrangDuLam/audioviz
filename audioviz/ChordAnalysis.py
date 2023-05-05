import os
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

import madmom
from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor
import librosa
import libfmp.b
import libfmp.c3
import libfmp.c4

from .colabInterface import *

def simple_chord(filename: str) -> None:
    
    dcp = DeepChromaProcessor()
    decode = DeepChromaChordRecognitionProcessor()
    chroma = dcp(filename)
    chrod_rec_res = decode(chroma)
    
    chord_seq = [c[2] for c in chrod_rec_res]
    time_slice = [np.round(c[0], 1) for c in chrod_rec_res]
    end_time = np.round(chrod_rec_res[-1][1], 1)
    duration = np.arange(0, end_time, 0.1)
    color_encode_list = list(set(chord_seq))
    
    chord_hm = np.ones((len(color_encode_list), len(duration)))
    for i in range(1, len(time_slice)):
        chord_hm[color_encode_list.index(chord_seq[i-1])][int(time_slice[i-1]*10):int(time_slice[i]*10)] = 0
        
    plt.figure(figsize=(30, 12))
    plt.imshow(chord_hm, interpolation='none', cmap='spring', aspect='auto')
    plt.xticks(np.arange(0, len(duration), 300), duration[::300])
    plt.xlabel('Time (seconds)')
    plt.yticks(np.arange(0, len(color_encode_list)), color_encode_list)
    plt.ylabel('Chord')
    plt.title('Chord recognition')

def compute_chromagram_from_filename(fn_wav, Fs=22050, N=4096, H=2048, gamma=None, version='STFT', norm='2'):
    """Compute chromagram for WAV file specified by filename

    Notebook: C5/C5S2_ChordRec_Templates.ipynb

    Args:
        fn_wav (str): Filenname of WAV
        Fs (scalar): Sampling rate (Default value = 22050)
        N (int): Window size (Default value = 4096)
        H (int): Hop size (Default value = 2048)
        gamma (float): Constant for logarithmic compression (Default value = None)
        version (str): Technique used for front-end decomposition ('STFT', 'IIS', 'CQT') (Default value = 'STFT')
        norm (str): If not 'None', chroma vectors are normalized by norm as specified ('1', '2', 'max')
            (Default value = '2')

    Returns:
        X (np.ndarray): Chromagram
        Fs_X (scalar): Feature reate of chromagram
        x (np.ndarray): Audio signal
        Fs (scalar): Sampling rate of audio signal
        x_dur (float): Duration (seconds) of audio signal
    """
    x, Fs = librosa.load(fn_wav, sr=Fs)
    x_dur = x.shape[0] / Fs
    if version == 'STFT':
        # Compute chroma features with STFT
        X = librosa.stft(x, n_fft=N, hop_length=H, pad_mode='constant', center=True)
        if gamma is not None:
            X = np.log(1 + gamma * np.abs(X) ** 2)
        else:
            X = np.abs(X) ** 2
        X = librosa.feature.chroma_stft(S=X, sr=Fs, tuning=0, norm=None, hop_length=H, n_fft=N)
    if version == 'CQT':
        # Compute chroma features with CQT decomposition
        X = librosa.feature.chroma_cqt(y=x, sr=Fs, hop_length=H, norm=None)
    if version == 'IIR':
        # Compute chroma features with filter bank (using IIR elliptic filter)
        X = librosa.iirt(y=x, sr=Fs, win_length=N, hop_length=H, center=True, tuning=0.0)
        if gamma is not None:
            X = np.log(1.0 + gamma * X)
        X = librosa.feature.chroma_cqt(C=X, bins_per_octave=12, n_octaves=7,
                                       fmin=librosa.midi_to_hz(24), norm=None)
    if norm is not None:
        X = libfmp.c3.normalize_feature_sequence(X, norm=norm)
    Fs_X = Fs / H
    return X, Fs_X, x, Fs, x_dur

def get_chord_labels(ext_minor='m', nonchord=False):
    """Generate chord labels for major and minor triads (and possibly nonchord label)

    Notebook: C5/C5S2_ChordRec_Templates.ipynb

    Args:
        ext_minor (str): Extension for minor chords (Default value = 'm')
        nonchord (bool): If "True" then add nonchord label (Default value = False)

    Returns:
        chord_labels (list): List of chord labels
    """
    chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chord_labels_maj = chroma_labels
    chord_labels_min = [s + ext_minor for s in chroma_labels]
    chord_labels = chord_labels_maj + chord_labels_min
    if nonchord is True:
        chord_labels = chord_labels + ['N']
    return chord_labels

def generate_chord_templates(nonchord=False):
    """Generate chord templates of major and minor triads (and possibly nonchord)

    Notebook: C5/C5S2_ChordRec_Templates.ipynb

    Args:
        nonchord (bool): If "True" then add nonchord template (Default value = False)

    Returns:
        chord_templates (np.ndarray): Matrix containing chord_templates as columns
    """
    template_cmaj = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]).T
    template_cmin = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]).T
    num_chord = 24
    if nonchord:
        num_chord = 25
    chord_templates = np.ones((12, num_chord))
    for shift in range(12):
        chord_templates[:, shift] = np.roll(template_cmaj, shift)
        chord_templates[:, shift+12] = np.roll(template_cmin, shift)
    return chord_templates

def chord_recognition_template(X, norm_sim='1', nonchord=False):
    """Conducts template-based chord recognition
    with major and minor triads (and possibly nonchord)

    Notebook: C5/C5S2_ChordRec_Templates.ipynb

    Args:
        X (np.ndarray): Chromagram
        norm_sim (str): Specifies norm used for normalizing chord similarity matrix (Default value = '1')
        nonchord (bool): If "True" then add nonchord template (Default value = False)

    Returns:
        chord_sim (np.ndarray): Chord similarity matrix
        chord_max (np.ndarray): Binarized chord similarity matrix only containing maximizing chord
    """
    chord_templates = generate_chord_templates(nonchord=nonchord)
    X_norm = libfmp.c3.normalize_feature_sequence(X, norm='2')
    chord_templates_norm = libfmp.c3.normalize_feature_sequence(chord_templates, norm='2')
    chord_sim = np.matmul(chord_templates_norm.T, X_norm)
    if norm_sim is not None:
        chord_sim = libfmp.c3.normalize_feature_sequence(chord_sim, norm=norm_sim)
    # chord_max = (chord_sim == chord_sim.max(axis=0)).astype(int)
    chord_max_index = np.argmax(chord_sim, axis=0)
    chord_max = np.zeros(chord_sim.shape).astype(np.int32)
    for n in range(chord_sim.shape[1]):
        chord_max[chord_max_index[n], n] = 1

    return chord_sim, chord_max

def chord_decoder(wave_filename: str) -> npt.ArrayLike :

    dcp = DeepChromaProcessor()
    decode = DeepChromaChordRecognitionProcessor()

    chroma = dcp(wave_filename)
    chordDecoded = np.array(decode(chroma))

    # csv formatter for rounding issues (too many 0s in )
    for i in chordDecoded :
        i[0] = np.round(i[0], 1)
        i[1] = np.round(i[1], 1)

    return chordDecoded

def plot_chord_recognition(wave_filename: str, anno_csv: str) -> None:

    '''
    To plot the chord recognition results of given input

    wave_filename: the exact .wav filename to analyze
    anno_csv: the segment annotation .csv 
    '''
    
    X, Fs_X, x, Fs, x_dur = compute_chromagram_from_filename(wave_filename)
    ann, color_ann = libfmp.c4.read_structure_annotation(os.path.join(anno_csv), fn_ann_color=anno_csv)
    
    chord_sim, chord_max = chord_recognition_template(X, norm_sim='max')
    chord_labels = get_chord_labels(nonchord=False)

    cmap = libfmp.b.compressed_gray_cmap(alpha=1, reverse=False)
    fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 0.03], 
                                          'height_ratios': [1.5, 3, 0.3]}, figsize=(8, 10))

    libfmp.b.plot_chromagram(X, ax=[ax[0,0], ax[0,1]], Fs=Fs_X, clim=[0, 1], xlabel='',
                         title='STFT-based chromagram (feature rate = %0.1f Hz)' % (Fs_X))
    libfmp.b.plot_segments_overlay(ann, ax=ax[0, 0], time_max=x_dur,
                               print_labels=False, colors=color_ann, alpha=0.1)

    libfmp.b.plot_matrix(chord_max, ax=[ax[1, 0], ax[1, 1]], Fs=Fs_X, 
                     title='Time–chord representation of chord recognition result',
                     ylabel='Chord', xlabel='')
    libfmp.b.plot_segments_overlay(ann, ax=ax[1, 0], time_max=x_dur,
                               print_labels=False, colors=color_ann, alpha=0.1)
    ax[1, 0].set_yticks(np.arange( len(chord_labels) ))
    ax[1, 0].set_yticklabels(chord_labels)
    ax[1, 0].grid()

    libfmp.b.plot_segments(ann, ax=ax[2, 0], time_max=x_dur, time_label='Time (seconds)',
                       colors=color_ann,  alpha=0.3)
    ax[2, 1].axis('off')
    plt.tight_layout()

    # To store the result which obtained by module madmom
    chordDetection = chord_decoder(wave_filename)
    header_row = np.array(['Start', 'End', 'Chord'])
    save_and_downloader('Chord.csv', header_row, chordDetection)


def plot_binary_template_chord_recognition(wave_filename: str, anno_csv: str) -> None:

    '''
    To plot the binary chord recognition results of given input

    wave_filename: the exact .wav filename to analyze
    anno_csv: the segment annotation .csv
    '''
    
    X, Fs_X, x, Fs, x_dur = compute_chromagram_from_filename(wave_filename)
    ann, color_ann = libfmp.c4.read_structure_annotation(os.path.join(anno_csv), fn_ann_color=anno_csv)
    chord_sim, chord_max = chord_recognition_template(X, norm_sim='max')

    chord_templates = generate_chord_templates()
    X_chord = np.matmul(chord_templates, chord_max)

    fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 0.03], 
                                              'height_ratios': [1, 1, 0.3]}, figsize=(8, 5))

    libfmp.b.plot_chromagram(X, ax=[ax[0, 0], ax[0, 1]], Fs=Fs_X, clim=[0, 1], xlabel='',
                            title='STFT-based chromagram (feature rate = %0.1f Hz)' % (Fs_X))
    libfmp.b.plot_segments_overlay(ann, ax=ax[0, 0], time_max=x_dur,
                               print_labels=False, colors=color_ann, alpha=0.1)
    
    libfmp.b.plot_chromagram(X_chord, ax=[ax[1, 0], ax[1, 1]], Fs=Fs_X, clim=[0, 1], xlabel='',
                            title='Binary templates of the chord recognition result')
    libfmp.b.plot_segments_overlay(ann, ax=ax[1, 0], time_max=x_dur,
                               print_labels=False, colors=color_ann, alpha=0.1)

    libfmp.b.plot_segments(ann, ax=ax[2, 0], time_max=x_dur, time_label='Time (seconds)',
                       colors=color_ann,  alpha=0.3)
    ax[2, 1].axis('off')
    plt.tight_layout()