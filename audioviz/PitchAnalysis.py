import librosa
import numpy as np
import plotly.graph_objects as go

from numpy import typing as npt

from .colabInterface import *


def pitch_estimation(y: npt.ArrayLike, sr:int, nfft: int = 2048) :

    '''
    To show the pitch esitmation result with STFT spectrogram

    y: input signal
    sr: sampling rate (22050 Hz by default)
    nfft: FFT points
    '''

    # https://community.plotly.com/t/how-to-add-lines-in-heatmap/30709
    S_dB = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=nfft)), ref=np.max)

    f0, _, _ = librosa.pyin(y,
                            fmin=librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C7'), frame_length = nfft)
    
    times = librosa.times_like(f0)
    freqs = librosa.fft_frequencies(sr = sr , n_fft = nfft)

    hmap = go.Heatmap(z=S_dB, x=times, y=freqs,
                    colorbar=dict(title='dB'), colorscale='Viridis')
    lines = go.Scatter(x=times, y=f0, name='Pitch',
                       mode='lines', line_color='red', line_width = 3.5)
    
    fig = go.Figure(data=[hmap, lines])
    fig.update_yaxes(type="log", range=[-1, 3])
    fig.update_layout(title='Pitch Estimation',
                   xaxis_title='Time (sec)',
                   yaxis_title='Frequency (Hz)')
    config = {'scrollZoom': True}
    fig.show(config=config)

    result = np.round(np.vstack((times, f0)).T, 3)
    save_and_downloader('Pitch.csv', np.array(['Time', 'Pitch']), result)


def plot_constant_q_transform(y: npt.ArrayLike, sr:int) :

    '''
    To show the constant Q transfrom spectrum

    y: input signal
    sr: sampling rate (22050 Hz by default)
    '''

    C = np.abs(librosa.cqt(y=y, sr=sr, n_bins=7*12)) # C1 to C7
    times = librosa.times_like(C)
    C_dB = librosa.amplitude_to_db(C, ref = np.max)

    fig = go.Figure(data=go.Heatmap(
        z=C_dB,
        x=times,
        colorscale='Viridis'))
    
    fig.update_layout(title='Pitch Classes',
                          xaxis_title='Time (sec)',
                          yaxis = dict(
                                    tickmode = 'array',
                                    tickvals = np.arange(0, 84, 12),
                                    ticktext = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
                                    ),
                          yaxis_title='Notes')

    config = {'scrollZoom': True}
    fig.show(config=config)


def pitch_class_histogram_chroma(y: npt.ArrayLike, sr: int, higher_resolution: bool) -> None :

    '''
    To show the occurence probability of each pitch by chroma-gram

    y: input signal
    sr: sampling rate (22050 Hz by default)
    higher_resolution: to provide more detailed statistics 
    save_to_csv: to save the statistics as .csv
    '''

    S = np.abs(librosa.stft(y))
    notes = np.array(librosa.key_to_notes('C:maj')) # For x-axis legend

    if not higher_resolution :

        chroma = librosa.feature.chroma_stft(S=S, sr=sr)
        valid_pitch = np.empty(np.shape(chroma)) # To count pitch
        valid_pitch[chroma < 1/np.sqrt(2)] = 0
        valid_pitch[chroma >= 1/np.sqrt(2)] = 1
        total = np.sum(valid_pitch)

        # To compute the probability
        # WARNING: (12,) means pure 1-D array
        occurProbs = np.empty((12,))
        for i in range(0, 12) :
            occurProbs[i] = np.sum(valid_pitch[i]) / total

        xLegend = notes

        fig = go.Figure(data=[go.Bar(
              x = xLegend, y = occurProbs * 100
        )])

        # The way to add color
        fig.data[0].marker.color = ['firebrick', 'forestgreen', 'firebrick', 'forestgreen',
                                    'firebrick', 'forestgreen', 'firebrick', 'forestgreen',
                                    'firebrick', 'forestgreen', 'firebrick', 'forestgreen']
        
        fig.update_layout(title='Pitch Classes',
                          yaxis_title='Occurrence',
                          yaxis_ticksuffix =' %')

        fig.show()

    
    if higher_resolution :

        chroma = librosa.feature.chroma_stft(S=S, sr=sr, n_chroma=120)
        valid_pitch = np.empty(np.shape(chroma)) # To count pitch
        valid_pitch[chroma < 1/np.sqrt(2)] = 0
        valid_pitch[chroma >= 1/np.sqrt(2)] = 1
        total = np.sum(valid_pitch)

        occurProbs = np.empty((120,))
        for i in range(0, 120) :
            occurProbs[i] = np.sum(valid_pitch[i]) / total
        occurProbs = np.roll(occurProbs, 12)
        
        xLegend = list()
        # To modify the axis values
        for i in range(120) :
            if i % 10 == 0 :
                xLegend.append(notes[i // 10])
            else :
                xLegend.append(" ")

        colors = list()
        
        # https://community.plotly.com/t/how-to-show-all-x-axis-tick-labels-on-a-go-bar-chart/59977
        # https://stackoverflow.com/questions/68070272/how-to-show-the-x-axis-date-ticks-neatly-or-in-a-form-of-certain-interval-in-a-p


        # Change to complmentary colors
        for i in range(120) :
            if i % 20 >=0 and i % 20 < 10 : colors.append('firebrick')
            else : colors.append('forestgreen')

        fig = go.Figure(data=[go.Bar(
               x = list(range(120)), y = occurProbs * 100
                )])

        fig.data[0].marker.color = colors
        
        fig.update_xaxes(nticks=12)
        fig.update_layout(title='Pitch Classes',
                          xaxis = dict(
                                    tickmode = 'array',
                                    tickvals = [4, 14, 24, 34, 44, 54, 64, 74, 84, 94, 104, 114],
                                    ticktext = notes
                                    ),
                          yaxis_title='Occurrence',
                          yaxis_ticksuffix =' %')

        config = {'scrollZoom': True}
        fig.show(config=config)

    
    result = np.vstack((xLegend, np.round(occurProbs, 4))).T
    save_and_downloader('Pitch_Class_Chroma.csv', np.array(['Note', 'Probability']), result)
       

def hzRevampHelper(hzStr: str) :
    '''
    String processing function used in f0-based pitch class statistics
    
    '''

    idx = max(hzStr.find('+'), hzStr.find('-'))

    return hzStr[:idx-1] + hzStr[idx:]


def pitch_class_histogram_f0(y: npt.ArrayLike, sr: int, higher_resolution: bool) -> None :

    '''
    To show the occurence probability of each pitch by pitch estimation

    y: input signal
    sr: sampling rate (22050 Hz by default)
    higher_resolution: to provide more detailed statistics 
    save_to_csv: to save the statistics as .csv
    '''

    notes = librosa.key_to_notes('C:maj')
    notesToRow = dict()
    for i in range(len(notes)) :
        notesToRow[notes[i]] = i
    
    f0, _, _ = librosa.pyin(y,
                        fmin=librosa.note_to_hz('C2'),
                        fmax=librosa.note_to_hz('C7'))
    f0Valid = [x for x in f0 if str(x) != 'nan'] # Clean the detection result array

    f0ToNotes = librosa.hz_to_note(f0Valid)
    chroma = list(map(lambda n : n[:-1], f0ToNotes)) # String processing
    
    if not higher_resolution :

        f0ToNotes = librosa.hz_to_note(f0Valid)
        chroma = list(map(lambda n : n[:-1], f0ToNotes)) # String processing
        pitchOccur = np.zeros((12,))

        for i in chroma :
            pitchOccur[notesToRow[i]] += 1
        occurProbs = pitchOccur / np.sum(pitchOccur)

        fig = go.Figure(data=[go.Bar(
              x = notes, y = occurProbs * 100
        )])

        # The way to add color
        fig.data[0].marker.color = ['firebrick', 'forestgreen', 'firebrick', 'forestgreen',
                                    'firebrick', 'forestgreen', 'firebrick', 'forestgreen',
                                    'firebrick', 'forestgreen', 'firebrick', 'forestgreen']
        
        fig.update_layout(title='Pitch Classes',
                   yaxis_title='Occurrence',
                   yaxis_ticksuffix =' %')

        fig.show()
    
    if higher_resolution :
        
        notes12 = librosa.key_to_notes('C:maj')
        notesCentDict = dict()
    
        for n in notes12 :
            for i in np.arange(-40, 60, 10) :

                if i < 0 :
                    noteCent = n + str(i)
                else :
                    noteCent = n + '+' + str(i)
                notesCentDict[noteCent] = 0

        f0, _, _ = librosa.pyin(y,
                            fmin=librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C7'))
        f0Valid = [x for x in f0 if str(x) != 'nan']
        f0ToNotes = librosa.hz_to_note(f0Valid, cents=True)

        categorizedNotes =list(map(hzRevampHelper, f0ToNotes))

        for i in categorizedNotes :
            notesCentDict[i] = notesCentDict.get(i, 0) + 1

        occurProbs = np.array(list(notesCentDict.values())) / sum(notesCentDict.values())
        notes = list(notesCentDict.keys())

        fig = go.Figure(data=[go.Bar(
                x = list(range(1200)), y = occurProbs * 100
            )])

        # The way to add color
        colors = []
        for i in range(120) :
            if i % 20 >=0 and i % 20 < 10 : colors.append('firebrick')
            else : colors.append('forestgreen')

        fig.data[0].marker.color = colors
        
        fig.update_layout(title='Pitch Classes',
                          # Refine the x-axis
                          xaxis = dict(
                                    tickmode = 'array',
                                    tickvals = [4, 14, 24, 34, 44, 54, 64, 74, 84, 94, 104, 114],
                                    ticktext = notes12
                                    ),
                   yaxis_title='Occurrence',
                   yaxis_ticksuffix =' %')
        config = {'scrollZoom': True}
        fig.show(config = config)

    result = np.vstack((notes, np.round(occurProbs, 4))).T
    save_and_downloader('Pitch_Class_Pitch.csv', np.array(['Notes', 'Probability']), result) 
    