from google.colab import drive
from google.colab import files
import functools
import os

from IPython.display import display
import ipywidgets as widgets

import soundfile as sf

from numpy import typing as npt


def core_download_and_save(filename: str, header_row: npt.ArrayLike, result_array: npt.ArrayLike, arg) :

    with open(filename, 'w') as out :
            print(*header_row, sep=',', file=out)
            for row in result_array :
                print(*row, sep=',', file=out)

    files.download(filename)

def noMsg(arg):
    print("No")

def save_and_downloader(filename: str, header_of_file: npt.ArrayLike, result_array: npt.ArrayLike) :

    buttonDownload = widgets.Button(description = 'Download & Save')
    buttonCancel =  widgets.Button(description = 'Cancel') 
    buttonDownload.on_click(functools.partial(core_download_and_save, filename, header_of_file, result_array))
    buttonCancel.on_click(noMsg)

    display(buttonDownload)
    display(buttonCancel)

def core_audio_download_and_save(filename: str, audio_array: npt.ArrayLike, sr: int, arg) :
    
    sf.write(filename, audio_array, sr, subtype='PCM_24')
    files.download(filename)


def audio_save_and_downloader(filename: str, audio_array: npt.ArrayLike, sr: int) :

    buttonDownload = widgets.Button(description = 'Download & Save')
    buttonCancel =  widgets.Button(description = 'Cancel') 
    buttonDownload.on_click(functools.partial(core_audio_download_and_save, filename, audio_array, sr))
    buttonCancel.on_click(noMsg)

    display(buttonDownload)
    display(buttonCancel)
