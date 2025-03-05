import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np

def waveform():
    root_dir = "data/single_channel/BadmintonCourt2/"
    source_type = os.listdir(root_dir)

    for item in source_type:
        speakers = os.listdir(root_dir + item)
        if not os.path.exists("/home/ubuntu/project/DOA/手撕代码/result/BadmintonCourt2/" + item):
            os.makedirs("/home/ubuntu/project/DOA/手撕代码/result/BadmintonCourt2/" + item)
        for speaker in speakers:
            if not os.path.exists("/home/ubuntu/project/DOA/手撕代码/result/BadmintonCourt2/" + item + '/' + speaker):
                os.makedirs("/home/ubuntu/project/DOA/手撕代码/result/BadmintonCourt2/" + item + '/' + speaker)
            audio_files = os.listdir(root_dir + item + '/' + speaker)
            for audio in audio_files:
                # Load the audio file
                audio_path = root_dir + item + '/' + speaker + '/' + audio
                y, sr = librosa.load(audio_path, sr=None)  # sr=None preserves the original sample rate

                # Create the plot
                plt.figure(figsize=(12, 6))

                # Display the waveform
                librosa.display.waveshow(y)

                plt.savefig("/home/ubuntu/project/DOA/手撕代码/result/BadmintonCourt2/" + item + '/' + speaker + '/' + audio.split('.')[0] + ".png")
                plt.close()


def log_mel():
    root_dir = "data/single_channel/BadmintonCourt2/"
    source_type = os.listdir(root_dir)

    for item in source_type:
        speakers = os.listdir(root_dir + item)
        if not os.path.exists("/home/ubuntu/project/DOA/手撕代码/result/BadmintonCourt2/" + item):
            os.makedirs("/home/ubuntu/project/DOA/手撕代码/result/BadmintonCourt2/" + item)
        for speaker in speakers:
            if not os.path.exists(
                    "/home/ubuntu/project/DOA/手撕代码/result/BadmintonCourt2/" + item + '/' + speaker + '/spectrogram/'):
                os.makedirs(
                    "/home/ubuntu/project/DOA/手撕代码/result/BadmintonCourt2/" + item + '/' + speaker + '/spectrogram/')
            audio_files = os.listdir(root_dir + item + '/' + speaker)
            for audio in audio_files:
                # Load the audio file
                audio_path = root_dir + item + '/' + speaker + '/' + audio
                y, sr = librosa.load(audio_path, sr=None)  # sr=None preserves the original sample rate

                # Extract log-mel spectrogram
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                log_S = librosa.power_to_db(S, ref=np.max)

                # Create the plot for log-mel spectrogram
                plt.figure(figsize=(12, 6))
                librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
                plt.colorbar(format='%+02.0f dB')
                plt.title('Log-Mel Spectrogram')
                plt.savefig(
                    "/home/ubuntu/project/DOA/手撕代码/result/BadmintonCourt2/" + item + '/' + speaker + '/spectrogram/' +
                    audio.split('.')[0] + "_logmel.png")
                plt.close()

