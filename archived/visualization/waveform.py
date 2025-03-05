import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
audio_path = '../../data/single_channel/librispeech/1088_287.wav'  # Replace with your audio file path
y, sr = librosa.load(audio_path, sr=None)  # sr=None preserves the original sample rate

# Create the plot
plt.figure(figsize=(12, 6))

# Display the waveform
librosa.display.waveshow(y)

plt.show()
