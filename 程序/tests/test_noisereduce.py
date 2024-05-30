import noisereduce as nr
from scipy.io import wavfile

fileName = 'Adjust_whitenoise_4M25-gain0.wav'
audio_path  = 'C:/Users/LSCM_/Downloads/20240529/4-mic demo/'+fileName

# fileName    = '[05-29-24]18_17_14.wav'
# audio_path  = 'C:/Users/LSCM_/Documents/'+fileName


# Load data from a WAV file (replace 'output.wav' with your actual file)
rate, data = wavfile.read(audio_path)

# Select a section of data that represents noise (e.g., noisy_part = data[10000:15000])
noisy_part = data[int(1.126*48e3):int(1.587*48e3)]
# # Perform noise reduction
reduced_noise = nr.reduce_noise(y = data, sr=rate, prop_decrease=0.97)

#, n_std_thresh_stationary=1.5,stationary=True,
#                               use_torch=False)

# reduced_noise = nr.reduce_noise(y=data,y_noise=noisy_part,sr=rate,stationary=True)

wavfile.write('processed.wav',data=reduced_noise,rate=rate)


# import torch
# from noisereduce.torchgate import TorchGate as TG
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# # Create TorchGating instance
# rate=48000
# tg = TG(sr=rate, nonstationary=True).to(device)

# # Apply Spectral Gate to noisy speech signal
# rate, data = wavfile.read(audio_path)
# enhanced_speech = tg(data)
# wavfile.write('processed.wav',data=enhanced_speech,rate=rate)