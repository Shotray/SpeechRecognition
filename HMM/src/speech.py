import scipy.io.wavfile
from matplotlib import pyplot as plt
from scipy.fftpack import dct
import numpy as np
import os

# draw picture
# def plt_plot(x, y, title, save, xlabel='Time(sec)', ylabel='Amplitude'):
#     plt.clf()
#     plt.figure(figsize=(10, 5), facecolor="white")
#     plt.plot(x, y, linewidth='0.5', color='#4169E1')
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     save = 'pic/'+save
#     if os.path.exists(save):
#         os.remove(save)
#     plt.savefig(save, dpi=500)
#     plt.close()


# get origin signal
# sample_rate, sig = scipy.io.wavfile.read('source/test.wav')
# sig = sig[:, 0]
# signal_num = np.arange(len(sig))
# print("sample rate: {}".format(sample_rate))
# print("signal : {}".format(sig))

# plt_plot(signal_num, sig, 'signal of Voice', 'origin.png')

# pre-emphasis
# x[t-1] = x[t] - alpha * x[t-1] 0.95<alpha<0.99
def pre_emhpasis(alpha,audio):
    emphasized_signal = np.append(audio[0], audio[1:] - alpha * audio[:-1])
    emphasized_signal_num = np.arange(len(emphasized_signal))
    # plt_plot(emphasized_signal_num, emphasized_signal,
    #          'Emphasized signal of Voice', 'emphasized_signal.png')
    return emphasized_signal

# window and fft
def process(emphasized_signal,sample_rate):
    frame_width = 0.025
    frame_shift = 0.01
    emlen = len(emphasized_signal)
    # for i in range(0, emlen, round(sample_rate * frame_shift)):
    #     temp = emphasized_signal[i: i + frame_width * sample_rate]

    sample_width = round(sample_rate * frame_width)
    sample_shift = round(sample_rate * frame_shift)
    frame_num = int(
        np.ceil(float(np.abs(emlen - sample_width)) / sample_shift))

    windowed = []
    mag_frames = []
    pow_frames = []
    for i in range(frame_num):
        temp = emphasized_signal[i*sample_shift:i*sample_shift + sample_width]
        weight = hamming(sample_width)
        temp = temp * weight
        windowed.append(temp)
        # x = dft(temp,sample_width)

        NFFT = sample_width
        mag_frames.append(np.absolute(np.fft.rfft(temp, NFFT)))

        pow_frames.append(
            (1.0/NFFT)*((np.absolute(np.fft.rfft(temp, NFFT)))**2))

    # reshape
    windowed_frame = np.array(windowed)
    windowed_frame = windowed_frame.flatten()

    # plt_plot(np.arange(len(windowed_frame)),
    #          windowed_frame, 'Hamming', 'hamming.png')

    # mag_frames_plot = np.array(mag_frames)
    # mag_frames_plot = mag_frames_plot.flatten()
    # plt_plot(np.arange(len(mag_frames_plot)), mag_frames_plot, 'FFT', 'FFT.png')

    return pow_frames, sample_width

# hamming window for alpha = 0.46164
# hanning window for alpha = 0.5
def hamming(n):
    x = np.arange(n)
    y = 0.53836 - 0.46164*np.cos((2*np.pi*x) / (n-1))
    return y

# mel fliter bank
def mel(frames, NFFT, sample_rate):
    mel_filter_channels = 40
    low_mel = 0
    high_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))

    # equally devided
    mel_points = np.linspace(low_mel, high_mel, mel_filter_channels+2)
    # convert mel to hz
    hz_points = (700*(10**(mel_points / 2595) - 1))

    bin = np.floor((NFFT + 1)*hz_points / sample_rate)
    fbank = np.zeros((mel_filter_channels, int(np.floor(NFFT / 2 +1
    ))))

    for m in range(1, mel_filter_channels + 1):
        f_m_minus = int(bin[m-1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(frames, fbank.T)
    filter_banks = np.where(
        filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    return filter_banks

# delta features
# 𝑑(𝑡)=(𝑐(𝑡+1)−𝑐(𝑡−1))/2
def delta_difference(mfcc, N=1):
    assert N >= 1
    num_frames = len(mfcc)
    denominator = 2 * sum(np.arange(1, N+1)**2)
    difference_mfcc = np.empty_like(mfcc)
    padded = np.pad(mfcc,((N,N),(0,0)),mode = 'edge')
    for i in range(num_frames):
        difference_mfcc[i] = np.dot(np.arange(-N,N+1),padded[i:i+2*N+1])/denominator
    return difference_mfcc
    
# normalization
# 𝑦 ̂_𝑡 [𝑗]=(𝑦_𝑡 [𝑗]−𝜇(𝑦[𝑗]))/𝜎(𝑦[𝑗]) 
def normalization(feat):
    return (feat - np.mean(feat, axis=0)) / np.std(feat, axis=0)


def get_mfcc(audio,samplerate = 48000):
    emphasized_signal = pre_emhpasis(0.97, audio)
    pow_frames, NFFT = process(emphasized_signal,samplerate)
    filter_banks = mel(pow_frames, NFFT, samplerate)

    num_cep = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_cep + 1)]
    energy = np.sum(pow_frames, axis=1)
    lenth = len(energy)
    energy = energy.reshape(lenth,1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)

    mfcc = np.concatenate((mfcc, 20*np.log10(energy)), axis=1)
    mfcc = np.concatenate((mfcc, delta_difference(mfcc,1),delta_difference(mfcc,2)),axis = 1)
    mfcc = normalization(mfcc)
    return mfcc

# if __name__ == '__main__':
#     emphasized_signal = pre_emhpasis(0.97)
#     pow_frames, NFFT = process(emphasized_signal)
#     filter_banks = mel(pow_frames, NFFT)

#     # print(np.array(pow_frames).shape)
#     # exit()

#     num_cep = 12
#     mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_cep + 1)]
#     energy = np.sum(pow_frames, axis=1)
#     lenth = len(energy)
#     energy = energy.reshape(lenth,1)
#     energy = np.where(energy == 0, np.finfo(float).eps, energy)

#     mfcc = np.concatenate((mfcc, 20*np.log10(energy)), axis=1)
#     mfcc = np.concatenate((mfcc, delta_difference(mfcc,1),delta_difference(mfcc,2)),axis = 1)
#     # print(np.array(mfcc).shape)
#     mfcc = normalization(mfcc)
#     print(mfcc)
