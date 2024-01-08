import time
import os, subprocess
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from math import log
import pickle
import librosa

from scipy.signal import savgol_filter
def smooth(signal: np.ndarray, window_size: int = 5, polynomial_order: int = 3, ax: int = 1, plot: bool = False) -> np.ndarray:
    smoothed_signal = savgol_filter(signal, window_size, polynomial_order, axis=ax)

    if plot:
        plt.figure(figsize=(16, 8))
        plt.plot(signal, label="Original")
        plt.plot(smoothed_signal, label="Smoothed")
        plt.legend()
        plt.savefig("smoothed_signal.png")

    return smoothed_signal

def scale(X: np.ndarray, x_min: float = 0, x_max: float = 1) -> np.ndarray:
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    X_scaled = (X - min_val) / (max_val - min_val) * (x_max - x_min) + x_min
    X_scaled[:, max_val == min_val] = x_min  # Handle division by zero
    return X_scaled

def global_scale(X: np.ndarray, x_min: float = 0, x_max: float = 1) -> np.ndarray:
    min_val = np.min(X)
    max_val = np.max(X)
    if max_val == min_val:
        return np.full(X.shape, x_min)
    
    return (X - min_val) / (max_val - min_val) * (x_max - x_min) + x_min

def newlogspace(start, stop, num=50, endpoint=True, base=10, dtype=None):
    return np.logspace(log(start)/log(base), log(stop)/log(base), num, endpoint, base, dtype)

def get_log_spaced_averages(FFT_features: np.ndarray, name_str: str, num: int = 30) -> np.ndarray:
    start_f = 1
    cutoff = -1

    frequency_bins = np.unique(newlogspace(start_f, FFT_features.shape[0], num=num).astype(int))[:cutoff]
    bin_avg_features = []

    prev_bin_stop = 0
    for bin_stop in frequency_bins:
        bin_fis = slice(prev_bin_stop, bin_stop)
        avg_feature = np.mean(FFT_features[bin_fis, :], axis=0)
        bin_avg_features.append(avg_feature)
        prev_bin_stop = bin_stop

    return np.array(bin_avg_features)

def extract_magnitudes(signal: np.ndarray, name_str: str, suppression_power: float) -> np.ndarray:
    magnitudes = get_log_spaced_averages(signal, name_str, num=64)
    magnitudes = librosa.amplitude_to_db(magnitudes, ref=np.max(magnitudes))
    magnitudes = global_scale(magnitudes)
    magnitudes **= suppression_power
    magnitudes_glob = global_scale(magnitudes.copy())
    magnitudes_loc = scale(magnitudes.copy())
    blend = 0.4
    magnitudes = (1 - blend) * magnitudes_glob + blend * magnitudes_loc

    return magnitudes

def get_chroma(signal, n_features, Fs, FFT_hop_length, norm_d = 2, min_cutoff = 0.1):
    chroma_cqt = librosa.feature.chroma_cqt(y=signal, sr=Fs, 
                                                hop_length=FFT_hop_length, fmin=None, norm=norm_d, threshold=min_cutoff, 
                                                tuning=None, n_chroma=n_features, n_octaves=6, bins_per_octave=n_features)
    
    chroma = librosa.feature.chroma_cens(y=signal, sr=Fs, 
                                       C=chroma_cqt, 
                                       hop_length=FFT_hop_length, fmin=None, tuning=None, 
                                       n_chroma=n_features, n_octaves=7, bins_per_octave=None, 
                                       cqt_mode='full', window=None, norm=norm_d,
                                       win_len_smooth=31, smoothing_window='hann')

    return chroma

def reencode_audio_to_mp3(input_file):
    base_filename = os.path.splitext(input_file)[0]
    output_file = f"{base_filename}_renc.mp3"

    command = ["ffmpeg", "-y", "-i", input_file, "-acodec", "libmp3lame", output_file]

    print(f"Re-encoding {input_file} to {output_file}...")
    try:
        subprocess.run(command, check=True)
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during audio re-encoding: {e}")
        return None

def save_obj(obj, name, feature_dict):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
    pkl_path = name + '.pkl'
    print("Audio features with shape %s saved to %s" %(feature_dict['features_array_harmonic'].shape, pkl_path))
    return pkl_path
    

def extract_audio_features(
        audio_path,
        output_folder='.',
        re_encode=1,
        n_chunks=1
        ):
    
    ################################################################################################

    #Initial FFT params:
    window_size_ms = 80
    window_step_ms = 40
    sample_rate = 24000

    #HPSS Decomposition:
    FFT_features_to_use = 961
    kernel_sizes = (16,16)     # first value = harmonic, the second = percussive.
    margin_to_use = (3,2)      # first value = harmonic, the second = percussive.

    ################################################################################################

    print('------------------- Extracting audio features... -------------------')

    if re_encode:
        mp3_path = reencode_audio_to_mp3(audio_path)
    else:
        mp3_path = audio_path

    source_file = mp3_path.split('/')[-1]
    pipeline_start = time.time()

    print("Loading %s into Librosa format..." %mp3_path)
    y, Fs = librosa.load(mp3_path, sr=sample_rate)
    print("Loaded audio file of shape %s with Librosa (fs = %d Hz)" %(str(y.shape), Fs))

    print("Running Short-Time Fourier Transform...")
    FFT_window_size = int(window_size_ms * Fs / 1000)
    FFT_hop_length  = int(window_step_ms * Fs / 1000)
    print("FFT_window_size: %d -- FFT_hop_length: %d" %(FFT_window_size, FFT_hop_length))

    # Compute Short-Time Fourier Transform:
    D = librosa.stft(y, 
                    n_fft=FFT_window_size,      #FFT window size
                    hop_length=FFT_hop_length)  #number audio of frames between STFT columns. If unspecified, defaults win_length / 4.

    fourier_features_per_second = D.shape[-1] / (y.shape[0] / Fs)

    print("Computed Fourier matrix, containing %d components for %d timesteps (%.4f features/s)"
        %(D.shape[0], D.shape[1], fourier_features_per_second))

    D_orig = D.copy()
    D = D[:FFT_features_to_use, :]
    print("Clipped the FFT feature spectrum from %s to shape: %s" %(str(D_orig.shape),str(D.shape)))

    print("Computing harmonic/percussive decompositions...")
    st = time.time()
    D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=margin_to_use, kernel_size=kernel_sizes, power=2.0, mask=False)

    print("Decomposition took %.2f seconds" %(time.time()-st))
    print("Done! Decomposed FFT_matrix shape: %s" %str(D_harmonic.shape))
    duration_seconds = D.shape[-1] / fourier_features_per_second

    print("Computing reconstructions...")
    D_harmonic_tmp, D_percussive_tmp = librosa.decompose.hpss(D_orig, margin=margin_to_use, kernel_size=kernel_sizes, power=2.0, mask=False)
    D_harmonic_time   = librosa.istft(D_harmonic_tmp, hop_length=FFT_hop_length, win_length=FFT_window_size)
    print(D_harmonic_time.shape)

    print("Computing log-spaced FFT features...")
    harm_features = extract_magnitudes(np.abs(D_harmonic),  'harm', suppression_power = 4)
    perc_features = extract_magnitudes(np.abs(D_percussive),'perc', suppression_power = 6)
    full_features = extract_magnitudes(np.abs(D), 'full', suppression_power = 2)

    print("Log-spaced FFT features computed! Shape: %s" %str(harm_features.shape))
    pipeline_duration_s = time.time()-pipeline_start
    print("Feature extraction pipeline took %.1f seconds for %d seconds of audio" %(pipeline_duration_s, duration_seconds))
    print("Parsing 1 second of audio would take %d ms" %(1000 * pipeline_duration_s / duration_seconds))

    harmonic_energy = np.linalg.norm(harm_features, axis = 0)

    print("Starting extraction of chroma features...")
    total_l = len(D_harmonic_time)
    chunk_length = total_l // n_chunks
    #Compute the remainder after division:
    remaining_n_samples = total_l - chunk_length*n_chunks
    print("Using %d chunks of %d samples: %d samples will be left at the end" %(n_chunks, chunk_length, remaining_n_samples))

    chroma_features = []
    n_features = harm_features.shape[0]

    if remaining_n_samples > 0:
        chunkable_signal = D_harmonic_time[:-remaining_n_samples]
        remaining_signal = D_harmonic_time[-remaining_n_samples:]

        print("chunkable signal length: %s" %str(chunkable_signal.shape))
        print("remaining signal length: %s" %str(remaining_signal.shape))

        total_frame_indices = np.arange(len(chunkable_signal))
        chunked_frame_indices = np.split(total_frame_indices, n_chunks)
        
        for i in range(n_chunks):
            print("Running chroma %d of %d" %(i, n_chunks))
            indices = chunked_frame_indices[i]
            chroma_features.append(get_chroma(chunkable_signal[indices], n_features, Fs, FFT_hop_length))

        #Final chunk:
        chroma_features.append(get_chroma(remaining_signal, n_features, Fs, FFT_hop_length))

    else:  
        total_frame_indices = np.arange(len(D_harmonic_time))
        chunked_frame_indices = np.split(total_frame_indices, n_chunks)

        for i in range(n_chunks):
            print("Running chroma %d of %d" %(i, n_chunks))
            indices = chunked_frame_indices[i]
            chroma_features.append(get_chroma(D_harmonic_time[indices], n_features, Fs, FFT_hop_length))


    # Combine all the results:
    chroma_features = np.concatenate(chroma_features, axis=1)
    chroma_features = chroma_features[:, :harmonic_energy.shape[0]]
    chroma_features = np.repeat(harmonic_energy[np.newaxis,:], harm_features.shape[0], axis=0) * chroma_features
    chroma_features = np.sqrt(chroma_features)

    print("------------------------------------------------------------------------")
    ################################################################################################

    now = datetime.now() # current date and time
    date_time_str = str(now.strftime("%m-%d-%Y__%H-%M-%S"))
    encoding_info = '%s_audio_features_%d_%d' %(str(source_file.split('.mp3')[0]), window_size_ms, window_step_ms)

    feature_dict = {}
    feature_dict['features_array_full'] = full_features
    feature_dict['features_array_harmonic']   = harm_features
    feature_dict['features_array_percussion'] = perc_features
    feature_dict['features_array_chroma'] = chroma_features

    feature_dict['metadata'] = {}
    feature_dict['metadata']['Audio_Sampling_Rate'] = Fs
    feature_dict['metadata']['window_size_ms'] = window_size_ms
    feature_dict['metadata']['window_step_ms'] = window_step_ms
    feature_dict['metadata']['FFT_features_to_use'] = FFT_features_to_use
    feature_dict['metadata']['duration_seconds'] = duration_seconds
    feature_dict['metadata']['features_per_second'] = full_features.shape[1] / duration_seconds
    feature_dict['metadata']['timestamp_generated'] = date_time_str
    feature_dict['metadata']['song_name'] = mp3_path.split('/')[-1][:-4]
    feature_dict['metadata']['encoding_info'] = encoding_info

    # Some final post-processing:
    log_features_to_use = feature_dict['features_array_harmonic'].shape[0]
    features_full       = feature_dict['features_array_full'].copy()[:log_features_to_use]
    features_harmonic   = feature_dict['features_array_harmonic'].copy()[:log_features_to_use]
    features_percussive = feature_dict['features_array_percussion'].copy()[:log_features_to_use]
    features_chroma     = feature_dict['features_array_chroma'].copy()[:log_features_to_use]

    features_harmonic = smooth(features_harmonic, window_size = 11, polynomial_order = 3, ax = 1, plot=False)
    features_harmonic[features_harmonic < 0.075] *= 0.5
    features_chroma      = smooth(features_chroma, window_size = 21, polynomial_order = 3, ax = 1, plot=False)
    features_percussive  = smooth(features_percussive, window_size = 5, polynomial_order = 3, ax = 1, plot=False)

    feature_dict['features_harmonic_preprocessed']   = features_harmonic
    feature_dict['features_percussive_preprocessed'] = features_percussive
    feature_dict['features_chroma_preprocessed']     = features_chroma

    #Dump to disk:
    filename = os.path.join(output_folder, encoding_info)
    pkl_path = save_obj(feature_dict, filename, feature_dict)
    print("features_per_second: %f" %feature_dict['metadata']['features_per_second'])

    return pkl_path, mp3_path



if __name__ == "__main__":
    pkl_path, mp3_path = extract_audio_features("/home/rednax/Music/versilov.mp3")
