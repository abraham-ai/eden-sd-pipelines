import os, sys, random, pickle, shutil, zipfile
import numpy as np
from scipy.signal import savgol_filter


#########################################################################

chroma_fraction         = 0.30
harmonic_power_f        = 1.5
harmonic_decay_f        = 0.80        # (0=fast, 0.9999 = slow)
harmonic_smooth_window  = 25
fade_to_black_s         = 0

nr_beat_bins = 3
percus_push_factor = 12.       # How much does a percussive beat push the perceptual change
base_decay = 0.40              # How slowely does a 'base-push' decay back to zero: slow_base[:, i] = max(slow_base[:, i-1] * decay, base[:, i])
min_v = 0.15   #  What is the minimum motion speed in latent space? (Relative to #nodes / minute setting)

outlier_removal_fraction = 0.70  # Beats with an amplitude above this percentile will get clipped (making softer beats more visually aparent)
percussive_threshold     = 0.40  # Anything lower than this fraction will get squashed to zero

# see also Planner.prep_audio_signals_for_render()

#########################################################################


def load_zip(zip_path):
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    try: shutil.rmtree('tmp_unzip/')
    except: pass
    zip_ref.extractall('tmp_unzip/')
    zip_ref.close()
    with open('tmp_unzip/features.pkl', 'rb') as f:
        feature_dict = pickle.load(f)

    audio_path = 'tmp_unzip/music.mp3'
    return feature_dict, audio_path

import matplotlib.pyplot as plt

def plot_signal(signal, range = None, title = '', path = None):
    if range is not None:
      if range[1] < 1:
        min_t, max_t = int(len(signal)*range[0]), int(len(signal)*range[1])
      else:
        min_t, max_t = range[0], range[1]

      signal = signal[min_t:max_t]

    plt.figure(figsize = (24,6))
    plt.plot(signal)
    plt.ylim([0,np.max(signal)*1.1])
    plt.title(title, fontsize = 12)
    if path is not None:
      plt.savefig(path)
    else:
      plt.show()

def normalize_full_signal(s):
    s = s - np.min(s)
    return s/np.max(s)

def smooth(signal, window_size = 5, polynomial_order = 3, axis = 1, plot=False):
    smoothed_signal = savgol_filter(signal, window_size, polynomial_order, axis=axis)
    return smoothed_signal

def bin_features(features, nr_bins=10):
    binned_features = np.zeros((nr_bins, features.shape[1]))
    bins = np.linspace(0, features.shape[0], num=nr_bins+1).astype(int)
    for i, bin_start in enumerate(bins[:-1]):
        binned_features[i, :] = np.mean(features[bin_start:bins[i+1], :], axis=0)
    return binned_features

def add_slowness(features, decay):
    # Add slowness to a feature signal by applying a decay factor such that
    #feature_value = max(prev_slow_features[i]*decay, feature_value)
    slow_features = np.zeros(features.shape)

    for i in range(1, features.shape[1]):
        slow_features[:, i] = np.maximum(slow_features[:, i-1] * decay, features[:, i])
    return slow_features

def get_diffs(features):
    norms = np.zeros(features.shape)
    zeros = np.zeros(features.shape[0])
    for i in range(1,features.shape[1]-1):
        f1 = features[:,i]
        f2 = features[:,i+1]
        norms[:,i] = np.linalg.norm(f2-f1)
        #norms[:,i] = np.maximum(zeros, f2-f1)
    return norms

import pandas as pd
def warp_signal(input_signal, fps, min_v = 0.25, power = 1, decay = 0.9, clip_fractions = [.1, .7], outlier_p = 3, end_slowdown_s = 5, plot=True, ranges = None, smooth_window = 71): 
  harmonic_energy = input_signal.copy()
  harmonic_energy = np.abs(get_diffs(harmonic_energy)) 
  harmonic_energy = add_slowness(harmonic_energy, decay = decay)
  
  harmonic_energy = np.sum(harmonic_energy, axis = 0)
  harmonic_energy = normalize_full_signal(harmonic_energy)

  ts = pd.Series(harmonic_energy)
  rolling_std = ts.rolling(window=smooth_window).std()
  rolling_std = rolling_std.fillna(rolling_std.mean()).values
  harmonic_energy = harmonic_energy * rolling_std
  
  harmonic_energy = smooth(harmonic_energy, window_size = smooth_window, polynomial_order=3, axis = 0)
  harmonic_energy = np.clip(harmonic_energy, np.percentile(harmonic_energy, outlier_p), np.percentile(harmonic_energy, 100-outlier_p))
  
  #Normalize and power up:
  harmonic_energy = harmonic_energy - np.min(harmonic_energy)
  harmonic_energy = (harmonic_energy / np.max(harmonic_energy)) + 1
  harmonic_energy = np.power(harmonic_energy, power)
  harmonic_energy = normalize_full_signal(harmonic_energy)
  
  harmonic_energy = np.clip(harmonic_energy, clip_fractions[0], clip_fractions[1])
  harmonic_energy = normalize_full_signal(harmonic_energy)
  
  # Therefore, the current minimum should become: 
  offset = min_v * np.mean(harmonic_energy) / (1-min_v)
  harmonic_energy = harmonic_energy + offset

  #End slowdown:
  if end_slowdown_s > 0:
    end_slowdown_samples = int(end_slowdown_s * fps)
    fade = np.ones((harmonic_energy.shape))
    fade[-end_slowdown_samples:] = np.linspace(1,0,end_slowdown_samples)
    harmonic_energy = harmonic_energy*fade
  harmonic_energy = harmonic_energy / np.mean(harmonic_energy)

  if plot:
    plot_signal(harmonic_energy, range = ranges, title = 'Tuned Energy signal', path = 'energy.jpg')
  return harmonic_energy



def create_audio_features(audio_path, verbose = 0):
  if '.zip' in audio_path:
    audio_features, audio_path = load_zip(audio_path)
  elif isinstance(audio_path, tuple):
    pickle_path, audio_path = audio_path
    with open(pickle_path, 'rb') as f:
      audio_features = pickle.load(f)
  else:
     raise ValueError('Audio path should be a zip file or a tuple of (features_pickle_path, audio_mp3_path)')

  if verbose > 0:
    for key in audio_features.keys():
      print(key)
      if not isinstance(audio_features[key], np.ndarray):
        print(audio_features[key])
      else:
        print(audio_features[key].shape)


  fps = audio_features['metadata']['features_per_second']
  harmonic_features = audio_features['features_array_harmonic'].copy()
  percussive_features = audio_features['features_array_percussion'].copy()
  try:
    chroma_features = audio_features['features_array_chroma'].copy()
  except:
    chroma_features = np.zeros(harmonic_features.shape)
  chroma_fraction = 0.0
  
  # Remove any nan values:
  if np.isnan(harmonic_features).any() or np.isnan(percussive_features).any():
      np.nan_to_num(harmonic_features, copy=False)
      np.nan_to_num(percussive_features, copy=False)

  harmonic_features = (1-chroma_fraction)*harmonic_features + chroma_fraction*chroma_features

  harmonic_energy = warp_signal(harmonic_features, fps, min_v = min_v, power = harmonic_power_f, 
    clip_fractions = [0.05, outlier_removal_fraction], 
    decay = harmonic_decay_f, 
    end_slowdown_s = 0*fade_to_black_s, 
    smooth_window = harmonic_smooth_window,
    ranges = None, plot=1)

  harmonic_energy_orig = harmonic_energy.copy()
  final_percus_features = bin_features(percussive_features, nr_bins=nr_beat_bins)

  # Clean up beat signals:
  for i in range(nr_beat_bins):
      max_beat = np.max(final_percus_features[i, :])
      final_percus_features[i, :] = np.clip(final_percus_features[i, :], 0, max_beat*outlier_removal_fraction)
      final_percus_features[i, :] = final_percus_features[i,:] - np.min(final_percus_features[i, :])

  # Normalize each percussive component independently:
  final_percus_features = final_percus_features - np.min(final_percus_features, axis=1)[:, np.newaxis]
  final_percus_features = final_percus_features / np.max(final_percus_features, axis=1)[:, np.newaxis]

  # These cutoff fractions are relative to the maximum = 1 (for each percussive component!!)
  # Clip outliers:
  for i in range(nr_beat_bins):
      final_percus_features[i, :] = np.clip(final_percus_features[i, :], 0.1, outlier_removal_fraction)

  # Normalize each percussive component independently:
  final_percus_features = final_percus_features - np.min(final_percus_features, axis=1)[:, np.newaxis]
  final_percus_features = final_percus_features / np.max(final_percus_features, axis=1)[:, np.newaxis]
  print("Final percussive signal of shape %s ready!" % str(final_percus_features.shape))

  '''
  Create final latent 'push' signal:
  '''
  harmonic_energy = harmonic_energy_orig.copy()

  #### MID ####
  treble = final_percus_features[1,:]
  treble = smooth(treble, axis = 0, polynomial_order = 2, window_size = 3)
  treble[treble < percussive_threshold] = treble[treble < percussive_threshold]**4
  final_percus_features[1,:] = treble

  #### SNARE ####
  snare = final_percus_features[-1,:]
  snare[snare < percussive_threshold] = snare[snare < percussive_threshold]**4
  final_percus_features[-1,:] = snare

  #### BASE ####
  # Slow down the variance in the base signal:
  base = final_percus_features[0, :]
  base = normalize_full_signal(base)
  base[base<percussive_threshold] = base[base<percussive_threshold]**4
  #base = base**2
  base = add_slowness(base[np.newaxis, :], decay=base_decay)[0]
  base = normalize_full_signal(base)    
  base = smooth(base, axis = 0, window_size = 5)
  base = normalize_full_signal(base)
  final_percus_features[0,:] = base

  final = percus_push_factor*base + harmonic_energy
  #final = percus_push_factor*snare + harmonic_energy

  final = normalize_full_signal(final)
  final = final + (min_v * np.mean(final) / (1-min_v))
  final = final / np.mean(final)
  harmonic_energy = final

  #plot_signal(harmonic_energy, range = None, title = 'harmonic_energy', path = "harmonic_energy.png")
  #plot_signal(base, range = None, title = 'base', path = "base.png")

  return harmonic_energy, final_percus_features, audio_features['metadata'], audio_path

