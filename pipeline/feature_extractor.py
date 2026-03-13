"""
feature_extractor.py
--------------------
PURPOSE: Extract meaningful features from cleaned EEG epochs.
We compress raw brain waves into numbers that describe brain state.

WHY THIS MATTERS:
Raw EEG = thousands of numbers per second, mostly redundant
Features = 20-50 numbers that capture what the brain is actually doing
This compression is what makes machine learning possible.
"""

import numpy as np
from scipy import signal
from scipy.integrate import simpson
import mne


# ─────────────────────────────────────────────
# FREQUENCY BANDS
# These are the universally agreed ranges in neuroscience
# Every BCI paper in the world uses these same bands
# ─────────────────────────────────────────────
FREQUENCY_BANDS = {
    'delta': (0.5, 4.0),   # Deep sleep, slow wave activity
    'theta': (4.0, 8.0),   # Dreaming, memory replay, REM ← our main target
    'alpha': (8.0, 13.0),  # Relaxed wakefulness, eyes closed
    'beta' : (13.0, 30.0), # Active thinking, concentration
}


def compute_power_spectral_density(epoch_data, sfreq):
    """
    Compute how much power exists at each frequency.

    WHY: Power Spectral Density (PSD) tells us the "energy" 
         at each frequency band. High theta power during sleep 
         = brain is in dreaming state.

    HOW IT WORKS:
    Uses Welch's method — splits the signal into overlapping 
    windows, computes FFT on each, then averages them.
    This gives a stable estimate of the frequency content.

    Parameters:
        epoch_data : numpy array (n_channels, n_samples)
        sfreq      : sampling frequency in Hz

    Returns:
        freqs : array of frequencies
        psd   : power at each frequency (n_channels, n_freqs)
    """
    freqs, psd = signal.welch(
        epoch_data,
        fs=sfreq,
        nperseg=int(sfreq * 4),  # 4-second windows
        noverlap=int(sfreq * 2)  # 50% overlap
    )
    return freqs, psd


def compute_band_power(freqs, psd, band):
    """
    Calculate total power within a specific frequency band.

    WHY: Instead of looking at every individual frequency,
         we sum up the power within each meaningful band.
         This gives us one number per band per channel.

    HOW IT WORKS:
    Uses Simpson's rule (numerical integration) to calculate
    the area under the PSD curve within the band limits.
    Area under the curve = total power in that band.

    Parameters:
        freqs : array of frequencies from PSD
        psd   : power spectrum (n_channels, n_freqs)
        band  : tuple (low_freq, high_freq)

    Returns:
        band_power : total power in band (one value per channel)
    """
    low, high = band
    
    # Find which frequency indices fall within our band
    freq_mask = (freqs >= low) & (freqs <= high)
    
    # Integrate (sum up) the power within those frequencies
    band_power = simpson(psd[:, freq_mask], x=freqs[freq_mask])
    
    return band_power


def extract_all_band_powers(epoch_data, sfreq):
    """
    Extract power for ALL frequency bands at once.

    WHY: This gives us a complete picture of what the brain
         is doing across all frequency ranges simultaneously.
         This is the most commonly used feature in BCI research.

    Returns:
        features : dict with band names as keys and 
                   power arrays as values
    """
    freqs, psd = compute_power_spectral_density(epoch_data, sfreq)
    
    features = {}
    for band_name, band_range in FREQUENCY_BANDS.items():
        features[band_name] = compute_band_power(freqs, psd, band_range)
    
    return features


def compute_relative_band_power(features):
    """
    Compute relative power: each band as percentage of total.

    WHY: Absolute power varies hugely between people and recording
         sessions (electrode placement, skull thickness, etc).
         Relative power is normalized — it tells us the PROPORTION
         of energy in each band, which is comparable across people.

    Example:
        Absolute theta power = 10 μV²
        Total power = 100 μV²
        Relative theta power = 10% ← this is comparable across subjects
    """
    # Sum all band powers per channel
    total_power = sum(features[band] for band in FREQUENCY_BANDS.keys())
    
    relative_features = {}
    for band_name in FREQUENCY_BANDS.keys():
        relative_features[f'relative_{band_name}'] = (
            features[band_name] / (total_power + 1e-10)  # +1e-10 avoids division by zero
        )
    
    return relative_features


def compute_theta_alpha_ratio(features):
    """
    Compute Theta/Alpha ratio.

    WHY: This ratio is a well-known marker of drowsiness and
         dream-like states. High theta + low alpha = dreaming.
         This single number captures the dreaming state better
         than looking at theta or alpha alone.

    This is an example of a DERIVED FEATURE — combining two
    basic features into one more meaningful number.
    """
    ratio = features['theta'] / (features['alpha'] + 1e-10)
    return ratio


def extract_features_from_epoch(epoch_data, sfreq):
    """
    Master function — extract ALL features from one epoch.

    This is what we'll call for every 30-second REM window.
    Returns a flat feature vector ready for the ML model.

    Parameters:
        epoch_data : numpy array (n_channels, n_samples)
        sfreq      : sampling frequency

    Returns:
        feature_vector : 1D numpy array of all features
        feature_names  : list of feature names (for interpretation)
    """
    all_features = []
    feature_names = []

    # 1. Absolute band powers
    band_powers = extract_all_band_powers(epoch_data, sfreq)
    for band_name, power in band_powers.items():
        for ch_idx, ch_power in enumerate(power):
            all_features.append(ch_power)
            feature_names.append(f'{band_name}_ch{ch_idx}')

    # 2. Relative band powers
    relative_powers = compute_relative_band_power(band_powers)
    for feat_name, power in relative_powers.items():
        for ch_idx, ch_power in enumerate(power):
            all_features.append(ch_power)
            feature_names.append(f'{feat_name}_ch{ch_idx}')

    # 3. Theta/Alpha ratio
    theta_alpha = compute_theta_alpha_ratio(band_powers)
    for ch_idx, ratio in enumerate(theta_alpha):
        all_features.append(ratio)
        feature_names.append(f'theta_alpha_ratio_ch{ch_idx}')

    feature_vector = np.array(all_features)

    return feature_vector, feature_names


if __name__ == "__main__":

    import os
    from pipeline.data_loader import load_sleep_recording
    from pipeline.preprocessor import bandpass_filter, remove_powerline_noise

    # Paths
    data_dir       = os.path.join(os.path.expanduser("~"), "mne_data", "physionet-sleep-data")
    psg_path       = os.path.join(data_dir, "SC4001E0-PSG.edf")
    hypnogram_path = os.path.join(data_dir, "SC4001EC-Hypnogram.edf")

    # Load and preprocess
    raw, annotations = load_sleep_recording(psg_path, hypnogram_path)
    raw = remove_powerline_noise(raw)
    raw = bandpass_filter(raw)

    # Get sampling frequency
    sfreq = raw.info['sfreq']
    print(f"Sampling frequency: {sfreq} Hz")

    # Get first 30 seconds of data as a test epoch
    data = raw.get_data()
    epoch_data = data[:, :int(sfreq * 30)]  # first 30 seconds

    print(f"Epoch shape: {epoch_data.shape} (channels x samples)")

    # Extract features
    feature_vector, feature_names = extract_features_from_epoch(epoch_data, sfreq)

    print(f"\nExtracted {len(feature_vector)} features")
    print("\nFirst 10 features:")
    for name, value in zip(feature_names[:10], feature_vector[:10]):
        print(f"  {name:35}: {value:.6f}")

    print("\nFeature extraction complete!")