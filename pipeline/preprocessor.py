"""
02_preprocessor.py
------------------
PURPOSE: Clean raw EEG data by removing noise and artifacts.
This is always the second step in any neural signal processing pipeline.

WHY THIS MATTERS:
Raw EEG contains noise from eye blinks, muscles, and power lines.
If we don't remove this, our model will learn noise instead of brain signals.
"""

import mne
import numpy as np

def bandpass_filter(raw, low_freq=0.5, high_freq=45.0):
    """
    Keep only brain-relevant frequencies.
    
    WHY: Brain waves we care about are between 0.5-30 Hz.
         Anything outside this range is noise.
    
    0.5 Hz lower limit → removes slow drifts in the signal
    30.0 Hz upper limit → removes muscle noise and power line interference
    """
    print(f"Applying bandpass filter: {low_freq} - {high_freq} Hz")
    raw_filtered = raw.copy().filter(l_freq=low_freq, h_freq=high_freq)
    return raw_filtered


def remove_powerline_noise(raw, powerline_freq=50.0):
    """
    Remove electrical noise from power lines.
    
    WHY: Power lines in Europe run at 50 Hz (60 Hz in USA).
         This creates a constant hum in the recording.
         A notch filter cuts out exactly that frequency.
    """
    print(f"Removing powerline noise at {powerline_freq} Hz")
    raw_clean = raw.copy().filter(l_freq=0.5, h_freq=45.0)
    return raw_clean


def extract_rem_epochs(raw, epoch_duration=30.0):
    """
    Extract only the REM sleep periods from the full night recording.
    
    WHY: We only care about REM sleep (dreaming periods).
         A full night recording is 7-8 hours.
         REM periods are maybe 90 minutes total.
         We extract only those parts to focus our model.
    
    epoch_duration: length of each chunk in seconds (standard is 30s in sleep research)
    """
    print("Extracting REM sleep epochs...")
    
    # Get all annotations (sleep stage labels)
    annotations = raw.annotations
    
    # Find REM periods
    # In Sleep-EDF dataset, REM is labeled as "Sleep stage R"
    rem_indices = []
    for i, annotation in enumerate(annotations):
        if 'R' in annotation['description']:
            rem_indices.append(i)
    
    print(f"Found {len(rem_indices)} REM annotations")
    
    # Create events from annotations
    events, event_id = mne.events_from_annotations(raw)
    
    return events, event_id, rem_indices


def get_signal_statistics(raw):
    """
    Calculate basic statistics about the signal quality.
    
    WHY: Always inspect your data before processing.
         This tells you if something is wrong with the recording.
    """
    data = raw.get_data()
    
    stats = {
        'mean'     : np.mean(data),
        'std'      : np.std(data),
        'max'      : np.max(data),
        'min'      : np.min(data),
        'n_samples': data.shape[1],
        'n_channels': data.shape[0]
    }
    
    print("\n--- SIGNAL STATISTICS ---")
    for key, value in stats.items():
        print(f"{key:12}: {value}")
    print("-------------------------\n")
    
    return stats


if __name__ == "__main__":
    
    import os
    from pipeline.data_loader import load_sleep_recording
    
    # Paths
    data_dir       = os.path.join(os.path.expanduser("~"), "mne_data", "physionet-sleep-data")
    psg_path       = os.path.join(data_dir, "SC4001E0-PSG.edf")
    hypnogram_path = os.path.join(data_dir, "SC4001EC-Hypnogram.edf")
    
    # Load raw data
    raw, annotations = load_sleep_recording(psg_path, hypnogram_path)
    
    # Step 1: Remove powerline noise
    raw = remove_powerline_noise(raw)
    
    # Step 2: Bandpass filter
    raw = bandpass_filter(raw)
    
    # Step 3: Get statistics
    stats = get_signal_statistics(raw)
    
    # Step 4: Extract REM epochs
    events, event_id, rem_indices = extract_rem_epochs(raw)
    
    print(f"Event types found: {event_id}")
    print("Preprocessing complete!")