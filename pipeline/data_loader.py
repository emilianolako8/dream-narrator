"""
01_data_loader.py
-----------------
PURPOSE: Load raw EEG sleep recordings from EDF files into Python.
This is always the first step in any neural signal processing pipeline.
"""

import mne
import os

def load_sleep_recording(psg_path, hypnogram_path):
    """
    Load a sleep recording and its sleep stage annotations.
    
    Parameters:
        psg_path       : path to the PSG (brain recording) EDF file
        hypnogram_path : path to the hypnogram (sleep stages) EDF file
    
    Returns:
        raw        : the EEG recording object
        annotations: sleep stage labels
    """

    print(f"Loading EEG recording from: {psg_path}")
    
    # Load the brain recording
    # preload=True means load everything into memory (fine for small files)
    raw = mne.io.read_raw_edf(psg_path, preload=True)
    
    print(f"Loading sleep stage annotations from: {hypnogram_path}")
    
    # Load the sleep stage labels
    annotations = mne.read_annotations(hypnogram_path)
    
    # Attach the labels to the recording
    raw.set_annotations(annotations)
    
    return raw, annotations


def print_recording_info(raw):
    """
    Print basic information about the recording.
    Useful for understanding what's inside the file.
    """
    print("\n--- RECORDING INFO ---")
    print(f"Duration        : {raw.times[-1] / 60:.1f} minutes")
    print(f"Sampling rate   : {raw.info['sfreq']} Hz")
    print(f"Channels        : {raw.ch_names}")
    print(f"Number of channels: {len(raw.ch_names)}")
    print("----------------------\n")


if __name__ == "__main__":
    
    # Paths to the downloaded data
    data_dir = os.path.join(os.path.expanduser("~"), "mne_data", "physionet-sleep-data")
    
    psg_path       = os.path.join(data_dir, "SC4001E0-PSG.edf")
    hypnogram_path = os.path.join(data_dir, "SC4001EC-Hypnogram.edf")
    
    # Load the data
    raw, annotations = load_sleep_recording(psg_path, hypnogram_path)
    
    # Print info
    print_recording_info(raw)
    
    print("Data loaded successfully!")
    print(f"Total annotations: {len(annotations)}")