"""
test_preprocessor.py
--------------------
PURPOSE: Unit tests for the preprocessing pipeline.

WHY TESTS MATTER:
Tests verify that each component works correctly in isolation.
When you change code later, tests tell you immediately if
something broke. This is called regression testing.

HOW TO RUN:
    pytest tests/
    pytest tests/test_preprocessor.py -v  (verbose)
"""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mne
from pipeline.preprocessor import bandpass_filter, remove_powerline_noise


# ─────────────────────────────────────────
# FIXTURES
# A fixture is reusable test data.
# pytest automatically injects fixtures
# into any test function that requests them.
# ─────────────────────────────────────────

@pytest.fixture
def sample_raw():
    """
    Create a synthetic EEG recording for testing.
    
    WHY SYNTHETIC DATA:
    We don't want tests to depend on downloaded files.
    Synthetic data is always available and controlled.
    We know exactly what it contains so we can verify outputs.
    """
    sfreq    = 100.0  # 100 Hz sampling rate
    duration = 60.0   # 60 seconds
    n_channels = 3

    # Create synthetic EEG-like data
    # Mix of brain frequencies + noise
    times = np.arange(0, duration, 1/sfreq)
    data  = np.zeros((n_channels, len(times)))

    for ch in range(n_channels):
        # Add delta wave (2 Hz)
        data[ch] += 50e-6 * np.sin(2 * np.pi * 2 * times)
        # Add theta wave (6 Hz)
        data[ch] += 30e-6 * np.sin(2 * np.pi * 6 * times)
        # Add alpha wave (10 Hz)
        data[ch] += 20e-6 * np.sin(2 * np.pi * 10 * times)
        # Add noise
        data[ch] += 5e-6 * np.random.randn(len(times))

    # Create MNE info object
    ch_names = [f'EEG_{i}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info     = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types
    )

    raw = mne.io.RawArray(data, info)
    return raw


# ─────────────────────────────────────────
# TESTS
# Each test function starts with "test_"
# pytest automatically finds and runs them
# ─────────────────────────────────────────

def test_bandpass_filter_returns_raw(sample_raw):
    """
    Test that bandpass filter returns an MNE Raw object.
    
    WHY: If the return type changes, everything downstream breaks.
    This is the most basic sanity check.
    """
    filtered = bandpass_filter(sample_raw)
    assert isinstance(filtered, mne.io.RawArray) or \
           hasattr(filtered, 'get_data'), \
           "bandpass_filter must return MNE Raw object"


def test_bandpass_filter_preserves_shape(sample_raw):
    """
    Test that filtering doesn't change data dimensions.
    
    WHY: Filters should clean the signal, not resize it.
    Shape changes would break all downstream processing.
    """
    original_shape = sample_raw.get_data().shape
    filtered       = bandpass_filter(sample_raw)
    filtered_shape = filtered.get_data().shape

    assert original_shape == filtered_shape, \
        f"Shape changed: {original_shape} → {filtered_shape}"


def test_bandpass_filter_removes_high_freq(sample_raw):
    """
    Test that bandpass filter actually removes high frequencies.
    
    WHY: This verifies the filter is actually working,
    not just passing data through unchanged.
    
    METHOD: Compare power above cutoff before and after filtering.
    """
    from scipy import signal as scipy_signal

    data_before = sample_raw.get_data()[0]
    filtered    = bandpass_filter(sample_raw, high_freq=30.0)
    data_after  = filtered.get_data()[0]

    sfreq = sample_raw.info['sfreq']

    # Compute power above 35 Hz (above our cutoff)
    freqs_b, psd_b = scipy_signal.welch(data_before, fs=sfreq)
    freqs_a, psd_a = scipy_signal.welch(data_after,  fs=sfreq)

    high_freq_mask    = freqs_b > 35
    power_before_high = np.mean(psd_b[high_freq_mask])
    power_after_high  = np.mean(psd_a[high_freq_mask])

    assert power_after_high < power_before_high, \
        "Bandpass filter should reduce high frequency power"


def test_bandpass_filter_preserves_brain_frequencies(sample_raw):
    """
    Test that filtering preserves brain-relevant frequencies.
    
    WHY: We don't want to destroy the signal we care about.
    After filtering, theta power (4-8 Hz) should still be present.
    """
    from scipy import signal as scipy_signal

    filtered  = bandpass_filter(sample_raw, low_freq=0.5, high_freq=45.0)
    data      = filtered.get_data()[0]
    sfreq     = sample_raw.info['sfreq']

    freqs, psd = scipy_signal.welch(data, fs=sfreq)

    # Check theta band (4-8 Hz) has meaningful power
    theta_mask  = (freqs >= 4) & (freqs <= 8)
    theta_power = np.mean(psd[theta_mask])

    assert theta_power > 0, \
        "Bandpass filter destroyed theta band power"


def test_remove_powerline_noise_returns_raw(sample_raw):
    """Test that powerline removal returns valid Raw object."""
    # Note: our implementation uses bandpass which is fine
    result = remove_powerline_noise(sample_raw)
    assert hasattr(result, 'get_data'), \
        "remove_powerline_noise must return MNE Raw object"


def test_remove_powerline_noise_preserves_shape(sample_raw):
    """Test that powerline removal preserves data shape."""
    original_shape = sample_raw.get_data().shape
    result         = remove_powerline_noise(sample_raw)
    result_shape   = result.get_data().shape

    assert original_shape == result_shape, \
        f"Shape changed: {original_shape} → {result_shape}"


def test_filter_chain(sample_raw):
    """
    Test the full filtering chain works without errors.
    
    WHY: Integration test — verifies components work together.
    Individual unit tests passing doesn't guarantee
    the chain works. This catches interaction bugs.
    """
    try:
        result = remove_powerline_noise(sample_raw)
        result = bandpass_filter(result)
        data   = result.get_data()
        assert data is not None
        assert not np.any(np.isnan(data)), \
            "Filter chain produced NaN values"
        assert not np.any(np.isinf(data)), \
            "Filter chain produced infinite values"
    except Exception as e:
        pytest.fail(f"Filter chain failed: {e}")