"""Module containing various utilities to test code"""

import numpy as np
from mne import create_info
from mne.epochs import EpochsArray

rng = np.random.RandomState(42)


def generate_epoch(n_epochs_cond=5, n_channels=10, n_times=20,
                   sfreq=1000., n_conditions=2):
    """Generate an epoch instance with an arbitrary number of timepoints,
    channels, trials, and conditions"""

    n_epochs = n_epochs_cond * n_conditions
    data = rng.randn(n_epochs, n_channels, n_times)
    conditions = np.repeat(np.arange(n_conditions), n_epochs_cond)
    events = np.array([np.arange(n_epochs),
                       [0] * n_epochs,
                       conditions]).T
    info = create_info(n_channels, sfreq, 'mag')
    epochs = EpochsArray(data, info, events)
    return epochs
