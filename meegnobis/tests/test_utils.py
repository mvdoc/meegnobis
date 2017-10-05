"""Tests for hh.utils"""
import numpy as np
import pytest

from mne import create_info, EpochsArray
from numpy.testing import assert_array_equal, assert_equal
from ..utils import convolve, moving_average
from ..testing import generate_epoch


def test_convolve():
    # first let's make sure it works as regulare np.convolve
    array = np.arange(10)
    filt = [0.5, 0.5]
    assert_array_equal(convolve(array, filt),
                       np.convolve(array, filt, mode='same'))
    # then if we repeat this a couple of times, we should get the same result
    array = np.repeat(array[None, :], 10, axis=0)
    array_npconv = np.convolve(array[0], filt, mode='same')
    array_conv = convolve(array, filt)
    assert_array_equal(array_conv,
                       np.repeat(array_npconv[None, :], 10, axis=0))
    # and if we do it one more time, it should work along the first dimension
    array_3d = np.stack((array, array))
    array_3d_npconv = np.stack((array_conv, array_conv))
    assert_array_equal(array_3d_npconv,
                       convolve(array_3d, filt))
    if np.__version__ < '1.12.0':
        array_4d = np.stack((array_3d, array_3d))
        with pytest.raises(ValueError):
            convolve(array_4d, filt)


def test_moving_average():
    # 800 ms of data at 1000 sfreq
    epoch = generate_epoch(n_times=800)

    epoch_ = moving_average(epoch, 0.02)
    # we want a copy
    assert(epoch is not epoch_)
    assert_equal(epoch.get_data().shape, epoch_.get_data().shape)

    # let's make a simple case to check that convolution works
    ntimes = 10
    nepochs = 4
    nchannels = 4
    sfreq = 2.
    array = np.arange(ntimes)
    array_2d = np.repeat(array[None, :], nchannels, axis=0)
    array_3d = np.repeat(array_2d[None, ...], nepochs, axis=0)
    assert_array_equal(array_3d.shape, (nepochs, nchannels, ntimes))

    info = create_info(nchannels, sfreq, 'mag')
    epochs = EpochsArray(array_3d, info)
    # check it fails if we have too large a window
    with pytest.raises(ValueError):
        moving_average(epochs, twindow=(ntimes+sfreq)/sfreq)
    # but it works in extreme cases as well
    moving_average(epochs, twindow=ntimes/sfreq)
    moving_average(epochs, twindow=1/sfreq)

    # let's check it works across ranges
    for i in range(1, ntimes + 1):
        twindow = 1/sfreq * i
        epochs_conv = moving_average(epochs, twindow)
        filt = np.ones(i)/float(i)
        assert_array_equal(epochs_conv.get_data(),
                           convolve(array_3d, filt))


