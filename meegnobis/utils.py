import mne
from mne.channels.layout import _pair_grad_sensors, _merge_grad_data
import numpy as np
from os.path import join as pjoin, abspath, dirname


def get_connectivity(ch_type):
    here = abspath(dirname(__file__))
    layout = {
        'mag': 'neuromag306mag',
        'grad': 'neuromag306planar',
        'cmb': pjoin(here, 'neuromag306cmb_neighb.mat')
    }

    connectivity, ch_names = mne.channels.read_ch_connectivity(layout[ch_type])
    return connectivity, ch_names


def combineplanar(evoked):
    """Combines planar gradiometer information, discarding magnetometers"""
    # get pairwise picks
    picks = _pair_grad_sensors(evoked.info, topomap_coords=False)
    # sort them
    picks = sorted(picks)
    # and get ch_names
    ch_names = [evoked.ch_names[i] for i in picks]
    # get them in order
    evoked_ = evoked.copy().pick_channels(ch_names)

    # merge the data
    data_merged = _merge_grad_data(evoked_.data)
    # name them as per fieldtrip syntax

    def lbl(ch1, ch2):
        return '+'.join((ch1, ch2[3:]))
    ch_names_merged = [lbl(*sorted(chs)) for chs in
                       zip(ch_names[:-1:2], ch_names[1::2])]

    # make it evoked
    info_merged = mne.create_info(ch_names_merged, evoked_.info['sfreq'])
    evoked_merged = mne.EvokedArray(data_merged, info_merged,
                                    tmin=evoked_.times[0])

    return evoked_merged


def convolve(array, filt):
    """A vectorized version of np.convolve, lifted from the example

    Parameters
    ----------
    array : np.ndarray
        multi dimensional array on which filt will be applied; it is vectorized
        along the last two dimensions, so for Epochs the dimensions should be
        (channel, epoch, time) in order to average across time
    filt : np.ndarray
        filter to be applied on array

    Returns
    -------
    array_convolve : np.ndarray (shape of array)
        the convolved array, computed using mode = 'same'
    """
    if np.__version__ >= '1.12.0':
        convvect = np.vectorize(lambda x, y: np.convolve(x, y, mode='same'),
                                signature='(n),(m)->(k)',
                                doc=__doc__)
        return convvect(array, filt)
    else:
        # we need to vectorize manually
        if array.ndim > 3:
            raise ValueError(
                "This function doesn't work on multidimensional "
                "arrays with more than 3 dimensions under numpy < 1.12.0")
        conv = lambda x: np.convolve(x, filt, mode='same')
        if array.ndim == 1:
            return conv(array)
        else:
            array_convolve = []
            for array0 in array:
                if array0.ndim == 1:
                    array_convolve.append(conv(array0))
                else:
                    array_convolve.append(
                        np.apply_along_axis(conv, 1, array0)
                    )
            return np.stack(array_convolve)


def moving_average(epoch, twindow):
    """

    Parameters
    ----------
    epoch : mne.Epochs object
    twindow : float; time window in s for which to compute a moving average

    Returns
    -------
    epoch_avg : mne.Epochs object
        a new Epochs object containing the epochs after having applied the
        moving average in time
    """
    epoch_ = epoch.copy()
    nsamples = np.ceil(float(twindow) * epoch.info['sfreq'])
    if nsamples > len(epoch.times):
        raise ValueError("Cannot compute moving average with time window "
                         "{0} and epoch length "
                         "{1}".format(twindow,
                                      epoch.times[-1] - epoch.times[0]))
    filt = np.ones(int(nsamples))/nsamples
    data = epoch_._data
    # by default we have (epochs, channels, times);
    # we need (channels, epochs, times)
    data = np.transpose(data, [1, 0, 2])
    data = convolve(data, filt)
    data = np.transpose(data, [1, 0, 2])
    epoch_._data = data
    return(epoch_)
