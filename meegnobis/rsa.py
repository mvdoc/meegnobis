"""Module containing functions for temporal RSA"""
import mne
import numpy as np
from itertools import islice
from mne import EpochsArray
from mne.cov import compute_whitener
from numpy.testing import assert_array_equal
from joblib.parallel import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedShuffleSplit


def fisher_correlation(x, y):
    """Computes correlation between x and y and fisher-transforms the output

    See scipy.spatial.cdist for more information"""
    return np.arctanh(1. - cdist(x, y, metric='correlation'))


def mean_group(array, targets):
    """Average rows of array according to the unique values of targets

    Arguments
    ---------
    array : (n_samples, n_features, n_times)
        array containing the epoch data
    targets : (n_samples, )
        array containing targets for the samples

    Returns
    -------
    avg_array : (n_unique_targets, n_features, n_times)
        array containing the average within each group
    unique_targets : (n_unique_targets, )
        array containing the unique targets, corresponding to the order in
        avg_array
    """
    unique_targets = np.unique(targets)
    n_unique_targets = len(unique_targets)
    avg_array = np.zeros((n_unique_targets, array.shape[1], array.shape[2]))
    for i, t in enumerate(unique_targets):
        mask = targets == t
        avg_array[i, ...] = array[mask, ...].mean(axis=0)
    return avg_array, unique_targets


def _compute_fold(epoch, targets, train, test, metric_fx=fisher_correlation,
                  cv_normalize_noise=None):
    """Computes pairwise metric across time for one fold

    Arguments
    ---------
    epoch : instance of mne.Epoch
    targets : array (n_trials,)
        target (condition) for each trials; they must be integers
    train : array-like of int
        indices of the training data
    test : array-like of int
        indices of the testing data
    metric_fx : function(x, y, targets_train, targets_test)
        any function that returns a scalar given two arrays.
        This condition must hold: metric_fx(x, y) == metric_fx(y, x)
    cv_normalize_noise : str | None (default None)
        Multivariate normalize the trials before computing the distance between
        pairwise conditions.
        Valid values are 'epoch' | 'baseline' | None:
            - 'epoch' computes the covariance matrix on the entire epoch;
            - 'baseline' uses only the baseline condition; requires to pass an
              array times

    Returns
    -------
    rdms: array (n_pairwise_targets, n_times, n_times)
    targets_pairs : list of len n_pairwise_targets
        the labels corresponding to each element in rdms
    """

    targets = np.asarray(targets)
    if targets.dtype != np.dtype('int64'):
        raise ValueError("targets must be integers, "
                         "not {0}".format(targets.dtype))

    if cv_normalize_noise not in ('epoch', 'baseline', None):
        raise ValueError(
            "cv_normalize_noise must be one of {0}".format(
                ('epoch', 'baseline', None)))

    # preallocate array
    unique_targets = np.unique(targets)
    n_unique_targets = len(unique_targets)
    n_times = len(epoch.times)
    n_triu = int(n_unique_targets * (n_unique_targets - 1) / 2
                 + n_unique_targets)
    rdms = np.zeros((n_triu, n_times, n_times))

    # impose conditions in epoch as targets, so that covariance
    # matrix is computed within each target
    events_ = epoch.events.copy()
    events_[:, 2] = targets
    epoch_ = EpochsArray(epoch._data, info=epoch.info, events=events_)
    epoch_.baseline = epoch.baseline

    # get training and testing data
    epoch_train = epoch_.copy()[train]
    targets_train = targets[train]
    assert(len(epoch_train) == len(targets_train))
    epoch_test = epoch_.copy()[test]
    targets_test = targets[test]
    assert(len(epoch_test) == len(targets_test))

    # perform multi variate noise normalization
    if cv_normalize_noise:
        tmax = 0. if cv_normalize_noise == 'baseline' else None
        cov_train = mne.compute_covariance(epoch_train,
                                           tmax=tmax, method='shrunk')
        W_train, ch_names = compute_whitener(cov_train, epoch_train.info)
        # whiten both training and testing set
        epo_data_train = np.array([np.dot(W_train, e)
                                   for e in epoch_train.get_data()])
        epo_data_test = np.array([np.dot(W_train, e)
                                  for e in epoch_test.get_data()])
    else:
        epo_data_train = epoch_train.get_data()
        epo_data_test = epoch_test.get_data()

    # average within train and test for each target
    epo_data_train, targets_train = mean_group(epo_data_train, targets_train)
    epo_data_test, targets_test = mean_group(epo_data_test, targets_test)
    # the targets should be the same in both training and testing set or else
    # we are correlating weird things together
    assert_array_equal(targets_train, targets_test)
    # compute pairwise metric
    idx = 0
    for t1 in range(n_times):
        for t2 in range(t1, n_times):
            # XXX: we should check whether the metric is symmetric to avoid
            # recomputing everything
            rdm = metric_fx(epo_data_train[..., t1],
                            epo_data_test[..., t2])
            # now we need to impose symmetry
            rdm += rdm.T
            rdm /= 2.
            # now store only the upper triangular matrix
            rdms[:, t1, t2] = rdm[np.triu_indices_from(rdm)]
            rdms[:, t2, t1] = rdms[:, t1, t2]
            idx += 1
    # return also the pairs labels. since we are taking triu, we loop first
    # across rows
    targets_pairs = []
    for i_tr, tr_lbl in enumerate(targets_train):
        for _, te_lbl in enumerate(targets_test[i_tr:]):
            targets_pairs.append('+'.join(map(str, [tr_lbl, te_lbl])))

    return rdms, targets_pairs


def compute_temporal_rdm(epoch, targets, metric_fx=fisher_correlation,
                         cv=StratifiedShuffleSplit(n_splits=10, test_size=0.5),
                         cv_normalize_noise=None,
                         n_jobs=1, batch_size=200):
    """Computes pairwise metric across time

    Arguments
    ---------
    epoch : instance of mne.Epoch
    targets : array (n_trials,)
        target (condition) for each trials
    metric_fx : function(x, y, train_targets, test_targets)
        any function that returns a scalar given two arrays.
        This condition must hold: metric_fx(x, y) == metric_fx(y, x)
    cv : instance of sklearn cross-validator
        (default StratifiedShuffleSplit(n_splits=10, test_size=0.5)
    cv_normalize_noise : str | None (default None)
        Multivariate normalize the trials before computing the distance between
        pairwise conditions.
        Valid values are 'epoch' | 'baseline' | None:
            - 'epoch' computes the covariance matrix on the entire epoch;
            - 'baseline' uses only the baseline condition; requires to pass an
              array times
    n_jobs : int (default 1)
    batch_size : int (default 200)
        size of the batches for cross-validation. To be used if the
        number of splits are very large in order to reduce the memory load

    Returns
    -------
    rdm: array (n_pairwise_targets, n_times, n_times)
        the cross-validated RDM over time
    targets_pairs : list of len n_pairwise_targets
        the labels for each row of rdm
    """
    splits = cv.split(targets, targets)
    n_splits = cv.get_n_splits(targets)

    # need to batch it for memory if more than 200 splits
    # trick from https://stackoverflow.com/questions/14822184/
    # is-there-a-ceiling-equivalent-of-operator-in-python/17511341#17511341
    n_batches = -(-n_splits // batch_size)
    print("Using n_batches {0}".format(n_batches))
    rdm = None
    for i_batch in range(n_batches):
        rdm_cv = Parallel(n_jobs=n_jobs)(
            delayed(_compute_fold)(epoch, targets, train, test,
                                   metric_fx=metric_fx,
                                   cv_normalize_noise=cv_normalize_noise)
            for train, test in islice(splits, batch_size))
        rdm_cv, targets_pairs = zip(*rdm_cv)
        if rdm is None:
            rdm = np.sum(rdm_cv, axis=0)
        else:
            rdm += np.sum(rdm_cv, axis=0)
        # remove to free up some memory
        del rdm_cv
    rdm /= n_splits

    return rdm, targets_pairs[0]


def make_pseudotrials(array, targets, navg=4):
    """Create pseudotrials by averaging within each group defined in `targets`
    a number `navg` of trials. The trials are randomly divided into groups of
    size `navg` for each target. If the number of trials in a group is not
    divisible by `navg`, one pseudotrials will be created by averaging the
    remainder trials.

    Parameters
    ----------
    array : (n_epochs, n_channels, n_times)
    targets : (n_epochs,)
    navg : int
        number of trials to average

    Returns
    -------
    avg_trials : (ceil(n_epochs/navg), n_channels, n_times)
        array containing the averaged trials
    avg_targets : (ceil(n_epochs/navg),)
        unique targets corresponding to the avg_trials
    """
    unique_targets, count_targets = np.unique(targets, return_counts=True)
    # count how many new trials we're getting for each condition
    n_splits_targets = -(-count_targets//navg)
    n_avg_trials = n_splits_targets.sum()
    # store the indices of the targets among which we'll be averaging
    idx_avg = dict()
    for i, t in enumerate(unique_targets):
        idx_target = np.where(targets == t)[0]
        # shuffle so we randomly pick them
        np.random.shuffle(idx_target)
        idx_avg[t] = np.array_split(idx_target, n_splits_targets[i])
    # now average
    avg_trials = np.zeros((n_avg_trials, array.shape[1], array.shape[2]))
    itrial = 0
    avg_targets = []
    for t in unique_targets:
        for idx in idx_avg[t]:
            avg_trials[itrial] = array[idx].mean(axis=0)
            avg_targets.append(t)
            itrial += 1
    avg_targets = np.asarray(avg_targets)
    return avg_trials, avg_targets
