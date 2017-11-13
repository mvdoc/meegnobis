"""Module containing functions for temporal RSA"""
from itertools import islice

import mne
import numpy as np
from joblib.parallel import Parallel, delayed
from mne import EpochsArray
from mne.cov import compute_whitener
from numpy.testing import assert_array_equal
from sklearn.model_selection import StratifiedShuffleSplit

from .metrics import CDIST_METRICS
from .utils import _npairs, _get_unique_targets
from .log import log

log.name = __name__


def mean_group(array, targets):
    """Average rows of array according to the unique values of targets

    Arguments
    ---------
    array : ndarray (n_samples, n_features, n_times)
        array containing the epoch data
    targets : ndarray (n_samples, )
        array containing targets for the samples

    Returns
    -------
    avg_array : ndarray (n_unique_targets, n_features, n_times)
        array containing the average within each group
    unique_targets : ndarray (n_unique_targets, )
        array containing the unique targets, corresponding to the order in
        avg_array
    """
    targets_ = np.array(targets)
    unique_targets = np.unique(targets_)
    n_unique_targets = len(unique_targets)
    avg_array = np.zeros((n_unique_targets, array.shape[1], array.shape[2]))
    for i, t in enumerate(unique_targets):
        mask = targets_ == t
        avg_array[i, ...] = array[mask, ...].mean(axis=0)
    return avg_array, unique_targets


def _run_metric_binarytargets(metric_fx, data_train, targets_train, data_test,
                              targets_test):
    """
    Parameters
    ----------
    data_train
    data_test
    targets_train
    targets_test

    Returns
    -------

    """
    # check if we have one of our metrics
    try:
        vectorized = metric_fx.is_vectorized
    except AttributeError:
        vectorized = False

    if vectorized:
        # the mtric should blow up if we get more than n_pairwise_targets
        metric_fx.fit(data_train, targets_train)
        rdm = metric_fx.score(data_test, targets_test)
    else:
        # we need to loop through all pairwise targets
        unique_targets = _get_unique_targets(targets_train, targets_test)
        n_unique_targets = len(unique_targets)
        n_pairwise_targets = _npairs(n_unique_targets)
        # preallocate output
        rdm = np.ones(n_pairwise_targets)
        idx = 0
        for p1 in range(n_unique_targets):
            for p2 in range(p1, n_unique_targets):
                target1 = unique_targets[p1]
                target2 = unique_targets[p2]
                mask_train = (targets_train == target1) | \
                             (targets_train == target2)
                mask_test = (targets_test == target1) | \
                            (targets_test == target2)
                # training
                metric_fx.fit(data_train[mask_train],
                              targets_train[mask_train])
                # score
                rdm[idx] = metric_fx.score(data_test[mask_test],
                                           targets_test[mask_test])
                idx += 1
    return rdm


def _run_metric(metric_fx, epoch_train, targets_train,
                epoch_test, targets_test):
    # check if metric is symmetric
    try:
        symmetric = metric_fx.is_symmetric
    except AttributeError:
        symmetric = False
    # get some vars
    n_times = epoch_train.shape[-1]
    times = range(n_times)
    unique_targets = _get_unique_targets(targets_train, targets_test)
    n_unique_targets = len(unique_targets)
    n_pairwise_targets = _npairs(n_unique_targets)
    n_pairwise_times = _npairs(n_times) if symmetric \
        else n_times * n_times
    # preallocate output
    rdms = np.zeros((n_pairwise_targets, n_pairwise_times))
    # compute pairwise metric
    idx = 0
    for t1 in range(n_times):
        start_t2 = t1 if symmetric else 0
        for t2 in range(start_t2, n_times):
            if idx % 1000 == 0:
                log.info("Running RDM for training, testing times "
                         "({0}, {1})".format(times[t1], times[t2]))
            # now store only the upper triangular matrix
            rdms[:, idx] = _run_metric_binarytargets(metric_fx,
                                                     epoch_train[..., t1],
                                                     targets_train,
                                                     epoch_test[..., t2],
                                                     targets_test)
            idx += 1
    return rdms


def _multiv_normalize(epoch_train, epoch_test, cv_normalize_noise=None):
    if cv_normalize_noise not in ('epoch', 'baseline', None):
        raise ValueError(
            "cv_normalize_noise must be one of {0}".format(
                ('epoch', 'baseline', None)))
    if cv_normalize_noise:
        log.info("Applying multivariate noise normalization "
                 "with method '{0}'".format(cv_normalize_noise))
        tmax = 0. if cv_normalize_noise == 'baseline' else None
        cov_train = mne.compute_covariance(epoch_train,
                                           tmax=tmax, method='shrunk')
        W_train, ch_names = compute_whitener(cov_train, epoch_train.info)
        # whiten both training and testing set
        epoch_train = np.array([np.dot(W_train, e)
                                for e in epoch_train.get_data()])
        epoch_test = np.array([np.dot(W_train, e)
                               for e in epoch_test.get_data()])
    else:
        epoch_train = epoch_train.get_data()
        epoch_test = epoch_test.get_data()
    return epoch_train, epoch_test


def _compute_fold(metric_fx, targets, train, test, epoch,
                  cv_normalize_noise=None, mean_groups=True):
    """Computes pairwise metric across time for one fold

    Arguments
    ---------
    metric_fx : function(x, y, targets_train, targets_test)
        any function that returns a scalar given two arrays.
    epoch : instance of mne.Epoch
    targets : array (n_trials,)
        target (condition) for each trials; they must be integers
    train : array-like of int
        indices of the training data
    test : array-like of int
        indices of the testing data
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
    rdms: array (n_pairwise_targets, n_pairwise_times)
        the cross-validated RDM over time; for efficiency purposes, it returns
        only the upper triangular matrix over time, thus it assumes that the
        metric is symmetric (this assumption doesn't hold for example for
        classification).
    targets_pairs : list of len n_pairwise_targets
        the labels corresponding to each element in rdms
    """

    targets = np.asarray(targets)
    if targets.dtype != np.dtype('int64'):
        raise ValueError("targets must be integers, "
                         "not {0}".format(targets.dtype))

    # impose conditions in epoch as targets, so that covariance
    # matrix is computed within each target
    events_ = epoch.events.copy()
    events_[:, 2] = targets
    epoch_ = EpochsArray(epoch.get_data(), info=epoch.info,
                         events=events_, tmin=epoch.times[0])
    epoch_.baseline = epoch.baseline

    # get training and testing data
    epoch_train = epoch_.copy()[train]
    targets_train = targets[train]
    assert(len(epoch_train) == len(targets_train))
    epoch_test = epoch_.copy()[test]
    targets_test = targets[test]
    assert(len(epoch_test) == len(targets_test))

    # perform multi variate noise normalization
    epoch_train, epoch_test = _multiv_normalize(epoch_train, epoch_test,
                                                cv_normalize_noise)
    if mean_groups:
        # average within train and test for each target
        epoch_train, targets_train = mean_group(epoch_train, targets_train)
        epoch_test, targets_test = mean_group(epoch_test, targets_test)
        # the targets should be the same in both training and testing
        # set or else we are correlating weird things together
        assert_array_equal(targets_train, targets_test)
    rdms = _run_metric(metric_fx, epoch_train, targets_train, epoch_test,
                       targets_test)

    # return also the pairs labels. since we are taking triu, we loop first
    # across rows
    unique_targets = _get_unique_targets(targets_train, targets_test)
    targets_pairs = []
    for i_tr, tr_lbl in enumerate(unique_targets):
        for _, te_lbl in enumerate(unique_targets[i_tr:]):
            targets_pairs.append('+'.join(map(str, [tr_lbl, te_lbl])))

    return rdms, targets_pairs


def compute_temporal_rdm(epoch, targets, metric='correlation',
                         cv=StratifiedShuffleSplit(n_splits=10, test_size=0.5),
                         cv_normalize_noise=None, metric_symmetric_time=True,
                         n_jobs=1, batch_size=200):
    """Computes pairwise metric across time

    Arguments
    ---------
    epoch : instance of mne.Epoch
    targets : array (n_trials,)
        target (condition) for each trials
    metric: str
        type of metric to use, one of 'cityblock', 'correlation', 'cosine',
        'dice', 'euclidean', 'hamming', 'sqeuclidean' for distances, or
        'linearsvm' for classification.
    cv : instance of sklearn cross-validator
        (default StratifiedShuffleSplit(n_splits=10, test_size=0.5)
    cv_normalize_noise : str | None (default None)
        Multivariate normalize the trials before computing the distance between
        pairwise conditions.
        Valid values are 'epoch' | 'baseline' | None:
            - 'epoch' computes the covariance matrix on the entire epoch;
            - 'baseline' uses only the baseline condition; requires to pass an
              array times
    metric_symmetric_time: bool (default True)
        Whether the metric can be considered symmetric for time. For example,
        this is true for distances because
        m(t_train, t_test) == m(t_test, t_train)
        but it's not true for classification, because the order matters for
        training. If `metric_symmetric_time` is set to True, only the upper
        diagonal matrix (for time) is computed; otherwise, the entire matrix
        is returned.
    n_jobs : int (default 1)
    batch_size : int (default 200)
        size of the batches for cross-validation. To be used if the
        number of splits are very large in order to reduce the memory load

    Returns
    -------
    rdm: ndarray (n_pairwise_targets, n_pairwise_times) or
                 (n_pairwise_targets, n_times * n_times)
        the cross-validated RDM over time; if `metric_symmetric_time` was set
        to True, then for efficiency purposes it returns
        only the upper triangular matrix over time, and the full matrix can be
        reconstructed with numpy.triu_indices_from. Otherwise, the entire
        flattened matrix over time is returned, and the full matrix can be
        reconstructed with np.reshape((n_times, n_times)). The order is row
        first, then columns.
    targets_pairs : list of len n_pairwise_targets
        the labels for each row of rdm
    """
    # set up metric
    if metric in CDIST_METRICS:
        metric_fx = CDIST_METRICS[metric]
    elif hasattr(metric, 'fit') and hasattr(metric, 'score'):
        metric_fx = metric
    else:
        raise ValueError(
            "I don't know how to deal with metric {0}. It's not one of {1} "
            "and it doesn't have fit/score attributes".format(
                metric, sorted(CDIST_METRICS.keys())))

    splits = cv.split(targets, targets)
    n_splits = cv.get_n_splits(targets)

    # need to batch it for memory if more than 200 splits
    # trick from https://stackoverflow.com/questions/14822184/
    # is-there-a-ceiling-equivalent-of-operator-in-python/17511341#17511341
    n_batches = -(-n_splits // batch_size)
    log.info("Using n_batches {0}".format(n_batches))
    rdm = None
    for i_batch in range(n_batches):
        log.info("Running batch {0}/{1}".format(i_batch+1, n_batches))
        rdm_cv = Parallel(n_jobs=n_jobs)(
            delayed(_compute_fold)(metric_fx, targets, train, test, epoch,
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
