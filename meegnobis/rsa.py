"""Module containing functions for temporal RSA"""
from itertools import islice

import mne
import numpy as np
from joblib.parallel import Parallel, delayed
from mne import EpochsArray
from mne.cov import compute_whitener
from numpy.testing import assert_array_equal
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing.label import LabelEncoder

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


def _get_mask_binary_trials(target1, target2, targets_train, targets_test):
    unique_targets = _get_unique_targets(targets_train, targets_test)
    target1 = unique_targets[target1]
    target2 = unique_targets[target2]
    mask_train = (targets_train == target1) | \
                 (targets_train == target2)
    mask_test = (targets_test == target1) | \
                (targets_test == target2)
    return mask_train, mask_test


def _get_combinations_triu(unique_targets):
    n_unique_targets = len(unique_targets)
    for p1 in range(n_unique_targets):
        for p2 in range(p1, n_unique_targets):
            yield unique_targets[p1], unique_targets[p2]


def _run_metric(metric_fx, epoch_train, targets_train,
                epoch_test, targets_test, time_diag_only=False):
    # check if metric is symmetric
    try:
        symmetric = metric_fx.is_symmetric
    except AttributeError:
        symmetric = False
    # get some vars
    n_times = epoch_train.shape[-1]
    unique_targets = _get_unique_targets(targets_train, targets_test)
    n_unique_targets = len(unique_targets)
    n_pairwise_targets = _npairs(n_unique_targets)
    if time_diag_only:
        n_pairwise_times = n_times
    else:
        n_pairwise_times = _npairs(n_times) if symmetric \
            else n_times * n_times

    # check if we have one of our metrics
    try:
        vectorized = metric_fx.is_vectorized
    except AttributeError:
        vectorized = False

    # preallocate output with ones; for classification we assume that
    # if target1 == target2, then acc = 1
    rdms = np.ones((n_pairwise_targets, n_pairwise_times))
    if vectorized:
        # we don't need to loop explicitly through pairwise targets
        itime = 0
        # now we can loop through time
        for t1 in range(n_times):
            # log.info("Training on time {0:.3f}".format(t1))
            start_t2 = t1 if symmetric or time_diag_only else 0
            end_t2 = t1 + 1 if time_diag_only else n_times
            # fit only once
            metric_fx.fit(epoch_train[..., t1], targets_train)
            for t2 in range(start_t2, end_t2):
                rdms[:, itime] = metric_fx.score(epoch_test[..., t2],
                                                 targets_test)
                itime += 1
    else:
        for ipair, (p1, p2) in enumerate(
                _get_combinations_triu(unique_targets)):
            log.info("Running for pair number {0} ({1}, {2})".format(ipair,
                                                                     p1, p2))
            if p1 != p2:
                mask_train, mask_test = _get_mask_binary_trials(p1, p2,
                                                                targets_train,
                                                                targets_test)
                data_train = epoch_train[mask_train]
                data_test = epoch_test[mask_test]
                targets_train_ = targets_train[mask_train]
                targets_test_ = targets_test[mask_test]
                itime = 0
                # now we can loop through time
                for t1 in range(n_times):
                    start_t2 = t1 if symmetric or time_diag_only else 0
                    end_t2 = t1 + 1 if time_diag_only else n_times
                    # fit only once
                    metric_fx.fit(data_train[..., t1], targets_train_)
                    for t2 in range(start_t2, end_t2):
                        rdms[ipair, itime] = \
                            metric_fx.score(data_test[..., t2], targets_test_)
                        itime += 1
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
                  cv_normalize_noise=None, mean_groups=False,
                  time_diag_only=False):
    """Computes pairwise metric across time for one fold

    Arguments
    ---------
    metric_fx : either an allowed string (e.g., 'correlation') or an object
        with attributes 'fit' and 'score' (similar to scikit-learn estimators)
    targets : array (n_trials,)
        target (condition) for each trials; they must be integers
    train : array-like of int
        indices of the training data
    test : array-like of int
        indices of the testing data
    epoch : instance of mne.Epoch
    cv_normalize_noise : str | None (default None)
        Multivariately normalize the trials before applying the metric
        (distance or classification). Normalization is performed with
        cross-validation, that is the covariance matrix is estimated in the
        training set, and then applied to the test set. Normalization is
        always performed on single trials, thus before averaging within each
        class if `mean_groups` is set to True.
        Valid values for `cv_normalize_noise` are 'epoch' | 'baseline' | None:
            - 'epoch' computes the covariance matrix on the entire epoch;
            - 'baseline' uses only the baseline condition; it requires `epoch`
            to have a valid baseline (i.e., to have performed baseline
            correction)
    mean_groups : bool (default False)
        Whether the trials belonging to each target should be averaged prior
        to running the metric. This is useful if the metric is a distance
        metric. Should be set to False for classification.
    time_diag_only : bool (default False)
        Whether to run only for the diagonal in time,
        e.g. for train_time == test_time

    Returns
    -------
    rdms: array (n_pairwise_targets, n_pairwise_times)
        the cross-validated RDM over time
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
                       targets_test, time_diag_only=time_diag_only)

    # return also the pairs labels. since we are taking triu, we loop first
    # across rows
    unique_targets = _get_unique_targets(targets_train, targets_test)
    targets_pairs = []
    for tr_lbl, te_lbl in _get_combinations_triu(unique_targets):
        targets_pairs.append('+'.join(map(str, [tr_lbl, te_lbl])))

    return rdms, targets_pairs


def compute_temporal_rdm(epoch, targets, metric='correlation',
                         cv=StratifiedShuffleSplit(n_splits=10, test_size=0.5),
                         cv_normalize_noise=None, mean_groups=False,
                         time_diag_only=False,
                         n_jobs=1, batch_size=200):
    """Computes pairwise metric across time

    Arguments
    ---------
    epoch : instance of mne.Epoch
    targets : array (n_trials,)
        target (condition) for each trials
    metric: str | BaseMetric | sklearn estimator
        type of metric to use, one of 'cityblock', 'correlation', 'cosine',
        'dice', 'euclidean', 'hamming', 'sqeuclidean' for distances.
        Alternatively, any object with attributes fit/score, similar to
        sklearn estimators.
    cv : instance of sklearn cross-validator
        Cross-validator used to split the data into training/test datasets
        (default StratifiedShuffleSplit(n_splits=10, test_size=0.5)
    cv_normalize_noise : str | None (default None)
        Multivariately normalize the trials before applying the metric
        (distance or classification). Normalization is performed with
        cross-validation, that is the covariance matrix is estimated in the
        training set, and then applied to the test set. Normalization is
        always performed on single trials, thus before averaging within each
        class if `mean_groups` is set to True.
        Valid values for `cv_normalize_noise` are 'epoch' | 'baseline' | None:
            - 'epoch' computes the covariance matrix on the entire epoch;
            - 'baseline' uses only the baseline condition; it requires `epoch`
            to have a valid baseline (i.e., to have performed baseline
            correction)
    mean_groups : bool (default False)
        Whether the trials belonging to each target should be averaged prior
        to running the metric. This is useful if the metric is a distance
        metric. Should be set to False for classification.
    time_diag_only : bool (default False)
        Whether to run only for the diagonal in time,
        e.g. for train_time == test_time
    n_jobs : int (default 1)
    batch_size : int (default 200)
        size of the batches for cross-validation. To be used if the
        number of splits are very large in order to reduce the memory load

    Returns
    -------
    rdm: ndarray (n_pairwise_targets, n_pairwise_times) or
                 (n_pairwise_targets, n_times * n_times)
        the cross-validated RDM over time; if the metric is symmetric,
        then for efficiency purposes only the upper triangular matrix over
        time is returned, and the full matrix can be reconstructed with
        `np.triu_indices_from`. Otherwise, the entire flattened matrix over
        time is returned, and the full matrix can be reconstructed with
        `np.reshape((n_times, n_times))`. The order is row first, then columns.
    targets_pairs : list-like (n_pairwise_targets,)
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

    # fix targets
    targets_, le = _conform_targets(targets)
    splits = cv.split(targets_, targets_)
    n_splits = cv.get_n_splits(targets_)

    # need to batch it for memory if more than 200 splits
    # trick from https://stackoverflow.com/questions/14822184/
    # is-there-a-ceiling-equivalent-of-operator-in-python/17511341#17511341
    n_batches = -(-n_splits // batch_size)
    log.info("Using n_batches {0}".format(n_batches))
    rdm = None
    for i_batch in range(n_batches):
        log.info("Running batch {0}/{1}".format(i_batch+1, n_batches))
        rdm_cv = Parallel(n_jobs=n_jobs)(
            delayed(_compute_fold)(metric_fx, targets_, train, test, epoch,
                                   mean_groups=mean_groups,
                                   cv_normalize_noise=cv_normalize_noise,
                                   time_diag_only=time_diag_only)
            for train, test in islice(splits, batch_size))
        rdm_cv, targets_pairs = zip(*rdm_cv)
        if rdm is None:
            rdm = np.sum(rdm_cv, axis=0)
        else:
            rdm += np.sum(rdm_cv, axis=0)
        # remove to free up some memory
        del rdm_cv
    rdm /= n_splits

    return rdm, _invert_targets_pairs(targets_pairs[0], le)


def _invert_targets_pairs(targets_pairs, label_encoder):
    """
    Given a list of targets pairs of the form 'target1+target2', revert back
    to the original labeling by using `label_encoder.inverse_transform`

    Parameters
    ----------
    targets_pairs : list or array-like of str
    label_encoder : fitted LabelEncoder

    Returns
    -------
    targets_pairs : list of str
        the inversed targets_pairs
    """
    t1t2 = [l.split('+') for l in targets_pairs]
    t1, t2 = zip(*t1t2)
    t1 = [str(label_encoder.inverse_transform(int(t))) for t in t1]
    t2 = [str(label_encoder.inverse_transform(int(t))) for t in t2]
    return ['+'.join(t) for t in zip(t1, t2)]


def _make_pseudotrials_array(array, targets, navg=4, rng=None):
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
    rng : random number generator

    Returns
    -------
    avg_trials : (ceil(n_epochs/navg), n_channels, n_times)
        array containing the averaged trials
    avg_targets : (ceil(n_epochs/navg),)
        unique targets corresponding to the avg_trials
    """
    if rng is None:
        rng = np.random.RandomState()

    unique_targets, count_targets = np.unique(targets, return_counts=True)
    # count how many new trials we're getting for each condition
    n_splits_targets = -(-count_targets//navg)
    n_avg_trials = n_splits_targets.sum()
    # store the indices of the targets among which we'll be averaging
    idx_avg = dict()
    for i, t in enumerate(unique_targets):
        idx_target = np.where(targets == t)[0]
        # shuffle so we randomly pick them
        rng.shuffle(idx_target)
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


def make_pseudotrials(epoch, targets, navg=4, rng=None):
    """Create pseudotrials by averaging within each group defined in `targets`
    a number `navg` of trials. The trials are randomly divided into groups of
    size `navg` for each target. If the number of trials in a group is not
    divisible by `navg`, one pseudotrials will be created by averaging the
    remainder trials.

    Parameters
    ----------
    epoch : mne.Epoch
    targets : (n_epochs,)
    navg : int
        number of trials to average
    rng : random number generator

    Returns
    -------
    avg_epoch : mne.Epoch
        epoch with ceil(n_epochs/navg) trials
    avg_targets : (ceil(n_epochs/navg),)
        unique targets corresponding to the avg_trials
    """
    data = epoch.get_data()
    targets_, le = _conform_targets(targets)
    data_avg, avg_targets = _make_pseudotrials_array(data, targets_,
                                                     navg=navg, rng=rng)

    avg_epoch = mne.EpochsArray(data_avg, tmin=epoch.times[-1],
                                info=epoch.info)
    # convert back to the original targets
    avg_targets = [le.inverse_transform(t) for t in avg_targets]
    avg_epoch.events[:, -1] = avg_targets
    avg_epoch.baseline = epoch.baseline

    return avg_epoch, avg_targets


def _conform_targets(targets):
    """
    Conform targets to  [0, n_targets-1].

    Parameters
    ----------
    targets : array (n_targets, )

    Returns
    -------
    targets_conformed : array (n_targets, )
        targets are between 0 and n_targets-1
    label_encoder : LabelEncoder
        fit on targets, used to invert back using
        label_encoder.inverse_transform
    """
    le = LabelEncoder()
    le.fit(targets)
    return le.transform(targets), le
