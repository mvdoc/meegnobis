"""Module containing test for rsa"""

import logging
from functools import partial

import mne
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal, \
    assert_array_almost_equal
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.svm import SVC

from meegnobis.utils import _npairs
from ..log import log
from ..rsa import mean_group, _compute_fold, compute_temporal_rdm,\
    _make_pseudotrials_array, CDIST_METRICS, _run_metric, make_pseudotrials
from ..testing import generate_epoch

rng = np.random.RandomState(42)
# silence the output for tests
mne.set_log_level("CRITICAL")
log.setLevel(logging.CRITICAL)


def test_mean_group():
    n_epochs_cond = 5
    n_conditions = 2
    epoch = generate_epoch(n_epochs_cond=n_epochs_cond,
                           n_conditions=n_conditions)
    array = epoch.get_data()
    targets = epoch.events[:, 2]

    avg_array, unique_targets = mean_group(array, targets)
    assert_array_equal(unique_targets, [0, 1])
    assert_array_equal(avg_array, np.asanyarray([array[:5].mean(axis=0),
                                                 array[5:].mean(axis=0)]))

    # check it works even it targets is a list
    avg_array, unique_targets = mean_group(array, targets.tolist())
    assert_array_equal(unique_targets, [0, 1])
    assert_array_equal(avg_array, np.asanyarray([array[:5].mean(axis=0),
                                                 array[5:].mean(axis=0)]))


@pytest.mark.parametrize("cv_normalize_noise", [None, 'epoch', 'baseline'])
def test_compute_fold(cv_normalize_noise):
    n_epochs_cond = 10
    n_conditions = 4
    epoch = generate_epoch(n_epochs_cond=n_epochs_cond,
                           n_conditions=n_conditions)
    targets = epoch.events[:, 2]
    train = [np.arange(n_epochs_cond / 2) + i*n_epochs_cond
             for i in range(n_conditions)]
    test = [np.arange(n_epochs_cond / 2, n_epochs_cond) + i*n_epochs_cond
            for i in range(n_conditions)]
    train = np.array(train).flatten().astype(int)
    test = np.array(test).flatten().astype(int)

    metric_fx = CDIST_METRICS['correlation']
    rdms, target_pairs = _compute_fold(metric_fx, targets, train, test, epoch,
                                       cv_normalize_noise=cv_normalize_noise,
                                       mean_groups=True)
    n_times = len(epoch.times)
    n_pairwise_conditions = _npairs(n_conditions)
    n_pairwise_times = _npairs(n_times)
    assert(rdms.shape == (n_pairwise_conditions, n_pairwise_times))
    assert(rdms.shape[0] == len(target_pairs))
    # target_pairs should be already sorted
    target_pairs_sorted = sorted(target_pairs)
    assert_array_equal(target_pairs, target_pairs_sorted)
    # we should only get the upper triangular with the diagonal
    unique_targets = np.unique(targets)
    target_pairs_ = []
    for i_tr, tr_lbl in enumerate(unique_targets):
        for i_te, te_lbl in enumerate(unique_targets[i_tr:]):
            target_pairs_.append('+'.join(map(str, [tr_lbl, te_lbl])))
    assert_array_equal(target_pairs_, target_pairs)

    # check that it fails if the targets are strings
    targets = list(map(str, targets))
    with pytest.raises(ValueError):
        rdms = _compute_fold(metric_fx, targets, train, test, epoch,
                             cv_normalize_noise=cv_normalize_noise,
                             mean_groups=True)


def test_compute_fold_valuerrorcov():
    with pytest.raises(ValueError):
        n_epochs_cond = 10
        n_conditions = 4
        epoch = generate_epoch(n_epochs_cond=n_epochs_cond,
                               n_conditions=n_conditions)
        targets = epoch.events[:, 2]
        train = [np.arange(n_epochs_cond / 2) + i*n_epochs_cond
                 for i in range(n_conditions)]
        test = [np.arange(n_epochs_cond / 2, n_epochs_cond) + i*n_epochs_cond
                for i in range(n_conditions)]
        train = np.array(train).flatten().astype(int)
        test = np.array(test).flatten().astype(int)

        _ = _compute_fold(CDIST_METRICS['correlation'],
                          targets, train, test, epoch,
                          cv_normalize_noise='thisshouldfail')


def test_compute_fold_values():
    n_epochs_cond = 1
    n_conditions = 4
    n_times = 10
    epoch = generate_epoch(n_epochs_cond=n_epochs_cond,
                           n_conditions=n_conditions,
                           n_times=n_times)
    targets = epoch.events[:, 2]
    # let's use the same train and test
    train = test = np.arange(len(targets))

    metric_euclidean = CDIST_METRICS['euclidean']

    rdms, target_pairs = _compute_fold(metric_fx=metric_euclidean,
                                       targets=targets, train=train, test=test,
                                       epoch=epoch)

    epo_data = epoch.get_data()
    idx = 0
    for i_tr in range(n_times):
        for i_te in range(i_tr, n_times):
            rdms_ = cdist(epo_data[..., i_tr], epo_data[..., i_te])
            # impose symmetry
            rdms_ += rdms_.T
            rdms_ /= 2.
            assert_array_equal(rdms[:, idx],
                               rdms_[np.triu_indices_from(rdms_)])
            idx += 1


@pytest.mark.parametrize("cv_normalize_noise", (None, 'epoch', 'baseline'))
@pytest.mark.parametrize("n_splits", (4, 10))
@pytest.mark.parametrize("batch_size", (2, 5))
@pytest.mark.parametrize("time_diag_only", (True, False))
def test_compute_temporal_rdm(cv_normalize_noise, n_splits, batch_size,
                              time_diag_only):
    """Mostly a smoke test for combinations of parameters"""
    n_epochs_cond = 20
    n_conditions = 4
    epoch = generate_epoch(n_epochs_cond=n_epochs_cond,
                           n_conditions=n_conditions)
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.5)

    rdm, target_pairs = compute_temporal_rdm(
        epoch, cv=cv, targets=epoch.events[:, 2],
        cv_normalize_noise=cv_normalize_noise,
        batch_size=batch_size, mean_groups=True,
        time_diag_only=time_diag_only)
    n_times = len(epoch.times)
    n_pairwise_conditions = _npairs(n_conditions)
    n_pairwise_times = n_times if time_diag_only else _npairs(n_times)
    assert_equal(rdm.shape[0], len(target_pairs))
    assert_equal(rdm.shape, (n_pairwise_conditions, n_pairwise_times))


def test_compute_temporal_rdm_targets():
    """Test that everything works even if targets are not standard (strings
    and not within [0 nconditions-1]"""
    n_epochs_cond = 20
    n_conditions = 4
    epoch = generate_epoch(n_epochs_cond=n_epochs_cond,
                           n_conditions=n_conditions)

    def _run_target(t):
        cv = StratifiedShuffleSplit(n_splits=4, test_size=0.5, random_state=43)
        return compute_temporal_rdm(epoch, cv=cv, targets=t, mean_groups=True)

    targets = epoch.events[:, 2]
    rdm, target_pairs = _run_target(targets)

    targets_str = list(map(lambda x: str(x + 100), epoch.events[:, 2]))
    rdm_str, target_pairs_str = _run_target(targets_str)

    targets_weird = list(map(lambda x: x + 100, epoch.events[:, 2]))
    rdm_w, target_pairs_w = _run_target(targets_weird)

    assert_array_equal(rdm, rdm_str)
    assert_array_equal(rdm, rdm_w)


def test_compute_temporal_rdm_batch_size():
    # Check we get the same results regardless of batch_size
    n_epochs_cond = 20
    n_conditions = 4
    epoch = generate_epoch(n_epochs_cond=n_epochs_cond,
                           n_conditions=n_conditions)
    # to avoid warnings
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.5, random_state=42)

    # one batch
    rdm1, target_pairs1 = compute_temporal_rdm(
        epoch, cv=cv, targets=epoch.events[:, 2],
        batch_size=20, mean_groups=True)
    # two batches
    rdm2, target_pairs2 = compute_temporal_rdm(
        epoch, cv=cv, targets=epoch.events[:, 2],
        batch_size=5, mean_groups=True)
    assert_array_almost_equal(rdm1, rdm2)
    assert_array_equal(target_pairs1, target_pairs2)


@pytest.mark.parametrize("addtargets", (0, 10, -100))
def test_make_pseudotrials(addtargets):
    n_epochs_cond = 20
    n_conditions = 4
    epoch = generate_epoch(n_epochs_cond=n_epochs_cond,
                           n_conditions=n_conditions)

    # add something to test that we can use arbitrary indexing of targets
    targets_conformed = epoch.events[:, 2]
    targets = targets_conformed + addtargets
    epoch_data = epoch.get_data()
    navg = 4
    rng2 = np.random.RandomState(52)
    avg_trials, avg_targets = _make_pseudotrials_array(epoch_data,
                                                       targets_conformed,
                                                       navg=navg, rng=rng2)
    rng2 = np.random.RandomState(52)
    avg_epoch, avg_targets_ = make_pseudotrials(epoch, targets, navg=navg,
                                                rng=rng2)
    assert_array_equal(avg_trials, avg_epoch.get_data())
    assert_array_equal(avg_targets, np.array(avg_targets_) - addtargets)
    assert_array_equal(avg_epoch.events[:, -1], avg_targets_)
    assert_equal(avg_epoch.baseline, epoch.baseline)
    # check we get the right shape of the data
    assert_equal(avg_trials.shape[0], -(-epoch_data.shape[0]//navg))
    assert_array_equal(avg_trials.shape[1:], epoch_data.shape[1:])
    assert_equal(len(avg_targets), len(avg_trials))
    assert_equal(len(np.unique(avg_targets)), n_conditions)

    # check we have randomization going on
    avg_trials2, avg_targets2 = _make_pseudotrials_array(epoch_data,
                                                         targets_conformed,
                                                         navg=navg)
    assert_array_equal(avg_targets, avg_targets2)
    assert(not np.allclose(avg_trials, avg_trials2))

    # check it works even with an odd number of trials
    epoch_data = epoch_data[1:]
    targets = targets[1:]
    targets_conformed = targets + addtargets
    # just to be sure it's odd
    assert(len(targets) % 2 == 1)
    assert(len(epoch_data) % 2 == 1)
    avg_trials, avg_targets = _make_pseudotrials_array(epoch_data,
                                                       targets_conformed,
                                                       navg=navg)
    assert_equal(avg_trials.shape[0], -(-epoch_data.shape[0]//navg))
    assert_equal(len(np.unique(avg_targets)), n_conditions)
    assert_equal(len(avg_targets), len(avg_trials))


def test_sklearn_clf():
    svc = SVC()

    n_epochs = 20
    n_sensors = 30
    n_times = 4
    n_conditions = 4

    data, targets = make_blobs(n_samples=n_epochs,
                               n_features=n_sensors,
                               centers=n_conditions)
    data_ = np.dstack(
        [data + np.random.randn(*data.shape) for _ in range(n_times)]
    )

    # smoke test
    rdms = _run_metric(svc, data_, targets, data_, targets)
    assert_equal(rdms.shape, (_npairs(n_conditions), n_times*n_times))

    # let's make an actual splitter
    data, targets = make_blobs(n_samples=n_epochs,
                               n_features=n_sensors,
                               centers=2, center_box=(-1, 1),
                               cluster_std=2.0)
    data_ = np.dstack(
        [data + np.random.randn(*data.shape) for _ in range(n_times)]
    )

    splitter = StratifiedShuffleSplit(n_splits=2)
    splits = list(splitter.split(targets, targets, groups=targets))
    rdms = []
    for train, test in splits:
        rdms.append(
            _run_metric(svc, data_[train], targets[train],
                        data_[test], targets[test])
        )

    # now let's run it manually
    manual_clf = []
    for train, test in splits:
        data_train = data_[train]
        data_test = data_[test]
        targets_train = targets[train]
        targets_test = targets[test]
        acc_times = []
        for t1 in range(n_times):
            for t2 in range(n_times):
                d_tr = data_train[..., t1]
                d_te = data_test[..., t2]
                svc.fit(d_tr, targets_train)
                acc_times.append(svc.score(d_te, targets_test))
        manual_clf.append(np.array(acc_times))

    # take only the "true" accuracy; the others are 1. by default
    rdms = np.stack(rdms)
    assert_array_equal(rdms[:, 0, :], np.ones((2, n_times*n_times)))
    assert_array_equal(rdms[:, 2, :], np.ones((2, n_times*n_times)))
    rdms = rdms[:, 1, :]

    assert_array_equal(rdms, manual_clf)

