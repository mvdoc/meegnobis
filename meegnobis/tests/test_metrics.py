"""Testing metrics"""
from ..metrics import CDIST_METRICS
from ..testing import generate_epoch
from scipy.spatial.distance import cdist
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal


@pytest.mark.parametrize('metric_name', CDIST_METRICS.keys())
def test_cdist_metrics_(metric_name):
    epoch = generate_epoch()
    data = epoch.get_data()
    metric = CDIST_METRICS[metric_name]
    fake_targets = np.arange(data.shape[0])

    # smoke test
    metric.fit(data[..., 0], fake_targets)
    metric.score(data[..., 0], fake_targets)

    # check it complains if we try to fit something with a different number
    # of targets
    targets = epoch.events[:, -1]
    assert(len(np.unique(targets)) == 2)
    assert(2 != data.shape[0])
    metric.fit(data[..., 0], targets)
    with pytest.raises(ValueError):
        metric.score(data[..., 0], targets)

    # now test we get the actual values
    epoch_train = generate_epoch(n_epochs_cond=1, n_times=1, n_conditions=10)
    data_train = epoch_train.get_data()
    targets_train = epoch_train.events[:, -1]
    epoch_test = generate_epoch(n_epochs_cond=1, n_times=1, n_conditions=10)
    data_test = epoch_test.get_data()
    targets_test = epoch_test.events[:, -1]
    # just to be sure
    assert_array_equal(targets_train, targets_test)
    # fit/score
    metric.fit(data_train[:, 0], targets_train)
    out = metric.score(data_test[:, 0], targets_test)
    # we should get a vector
    assert(out.ndim == 1)
    out_cdist = cdist(data_train[:, 0], data_test[:, 0],
                      metric=metric_name)
    out_cdist += out_cdist.T
    out_cdist /= 2
    out_cdist = out_cdist[np.triu_indices_from(out_cdist)]
    assert_array_equal(out, out_cdist)

