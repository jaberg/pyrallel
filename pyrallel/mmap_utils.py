"""Utilities for Memory Mapping cross validation folds

Author: Olivier Grisel <olivier@ogrisel.com>
Licensed: MIT
"""
import os
from IPython.parallel import interactive


@interactive
def persist_cv_splits(X, y, name=None, n_cv_iter=5, suffix="_cv_%03d.pkl",
                      train_size=None, test_size=0.25, random_state=None,
                      folder='.'):
    """Materialize randomized train test splits of a dataset."""
    from sklearn.externals import joblib
    from sklearn.cross_validation import ShuffleSplit
    import os
    import uuid

    if name is None:
        name = uuid.uuid4().get_hex()

    cv = ShuffleSplit(X.shape[0], n_iter=n_cv_iter,
                      test_size=test_size, random_state=random_state)
    cv_split_filenames = []

    for i, (train, test) in enumerate(cv):
        cv_fold = (X[train], y[train], X[test], y[test])
        cv_split_filename = os.path.join(folder, name + suffix % i)
        cv_split_filename = os.path.abspath(cv_split_filename)
        joblib.dump(cv_fold, cv_split_filename)
        cv_split_filenames.append(cv_split_filename)

    return cv_split_filenames


def warm_mmap_on_cv_splits(client, cv_split_filenames):
    """Trigger a disk load on all the arrays of the CV splits

    Assume the files are shared on all the hosts using NFS.
    """
    # First step: query cluster to fetch one engine id per host
    all_engines = client[:]

    @interactive
    def hostname():
        import socket
        return socket.gethostname()

    hostnames = all_engines.apply(hostname).get_dict()
    one_engine_per_host = dict((hostname, engine_id)
                               for engine_id, hostname
                               in hostnames.items())
    hosts_view = client[one_engine_per_host.values()]

    # Second step: for each data file and host, mmap the arrays of the file
    # and trigger a sequential read of all the arrays' data
    @interactive
    def load_in_memory(filenames):
        from sklearn.externals import joblib
        for filename in filenames:
            arrays = joblib.load(filename, mmap_mode='r')
            for array in arrays:
                array.sum()  # trigger the disk read

    cv_split_filenames = [os.path.abspath(f) for f in cv_split_filenames]
    hosts_view.apply_sync(load_in_memory, cv_split_filenames)


if 0:
    # XXX delete all this  - what's it for?
  class MemmapOnWorkers(object):
    def __init__(self, pre_warm=True, collect_files_on_reset=False):
        self._temp_files = []
        # Mark the files for garbage collection
        if collect_files_on_reset:
            self._temp_files.extend(cv_split_filenames)

        # Warm the OS disk cache on each host with sequential reads instead
        # of having concurrent evaluation tasks compete for the the same host
        # disk resources later.
        if pre_warm:
            warm_mmap_on_cv_splits(self.lb_view.client, cv_split_filenames)

        cv_split_filenames=()



