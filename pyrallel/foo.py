from functools import partial
from IPython.parallel import interactive
from IPython.parallel import TaskAborted
from IPython.display import clear_output
from scipy.stats import sem
from IPython.parallel import Client

from pyrallel import mmap_utils, hyperselect

from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from collections import OrderedDict
import numpy as np


from hyperopt.pyll import scope
from hyperopt import hp
import hyperopt
import sklearn.decomposition
import sklearn.mixture
import sklearn.tree
import sklearn.svm

scope.define(sklearn.decomposition.PCA)
scope.define(sklearn.mixture.GMM)
scope.define(sklearn.tree.DecisionTreeClassifier)
scope.define(sklearn.svm.SVC)


def compute_evaluation(config, cv_split_filename,
                       train_fraction=1.0, mmap_mode='r'):
    """Evaluate a model on a given CV split"""
    # All module imports should be executed in the worker namespace to make
    # possible to run an an engine node.
    from time import time, sleep
    from sklearn.externals import joblib

    time_start = time()

    pre_processing = config['pre_processing']
    classifier = config['classifier']

    X_train, y_train, X_test, y_test = joblib.load(
        cv_split_filename, mmap_mode=mmap_mode)

    # Slice a subset of the training set for plotting learning curves
    n_samples_train = int(train_fraction * X_train.shape[0])
    X_train = X_train[:n_samples_train]
    y_train = y_train[:n_samples_train]

    pre_processing.fit(X_train)
    X_train_pp = pre_processing.transform(X_train)

    # Fit model and measure training time
    tick = time()
    classifier.fit(X_train_pp, y_train)
    train_time = time() - tick

    # Compute score on training set
    train_score = classifier.score(X_train_pp, y_train)

    # Compute score on test set
    X_test_pp = pre_processing.transform(X_test)
    test_score = classifier.score(X_test_pp, y_test)

    sleep(3)

    time_end = time()

    # Wrap evaluation results in a simple tuple datastructure
    return {
            'loss': 1 - test_score,
            'loss_': {
                'duration': time_end - time_start,
                'erate': 1 - test_score,  #XXX should be validation error
                },
            'test_score': test_score,
            'train_score': train_score,
            'train_time': train_time,
            'train_fraction': train_fraction,
            }


def main():
    client = Client()
    print 'n. clients: ', len(client)

    digits = load_digits()

    X = MinMaxScaler().fit_transform(digits.data)
    y = digits.target

    pre_processing = hp.choice('preproc_algo', [
        scope.PCA(
            n_components=1 + hp.qlognormal(
                'pca_n_comp', np.log(10), np.log(10), 1),
            whiten=hp.choice(
                'pca_whiten', [False, True])),
        scope.GMM(
            n_components=1 + hp.qlognormal(
                'gmm_n_comp', np.log(100), np.log(10), 1),
            covariance_type=hp.choice(
                'gmm_covtype', ['spherical', 'tied', 'diag', 'full'])),
        ])

    classifier = hp.choice('classifier', [
        scope.DecisionTreeClassifier(
            criterion=hp.choice('dtree_criterion', ['gini', 'entropy']),
            max_features=hp.uniform('dtree_max_features', 0, 1),
            max_depth=hp.quniform('dtree_max_depth', 1, 25, 1)),
        scope.SVC(
            C=hp.lognormal('svc_rbf_C', 0, 3),
            kernel='rbf',
            gamma=hp.lognormal('svc_rbf_gamma', 0, 2),
            tol=hp.lognormal('svc_rbf_tol', np.log(1e-3), 1)),
        ])

    sklearn_space = {'pre_processing': pre_processing,
                     'classifier': classifier}

    digits_cv_split_filenames = mmap_utils.persist_cv_splits(
                X, y, name='digits_10', n_cv_iter=10)

    mmap_utils.warm_mmap_on_cv_splits(client, digits_cv_split_filenames)

    trials = hyperselect.IPythonTrials(client)
    trials.fmin(
        partial(compute_evaluation,
            cv_split_filename=digits_cv_split_filenames[0],
            ),
        sklearn_space,
        algo=hyperopt.tpe.suggest,
        max_evals=30,
        verbose=1,
        )
    trials.wait()
    print trials.best_trial

