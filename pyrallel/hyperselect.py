"""Utilities for Parallel Model Selection with IPython

Author: Olivier Grisel <olivier@ogrisel.com>
Licensed: MIT
"""
from time import sleep
from collections import namedtuple
import os

from IPython.parallel import interactive
from IPython.parallel import TaskAborted
from IPython.display import clear_output
from scipy.stats import sem
import numpy as np

from sklearn.utils import check_random_state
try:
    # sklearn 0.14+
    from sklearn.grid_search import ParameterGrid
except ImportError:
    # sklearn 0.13
    from sklearn.grid_search import IterGrid as ParameterGrid

from pyrallel.mmap_utils import warm_mmap_on_cv_splits
from pyrallel.mmap_utils import persist_cv_splits

from hyperopt.base import Trials
from hyperopt.base import Ctrl
from hyperopt.fmin import Domain
from hyperopt.fmin import FMinIter
from hyperopt import rand, tpe
from hyperopt.base import JOB_STATE_NEW
from hyperopt.base import JOB_STATE_RUNNING
from hyperopt.base import JOB_STATE_DONE
from hyperopt.base import JOB_STATE_ERROR
from hyperopt.base import StopExperiment
from hyperopt.base import spec_from_misc

def is_waiting(task):
    return 


def is_aborted(task):
    return isinstance(getattr(task, '_exception', None), TaskAborted)


class IPythonTrials(Trials):
    def _insert_trial_docs(self, docs):
        for doc in docs:
            assert 'ar' in doc
        rval = [doc['tid'] for doc in docs]
        self._dynamic_trials.extend(docs)
        return rval

    def refresh(self):
        for tt in self._dynamic_trials:
            if tt['state'] == JOB_STATE_NEW and tt['ar'] != None:
                tt['state'] == JOB_STATE_RUNNING
            if tt['ar'] and tt['ar'].ready():
                tt['result'] = tt['ar'].get()
                tt['state'] = JOB_STATE_DONE
            # XXX mark errors

        if self._exp_key is None:
            self._trials = [tt for tt in self._dynamic_trials
                if tt['ar'] and tt['ar'].metadata['status'] == 'ok'] # XXX correct?
        else:
            self._trials = [tt for tt in self._dynamic_trials
                if (tt['ar'] and tt['ar'].metadata['status'] == 'ok' # XXX correct?
                    and tt['exp_key'] == self._exp_key
                    )]
        self._ids.update([tt['tid'] for tt in self._trials])


@interactive
def call_domain(domain, *args, **kwargs):
    return domain.evaluate(*args, ctrl=None, **kwargs)


class SequentialSeach(object):
    """"Async Randomized Parameter search."""

    def __init__(self, load_balanced_view, random_state=0, algo=rand.suggest):
        self.lb_view = load_balanced_view
        self.random_state = random_state
        self._temp_files = []
        self.algo = algo

    def reset(self):
        # Collect temporary files:
        for filename in self._temp_files:
            os.unlink(filename)
        del self._temp_files[:]

    def launch_for_splits(self, eval_fn, space, cv_split_filenames=(),
                          pre_warm=True, collect_files_on_reset=False,
                          max_evals=50):
        """Launch a Grid Search on precomputed CV splits."""

        # Abort any existing processing and erase previous state
        self.reset()

        # Mark the files for garbage collection
        if collect_files_on_reset:
            self._temp_files.extend(cv_split_filenames)

        # Warm the OS disk cache on each host with sequential reads instead
        # of having concurrent evaluation tasks compete for the the same host
        # disk resources later.
        if pre_warm:
            warm_mmap_on_cv_splits(self.lb_view.client, cv_split_filenames)

        # Randomize the grid order
        random_state = check_random_state(self.random_state)

        domain = Domain(eval_fn, space,
            rseed=int(random_state.randint(2^20)))

        trials = self.trials = IPythonTrials()

        while len(trials.trials) < max_evals:
            trials.refresh()
            if trials.count_by_state_unsynced(JOB_STATE_NEW):
                sleep(1e-3)
                continue

            new_ids = trials.new_trial_ids(1)
            new_trials = self.algo(new_ids, domain, self.trials)
            if new_trials is StopExperiment:
                stopped = True
                break
            elif len(new_trials) == 0:
                break
            else:
                assert len(new_trials) == 1

                task = self.lb_view.apply(
                    call_domain,
                    domain,
                    spec_from_misc(new_trials[0]['misc']),
                    )

                new_trials[0]['ar'] = task
                self.trials._insert_trial_docs(new_trials)

        # Make it possible to chain method calls
        return self


    def report(self, n_top=5):
        bests = self.find_bests(n_top=n_top)
        output = "Progress: {0:02d}% ({1:03d}/{2:03d})\n".format(
            int(100 * self.progress()), self.completed(), self.total())
        for i, best in enumerate(bests):
            output += ("\nRank {0}: validation: {1:.5f} (+/-{2:.5f})"
                       " train: {3:.5f} (+/-{4:.5f}):\n {5}".format(
                       i + 1, *best))
        return output

    def __repr__(self):
        return self.report()

    def monitor(self, plot=False):
        try:
            while not self.done():
                self.lb_view.spin()
                if plot:
                    import pylab as pl
                    pl.clf()
                    self.boxplot_parameters()
                clear_output()
                print(self.report())
                if plot:
                    pl.show()
                sleep(1)
        except KeyboardInterrupt:
            print("Monitoring interrupted.")


