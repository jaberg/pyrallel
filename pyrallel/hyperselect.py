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
    def __init__(self, client):
        Trials.__init__(self)
        self._client = client

    def _insert_trial_docs(self, docs):
        rval = [doc['tid'] for doc in docs]
        self._dynamic_trials.extend(docs)
        return rval

    def refresh(self):
        for tt in self._dynamic_trials:
            if tt['ar'] and tt['ar'].ready():
                try:
                    tt['result'] = tt['ar'].get()
                    tt['state'] = JOB_STATE_DONE
                except Exception, e:
                    tt['error'] = str(e)
                    tt['state'] = JOB_STATE_ERROR
                tt['ar_meta'] = tt['ar'].metadata
                tt['ar'] = None
            elif tt['ar']:
                #print dir(tt['ar'])
                #print dir(tt['ar'].metadata)
                #print 'sent', tt['ar'].sent
                #print 'elapsed', tt['ar'].elapsed
                #print 'prog', tt['ar'].progress
                #print 'succ', tt['ar'].successful
                #print tt['ar'].metadata
                if tt['ar'].sent:
                    tt['state'] = JOB_STATE_RUNNING
            #elif (tt['state'] != JOB_STATE_RUNNING):
                    #and tt['ar'].metadata['started']):
            #if tt['state'] == JOB_STATE_NEW:
                #print id(tt['ar']), tt['ar'], tt['ar'].metadata #['status']
            # XXX mark errors

        Trials.refresh(self)

    def fmin(self, fn, space, algo, max_evals,
        random_state=0,
        verbose=0,
        ):
        lb_view = self._client.load_balanced_view()
        random_state = check_random_state(random_state)

        domain = Domain(fn, space,
            rseed=int(random_state.randint(2^20)))

        while len(self.trials) < max_evals:
            if lb_view.queue_status()['unassigned']:
                sleep(1e-3)
                continue
            self.refresh()
            if verbose:
                print 'fmin : %4i/%4i/%4i/%4i  %f' % (
                    self.count_by_state_unsynced(JOB_STATE_NEW),
                    self.count_by_state_unsynced(JOB_STATE_RUNNING),
                    self.count_by_state_unsynced(JOB_STATE_DONE),
                    self.count_by_state_unsynced(JOB_STATE_ERROR),
                    min([float('inf')] + [l for l in self.losses() if l is not None])
                    )

            new_ids = self.new_trial_ids(1)
            new_trials = algo(new_ids, domain, self)
            if new_trials is StopExperiment:
                stopped = True
                break
            elif len(new_trials) == 0:
                break
            else:
                assert len(new_trials) == 1

                task = lb_view.apply_async(
                    call_domain,
                    domain,
                    spec_from_misc(new_trials[0]['misc']),
                    )

                # -- XXX bypassing checks because 'ar'
                # is not ok for SONify... but should check
                # for all else being SONify
                tid, = self.insert_trial_docs(new_trials)
                assert self._dynamic_trials[-1]['tid'] == tid
                self._dynamic_trials[-1]['ar'] = task

    def wait(self):
        while True:
            self.refresh()
            if self.count_by_state_unsynced(JOB_STATE_NEW):
                sleep(1e-3)
                continue
            if self.count_by_state_unsynced(JOB_STATE_RUNNING):
                sleep(1e-3)
                continue
            break


@interactive
def call_domain(domain, *args, **kwargs):
    try:
        return domain.evaluate(*args, ctrl=None, **kwargs)
    except Exception, e:
        # XXX DO NOT CATCH ALL ERRORS HERE
        #     THE memo-evaluation can raise errors, which
        #     should be caught by user code, so the user
        #     code should also evaluate the memo
        #     Bassicaly, forget about the simple mode and
        #     require the pass_ctrl mode in which
        #     hyperparameters are passed directly to the
        #     evaluate fn.
        return {
                'loss': 1.0,
                'status': 'fail',
                'error': str(e),
                }


def report(trials, n_top=5):
    bests = trials.find_bests(n_top=n_top)
    output = "Progress: {0:02d}% ({1:03d}/{2:03d})\n".format(
        int(100 * self.progress()), self.completed(), self.total())
    for i, best in enumerate(bests):
        output += ("\nRank {0}: validation: {1:.5f} (+/-{2:.5f})"
                   " train: {3:.5f} (+/-{4:.5f}):\n {5}".format(
                   i + 1, *best))
    return output

