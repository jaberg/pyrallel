"""Utilities for Parallel Model Selection with IPython

Author: Olivier Grisel <olivier@ogrisel.com>
Licensed: MIT
"""
from time import sleep

from hyperopt.base import Trials
from hyperopt.fmin import Domain
from hyperopt.fmin import FMinIter
from hyperopt.base import JOB_STATE_NEW
from hyperopt.base import JOB_STATE_RUNNING
from hyperopt.base import JOB_STATE_DONE
from hyperopt.base import JOB_STATE_ERROR
from hyperopt.base import StopExperiment
from hyperopt.base import spec_from_misc

from IPython.parallel import interactive

import numpy as np

from sklearn.utils import check_random_state


@interactive
def _call_domain(domain, *args, **kwargs):
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
                    _call_domain,
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

