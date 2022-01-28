"""
=========================================================
Compute differences of whole-brain resolution metrics for WH data set.
e.g. run WH_ResolutionMetrics_differences.py RA_config 1
=========================================================

"""

import sys

import importlib
import time

import mne

print('MNE Version: %s\n\n' % mne.__version__)  # just in case

# get analysis parameters from config file
module_name = sys.argv[1]

C = importlib.import_module(module_name)

# list of parameters settings to apply

# the following will be iterated per item in paramlist
functions = ['psf', 'ctf']  # type of resolution functions
metrics = ['peak_err', 'sd_ext', 'peak_amp', 'sum_amp']  # type of resolution metrics

# 'paramlist' can contain parameter combinations that are not necessarily
# nested.

# Methods subtractions to perform
methods_contr = [('MNE', 'dSPM'), ('MNE', 'sLOR'), ('MNE', 'eLOR'),
                 ('sLOR', 'eLOR'), ('dSPM', 'sLOR'),
                 ('MNE', 'LCMV_-200_0ms'), ('MNE', 'LCMV_50_250ms'),
                 ('MNE', 'MNE_dep80'),
                 ('LCMV_-200_0ms', 'LCMV_50_250ms')]

# parameters as specified in RA_ResolutionMetrics.py for the methods contrast
paramlist = [
    dict(functions=functions, metrics=metrics,
         methods_contr=methods_contr,
         chtype='eegmeg', snr=3., loose=0., depth=0.8)
]

# for filenames, remove in future
st_duration = C.res_st_duration
origin = C.res_origin

# get subject ID to process
# qsub start at 0, thus +1 here
sbj_ids = [int(sys.argv[2]) + 1]

for sbj in sbj_ids:

    # time whole subject processing
    t0 = time.time()

    subject = 'Sub%02d' % sbj

    print('###\nWorking hard on %s.\n###' % (subject))

    for params in paramlist:

        # paramters for resolution matrix and metrics
        functions = params['functions']
        metrics = params['metrics']
        methods_contr = params['methods_contr']
        chtype = params['chtype']
        snr = params['snr']
        loose = params['loose']
        depth = params['depth']

        lambda2 = 1. / snr ** 2

        for methods in methods_contr:  # which methods to subtract

            method1 = methods[0]
            method2 = methods[1]

            for function in functions:

                for metric in metrics:

                    # for filenames
                    lamb2_str = str(lambda2).replace('.', '')
                    if len(lamb2_str) > 3:
                        lamb2_str = lamb2_str[:3]

                        if loose is None:
                            loose = 0
                        loo_str = 'loo%s' % str(int(100 * loose))

                        if depth is None:
                            depth = 0
                        dep_str = 'dep%s' % str(int(100 * depth))

                    stctext1 = '%s_%s_%s' % (function, metric, method1)

                    stctext2 = '%s_%s_%s' % (function, metric, method2)

                    fname_stc1 = C.fname_STC(C, C.resolution_subdir, subject,
                                             stctext1)
                    fname_stc2 = C.fname_STC(C, C.resolution_subdir, subject,
                                             stctext2)

                    stc1 = mne.read_source_estimate(fname_stc1)
                    stc2 = mne.read_source_estimate(fname_stc2)

                    # Compute difference distributions for metrics
                    stc_diff = stc1 - stc2

                    contr_str = '%s-%s' % (method1, method2)
                    stctext3 = '%s_%s_%s' % (function, metric, contr_str)

                    fname_stc3 = C.fname_STC(C, C.resolution_subdir, subject,
                                             stctext3)

                    # save STC with difference distribution
                    print('Writing difference distribution to %s.' % fname_stc3)
                    stc_diff.save(fname_stc3)
