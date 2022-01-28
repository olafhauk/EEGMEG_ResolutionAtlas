"""
=========================================================
Grand-average of morphed STCs for resolution metrics
for WH data set.
Doesn't run in parallel mode.
e.g.: run RA_AvgSTCs.py RA_config
=========================================================

"""

print(__doc__)

import os
import os.path as op

import sys

import importlib
import glob

import numpy as np

import mne
print('MNE Version: %s\n\n' % mne.__version__)  # just in case

## get analysis parameters from config file

module_name = sys.argv[1]

C = importlib.import_module(module_name)

# list of parameters settings to apply

# the following will be iterated per item in paramlist
functions = ['psf', 'ctf']  # type of resolution functions
metrics = ['peak_err', 'sd_ext', 'peak_amp', 'sum_amp']  # type of resolution metrics

# 'paramlist' can contain parameter combinations that are not necessarily
# nested.

# inverse methods and contrasts to average
methods = ['MNE', 'sLOR', 'dSPM', 'eLOR', 'LCMV_-200_0ms',
           'LCMV_50_250ms', 'MNE', 'MNE_dep80',
           ('MNE', 'MNE_dep80'), ('MNE', 'dSPM'),
           ('MNE', 'sLOR'), ('MNE', 'eLOR'), ('sLOR', 'eLOR'),
           ('dSPM', 'sLOR'), ('MNE', 'LCMV_-200_0ms'),
           ('MNE', 'LCMV_50_250ms'), ('LCMV_-200_0ms', 'LCMV_50_250ms')]

paramlist = [
    dict(functions=functions, metrics=metrics,
         methods=methods, chtype='eegmeg', snr=3., loose=0., depth=0.)
]

# for filenames, remove in future
st_duration = C.res_st_duration
origin = C.res_origin


# create dir for average if necessary
fname_avg = C.fname_STC(C, C.resolution_subdir, 'fsaverage', '')
if not op.exists(fname_avg):
    os.mkdir(fname_avg)

for params in paramlist:

        # paramters for resolution matrix and metrics
        functions = params['functions']
        metrics = params['metrics']
        methods = params['methods']
        chtype = params['chtype']
        snr = params['snr']
        loose = params['loose']
        depth = params['depth']

        lambda2 = 1. / snr ** 2

        for method in methods:  # which methods to subtract

            if type(method) is tuple:  # if contrast specified

                method_str = '%s-%s' % (method[0], method[1])

            else:  # if just one method specified

                method_str = method

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

                    stctext = '%s_%s_%s_mph' % (function, metric, method_str)

                    stcs = []  # initialise

                    correlations = np.array([])
                    corrtext = '%s_%s_%s_corr.txt' % (function, metric, method_str)
                    for sbj in C.subjs:

                        subject = 'Sub%02d' % sbj

                        fname_mph = C.fname_STC(C, C.resolution_subdir, subject,
                                                stctext)

                        # READ EXISTING SOURCE ESTIMATE
                        print('Reading: %s.' % fname_mph)
                        stc = mne.read_source_estimate(fname_mph, subject)

                        stcs.append(stc)

                        # Read correlations 
                        fname_corr = C.fname_STC(C, C.resolution_subdir, subject, corrtext)
                        correlations = np.append(correlations, np.loadtxt(fname_corr))

                    # average STCs across subjects
                    print('Averaging %d STC files.' % len(stcs))

                    avg = np.average([s.data for s in stcs], axis=0)

                    # turn average into source estimate object
                    avg_stc = mne.SourceEstimate(avg, stcs[0].vertices,
                                                 stcs[0].tmin, stcs[0].tstep)

                    fname_avg = C.fname_STC(C, C.resolution_subdir,
                                            'fsaverage', stctext)

                    print('###\nWriting grand-average STC file %s.\n###' % fname_avg)

                    avg_stc.save(fname_avg)

                    # Write correlations
                    fname_corr = C.fname_STC(C, C.resolution_subdir,
                                             'fsaverage', corrtext)
                    avg = correlations.mean()
                    sd = correlations.std()
                    print('Writing correlations with depth to %s.' % fname_corr)
                    print('%.2f (%.2f)' % (avg, sd))
                    # Write individual correlations with mean and st. dev.
                    correlations = np.append(correlations, [avg, sd])
                    np.savetxt(fname_corr, correlations, '%.2f')

            # average individual PSFs and CTFs
            if type(method) is not tuple:  # if no subtraction

                for [si, sbj] in enumerate(C.subjs):

                    subject = 'Sub%02d' % sbj

                    # read data covariance matrix for LCMV beamformer
                    # covariance matrix (filter with wildcard)
                    filetext = '%s_PSF*mph-lh.stc' % (method)
                    fname_stc = C.fname_STC(
                        C, C.resolution_subdir, subject, filetext)

                    # get list of matching filenames for PSFs
                    fname_stcs = glob.glob(fname_stc)  # be careful if multiple options present

                    # now append filenames for CTFs
                    filetext = '%s_CTF*mph-lh.stc' % (method)
                    fname_stc = C.fname_STC(
                        C, C.resolution_subdir, subject, filetext)

                    # add list items to existing file list
                    fname_stcs += glob.glob(fname_stc)

                    # initialise list for STCs
                    if si == 0:

                        stcs = [[] for i in range(0, len(fname_stcs))]

                    for [fi, ff] in enumerate(fname_stcs):

                        # READ EXISTING SOURCE ESTIMATE
                        print('Reading: %s.' % ff)
                        stc = mne.read_source_estimate(ff, subject)

                        stcs[fi].append(stc)

                # average individual PSFs and CTFs
                # average STCs across subjects
                for [fi, ff] in enumerate(fname_stcs):

                    avg = np.average([s.data for s in stcs[fi]], axis=0)

                    # turn average into source estimate object
                    avg_stc = mne.SourceEstimate(avg, stcs[0][0].vertices,
                                                 stcs[0][0].tmin, stcs[0][0].tstep)

                    # get only filename without path and suffix
                    filetext = ff.split('/')[-1][:-7]

                    fname_avg = C.fname_STC(
                        C, C.resolution_subdir, 'fsaverage', filetext)

                    print('###\nWriting grand-average STC file %s.\n###' % fname_avg)

                    avg_stc.save(fname_avg)
