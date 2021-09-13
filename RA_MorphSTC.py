"""
=========================================================
Morph STCs of resolution metrics for W&H data set.
e.g.: run RA_MorphSTC.py RA_config 11
=========================================================

"""

import sys

import importlib
import glob

import numpy as np

import mne
from mne.source_space import compute_distance_to_sensors
print('MNE Version: %s\n\n' % mne.__version__)  # just in case

# get analysis parameters from config file
module_name = sys.argv[1]

C = importlib.import_module(module_name)
importlib.reload(C)

# list of parameters settings to apply

# the following will be iterated per item in paramlist
functions = ['psf', 'ctf']  # type of resolution functions

metrics = ['peak_err', 'sd_ext', 'peak_amp', 'sum_amp']  # type of resolution metrics

# 'paramlist' can contain parameter combinations that are not necessarily
# nested.
# inverse methods and contrasts to morph
methods = ['MNE', 'MNE_dep80', 'sLOR', 'dSPM', 'eLOR', 'LCMV_-200_0ms',
           'LCMV_50_250ms', ('MNE', 'MNE_dep80'), ('MNE', 'dSPM'),
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

# ## get analysis parameters from config file

module_name = sys.argv[1]

C = importlib.import_module(module_name)
importlib.reload(C)

# get subject ID to process
# qsub start at 0, thus +1 here
sbj_ids = [int(sys.argv[2]) + 1]

# hack to have variables via qsub
stc_path, stc_type, metric = '', '', ''

###
for sbj in sbj_ids:

    # only one morph_mat per subject needed
    morph_mat = []

    subject = 'Sub%02d' % sbj

    print('###\nAbout to morph STCs for %s.\n###' % (subject))

    # Forward solution to compute source depth and correlations
    fwd_fname = C.fname_ForwardSolution(C, subject, 'EEGMEG')
    print('###\nReading EEGMEG forward solutions: %s .\n###' % (fwd_fname))
    fwd = mne.read_forward_solution(fwd_fname)

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

                    stctext = '%s_%s_%s' % (function, metric, method_str)

                    fname_stc = C.fname_STC(C, C.resolution_subdir, subject,
                                            stctext)

                    fname_mph = C.fname_STC(C, C.resolution_subdir, subject,
                                            stctext + '_mph')

                    # read existing source estimate
                    print('Reading: %s.' % fname_stc)
                    stc = mne.read_source_estimate(fname_stc, subject)

                    if morph_mat == []:

                        print('Computing morphing matrix.')
                        morph_mat = mne.compute_source_morph(
                            src=stc, subject_from=subject, subject_to=C.stc_morph,
                            subjects_dir=C.subjects_dir)

                        fname_mphmat = C.fname_STC(
                            C, C.resolution_subdir, subject, 'mphmat')

                        morph_mat.save(fname_mphmat, overwrite=True)

                    stc_mph = morph_mat.apply(stc)

                    print('Writing morphed to: %s.' % fname_mph)
                    stc_mph.save(fname_mph)

                    # Correlation with source depth
                    # Compute minimum Euclidean distances between vertices and MEG sensors
                    depths = compute_distance_to_sensors(src=fwd['src'], info=fwd['info']).min(axis=1)
                    fname_corr = C.fname_STC(C, C.resolution_subdir, subject, stctext + '_corr.txt')
                    print('Writing correlations with depth to %s.' % fname_corr)
                    # save depth correlation to text file
                    np.savetxt(fname_corr, [np.corrcoef(depths, stc.data.squeeze())[0, 1]], "%.2f")

            # plot individual PSFs and CTFs

            if type(method) is not tuple:  # if no subtraction

                # read data covariance matrix for LCMV beamformer
                # covariance matrix (filter with wildcard)
                filetext = '%s_PSF*-lh.stc' % (method)
                fname_stc = C.fname_STC(
                    C, C.resolution_subdir, subject, filetext)

                # get list of matching filenames for PSFs
                fname_stcs = glob.glob(fname_stc)  # be careful if multiple options present

                # now append filenames for CTFs
                filetext = '%s_CTF*-lh.stc' % (method)
                fname_stc = C.fname_STC(
                    C, C.resolution_subdir, subject, filetext)

                # add list items to existing file list
                fname_stcs += glob.glob(fname_stc)

                # don't morph what's already morphed
                # only read one STC for left hemisphere
                good_fnames = []  # file names without 'mph'
                for ff in fname_stcs:

                    if ('mph' in ff) or ('-rh.stc' in ff):

                        good_fnames.append(ff)

                for ff in good_fnames:

                    print('Reading: %s.' % ff)
                    stc = mne.read_source_estimate(ff, subject)

                    stc_mph = morph_mat.apply(stc)

                    fname_stc_mph = ff.replace('-lh.stc', '_mph')

                    print('Writing morphed to: %s.' % fname_stc_mph)

                    stc_mph.save(fname_stc_mph)
