"""
=========================================================
Compute whole-brain resolution metrics for WH data set.
e.g. run RA_ResolutionMetrics.py RA_config 1
=========================================================

"""

import os
import sys

import importlib
import glob

from copy import deepcopy

import time

import mne

from mne.minimum_norm import (make_inverse_resolution_matrix,
                              resolution_metrics)
from mne.source_estimate import SourceEstimate

from mne.beamformer import make_lcmv, make_lcmv_resolution_matrix

print('MNE Version: %s\n\n' % mne.__version__)  # just in case

# import config file with analysis parameters
module_name = sys.argv[1]

C = importlib.import_module(module_name)

# list of parameters settings to apply

# the following will be iterated per item in paramlist
chtype = 'eegmeg'
snr = 3.
loose = 0.
depth = 0.

functions = ['psf', 'ctf']  # type of resolution functions
metrics = ['peak_err', 'sd_ext', 'peak_amp', 'sum_amp']  # type of resolution metrics

# # for MNE-type estimators (MNE name and short label for filenames)
# methods = [
#     ('MNE', 'MNE'),
#     ('sLORETA', 'sLOR'),
#     ('dSPM', 'dSPM'),
#     ('eLORETA', 'eLOR')
# ]  # inverse methods
# paramlist = [dict(functions=functions, metrics=metrics, methods=methods,
#                   chtype=chtype, snr=snr, loose=loose, depth=depth,
#                   cov_dats=[])]

# # append depth-weighted MNE
# methods = [('MNE', 'MNE_dep80')]  # inverse methods, (for MNE, for filenames)
# depth = .8
# paramlist.append(dict(functions=functions, metrics=metrics, methods=methods,
#                  chtype=chtype, snr=snr, loose=loose, depth=depth,
#                  cov_dats=[]))

# append parameter list for LCMV beamformer
methods = [('LCMV', 'LCMV')]  # inverse methods
depth = 0.
cov_dats = [[-.2, 0], [0.05, 0.25]]  # latencies for data covariance
# paramlist.append(dict(functions=functions, metrics=metrics, methods=methods,
#                       chtype=chtype, snr=snr, loose=loose, depth=depth,
#                       cov_dats=cov_dats))

paramlist = [dict(functions=functions, metrics=metrics, methods=methods,
                  chtype=chtype, snr=snr, loose=loose, depth=depth,
                  cov_dats=cov_dats)]

# for filenames, remove in future
st_duration = C.res_st_duration
origin = C.res_origin

# get subject ID to process
# qsub start at 0, thus +1 here
sbj_ids = [int(sys.argv[2]) + 1]


def functions_and_metrics(resmat, src, functions, metrics, subject, stctext):
    """
    Iterate over resolution functions and metrics, save to STC.

    Parameters:
    resmat: array
        resolution matrix
    src: instance of SourceSpace
        The source space to use for metrics (e.g. from forward solution).
    functions: str 'psf' | 'ctf'
        The resolution function for which to compute metrics.
    metrics: list of str
        The resolution metrics to compute.
    subject: str
        The subject name (e.g. 'Sub01').
    stctext: str
        Text for STC filename, function and metric will be pre-pended.

    Returns:
    nothing
    """
    # After resolution matrix, compute resolution metrics
    for function in functions:

        for metric in metrics:

            # compute resolution metrics
            resmet = resolution_metrics(resmat, src,
                                        function=function,
                                        metric=metric)

            # prepend psf/ctf and resolution metric
            filetext = '%s_%s_%s' % (function, metric, stctext)
            fname_stc = C.fname_STC(C, C.resolution_subdir, subject, filetext)

            # save STC with resolution metrics
            print('Writing resolution metrics to %s.' % fname_stc)
            resmet.save(fname_stc)


def save_resolution_matrix(fname, resmat, src, gap=10):
    """Save PSFs from resolution matrix as STCs.

    Parameters:
    fname: str
        File name for STC output.
    resmat: array
        resolution matrix
    src: Source Space
        Source space used to create resolution matrix.
    gap: int
        Every gap-th PSF will be used (default 10).

    Returns:
        nothing
    """
    # vertices used in forward and inverse operator
    vertno_lh = src[0]['vertno']
    vertno_rh = src[1]['vertno']
    vertices = [vertno_lh, vertno_rh]

    # get desired columns from resolution matrix
    data = resmat[:, ::gap]

    # turn columns into source estimate object
    stc = SourceEstimate(data=data, vertices=vertices,
                         tmin=0, tstep=0.001)

    print('Saving PSFs from resmat to %s.' % fname)
    stc.save(fname)

    return


# Main
for sbj in sbj_ids:

    # time whole subject processing
    t0 = time.time()

    subject = 'Sub%02d' % sbj

    print('###\nWorking very hard on %s.\n###' % (subject))

    # path to output
    fname_stc = C.fname_STC(C, C.resolution_subdir, subject, '')

    # create output path if necessary
    if not os.path.exists(fname_stc):

        print('Creating %s.' % fname_stc)

        os.makedirs(fname_stc)

    fwd_fname = C.fname_ForwardSolution(C, subject, 'EEGMEG')

    print('###\nReading EEGMEG forward solutions: %s .\n###' % (fwd_fname))

    fwd = mne.read_forward_solution(fwd_fname)

    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)

    info = fwd['info']

    # covariance matrix (get best one with wildcard)
    # decided to use shrunk
    fname_cov = C.fname_cov(
        C, subject, st_duration, origin, C.res_cov_latwin, '*shrunk*', '')

    print('Reading covariance matrix %s' % fname_cov)

    # method may be underspecified, since it may be ranked differently for different subjects
    fname_cov = glob.glob(fname_cov)[0]  # be careful if multiple options present

    print('###\nReading noise covariance matrix from: %s.\n###' % (fname_cov))

    noise_cov = mne.read_cov(fname_cov)

    # # adjust info, since it was obtained from forward solution
    # info['projs'] = noise_cov['projs']
    # info['comps'] = ''

    # # regularise empirical noise covariance matrix
    # noise_cov = mne.cov.regularize(noise_cov, info, mag=0.05, grad=0.05,
    #                                eeg=0.05, rank='info')

    for params in paramlist:

        # parameters for resolution matrix and metrics
        functions = params['functions']
        metrics = params['metrics']
        methods = params['methods']
        chtype = params['chtype']
        snr = params['snr']
        loose = params['loose']
        depth = params['depth']
        cov_dats = params['cov_dats']

        if chtype == 'eegmeg':
            meg = True
            eeg = True
        elif chtype == 'meg':
            meg = True
            eeg = False
        elif chtype == 'eeg':
            meg = False
            eeg = True
        else:
            print('Modality not defined: %s.' % chtype)

        lambda2 = 1. / snr ** 2

        fwd_use = mne.pick_types_forward(fwd, meg=meg, eeg=eeg)

        # channel names to use
        ch_names = fwd_use['info']['ch_names']

        # read info for make_lcmv() from epochs
        epo_fname = C.fname_epo(C, subject, st_duration, origin)

        print('Reading info for epochs from: %s' % epo_fname)
        info = mne.io.read_info(epo_fname)

        # restrict noise covariance to channels in forward solution
        noise_cov_use = deepcopy(noise_cov)

        # getting the right channels for covariance matrix
        noise_cov_use.pick_channels(ch_names)

        # read inverse operator
        res_cov_latwin = [-0.2, 0.]  # latencies of noise covariance matrix
        inv_fname = C.fname_InverseOperator(C, subject, st_duration, origin,
                                            res_cov_latwin, chtype.upper(),
                                            loose, depth)

        print('\n###\nReading inverse operator from %s.\n###\n' % (inv_fname))

        invop = mne.minimum_norm.read_inverse_operator(inv_fname)

        for (method, meth_name) in methods:

            print('Method: %s (%s).' % (method, meth_name))

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

            if method == 'LCMV':  # for LCMV beamformer

                print('Doing LCMV.')

                for cov_dat in cov_dats:

                    # read data covariance matrix for LCMV beamformer
                    # covariance matrix (filter with wildcard)
                    fname_cov = C.fname_cov(C, subject, st_duration, origin,
                                            cov_dat, '*shrunk*', '')

                    # method may be underspecified, since it may be ranked differently for different subjects
                    fname_cov = glob.glob(fname_cov)[0]  # be careful if multiple options present

                    print('###\nReading data covariance matrix for LCMV from: %s.\n###' % (fname_cov))

                    data_cov = mne.read_cov(fname_cov)

                    # restrict data covariance to channels in forward solution
                    data_cov_use = deepcopy(data_cov)

                    # pick right channels
                    data_cov_use.pick_channels(ch_names)

                    noise_cov_lcmv = deepcopy(noise_cov_use)

                    # rank = 'info' in make_lcmv did not get the rank right
                    rank_noise = mne.compute_rank(noise_cov_lcmv, rank='info', info=info)

                    bf_filts = make_lcmv(info, fwd_use, data_cov_use, reg=0.05,
                                         noise_cov=noise_cov_lcmv,
                                         pick_ori=None, rank=rank_noise,
                                         weight_norm='unit-noise-gain',
                                         reduce_rank=False,
                                         depth=None,
                                         verbose=False)

                    resmat = make_lcmv_resolution_matrix(bf_filts, fwd_use, info)

                    cov_lat_str = C._lat_str(cov_dat[0], cov_dat[1])

                    meth_str = '%s_%s' % ('LCMV', cov_lat_str)

                    functions_and_metrics(resmat, fwd_use['src'], functions, metrics,
                                          subject, meth_str)

                    # Save some PSFs
                    filetext = meth_str + '_PSFs'
                    fname = fname_stc = C.fname_STC(
                        C, C.resolution_subdir, subject, filetext)

                    save_resolution_matrix(fname, resmat, invop['src'], gap=10)

            else:  # for MNE-type estimators

                # Compute resolution matrix
                resmat = make_inverse_resolution_matrix(fwd_use, invop,
                                                        method=method,
                                                        lambda2=lambda2)

                functions_and_metrics(
                    resmat, fwd_use['src'], functions, metrics, subject,
                    meth_name)

                # Save some PSFs
                filetext = meth_name + '_PSFs'
                fname = fname_stc = C.fname_STC(
                    C, C.resolution_subdir, subject, filetext)

                save_resolution_matrix(fname, resmat, invop['src'], gap=10)
