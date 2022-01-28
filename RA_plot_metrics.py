"""
=========================================================
Plot whole-brain resolution metrics for WH dataset.
run RA_plot_metrics.py RA_config
Run in command window - currently cannot run headless.
=========================================================
"""

import sys

import os
from os import path as op

import numpy as np

import importlib

import matplotlib
from matplotlib import pyplot as plt

import mne

print('MNE Version: %s\n\n' % mne.__version__)  # just in case

module_name = sys.argv[1]
C = importlib.import_module(module_name)

inpath = op.join(C.resolution_path, C.resolution_subdir, 'fsaverage')

# it should exist, but...
if not op.exists(C.figures_dir):
    os.mkdir(C.figures_dir)

# list of parameters settings to apply

# the following will be iterated per item in paramlist
functions = ['psf', 'ctf']  # type of resolution functions
metrics = ['peak_err', 'sd_ext', 'peak_amp', 'sum_amp']  # type of resolution metrics

# 'paramlist' can contain parameter combinations that are not necessarily
# nested.

# inverse methods and contrasts to average
# (method/contrast, max scaling for (err, ext, amp))

# All
methods = [('MNE', (5., 5., 1.)),
           ('MNE_dep80', (5., 5., 1.)),
           ('sLOR', (5., 5., 1.)),
           ('dSPM', (5., 5., 1.)),
           ('eLOR', (5., 5., 1.)),
           ('LCMV_-200_0ms', (5., 5., 1.)),
           ('LCMV_50_250ms', (5., 5., 1.)),
           (('MNE', 'MNE_dep80'), (5., 1., .5)),
           (('MNE', 'dSPM'), (5., 1., .5)),
           (('MNE', 'sLOR'), (5., 1., .5)),
           (('MNE', 'eLOR'), (5., 1., .5)),
           (('sLOR', 'eLOR'), (5., 1., .5)),
           (('dSPM', 'sLOR'), (5., 1., .5)),
           (('MNE', 'LCMV_-200_0ms'), (5., 5., .5)),
           (('MNE', 'LCMV_50_250ms'), (5., 5., .5)),
           (('LCMV_-200_0ms', 'LCMV_50_250ms'), (5., 1., .5))]

# LCMV only
# methods = [('LCMV_-200_0ms', (8., 8., 1.)),
#            ('LCMV_50_250ms', (8., 8., 1.))]
#            # (('MNE', 'LCMV_-200_0ms'), (2., 5., .5)),
#            # (('MNE', 'LCMV_50_250ms'), (2., 5., .5)),
#            # (('LCMV_-200_0ms', 'LCMV_50_250ms'), (2., 5., .5))]

paramlist = [
    dict(functions=functions, metrics=metrics, methods=methods,
         chtype='eegmeg', snr=3., loose=0., depth=0.)
]

# for filenames, remove in future
st_duration = C.res_st_duration
origin = C.res_origin

subject = 'fsaverage'

# change font for pyplot (histograms)
font = {'family' : 'normal',
        'weight' : 'regular',
        'size'   : 20}
matplotlib.rc('font', **font)

def data_from_stcs(stcs, times=[0.]):
    """Collect data across STCs and samples (e.g. to plot histogram).
    Parameters:
    stcs: list | instance of Source Estimate
        The STC(s) with relevant data.
    times: list
        The time points (s) to take into account.

    Returns:
    data: 1D numpy array
        Data across STCs and samples as flattened array.
    """
    if type(stcs) is not list:
        stcs = [stcs]

    # get data from STCs
    data = []
    for stc in stcs:
        idx = stc.time_as_index(times)
        data.append(stc.data[:, idx])

    # turn everything into flattened numpy array
    data = np.array(data).flatten()

    return data


def define_hist_range(metric, method='mne'):
    """ Define the x-range for histogram for metrics.
    Parameters:
    metric: str
        The resolution metric used in the histogram.

    Returns:
    hist_range: tuple | None
        The min/max range for x-axis of the histogram.
    """
    if method == 'mne':
        if metric == 'peak_err':
            hist_range = (0., 7.)
        elif metric == 'sd_ext':
            hist_range = (0., 6.)
        else:
            hist_range = None
    elif method == 'lcmv':
        if metric == 'peak_err':
            hist_range = (0., 10.)
        elif metric == 'sd_ext':
            hist_range = (0., 10.)
        else:
            hist_range = None

    return hist_range


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

    # methods and thresholds for plotting
    for (method, threshs) in methods:  # which methods to subtract

        if type(method) == tuple:  # if contrast specified

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

                # select appropriate threshold for plotting
                if 'err' in metric:
                        thresh = threshs[0]
                elif 'ext' in metric:
                    thresh = threshs[1]
                elif 'amp' in metric:
                    thresh = threshs[2]
                else:
                    print('Not recognised this metric %s.' % metric)

                fname_mph = C.fname_STC(C, C.resolution_subdir, subject,
                                        stctext)

                # READ EXISTING SOURCE ESTIMATE
                print('Reading: %s.' % fname_mph)
                stc = mne.read_source_estimate(fname_mph, subject)

                # time_label = '%s %s %s' % (method, function, metric)
                time_label = ''  # easier for paper figures

                # positive or symmetric colorbar
                if type(method) == tuple:  # if contrast specified
                    clim = dict(kind='value', pos_lims=[0, thresh / 2., thresh])
                else:  # if just one method specified
                    clim = dict(kind='value', lims=[0, thresh / 2., thresh])

                print('Plotting.')
                brain = stc.plot(
                    time_label=time_label, subjects_dir=C.subjects_dir,
                    subject='fsaverage', colorbar=False, background='white',
                    clim=clim)

                fig_fname = op.join(C.figures_dir, stctext + '.jpg')

                print('Saving figure to %s.' % fig_fname)
                brain.save_image(fig_fname)
                brain.close()

                # plot histogram
                data = data_from_stcs(stc, times=[0.])

                fig, ax = plt.subplots()
                if method_str[:4] == 'LCMV':
                    methtxt = 'lcmv'
                else:
                    methtxt = 'mne'
                hist_range = define_hist_range(metric, methtxt)
                hist_bins = ax.hist(data, 100, hist_range)
                avg = data.mean()
                bin_max = hist_bins[0].max()
                ax.axvline(avg, color='k')
                ax_text = '%.1f(%.1f)|%.1f' % (avg, data.std(), np.median(data))
                axt = ax.text(avg + 0.2, bin_max - bin_max / 10., ax_text)
                axt.set_fontsize(26.)
                fig_fname = op.join(C.figures_dir, stctext + '_hist.jpg')
                fig.savefig(fig_fname)
                plt.close(fig)
