"""
============================================================
PSFs and CTFs for 2 example ROIs for LCMV beamformers
============================================================

Auditory and Visual labels.
"""

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import (read_inverse_operator,
                              make_inverse_resolution_matrix,
                              get_point_spread, get_cross_talk)
from mne.beamformer import make_lcmv, make_lcmv_resolution_matrix

print(__doc__)

###############################################################################
# Load forward solution and inverse operator
# ------------------------------------------
#
# We need a matching forward solution and inverse operator to compute
# resolution matrices for different methods.

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-fixed-inv.fif'

# folder for output images
images_path = '/home/olaf/documents/Papers/EEGMEGResolutionAtlas_Apr19/NI - SI/Figures/images/'

forward = mne.read_forward_solution(fname_fwd)
# Convert forward solution to fixed source orientations
mne.convert_forward_solution(
    forward, surf_ori=True, force_fixed=True, copy=False)

inverse_operator = read_inverse_operator(fname_inv)

src = forward['src']

# Labels used in this example
use_labels = ['superiortemporal-lh', 'lateraloccipital-lh']

###############################################################################
# Read labels
# --------------------------------------------------
#
# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot('sample', parc='aparc',
                                    subjects_dir=subjects_dir)

# get just the desired labels
labels = [ll for ll in labels if ll.name in use_labels]

# split superior temporal, use posterior half
# note sequence here is alphabetical
subjects_dir = sample.data_path() + '/subjects'
labels[1] = labels[1].split(2, subjects_dir=subjects_dir)[0]

n_labels = len(labels)

label_names = [label.name for label in labels]


###############################################################################
# Now check LCMV beamformer.
# In this case, PSFs and CTFs will be different and we have to look at them
# separately

fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

# Read raw data
raw = mne.io.read_raw_fif(raw_fname)

# only pick good EEG/MEG sensors
raw.info['bads'] += ['EEG 053']  # bads + 1 more
picks = mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads')

# Find events
events = mne.find_events(raw)

# event_id = {'aud/l': 1, 'aud/r': 2, 'vis/l': 3, 'vis/r': 4}
event_id_aud = {'aud/l': 1, 'aud/r': 2}
event_id_vis = {'vis/l': 3, 'vis/r': 4}

tmin, tmax = -.2, .5  # epoch duration (s)
epochs_aud = mne.Epochs(raw, events, event_id=event_id_aud, tmin=tmin, tmax=tmax,
                        picks=picks, baseline=(-.2, 0.), preload=True)
epochs_vis = mne.Epochs(raw, events, event_id=event_id_vis, tmin=tmin, tmax=tmax,
                        picks=picks, baseline=(-.2, 0.), preload=True)
del raw

# covariance matrix for post-stimulus interval (around main evoked responses)
tmin, tmax = 0., .4
cov_aud = mne.compute_covariance(epochs_aud, tmin=tmin, tmax=tmax,
                                 method='empirical')
cov_vis = mne.compute_covariance(epochs_vis, tmin=tmin, tmax=tmax,
                                 method='empirical')

info = epochs_aud.info
del epochs_aud, epochs_vis

# read noise covariance matrix
noise_cov = mne.read_cov(fname_cov)

# regularize noise covariance
noise_cov = mne.cov.regularize(noise_cov, info, mag=0.1, grad=0.1,
                               eeg=0.1, rank='info')

# compute LCMV beamformer filters for auditory and visual epochs separately
filters_aud = make_lcmv(info, forward, cov_aud, reg=0.05,
                        noise_cov=noise_cov,
                        pick_ori=None, rank='info',
                        weight_norm='unit-noise-gain',
                        reduce_rank=False,
                        depth=None,
                        verbose=False)

filters_vis = make_lcmv(info, forward, cov_vis, reg=0.05,
                        noise_cov=noise_cov,
                        pick_ori=None, rank='info',
                        weight_norm='unit-noise-gain',
                        reduce_rank=False,
                        depth=None,
                        verbose=False)

# filters for baseline activity
filters_noi = make_lcmv(info, forward, noise_cov, reg=0.05,
                        noise_cov=noise_cov,
                        pick_ori=None, rank='info',
                        weight_norm='unit-noise-gain',
                        reduce_rank=False,
                        depth=None,
                        verbose=False)

rm_bf_aud = make_lcmv_resolution_matrix(filters_aud, forward, info)
rm_bf_vis = make_lcmv_resolution_matrix(filters_vis, forward, info)
rm_bf_noi = make_lcmv_resolution_matrix(filters_noi, forward, info)
# Compute resolution matrices for MNE
rm_mne = make_inverse_resolution_matrix(forward, inverse_operator,
                                        method='MNE', lambda2=1. / 3.**2)

# Compute first PCA component across PSFs and CTFs within labels.
n_comp = 5
stcs_psf_bf_aud = get_point_spread(
    rm_bf_aud, src, labels, mode='pca', n_comp=n_comp, norm=None,
    return_pca_vars=False)
stcs_ctf_bf_aud = get_cross_talk(
    rm_bf_aud, src, labels, mode='pca', n_comp=n_comp, norm=None,
    return_pca_vars=False)
stcs_psf_bf_noi = get_point_spread(
    rm_bf_noi, src, labels, mode='pca', n_comp=n_comp, norm=None,
    return_pca_vars=False)
del rm_bf_aud

stcs_psf_bf_vis = get_point_spread(
    rm_bf_vis, src, labels, mode='pca', n_comp=n_comp, norm=None,
    return_pca_vars=False)
stcs_ctf_bf_vis = get_cross_talk(
    rm_bf_vis, src, labels, mode='pca', n_comp=n_comp, norm=None,
    return_pca_vars=False)
del rm_bf_vis

# for MNE
stcs_psf_mne, pca_vars_mne = get_point_spread(
    rm_mne, src, labels, mode='pca', n_comp=n_comp, norm=None,
    return_pca_vars=True)
del rm_mne

for [psf, label] in zip(stcs_psf_bf_aud, labels):

    # only plot first component
    psf.crop(tmin=0., tmax=0.)
    max_val = np.abs(np.max(psf.data))

    ###############################################################################
    # Point-spread function for the lateral occipital label in the left hemisphere

    brain = psf.plot(subjects_dir=subjects_dir, subject='sample',
                     hemi='lh', views='lateral', background='white',
                     clim=dict(kind='value', pos_lims=(0, max_val / 2., max_val)))

    [brain.add_label(ll, borders=True) for ll in labels]

    brain.save_image(images_path + 'Fig7_BF_PSF_Aud' + label.name[:-3] + '.jpg')

    add_text = 'BF PSF Aud ' + label.name[:-3]
    brain.add_text(0.1, 0.9, add_text, 'title', font_size=16)

for [psf, label] in zip(stcs_psf_bf_vis, labels):

    # only plot first component
    psf.crop(tmin=0., tmax=0.)
    max_val = np.abs(np.max(psf.data))

    ###############################################################################
    # Point-spread function for the lateral occipital label in the left hemisphere

    brain = psf.plot(subjects_dir=subjects_dir, subject='sample',
                     hemi='lh', views='lateral', background='white',
                     clim=dict(kind='value', pos_lims=(0, max_val / 2., max_val)))

    [brain.add_label(ll, borders=True) for ll in labels]

    brain.save_image(images_path + 'Fig7_BF_PSF_Vis' + label.name[:-3] + '.jpg')

    add_text = 'BF PSF Vis ' + label.name[:-3]
    brain.add_text(0.1, 0.9, add_text, 'title', font_size=16)

# for baseline noise only produce PSF (CTFs are similar)
for [psf, label] in zip(stcs_psf_bf_noi, labels):

    # only plot first component
    psf.crop(tmin=0., tmax=0.)
    max_val = np.abs(np.max(psf.data))

    ###############################################################################
    # Point-spread function for the lateral occipital label in the left hemisphere

    brain = psf.plot(subjects_dir=subjects_dir, subject='sample',
                     hemi='lh', views='lateral', background='white',
                     clim=dict(kind='value', pos_lims=(0, max_val / 2., max_val)))

    [brain.add_label(ll, borders=True) for ll in labels]

    brain.save_image(images_path + 'Fig7_BF_PSF_Noi' + label.name[:-3] + '.jpg')

    add_text = 'BF PSF Noi ' + label.name[:-3]
    brain.add_text(0.1, 0.9, add_text, 'title', font_size=16)

# PSF only for MNE (same as CTF)
for [psf, label] in zip(stcs_psf_mne, labels):

    # only plot first component
    psf.crop(tmin=0., tmax=0.)
    max_val = np.abs(np.max(psf.data))

    ###############################################################################
    # Point-spread function for the lateral occipital label in the left hemisphere

    brain = psf.plot(subjects_dir=subjects_dir, subject='sample',
                     hemi='lh', views='lateral', background='white',
                     clim=dict(kind='value', pos_lims=(0, max_val / 2., max_val)))

    [brain.add_label(ll, borders=True) for ll in labels]

    brain.save_image(images_path + 'Fig7_BF_PSF_MNE' + label.name[:-3] + '.jpg')

    add_text = 'MNE PSF ' + label.name[:-3]
    brain.add_text(0.1, 0.9, add_text, 'title', font_size=16)

# Maximum for scaling across plots, across CTFs
# max_val = np.max([pp.data[:,0] for pp in stcs_ctf_lor])

for [ctf, label] in zip(stcs_ctf_bf_aud, labels):

    # only plot first component
    ctf.crop(tmin=0., tmax=0.)
    max_val = np.abs(np.max(ctf.data))

    brain = ctf.plot(subjects_dir=subjects_dir, subject='sample',
                     hemi='lh', views='lateral', background='white',
                     clim=dict(kind='value', pos_lims=(0, max_val / 2., max_val)))

    [brain.add_label(ll, borders=True) for ll in labels]

    brain.save_image(images_path + 'Fig7_BF_CTF_Aud' + label.name[:-3] + '.jpg')

    add_text = 'BF CTF Aud ' + label.name[:-3]
    brain.add_text(0.1, 0.9, add_text, 'title', font_size=16)

for [ctf, label] in zip(stcs_ctf_bf_vis, labels):

    # only plot first component
    ctf.crop(tmin=0., tmax=0.)
    max_val = np.abs(np.max(ctf.data))

    brain = ctf.plot(subjects_dir=subjects_dir, subject='sample',
                     hemi='lh', views='lateral', background='white',
                     clim=dict(kind='value', pos_lims=(0, max_val / 2., max_val)))

    [brain.add_label(ll, borders=True) for ll in labels]

    brain.save_image(images_path + 'Fig7_BF_CTF_Vis' + label.name[:-3] + '.jpg')

    add_text = 'BFr CTF Vis ' + label.name[:-3]
    brain.add_text(0.1, 0.9, add_text, 'title', font_size=16)
