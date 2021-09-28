"""
============================================================
PSFs and CTFs for 3 example ROIs
============================================================

Temporal Pole, Pars Triangularis, Pars Opercularis
"""

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import (read_inverse_operator,
                              make_inverse_resolution_matrix,
                              get_point_spread, get_cross_talk)
from mne.beamformer import make_lcmv, make_lcmv_resolution_matrix

from mne.viz import circular_layout, plot_connectivity_circle

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

# Labels used in this example
use_labels = ['parsopercularis-lh', 'parstriangularis-lh', 'middletemporal-lh']

# Compute resolution matrices for MNE
rm_mne = make_inverse_resolution_matrix(forward, inverse_operator,
                                        method='MNE', lambda2=1. / 3.**2)
src = inverse_operator['src']

###############################################################################
# Read labels
# --------------------------------------------------
#
# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot('sample', parc='aparc',
                                    subjects_dir=subjects_dir)

# get just the desired labels
labels = [ll for ll in labels if ll.name in use_labels]

# split middle temporal, use anterior quarter
# note sequence here is alphabetical
subjects_dir = sample.data_path() + '/subjects'
labels[0] = labels[0].split(4, subjects_dir=subjects_dir)[-1]

n_labels = len(labels)

label_names = [label.name for label in labels]

###############################################################################
# Compute point-spread function summaries (PCA) for all labels
# ------------------------------------------------------------
#
# We summarise the PSFs per label by their first five principal components, and
# use the first component to evaluate label-to-label leakage below.

# Compute first PCA component across PSFs within labels.
# Note the differences in explained variance, probably due to different
# spatial extents of labels.
n_comp = 5
stcs_psf_mne, pca_vars_mne = get_point_spread(
    rm_mne, src, labels, mode='pca', n_comp=n_comp, norm=None,
    return_pca_vars=True)
n_verts = rm_mne.shape[0]
del rm_mne

###############################################################################
# We can show the explained variances of principal components per label. Note
# how they differ across labels, most likely due to their varying spatial
# extent.

with np.printoptions(precision=1):
    for [name, var] in zip(label_names, pca_vars_mne):
        print(f'{name}: {var.sum():.1f}% {var}')

###############################################################################
# The output shows the summed variance explained by the first five principal
# components as well as the explained variances of the individual components.
#
# Evaluate leakage based on label-to-label PSF correlations
# ---------------------------------------------------------
#
# Note that correlations ignore the overall amplitude of PSFs, i.e. they do
# not show which region will potentially be the bigger "leaker".

# get PSFs from Source Estimate objects into matrix
psfs_mat = np.zeros([n_labels, n_verts])
# Leakage matrix for MNE, get first principal component per label
for [i, s] in enumerate(stcs_psf_mne):
    psfs_mat[i, :] = s.data[:, 0]
# Compute label-to-label leakage as Pearson correlation of PSFs
# Sign of correlation is arbitrary, so take absolute values
leakage_mne = np.abs(np.corrcoef(psfs_mat))

print('Correlations:')
with np.printoptions(precision=1):
    for ii in [[0, 1], [0, 2], [1, 2]]:
        print(f'{label_names[ii[0]]} | {label_names[ii[1]]}: {leakage_mne[ii[0], ii[1]]:.1f}')

###############################################################################
# Most leakage occurs for neighbouring regions, but also for deeper regions
# across hemispheres.
#
# Save the figure (optional)
# --------------------------
#
# Matplotlib controls figure facecolor separately for interactive display
# versus for saved figures. Thus when saving you must specify ``facecolor``,
# else your labels, title, etc will not be visible::
#
#     >>> fname_fig = data_path + '/MEG/sample/plot_label_leakage.png'
#     >>> fig.savefig(fname_fig, facecolor='black')
#
# Plot PSFs for individual labels
# -------------------------------
#
# Let us confirm for left and right lateral occipital lobes that there is
# indeed no leakage between them, as indicated by the correlation graph.
# We can plot the summary PSFs for both labels to examine the spatial extent of
# their leakage.

for [psf, label] in zip(stcs_psf_mne, labels):

    # only plot first component
    psf.crop(tmin=0., tmax=0.)
    # Maximum for scaling
    max_val = np.max(np.abs(psf.data[:,0]))

    brain = psf.plot(subjects_dir=subjects_dir, subject='sample',
                     hemi='lh', views='lateral', background='white',
                     clim=dict(kind='value', pos_lims=(0, max_val / 2., max_val)))

    # show all labels as borders
    [brain.add_label(ll, borders=True) for ll in labels]

    brain.save_image(images_path + 'Fig3_MNE_PSF_' + label.name[:-3] + '.jpg')

    brain.add_text(0.1, 0.9, label.name[:-3], 'title', font_size=16)

###############################################################################
# Now check if things look better for a method with zero peak localisation
# error, such as eLORETA. In this case, PSFs and CTFs will be different and we
# have to look at them separately

# method for MNE-Python and label for filename
method = ('eLORETA', 'eLOR')

# Compute resolution matrices for eLORETA
rm_lor = make_inverse_resolution_matrix(forward, inverse_operator,
                                        method=method[0], lambda2=1. / 3.**2)

# Compute first PCA component across PSFs and CTFs within labels.
n_comp = 5
stcs_psf_lor = get_point_spread(
    rm_lor, src, labels, mode='pca', n_comp=n_comp, norm=None,
    return_pca_vars=False)
stcs_ctf_lor = get_cross_talk(
    rm_lor, src, labels, mode='pca', n_comp=n_comp, norm=None,
    return_pca_vars=False)
del rm_lor

# Maximum for scaling across plots, across PSFs
# max_val = np.max([pp.data[:,0] for pp in stcs_psf_lor])

for [psf, label] in zip(stcs_psf_lor, labels):

    # only plot first component
    psf.crop(tmin=0., tmax=0.)
    max_val = np.max(np.abs(psf.data[:,0]))

    ###############################################################################
    # Point-spread function for the lateral occipital label in the left hemisphere

    brain = psf.plot(subjects_dir=subjects_dir, subject='sample',
                     hemi='lh', views='lateral', background='white',
                     clim=dict(kind='value', pos_lims=(0, max_val / 2., max_val)))

    [brain.add_label(ll, borders=True) for ll in labels]

    brain.save_image(images_path + 'Fig3_%s_PSF_%s%s' % (label.name[:-3], method[1], '.jpg'))

    add_text = 'Lor PSF ' + label.name[:-3]
    brain.add_text(0.1, 0.9, add_text, 'title', font_size=16)

# Maximum for scaling across plots, across CTFs
# max_val = np.max([pp.data[:,0] for pp in stcs_ctf_lor])

for [ctf, label] in zip(stcs_ctf_lor, labels):

    # only plot first component
    ctf.crop(tmin=0., tmax=0.)
    max_val = np.max(np.abs(ctf.data[:,0]))

    brain = ctf.plot(subjects_dir=subjects_dir, subject='sample',
                     hemi='lh', views='lateral', background='white',
                     clim=dict(kind='value', pos_lims=(0, max_val / 2., max_val)))

    [brain.add_label(ll, borders=True) for ll in labels]

    brain.save_image(images_path + 'Fig3_%s_CTF_%s%s' % (label.name[:-3], method[1], '.jpg'))

    add_text = 'Lor CTF ' + label.name[:-3]
    brain.add_text(0.1, 0.9, add_text, 'title', font_size=16)
