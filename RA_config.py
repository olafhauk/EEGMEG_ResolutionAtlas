"""
=======================================
Config file for Wakeman&Henson data set
=======================================
"""

# not all variables need for resolution atlas

import os.path as op

##########################################################
## GENERAL
##########################################################

# path to unmaxfiltered raw data
raw_path_in = '/imaging/rh01/Methods/DanData/RawFIF/'

# output path for maxfiltered raw data
raw_path_sss = '/group/erp/data/olaf.hauk/MEG/WakemanHensonEMEG/data/RAW'

# path to MRI/MEG transformations
trans_path = '/group/erp/data/olaf.hauk/MEG/WakemanHensonEMEG/data/TRANS'

# where MRIs are
subjects_dir = '/group/erp/data/olaf.hauk/MEG/WakemanHensonEMEG/MRI/'

# raw_path_sss = '/imaging/rh01/Methods/DanData/Raw'

# subject numbers (from Roni's Matlab script)
subjs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# sessions per subject
sess = [1, 2, 3, 4, 5, 6]
# sess = [1, 2]

# reference session for trans option
ref_sess = 4

# path to results from Resolution metrics
resolution_path = '/group/erp/data/olaf.hauk/MEG/WakemanHensonEMEG/ResolutionAtlas/'
# subdirectory of resolution_path for subject-specific subdirectories for STC files
resolution_subdir = 'ResolutionMetrics'

figures_dir = '/group/erp/data/olaf.hauk/MEG/WakemanHensonEMEG/ResolutionAtlas/ResolutionMetrics/Figures'

##########################################################
## MAXFILTER
##########################################################

# maxfilter parameters
MF = {# 'NM_cmd': '/imaging/local/software/neuromag/bin/util/x86_64-pc-linux-gnu/maxfilter-2.2',
      'st_duration': [10.],  # list
      'origin': [(0., 0., 0.045)],  # list
      'ref_sess': 4}

st_string_stem = 'ST'
ori_string_stem = 'O'

# only for use of post-MF processing
# different py-scripts for MNE/NM maxfilter options
MF_method = '_MN'  # '_NM'


##########################################################
### FILTERING AND BAD CHANNELS
##########################################################

freq_l = 0.1  # Hz, high-pass filter
freq_h = 40.  # Hz, low-pass filter

##########################################################
### EPOCHING
##########################################################

# path to evoked responses
evo_path = '/group/erp/data/olaf.hauk/MEG/WakemanHensonEMEG/data/AVG/'

# epoch start/end latencies (s)
epo_t1, epo_t2 = -0.2, 0.5

# path to epoched data, e.g. for measurement info
epo_path = '/group/erp/data/olaf.hauk/MEG/WakemanHensonEMEG/data/EPO/'

##########################################################
### COVARIANCE
##########################################################

cov_path = '/group/erp/data/olaf.hauk/MEG/WakemanHensonEMEG/data/COV/'

cov_latwins = [
    [-0.4, 0.],
    [-0.3, 0.],
    [-0.2, 0.],
    [-0.1, 0.],
    [-0.2, 0.1],
    [-0.2, 0.25],
    [-0.2, 0.5],
    [0.05, 0.5],
    [0.05, 0.25]
]

# cov_methods = ['auto'] # list of lists if best covmat to be chosen
# cov_methods = ['empirical', 'shrunk', 'ledoit_wolf'] # list
cov_methods = ['empirical']


##########################################################
### SOURCE SPACE
##########################################################

src_spacing = 'oct6'

##########################################################
### BEM
##########################################################

# where BEM figures will be written to
bem_path = subjects_dir  # will be in subject/bem

bem_log_file = '/group/erp/data/olaf.hauk/MEG/WakemanHensonEMEG/MRI/BEMs/BEM_Model_and_SourceSpace.log'

bem_fig_dir = '/group/erp/data/olaf.hauk/MEG/WakemanHensonEMEG/data/Figures/BEM_plot'

coor_fig_dir = '/group/erp/data/olaf.hauk/MEG/WakemanHensonEMEG/data/Figures/CoorTrans_plot'

bem_ico = 4

bem_conductivity_1 = (0.3,)  # for single layer
bem_conductivity_3 = (0.3, 0.006, 0.3)  # for three layers


##########################################################
### FORWARD AND INVERSE OPERATORS
##########################################################

# for use in forward solution
fwd_filt_suf = '_01_400Hz_raw.fif'  # filter, suffix for filenames

fwd_st = 10.
fwd_origin = (0., 0., 0.045)

# for inverse operator
inv_filt_suf = '_01_400Hz_raw.fif'  # filter, suffix for filenames

# inv_lat_str = '-200_400ms'
inv_cov_latwin = [-0.2, 0.]

inv_cov_method = 'empirical'  # used for file filter, number of method name

# regularisation parameter for 'empirical' covariance matrices
inv_lambda_empirical = {'eeg': 0.1, 'mag': 0.1, 'grad': 0.1}


##########################################################
### INVERSE OPERATOR
##########################################################

stc_path = '/group/erp/data/olaf.hauk/MEG/WakemanHensonEMEG/ResolutionAtlas/STC/'
stc_method = 'MNE'

# regularisation SNR
stc_snr = 3.

# inv_modalities = ['EEGMEG', 'MEG', 'EEG'] # which invop files to apply
inv_modalities = ['EEGMEG', 'MEG']

# orientation constraint
inv_loose = [None, 0.2, 0.6]

# depth weighting
inv_depth = [None, 0.4, 0.8]

# brain to morph individual STCs to
stc_morph = 'fsaverage'


##########################################################
### LCMV BEAMFORMER
##########################################################

lcmv_datcov_latwin = [0.05, 0.25]

lcmv_noisecov_latwin = [-0.3, 0.]

# type of covariance matrix
lcmv_dat_method = 'empirical'

lcmv_noise_method = 'empirical'

# regularisation parameter for 'empirical' covariance matrices
lcmv_lambda_empirical = {'eeg': 0.1, 'mag': 0.1, 'grad': 0.1}

# which forward solutions to use
lcmv_modalities = ['EEGMEG', 'MEG']


##########################################################
### Resolution metrics
##########################################################

# Maxfilter parameters to use for filenames
res_st_duration = 10.
res_origin = [0., 0., 0.045]

res_lambda_empirical = {'eeg': 0.2, 'mag': 0.2, 'grad': 0.2}

res_cov_latwin = [-0.2, 0.]

# list of regularisation parameters
res_lambda2s = [1., 1. / 9.]

# list of sensor configurations for which to compute metrics
res_modalities = ['EEGMEG', 'MEG', 'EEG']

# list of tuples with measurement modalities to be comparede
modal_contr = [('EEGMEG', 'MEG'), ('EEGMEG', 'EEG'), ('MEG', 'EEG')]

# Type of inverse method, 'MNE' | 'sLORETA' | 'dSPM' | 'LCMV'
res_inv_types = ['MNE', 'sLORETA', 'dSPM', 'eLORETA', 'LCMV']

# Contrasts for methods comparisons
meth_contr = [('MNE', 'dSPM'), ('MNE', 'sLORETA'), ('MNE', 'eLORETA'),
              ('sLORETA', 'eLORETA'), ('dSPM', 'sLORETA'), ('MNE', 'LCMV'),
              ('dSPM', 'LCMV'), ('sLORETA', 'LCMV')]

# list of tuples with MNE depth-weightings to be compared
meth_contr_dep = [(0., .8)]


###########################################################
# FUNCTIONS FOR FILENAMES
###########################################################

def fname_epo(C, subject, st_duration, origin):

    subject = str(subject)

    st_string = _st_str(C.st_string_stem, st_duration)

    # for file name only use z-component in mm
    ori_string = C.ori_string_stem + str(origin[2] * 1000).split('.')[0]

    lat_str = _lat_str(C.epo_t1, C.epo_t2)

    filt_str = _filt_str(C.freq_l, C.freq_h)

    epo_fname = op.join(C.epo_path, subject, subject + '_' + st_string + '_' + ori_string + C.MF_method + 
                        '_' + lat_str + "_" + filt_str + "-epo.fif")

    return epo_fname


def fname_cov(C, subject, st_duration, origin, latwin, method, cov_i):

    subject = str(subject)

    st_string = _st_str(C.st_string_stem, st_duration)

    # for file name only use z-component in mm
    ori_string = C.ori_string_stem + str(origin[2] * 1000).split('.')[0]

    lat_str = _lat_str(C.epo_t1, C.epo_t2)

    filt_str = _filt_str(C.freq_l, C.freq_h)

    cov_str = _cov_str(latwin)

    method_now = method + str(cov_i)  # add the ranking of covmat

    cov_fname = op.join(C.cov_path, subject, subject + '_' + st_string + '_' + ori_string + 
                        C.MF_method + '_' + lat_str + '_' + filt_str + '_' + cov_str + '_' + method_now + '-cov.fif')

    return cov_fname


def fname_evo(C, subject, st_duration, origin):

    subject = str(subject)

    st_string = _st_str(C.st_string_stem, st_duration)

    # for file name only use z-component in mm
    ori_string = C.ori_string_stem + str(origin[2] * 1000).split('.')[0]

    lat_str = _lat_str(C.epo_t1, C.epo_t2)

    filt_str = _filt_str(C.freq_l, C.freq_h)

    evo_fname = op.join(C.evo_path, subject, subject + '_' + st_string + '_' + ori_string +
                        C.MF_method + '_' + lat_str + '_' + filt_str + '-ave.fif')

    return evo_fname


def fname_src_space(C, subject):

    subject = str(subject)

    src_fname = op.join(C.bem_path, subject, 'bem', subject + '_' + C.src_spacing + '-src.fif')

    return src_fname


def fname_BEM(C, subject, modality):
    # modality: e.g. EEG, MEG or EEGMEG

    subject = str(subject)

    bem_fname = op.join(C.bem_path, subject, 'bem', subject + '_' + modality + '-bem.fif')

    return bem_fname


def fname_ForwardSolution(C, subject, modality):
    # modality: e.g. EEG, MEG or EEGMEG

    subject = str(subject)

    fwd_fname = op.join(C.evo_path, subject, subject + '_' + C.MF_method + '_' + modality + '-fwd.fif')

    return fwd_fname


def fname_InverseOperator(C, subject, st_duration, origin, latwin, modality, loose, depth):
    # modality: e.g. EEG, MEG or EEGMEG

    subject = str(subject)

    st_string = _st_str(C.st_string_stem, st_duration)

    # for file name only use z-component in mm
    ori_string = C.ori_string_stem + str(origin[2]*1000).split('.')[0]

    filt_str = _filt_str(C.freq_l, C.freq_h)

    cov_str = _cov_str(latwin)

    if loose is None:
        loose = 0

    loo_str = '_loo%s' % str(int(100 * loose))

    if depth is None:
        depth = 0

    dep_str = '_dep%s' % str(int(100*depth))

    inv_fname = op.join(C.evo_path, subject, subject + '_' + st_string + '_' + ori_string +
                        C.MF_method + '_' + filt_str + '_' + cov_str + '_' + C.inv_cov_method +
                        '_' + modality + loo_str + dep_str + '-inv.fif')

    return inv_fname


def fname_STC(C, folder, subject, mytext):

    subject = str(subject)

    stc_fname = op.join(C.resolution_path, folder, subject, mytext)

    return stc_fname

###########################################################
# UTILITY FUNCTIONS
###########################################################


def _st_str(st_string_stem, st_duration):

    if st_duration is None:
        st_str = st_string_stem + '0'
    else:
        st_str = st_string_stem + str(st_duration).replace(".", "-")

    return st_str


def _lat_str(t1, t2):

    lat_tmp = str(int(1000 * t1)) + '_' + str(int(1000 * t2)) + "ms"
    lat_str = lat_tmp.replace(".", "")

    return lat_str


def _filt_str(l_freq, h_freq):

    filt_tmp = str(l_freq) + '_' + str(h_freq) + "Hz"
    filt_str = filt_tmp.replace(".", "")

    return filt_str


def _cov_str(latwin):

    if latwin is not None:
        tmin_str = str(int(1000 * latwin[0]))  # time interval as string in ms
        tmax_str = str(int(1000 * latwin[1]))
        cov_str = 'cov' + tmin_str + '_' + tmax_str
    else:
        cov_str = 'cov_id'  # covariance identity matrix

    return cov_str
