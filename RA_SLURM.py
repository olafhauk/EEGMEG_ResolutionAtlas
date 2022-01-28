#!/imaging/local/software/miniconda/envs/mne0.23/bin/python
"""
==========================================
Submit sbatch jobs for WH Resolution analysis
==========================================

"""
print(__doc__)

import subprocess
from os import path as op

# wrapper to run python script via qsub. Python3
fname_wrap = op.join('/', 'home', 'olaf', 'MEG', 'WakemanHensonEMEG', 'Python2SLURM_ResolutionAtlas.sh')

# indices of subjects to process
subjs = range(0, 16)

job_list = \
[
    #####
    # COMPUTING and MORPHING
    # #####
    # Compute resolution metrics
    {'N':   'RA_Metrics',                                       # job name
     'Py':  'RA_ResolutionMetrics',                         # Python script
     'Cf':  'RA_config',                     # configuration script
     'Ss':  subjs,                                          # subject indices
     'mem': '4G',                                          # memory for qsub process
     'dep': ''},
    # Compute contrasts among methods
    {'N':   'RA_Diff',                                       # job name
     'Py':  'RA_ResolutionMetrics_differences',                         # Python script
     'Cf':  'RA_config',                     # configuration script
     'Ss':  subjs,                                          # subject indices
     'mem': '1G',                                          # memory for qsub process
     'dep': 'RA_Metrics'},
    # Morph STCs to fsaverage
    {'N':   'RA_Mph',                                       # job name
     'Py':  'RA_MorphSTC',                         # Python script
     'Cf':  'RA_config',                     # configuration script
     'Ss':  subjs,                                          # subject indices
     'mem': '1G',                                          # memory for qsub process
     'dep': 'RA_Diff'},

    # Grand-average STCs
    # {'N':   'RA_Avg',                                       # job name
    #  'Py':  'RA_AvgSTCs',                         # Python script
    #  'Cf':  'RA_config',                     # configuration script
    #  'Ss':  [99],                                          # subject indices
    #  'mem': '1G',                                          # memory for qsub process
    #  'dep': ''},
]

# directory where python scripts are
dir_py = op.join('/', 'home', 'olaf', 'MEG', 'WakemanHensonEMEG', 'ResolutionAtlas')

# directory for qsub output
dir_qsub = op.join('/', 'home', 'olaf', 'MEG', 'WakemanHensonEMEG', 'ResolutionAtlas', 'sbatch_out')

# keep track of qsub Job IDs
Job_IDs = {}

for job in job_list:

    for Ss in job['Ss']:

        Ss = str(Ss)  # turn into string for filenames etc.

        N = Ss + job['N']
        Py = op.join(dir_py, job['Py'])
        Cf = job['Cf']
        mem = job['mem']

        # files for sbatch output
        file_out = op.join(dir_qsub, job['N'] + '_' + Cf + '-%s.out' % str(Ss))
        file_err = op.join(dir_qsub, job['N'] + '_' + Cf + '-%s.err' % str(Ss))

        # if job dependent of previous job, get Job ID and produce command
        if 'dep' in job:  # if dependency on previous job specified
            if job['dep'] == '':
                dep_str = ''
            else:
                job_id = Job_IDs[Ss + job['dep'], Ss]
                dep_str = '--dependency=afterok:%s' % (job_id)
        else:
            dep_str = ''

        if 'node' in job:  # if node constraint present (e.g. Maxfilter)
            node_str = job['node']
        else:
            node_str = ''

        if 'var' in job:  # if variables for python script specified
            var_str = job['var']
        else:
            var_str = ''

        # sbatch command string to be executed
        sbatch_cmd = 'sbatch \
                        -o %s \
                        -e %s \
                        --export=pycmd="%s.py %s",subj_idx=%s,var=%s \
                        --mem-per-cpu=%s -t 1-00:00:00 -J %s %s\
                        %s' \
                        % (file_out, file_err, Py, Cf, Ss, var_str, mem, N, dep_str, fname_wrap)

        # format string for display
        print_str = sbatch_cmd.replace(' ' * 25, '  ')
        print('\n%s\n' % print_str)

        # execute qsub command
        proc = subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE, shell=True)

        # get linux output
        (out, err) = proc.communicate()

        # keep track of Job IDs from sbatch, for dependencies
        Job_IDs[N, Ss] = str(int(out.split()[-1]))

# Done