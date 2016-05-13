import os
import os.path as op
from src import utils

mris_ca_label = 'mris_ca_label {subject} {hemi} sphere.reg {freesurfer_home}/average/{hemi}.{atlas_type}.gcs {subjects_dir}/{subject}/label/{hemi}.{atlas}.annot -orig white'

def create_freesurfer_annotation_file(subject, atlas, subjects_dir='', freesurfer_home='', overwrite_annot_file=True, print_only=False):
    '''
    Creates the annot file by using the freesurfer mris_ca_label function

    Parameters
    ----------
    subject: subject name
    atlas: One of the three atlases included with freesurfer:
        Possible values are aparc.DKTatlas40, aparc.a2009s or aparc
        https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation
    subjects_dir: subjects dir. If empty, get it from the environ
    freesurfer_home: freesurfer home. If empty, get it from the environ
    overwrite_annot_file: If False and the annot file already exist, the function return True. If True, the function
        delete first the annot files if exist.
    print_only: If True, the function will just prints the command without executing it

    Returns
    -------
        True if the new annot files exist
    '''
    atlas_types = {'aparc': 'curvature.buckner40.filled.desikan_killiany',
                   'aparc.a2009s': 'destrieux.simple.2009-07-28',
                   'aparc.DKTatlas40': 'DKTatlas40'}
    atlas_type = atlas_types[atlas]
    check_env_var('FREESURFER_HOME', freesurfer_home)
    check_env_var('SUBJECTS_DIR', subjects_dir)
    annot_files_exist = True
    for hemi in ['rh','lh']:
        annot_fname = op.join(subjects_dir, subject, 'label', '{}.{}.annot'.format(hemi, atlas))
        if overwrite_annot_file and op.isfile(annot_fname):
            os.remove(annot_fname)
        rs = utils.partial_run_script(locals(), print_only=print_only)
        rs(mris_ca_label)
        annot_files_exist = annot_files_exist and op.isfile(annot_fname)
    return annot_files_exist


def check_env_var(var_name, var_val):
    if var_val == '':
        var_val = os.environ.get(var_name, '')
        if var_val  == '':
            raise Exception('No {}!'.format(var_name))