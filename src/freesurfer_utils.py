import os
import os.path as op
import re
from functools import partial

import numpy as np

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
    utils.check_env_var('FREESURFER_HOME', freesurfer_home)
    utils.check_env_var('SUBJECTS_DIR', subjects_dir)
    annot_files_exist = True
    for hemi in ['rh','lh']:
        annot_fname = op.join(subjects_dir, subject, 'label', '{}.{}.annot'.format(hemi, atlas))
        if overwrite_annot_file and op.isfile(annot_fname):
            os.remove(annot_fname)
        rs = utils.partial_run_script(locals(), print_only=print_only)
        rs(mris_ca_label)
        annot_files_exist = annot_files_exist and op.isfile(annot_fname)
    return annot_files_exist


def import_freesurfer_lut(subjects_dir='', fs_lut=''):
    """
    Import Look-up Table with colors and labels for anatomical regions.
    It's necessary that Freesurfer is installed and that the environmental
    variable 'FREESURFER_HOME' is present.

    Parameters
    ----------
    subjects_dir : str
        path to the subjects dir
    fs_lut : str
        path to file called FreeSurferColorLUT.txt

    Returns
    -------
    idx : list of int
        indices of regions
    label : list of str
        names of the brain regions
    rgba : numpy.ndarray
        one row is a brain region and the columns are the RGBA colors
    """
    if fs_lut == '':
        try:
            fs_home = os.environ['FREESURFER_HOME']
        except KeyError:
            raise OSError('FREESURFER_HOME not found')
        else:
            if fs_home != '':
                fs_lut = op.join(fs_home, 'FreeSurferColorLUT.txt')
            else:
                fs_lut = op.join(subjects_dir, 'FreeSurferColorLUT.txt')

    idx = np.genfromtxt(fs_lut, dtype=None, usecols=(0))
    label = utils.fix_bin_str_in_arr(np.genfromtxt(fs_lut, dtype=None, usecols=(1)))
    rgba = np.genfromtxt(fs_lut, dtype=None, usecols=(2, 3, 4, 5))
    lut = {'index':idx, 'label':label, 'RGBA':rgba}
    return lut


def extend_subcorticals_excludes(excludes=[]):
    excludes.extend(['ctx', 'Line', 'CSF', 'Lesion', 'undetermined', 'vessel', 'F3orb', 'aOg', 'lOg', 'mOg', 'pOg',
         'Porg', 'Aorg', 'F1', 'Chiasm', 'Corpus_Callosum', 'WM', 'wm', 'Dura', 'Brain-Stem', 'abnormality',
         'Epidermis', 'Tissue', 'Muscle', 'Cranium', 'Ear', 'Adipose', 'Spinal', 'Nerve', 'Bone', 'unknown',
         'Air', 'Fat', 'Tongue','Nasal', 'Globe', 'Teeth', 'Cbm', 'lh.', 'rh.', 'IliacA', 'SacralA',
         'ObturatorA', 'PudendalA', 'UmbilicalA', 'RectalA', 'IliacV', 'ObturatorV', 'PudendalV',
         'Lymph', 'AIPS', 'IPL', 'Visual', 'right_', 'left_', 'Brainstem', 'CST', 'AAA', 'choroid-plexus',
         'LongFas', 'Bundle', 'Gyrus', 'Tract', 'Cornea', 'Diploe', 'Humor', 'Lens', 'Table',
         'Periosteum', 'Endosteum', 'R-C-S', 'Iris', 'IntCapsule', 'Interior', 'Skull', 'White', 'white',
         'fossa', 'Scalp', 'Hematoma', 'brainstem', 'DCG', 'SCP', 'Floculus', 'CblumNodulus',
         'pathway', 'GC-DG', 'HATA', 'fimbria', 'ventricle', 'Ventricle', 'molecular', 'Cerebral_Cortex', 'Background',
         'Voxel-Unchanged', 'Head', 'Fluid', 'Sinus', 'Eustachian', 'V1', 'V2', 'BA', 'Aorta',
         'MT', 'Tumor', 'GrayMatter', 'SUSPICIOUS', 'fmajor', 'fminor', 'CC', 'LAntThalRadiation',
         'LUncinateFas', 'RAntThalRadiation', 'RUncinateFas', 'Vent', 'SLF', 'Cerebral-Exterior'])
    return excludes


def get_subcortical_regions(excludes=[], output_fname='', input_fname='', extra=[]):
    if input_fname != '' and op.isfile(input_fname):
        header_fname = '{0}_header{1}'.format(*os.path.splitext(input_fname))
        if op.isfile(header_fname):
            regions = utils.fix_bin_str_in_arr(np.genfromtxt(input_fname, dtype=None))
            header = utils.fix_bin_str_in_arr(np.genfromtxt(header_fname, dtype=None))
            return regions, header
        else:
            print("get_subcortical_regions: Can't find header file!")
    regions, header = [], []
    excludes = extend_subcorticals_excludes(excludes)
    lut = import_freesurfer_lut()
    compiled_excludes = re.compile('|'.join(excludes))
    _region_are_excluded = partial(region_are_excluded, compiled_excludes=compiled_excludes)
    for region in lut['label']:
        if not _region_are_excluded(region):
            regions.append(region)
            header.append(fix_region_name(region))
    for region in extra:
        if region not in regions:
            regions.append(region)
            header.append(fix_region_name(region))
    regions, header = sort_regions(regions, header)
    if output_fname != '':
        utils.write_arr_to_file(regions, output_fname)
        header_fname = '{0}_header{1}'.format(*os.path.splitext(output_fname))
        utils.write_arr_to_file(header, header_fname)
    return regions, header


def sort_regions(regions, header):
    header, regions = utils.sort_two_arrays(header, regions, utils.natural_keys)
    lh_indices = np.where(np.char.find(header, '-lh') > -1)[0]
    for ind in range(len(lh_indices)):
        lh_indice = lh_indices[ind]
        if '-lh' not in header[lh_indice]:
            raise Exception('Ahhh!!!!')
        rh_indice = header.index(header[lh_indice].replace('-lh', '-rh'))
        if rh_indice == lh_indice + 1:
            continue
        header.insert(lh_indice + 1, header[rh_indice])
        header.pop(rh_indice + 1)
        regions.insert(lh_indice + 1, regions[rh_indice])
        regions.pop(rh_indice + 1)
        lh_indices = np.where(np.char.find(header, '-lh') > -1)[0]
        # for ind, lh_indice in enumerate(lh_indices[ind + 1:]):
        #     if '-lh' in lh_indice:
        #         lh_indices[ind + 1] += 1

        # header[lh_indice + 1], header[rh_indice] = header[rh_indice], header[lh_indice + 1]
        # regions[lh_indice + 1], regions[rh_indice] = regions[rh_indice], regions[lh_indice + 1]
    return regions, header


def fix_region_name(region):
    for hemi_big, hemi_small in zip(['Right-', 'Left-'], ['-rh', '-lh']):
        if region.startswith(hemi_big):
            region = '{}{}'.format(region[len(hemi_big):], hemi_small)
            break
    region = region.replace("'", '')
    region = region.lower()
    return region


def region_are_excluded(region, compiled_excludes):
    if isinstance(region, np.bytes_):
        region = region.astype(str)
    return not compiled_excludes.search(region) is None