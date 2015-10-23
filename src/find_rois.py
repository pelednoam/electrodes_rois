import numpy as np
import os
from functools import partial
import re
import nibabel as nib
from collections import Counter
from scipy.spatial.distance import cdist
import mne
from mne.surface import read_surface
from itertools import product
import csv
import glob
import traceback
import shutil
import string


def identify_roi_from_atlas(elecs_names, elecs_pos, elcs_ori, approx=4, elc_length=1, nei_dimensions=None,
    atlas=None, enlarge_if_no_hit=False, subjects_dir=None, subject=None, n_jobs=6):

    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    # get the segmentation file
    asegf = os.path.join(subjects_dir, subject, 'mri', 'aseg.mgz')
    if not os.path.isfile(asegf):
        asegf = os.path.join(subjects_dir, subject, 'mri', 'aparc+aseg.mgz')
    aseg_header = nib.load(asegf).get_data()

    # load the surfaces and annotation
    # uses the pial surface, this change is pushed to MNE python
    parcels, pia_verts = {}, {}
    for hemi in ['rh', 'lh']:
        parcels[hemi] = mne.read_labels_from_annot(subject, parc=atlas, hemi=hemi,
            subjects_dir=subjects_dir, surf_name='pial')
        pia_verts[hemi], _ = nib.freesurfer.read_geometry(
            os.path.join(subjects_dir, subject, 'surf', '{}.pial'.format(hemi)))
    pia = np.vstack((pia_verts['lh'], pia_verts['rh']))
    len_lh_pia = len(pia_verts['lh'])
    lut = import_freesurfer_lut()

    elecs = []
    for elec_pos, elec_name, elc_ori in zip(elecs_pos, elecs_names, elcs_ori):
        regions, regions_hits, subcortical_regions, subcortical_hits = \
            identify_roi_from_atlas_per_electrode(elec_pos, pia, len_lh_pia,
                parcels, lut, aseg_header, approx, elc_length, nei_dimensions, elc_ori,
                enlarge_if_no_hit, subjects_dir, subject)
        regions_hits, subcortical_hits = np.array(regions_hits), np.array(subcortical_hits)
        regions_probs = np.hstack((regions_hits, subcortical_hits)) / float(np.sum(regions_hits) + np.sum(subcortical_hits))
        elecs.append({'name': elec_name, 'cortical_rois': regions, 'subcortical_rois': subcortical_regions,
            'cortical_probs': regions_probs[:len(regions)],
            'subcortical_probs': regions_probs[len(regions):]})

    return elecs


def identify_roi_from_atlas_per_electrode(pos, pia, len_lh_pia, parcels, lut, aseg_header,
    approx=4, elc_length=1, nei_dimensions=None, elc_ori=None,
    enlarge_if_no_hit=False, subjects_dir=None, subject=None):
    '''
    Find the surface labels contacted by an electrode at this position
    in RAS space.

    Parameters
    ----------

    pos : np.ndarray
        1x3 matrix holding position of the electrode to identify
    approx : int
        Number of millimeters error radius
    atlas : str or None
        The string containing the name of the surface parcellation,
        does not apply to subcortical structures. If None, aparc is used.
    '''

    # find closest vertex
    closest_vert = np.argmin(cdist(pia, [pos]))

    # we force the label to only contact one hemisphere even if it is
    # beyond the extent of the medial surface
    hemi_str = 'lh' if closest_vert<len_lh_pia else 'rh'
    hemi_code = 0 if hemi_str=='lh' else 1

    if hemi_str == 'rh':
        closest_vert -= len_lh_pia

    surf_fname = os.path.join(subjects_dir, subject, 'surf', hemi_str + '.pial')
    verts, _ = read_surface(surf_fname)
    closest_vert_pos = verts[closest_vert]

    we_have_a_hit = False
    while not we_have_a_hit:
        # grow the area of surface surrounding the vertex
        radius_label, = mne.grow_labels(subject, closest_vert, approx, hemi_code,
            subjects_dir=subjects_dir, surface='pial')

        bins = calc_neighbors(closest_vert_pos, approx + elc_length, nei_dimensions, calc_bins=True)
        elc_line = [pos + elc_ori*t for t in np.linspace(-elc_length/2.0, elc_length/2.0, 100)]

        # excludes=['white', 'WM', 'Unknown', 'White', 'unknown', 'Cerebral-Cortex']
        excludes=['Unknown', 'unknown', 'Cerebral-Cortex']
        compiled_excludes = re.compile('|'.join(excludes))
        _region_is_excluded = partial(region_is_excluded, compiled_excludes=compiled_excludes)

        regions, regions_hits = [], []
        for parcel in parcels[hemi_str]:
            if _region_is_excluded(str(parcel.name)):
                continue
            intersect_verts = np.intersect1d(parcel.vertices, radius_label.vertices)
            if len(intersect_verts)>0:
                hits = calc_hits_in_neighbors_from_line(elc_line, verts[intersect_verts], bins, approx)
                if hits > 0:
                    regions_hits.append(hits)
                    #force convert from unicode
                    regions.append(str(parcel.name))

        subcortical_regions, subcortical_hits = identify_roi_from_aparc(pos, elc_line, elc_length, lut, aseg_header, approx=approx,
            nei_dimensions=nei_dimensions, subcortical_only=True, excludes=excludes)

        we_have_a_hit = not electrode_is_only_in_white_matter(regions, subcortical_regions) or not enlarge_if_no_hit
        if not we_have_a_hit:
            approx += .5
            elc_length += 1

    return regions, regions_hits, subcortical_regions, subcortical_hits


def electrode_is_only_in_white_matter(regions, subcortical_regions):
    return len(regions) == 0 and len(subcortical_regions)==1 and \
        subcortical_regions[0] in ['{}-Cerebral-White-Matter'.format(hemi) \
        for hemi in ['Right', 'Left']]

# def calc_hits_in_neighbors_from_line(line, points, neighb, approx):
#     bins = [np.sort(np.unique(neighb[:, idim])) for idim in range(points.shape[1])]
#     hist, bin_edges = np.histogramdd(points, bins=bins, normed=False)
#     hist_bin_centers_list = [bin_edges[d][:-1] + (bin_edges[d][1:] - bin_edges[d][:-1])/2. for d in range(len(bin_edges))]
#     indices3d = product(*[range(len(bin_edges[d])-1) for d in range(len(bin_edges))])
#     ret_indices = []
#     for i,j,k in indices3d:
#         bin_center = [bin[ind] for bin,ind in zip(hist_bin_centers_list, (i, j, k))]
#         dist = min(cdist(line, [bin_center]))
#         if hist[i,j,k]>0 and dist < approx:
#             ret_indices.append((i,j,k))
#     hits = len(ret_indices)


def calc_hits_in_neighbors_from_line(line, points, neighb, approx):
    bins = [np.sort(np.unique(neighb[:, idim])) for idim in range(points.shape[1])]
    hist, bin_edges = np.histogramdd(points, bins=bins, normed=False)
    hist_bin_centers_list = [bin_edges[d][:-1] + (bin_edges[d][1:] - bin_edges[d][:-1])/2.
        for d in range(len(bin_edges))]
    bin_centers = list(product(*hist_bin_centers_list))
    dists = np.min(cdist(line, bin_centers), 0).reshape(hist.shape)
    hits = len(np.where((hist > 0) & (dists<approx))[0])
    return hits


def identify_roi_from_aparc(pos, elc_line, elc_length, lut, aseg_header, approx=4, nei_dimensions=None,
    subcortical_only=False, excludes=None):
    '''
    Find the volumetric labels contacted by an electrode at this position
    in RAS space.

    Parameters
    ----------

    pos : np.ndarray
        1x3 matrix holding position of the electrode to identify
    approx : int
        Number of millimeters error radius
    subcortical_only : bool
        if True, exclude cortical labels
    '''

    def find_neighboring_regions(pos, elc_length, elc_line, aseg_header, lut, approx, dimensions, excludes):
        compiled_excludes = re.compile('|'.join(excludes))
        _region_is_excluded = partial(region_is_excluded, compiled_excludes=compiled_excludes)
        neighb = calc_neighbors(pos, elc_length + approx, dimensions)
        dists = np.min(cdist(elc_line, neighb), 0)
        # import matplotlib.pyplot as plt
        # plt.hist(cdist([pos], neighb)[0])
        # plt.hist(dists)
        neighb = neighb[np.where(dists<approx)]
        regions = []
        for nei in neighb:
            nei_regions = set()
            round_neis = round_coords(nei)
            for round_nei in round_neis:
                cx, cy, cz = map(int, round_nei)
                d_type = aseg_header[cx, cy, cz]
                label_index = np.where(lut['index']==d_type)[0][0]
                region = lut['label'][label_index]
                if not _region_is_excluded(region):
                    nei_regions.add(region)
            for region in nei_regions:
                regions.append(region)

        # regions = exclude_regions(regions, excludes)
        cnt = Counter(regions)
        regions, hits = [], []
        for region, count in cnt.iteritems():
            regions.append(region)
            hits.append(count)
        return regions, hits
        # return np.unique(regions).tolist()

    def round_coords(pos):
        rounds = [[np.floor(pos[d]), np.ceil(pos[d])] for d in range(3)]
        coords = list(product(*rounds))
        return coords

    def to_ras(points):
        RAS_AFF = np.array([[-1, 0, 0, 128],
            [0, 0, -1, 128],
            [0, 1, 0, 128],
            [0, 0, 0, 1]])
        return [np.around(np.dot(RAS_AFF, np.append(p, 1)))[:3] for p in points]


    if subcortical_only:
        excludes.append('ctx')

    RAS_AFF = np.array([[-1, 0, 0, 128],
        [0, 0, -1, 128],
        [0, 1, 0, 128],
        [0, 0, 0, 1]])
    # ras_pos = np.around(np.dot(RAS_AFF, np.append(pos, 1)))[:3]
    ras_pos = np.dot(RAS_AFF, np.append(pos, 1))[:3]
    ras_elc_line = [np.dot(RAS_AFF, np.append(p, 1))[:3] for p in elc_line]
    return find_neighboring_regions(ras_pos, elc_length, ras_elc_line, aseg_header, lut, approx, nei_dimensions, excludes)


def import_freesurfer_lut(fs_lut=None):
    """
    Import Look-up Table with colors and labels for anatomical regions.
    It's necessary that Freesurfer is installed and that the environmental
    variable 'FREESURFER_HOME' is present.

    Parameters
    ----------
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
    if fs_lut is None:
        try:
            fs_home = os.environ['FREESURFER_HOME']
        except KeyError:
            raise OSError('FREESURFER_HOME not found')
        else:
            fs_lut = os.path.join(fs_home, 'FreeSurferColorLUT.txt')

    idx = np.genfromtxt(fs_lut, dtype=None, usecols=(0))
    label = np.genfromtxt(fs_lut, dtype=None, usecols=(1))
    rgba = np.genfromtxt(fs_lut, dtype=None, usecols=(2, 3, 4, 5))
    lut = {'index':idx, 'label':label, 'RGBA':rgba}
    return lut


def exclude_regions(regions, excludes):
    if (excludes):
        excluded = compile('|'.join(excludes))
        regions = [x for x in regions if not excluded.search(x)]
    return regions


def region_is_excluded(region, compiled_excludes):
    return not compiled_excludes.search(region) is None


def calc_neighbors(pos, approx=None, dimensions=None, calc_bins=False):
    if not approx is None:
        sz = int(np.around(approx * 2 + (2 if calc_bins else 1)))
        sx = sy = sz
    elif not dimensions is None:
        sx, sy, sz = dimensions
    else:
        raise Exception('approx and dimensions are None!')

    x, y, z = np.meshgrid(range(sx), range(sy), range(sz))

    # approx is in units of millimeters as long as we use the RAS space
    # segmentation
    neighb = np.vstack((np.reshape(x, (1, sx ** 3)),
        np.reshape(y, (1, sy ** 3)),
        np.reshape(z, (1, sz ** 3)))).T - approx

    if calc_bins:
        neighb = neighb.astype(float)
        neighb -= 0.5

    return pos + neighb


def get_electrodes(subject, elecs_dir='', delimiter=','):
    if elecs_dir=='':
        elecs_dir = get_electrodes_dir()
    elec_file = os.path.join(elecs_dir, '{}.csv'.format(subject))
    data = np.genfromtxt(elec_file, dtype=str, delimiter=delimiter)
    pos = data[1:, 1:].astype(float)
    names = data[1:, 0]
    names = np.array([name.strip() for name in names])
    return names, pos


def get_electrodes_dir():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    elec_dir = os.path.join(os.path.split(curr_dir)[0], 'electrodes')
    return elec_dir


def write_results_to_csv(results, atlas, elecs_dir='', post_fix='',
        write_only_cortical=False, write_only_subcortical=False):

    if elecs_dir=='':
        elecs_dir = get_electrodes_dir()

    cortical_rois, subcortical_rois = [], []
    for elecs in results.itervalues():
        for elc in elecs:
            cortical_rois.extend(elc['cortical_rois'])
            subcortical_rois.extend(elc['subcortical_rois'])
    cortical_rois = list(np.unique(cortical_rois))
    subcortical_rois = list(np.unique(subcortical_rois))

    for subject, elecs in results.iteritems():
        write_values(elecs, ['electrode'] + cortical_rois + subcortical_rois, [cortical_rois, subcortical_rois],
            ['cortical_rois','subcortical_rois'], ['cortical_probs', 'subcortical_probs'],
            os.path.join(elecs_dir, '{}_{}_electrodes_all_rois{}.csv'.format(subject, atlas, post_fix)))

        if write_only_cortical:
            write_values(elecs, ['electrode'] + cortical_rois, [cortical_rois],['cortical_rois'], ['cortical_probs'],
                os.path.join(elecs_dir, '{}_{}_electrodes_cortical_rois{}.csv'.format(subject, atlas, post_fix)))

        if write_only_subcortical:
            write_values(elecs, ['electrode']  + subcortical_rois, [subcortical_rois],
                ['subcortical_rois'], ['subcortical_probs'],
                os.path.join(elecs_dir, '{}_{}_electrodes_subcortical_rois{}.csv'.format(subject, atlas, post_fix)))


def write_values(elecs, header, rois_arr, rois_names, probs_names, file_name):
    with open(file_name, 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerow(header)
        for elc in elecs:
            values = [elc['name']]
            for rois, rois_field, prob_field in zip(rois_arr, rois_names, probs_names):
                for col, roi in enumerate(rois):
                    if roi in elc[rois_field]:
                        index = elc[rois_field].index(roi)
                        values.append(str(elc[prob_field][index]))
                    else:
                        values.append(0.)
            writer.writerow(values)


def get_electrodes_orientation(elecs_names, elecs_pos):
    elcs_oris = []
    for elc_name, elc_pos in zip(elecs_names, elecs_pos):
        next_elc = '{}{}'.format(elc_name[:3], int(elc_name[-1])+1)
        ori = 1
        if next_elc not in elecs_names:
            next_elc = '{}{}'.format(elc_name[:3], int(elc_name[-1])-1)
            ori = -1
        next_elc_index = np.where(elecs_names==next_elc)[0][0]
        next_elc_pos = elecs_pos[next_elc_index]
        dist = np.linalg.norm(next_elc_pos-elc_pos)
        elc_ori = ori * (next_elc_pos-elc_pos) / dist # norm(elc_ori)=1mm
        elcs_oris.append(elc_ori)
        # print(elc_name, elc_pos, next_elc, next_elc_pos, elc_line(1))
    return elcs_oris


def get_subjects():
    files = glob.glob(os.path.join(get_electrodes_dir(), '*.csv'))
    names = set()
    for full_file_name in files:
        file_name = os.path.split(full_file_name)[1]
        if '_' not in file_name:
            names.add(os.path.splitext(file_name)[0])
    return names


def check_for_annot_file(subject, subjects_dir, atlas, fsaverage='fsaverage5c', overwrite=False, n_jobs=6):
    annot_file = os.path.join(subjects_dir, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))
    if overwrite or not os.path.isfile(annot_file.format(hemi='rh')) or not os.path.isfile(annot_file.format(hemi='lh')):
        morph_labels_from_fsaverage(subject, subjects_dir, atlas, n_jobs=n_jobs, fsaverage=fsaverage, overwrite=overwrite)
        labels_to_annot(subject, subjects_dir, atlas, overwrite=overwrite)


def morph_labels_from_fsaverage(subject, subjects_dir='', aparc_name='aparc250', fs_labels_fol='', sub_labels_fol='', n_jobs=6, fsaverage='fsaverage', overwrite=False):
    if subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    subject_dir = os.path.join(subjects_dir, subject)
    labels_fol = os.path.join(subjects_dir, fsaverage, 'label', aparc_name) if fs_labels_fol=='' else fs_labels_fol
    sub_labels_fol = os.path.join(subject_dir, 'label', aparc_name) if sub_labels_fol=='' else sub_labels_fol
    if not os.path.isdir(sub_labels_fol):
        os.makedirs(sub_labels_fol)
    for label_file in glob.glob(os.path.join(labels_fol, '*.label')):
        local_label_name = os.path.join(sub_labels_fol, '{}.label'.format(os.path.splitext(os.path.split(label_file)[1])[0]))
        if os.path.isfile(local_label_name) and overwrite:
            os.remove(local_label_name)
        if not os.path.isfile(local_label_name) or overwrite:
            fs_label = mne.read_label(label_file)
            fs_label.values.fill(1.0)
            sub_label = fs_label.morph(fsaverage, subject, grade=None, n_jobs=n_jobs, subjects_dir=subjects_dir)
            sub_label.save(local_label_name)


def labels_to_annot(subject, subjects_dir='', aparc_name='aparc250', labels_fol='', overwrite=True):
    if subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    subject_dir = os.path.join(subjects_dir, subject)
    labels_fol = os.path.join(subject_dir, 'label', aparc_name) if labels_fol=='' else labels_fol
    labels = []
    for label_file in glob.glob(os.path.join(labels_fol, '*.label')):
        # print label_file
        try:
            label = mne.read_label(label_file)
            labels.append(label)
        except:
            print('error reading the label!')
            print(traceback.format_exc())

    mne.write_labels_to_annot(subject=subject, labels=labels, parc=aparc_name, overwrite=overwrite,
                              subjects_dir=subjects_dir)


def get_all_labels_and_segmentations(subject, atlas):
    percs, segs = [], []
    all_segs = import_freesurfer_lut()['label']
    excludes=['Unknown', 'unknown', 'Cerebral-Cortex', 'ctx']
    compiled_excludes = re.compile('|'.join(excludes))
    _region_is_excluded = partial(region_is_excluded, compiled_excludes=compiled_excludes)
    for seg in all_segs:
        if not _region_is_excluded(seg):
            segs.append(seg)

    for hemi in ['rh', 'lh']:
        annot_file = os.path.join(subjects_dir, subject, 'label', '{}.{}.annot'.format(hemi, atlas))
        labels = mne.read_labels_from_annot(subject, surf_name='pial', annot_fname=annot_file)
        for label in labels:
            percs.append(label.name)

    return percs, segs


def prepare_local_subjects_folder(neccesary_files, subject, remote_subject_dir, local_subjects_dir, print_traceback=False):
    local_subject_dir = os.path.join(local_subjects_dir, subject)
    for fol, files in neccesary_files.iteritems():
        if not os.path.isdir(os.path.join(local_subject_dir, fol)):
            os.makedirs(os.path.join(local_subject_dir, fol))
        for file_name in files:
            try:
                if not os.path.isfile(os.path.join(local_subject_dir, fol, file_name)):
                    shutil.copyfile(os.path.join(remote_subject_dir, fol, file_name),
                                os.path.join(local_subject_dir, fol, file_name))
            except:
                if print_traceback:
                    print(traceback.format_exc())
    all_files_exists = True
    for fol, files in neccesary_files.iteritems():
        for file_name in files:
            if not os.path.isfile(os.path.join(local_subject_dir, fol, file_name)):
                print("The file {} doesn't exist in the local subjects folder!!!".format(file_name))
                all_files_exists = False
    if not all_files_exists:
        raise Exception('Not all files exist in the local subject folder!!!')


def run_for_all_subjects(subjects, atlas, error_radius, elc_length, subjects_dir, template_brain='fsaverage',
        neccesary_files=None, remote_subject_dir_template='', output_files_post_fix='', overwrite=False,
        overwrite_annotation=False, write_only_cortical=False, write_only_subcortical=False,
        enlarge_if_no_hit=False, n_jobs=6):

    ok_subjects, bad_subjects = [], []
    results = {}
    for subject in subjects:
        output_file = os.path.join(get_electrodes_dir(), '{}_{}_electrodes_all_rois{}.csv'.format(subject, atlas, output_files_post_fix))
        if not os.path.isfile(output_file) or overwrite:
            try:
                if not neccesary_files is None:
                    remote_subject_dir = build_remote_subject_dir(remote_subject_dir_template, subject)
                    prepare_local_subjects_folder(neccesary_files, subject, remote_subject_dir, subjects_dir,
                        print_traceback=True)
                check_for_annot_file(subject=subject, subjects_dir=subjects_dir, atlas=atlas, fsaverage=template_brain,
                    overwrite=overwrite_annotation, n_jobs=n_jobs)
                elecs_names, elecs_pos = get_electrodes(subject)
                elcs_ori = get_electrodes_orientation(elecs_names, elecs_pos)
                elecs = identify_roi_from_atlas(elecs_names, elecs_pos, elcs_ori,
                    atlas=atlas, approx=error_radius, elc_length=elc_length,
                    enlarge_if_no_hit=enlarge_if_no_hit, subjects_dir=subjects_dir,
                    subject=subject, n_jobs=n_jobs)
                results[subject] = elecs
                ok_subjects.append(subject)
            except:
                bad_subjects.append(subject)
                print(traceback.format_exc())

    write_results_to_csv(results, atlas, post_fix=output_files_post_fix,
        write_only_cortical=write_only_cortical, write_only_subcortical=write_only_subcortical)

    print('ok subjects:')
    print(ok_subjects)
    print('bad_subjects:')
    print(bad_subjects)


def build_remote_subject_dir(remote_subject_dir_template, subject):
    if isinstance(remote_subject_dir_template, dict):
        if 'func' in remote_subject_dir_template:
            template_val = remote_subject_dir_template['func'](subject)
            remote_subject_dir = remote_subject_dir_template['template'].format(subject=template_val)
        else:
            remote_subject_dir = remote_subject_dir_template['template'].format(subject=subject)
    else:
        remote_subject_dir = remote_subject_dir_template.format(subject=subject)
    return remote_subject_dir


if __name__ == '__main__':
    subjects_dir = [s for s in ['/home/noam/subjects', '/homes/5/npeled/space3/subjects'] if os.path.isdir(s)][0]
    freesurfer_home = [s for s in ['/usr/local/freesurfer/stable5_3_0', '/home/noam/freesurfer'] if os.path.isdir(s)][0]
    os.environ['SUBJECTS_DIR'] = subjects_dir
    os.environ['FREESURFER_HOME'] = freesurfer_home
    atlas = 'laus250'
    neccesary_files = {'mri': ['aseg.mgz'], 'surf': ['rh.pial', 'lh.pial', 'rh.sphere.reg', 'lh.sphere.reg', 'lh.white', 'rh.white']}
    remote_subject_dir_template = {'template':'/space/huygens/1/users/mia/subjects/{subject}_SurferOutput', 'func':string.upper}
    template_brain = 'fsaverage5c'
    subjects = ['mg78'] # get_subjects()
    error_radius = 3
    elc_length = 4
    output_files_post_fix = '_cigar_r_{}_l_{}'.format(error_radius, elc_length)
    overwrite = True
    overwrite_annotation = False
    write_only_cortical=False
    write_only_subcortical=False
    enlarge_if_no_hit=True
    n_jobs = 6

    run_for_all_subjects(subjects, atlas, error_radius, elc_length,
        subjects_dir, template_brain, neccesary_files,
        remote_subject_dir_template, output_files_post_fix, overwrite, overwrite_annotation,
        write_only_cortical, write_only_subcortical, enlarge_if_no_hit, n_jobs)