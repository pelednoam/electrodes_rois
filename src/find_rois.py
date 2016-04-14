import numpy as np
import os
import os.path as op
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
import logging
from src import utils
from src import labels_utils as lu
from src import colors_utils as cu

LINKS_DIR = utils.get_links_dir()


def identify_roi_from_atlas(labels, elecs_names, elecs_pos, elcs_ori=None, approx=4, elc_length=1,
    nei_dimensions=None, atlas=None, elecs_dists=None, strech_to_dist=False, enlarge_if_no_hit=False,
    bipolar_electrodes=False, subjects_dir=None, subject=None, aseg_atlas=True, n_jobs=6):

    if subjects_dir is None or subjects_dir == '':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject == '':
        subject = os.environ['SUBJECT']

    # get the segmentation file
    aseg_fname = op.join(subjects_dir, subject, 'mri', 'aseg.mgz')
    asegf = aseg_fname
    aseg_atlas_fname = op.join(subjects_dir, subject, 'mri', '{}+aseg.mgz'.format(atlas))
    lut_atlast_fname = op.join(subjects_dir, subject, 'mri', '{}ColorLUT.txt'.format(atlas))
    lut_fname = ''
    if aseg_atlas:
        if op.isfile(aseg_atlas_fname) and op.isfile(lut_atlast_fname):
            asegf = aseg_atlas_fname
            lut_fname = lut_atlast_fname
        else:
            print('{} doesnot exist!'.format(aseg_atlas_fname))
    if not op.isfile(asegf):
        asegf = op.join(subjects_dir, subject, 'mri', 'aparc+aseg.mgz')
    try:
        aseg = nib.load(asegf)
        aseg_data = aseg.get_data()
        # np.save(op.join(subjects_dir, subject, 'mri', 'aseg.npy'), aseg_data)
    except:
        backup_aseg_file = op.join(subjects_dir, subject, 'mri', 'aseg.npy')
        if op.isfile(backup_aseg_file):
            aseg_data = np.load(backup_aseg_file)
        else:
            print('!!!!! Error in loading aseg file !!!!! ')
            print('!!!!! No subcortical labels !!!!!')
            aseg_data = None

    lut = import_freesurfer_lut(subjects_dir, lut_fname)

    # load the surfaces and annotation
    # uses the pial surface, this change is pushed to MNE python
    pia_verts = {}
    for hemi in ['rh', 'lh']:
        pia_verts[hemi], _ = nib.freesurfer.read_geometry(
            op.join(subjects_dir, subject, 'surf', '{}.pial'.format(hemi)))
    pia = np.vstack((pia_verts['lh'], pia_verts['rh']))
    len_lh_pia = len(pia_verts['lh'])

    elecs = []
    if elcs_ori is None:
        elcs_ori = [None] * len(elecs_pos)
    elecs_data = [(elc_num, elec_pos, elec_name, elc_ori, elc_dist) for \
        elc_num, (elec_pos, elec_name, elc_ori, elc_dist) in enumerate(zip(elecs_pos, elecs_names, elcs_ori, elecs_dists))]
    N = len(elecs_data)
    elecs_data_chunks = utils.chunks(elecs_data, len(elecs_data) / n_jobs)
    params = [(elecs_data_chunk, subject, subjects_dir, labels, atlas, aseg_data, lut, pia, len_lh_pia, approx, elc_length, nei_dimensions,
               strech_to_dist, enlarge_if_no_hit, bipolar_electrodes, N) for elecs_data_chunk in elecs_data_chunks]
    print('run with {} jobs'.format(n_jobs))
    results = utils.run_parallel(_find_elecs_roi_parallel, params, n_jobs)
    for results_chunk in results:
        for elec_name, regions, regions_hits, subcortical_regions, subcortical_hits, approx, elc_length in results_chunk:
            regions_probs = np.hstack((regions_hits, subcortical_hits)) / float(np.sum(regions_hits) + np.sum(subcortical_hits))
            if not np.allclose([np.sum(regions_probs)],[1.0]):
                print('Warning!!! {}: sum(regions_probs) = {}!'.format(elec_name, sum(regions_probs)))
            elecs.append({'name': elec_name, 'cortical_rois': regions, 'subcortical_rois': subcortical_regions,
                'cortical_probs': regions_probs[:len(regions)],
                'subcortical_probs': regions_probs[len(regions):], 'approx': approx, 'elc_length': elc_length})
    return elecs


def _find_elecs_roi_parallel(params):
    results = []
    elecs_data_chunk, subject, subjects_dir, labels, atlas, aseg_data, lut, pia, len_lh_pia, approx, elc_length,\
        nei_dimensions, strech_to_dist, enlarge_if_no_hit, bipolar_electrodes, N = params
    for elc_num, elec_pos, elec_name, elc_ori, elc_dist in elecs_data_chunk:
        print('{}: {} / {}'.format(elec_name, elc_num, N))
        regions, regions_hits, subcortical_regions, subcortical_hits, approx, elc_length = \
            identify_roi_from_atlas_per_electrode(labels, elec_pos, pia, len_lh_pia, atlas, lut,
                aseg_data, approx, elc_length, nei_dimensions, elc_ori, elc_dist, strech_to_dist,
                enlarge_if_no_hit, bipolar_electrodes, subjects_dir, subject, n_jobs=1)
        results.append((elec_name, regions, regions_hits, subcortical_regions, subcortical_hits, approx, elc_length))
    return results


def identify_roi_from_atlas_per_electrode(labels, pos, pia, len_lh_pia, atlas, lut, aseg_data,
      approx=4, elc_length=1, nei_dimensions=None, elc_ori=None, elc_dist=0, strech_to_dist=False,
      enlarge_if_no_hit=False, bipolar_electrodes=False, subjects_dir=None, subject=None, n_jobs=1):
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

    surf_fname = op.join(subjects_dir, subject, 'surf', hemi_str + '.pial')
    verts, _ = read_surface(surf_fname)
    closest_vert_pos = verts[closest_vert]

    we_have_a_hit = False
    if strech_to_dist and bipolar_electrodes and elc_length < elc_dist:
        elc_length = elc_dist
    while not we_have_a_hit:
        # grow the area of surface surrounding the vertex
        # radius_label, = mne.grow_labels(subject, closest_vert, approx, hemi_code,
        #     subjects_dir=subjects_dir, surface='pial')

        # bins = calc_neighbors(closest_vert_pos, approx + elc_length, nei_dimensions, calc_bins=True)
        bins = calc_neighbors(pos, approx + elc_length, nei_dimensions, calc_bins=True)
        if not elc_ori is None:
            elc_line = [pos + elc_ori*t for t in np.linspace(-elc_length/2.0, elc_length/2.0, 100)]
        else:
            elc_line = [pos]

        # excludes=['white', 'WM', 'Unknown', 'White', 'unknown', 'Cerebral-Cortex']
        excludes=['Unknown', 'unknown', 'Cerebral-Cortex', 'corpuscallosum']
        compiled_excludes = re.compile('|'.join(excludes))
        _region_is_excluded = partial(region_is_excluded, compiled_excludes=compiled_excludes)

        regions, regions_hits = [], []
        # parcels_files = glob.glob(op.join(subjects_dir, subject, 'label', atlas, '*.label'))

        # files_chunks = utils.chunks(parcels_files, len(parcels_files) / n_jobs)
        # params = [(files_chunk, hemi_str, verts, elc_line, bins, approx, _region_is_excluded) for files_chunk in files_chunks]
        # results = utils.run_parallel(_calc_hits_parallel, params, n_jobs)
        # for chunk in results:
        #     for parcel_name, hits in chunk:
        results = calc_hits(labels, hemi_str, verts, elc_line, bins, approx, _region_is_excluded)
        for parcel_name, hits in results:
            if hits > 0:
                regions.append(parcel_name)
                regions_hits.append(hits)

        if aseg_data is not None:
            subcortical_regions, subcortical_hits = identify_roi_from_aparc(pos, elc_line, elc_length, lut, aseg_data,
                approx=approx, nei_dimensions=nei_dimensions, subcortical_only=True, excludes=excludes)
        else:
            subcortical_regions, subcortical_hits = [], []

        we_have_a_hit = not electrode_is_only_in_white_matter(regions, subcortical_regions) or not enlarge_if_no_hit
        if not we_have_a_hit:
            approx += .5
            elc_length += 1
            print('No hit! Recalculate with a bigger cigar')

    return regions, regions_hits, subcortical_regions, subcortical_hits, approx, elc_length


def calc_hits(labels, hemi_str, surf_verts, elc_line, bins, approx, _region_is_excluded):
    res = []
    res.append(('', 0))
    for label in labels:
        label = utils.Bag(label)
        # parcel = mne.read_label(parcel_file)
        if label.hemi != hemi_str:
            continue
        elif _region_is_excluded(str(label.name)):
            continue
        else:
            hits = calc_hits_in_neighbors_from_line(elc_line, surf_verts[label.vertices], bins, approx)
        res.append((str(label.name), hits))
    return res


def read_labels_vertices(subjects_dir, subject, atlas, read_labels_from_annotation=False, overwrite=False, n_jobs=1):
    res_file = op.join(subjects_dir, subject, 'label', '{}.pkl'.format(atlas))
    if not overwrite and op.isfile(res_file):
        labels = utils.load(res_file)
    else:
        if read_labels_from_annotation:
            annot_labels = mne.read_labels_from_annot(subject, atlas)
            labels = list([{'name': label.name, 'hemi': label.hemi, 'vertices': label.vertices}
                           for label in annot_labels])
        else:
            labels_files = glob.glob(op.join(subjects_dir, subject, 'label', atlas, '*.label'))
            files_chunks = utils.chunks(labels_files, len(labels_files) / n_jobs)
            results = utils.run_parallel(_read_labels_vertices, files_chunks, n_jobs)
            labels = []
            for labels_chunk in results:
                labels.extend(labels_chunk)
        utils.save(labels, res_file)
    return labels


def _read_labels_vertices(files_chunk):
    labels = []
    for label_file in files_chunk:
        label = mne.read_label(label_file)
        labels.append({'name': label.name, 'hemi': label.hemi, 'vertices': label.vertices})
    return labels


# def parcels_volume_search(elc_line, pos, approx, dimensions):
#     neighb = calc_neighbors(pos, elc_length + approx, dimensions)
#     dists = np.min(cdist(elc_line, neighb), 0)
#     neighb = neighb[np.where(dists<approx)]


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


#todo: don't have to calculate the hist if the dist is above some threshold
def calc_hits_in_neighbors_from_line(line, points, neighb, approx):
    bins = [np.sort(np.unique(neighb[:, idim])) for idim in range(points.shape[1])]
    hist, bin_edges = np.histogramdd(points, bins=bins, normed=False)
    if np.sum(hist > 0) == 0:
        return 0
    else:
        hist_bin_centers_list = [bin_edges[d][:-1] + (bin_edges[d][1:] - bin_edges[d][:-1])/2.
            for d in range(len(bin_edges))]
        bin_centers = list(product(*hist_bin_centers_list))
        dists = np.min(cdist(line, bin_centers), 0).reshape(hist.shape)
        hits = len(np.where((hist > 0) & (dists<approx))[0])
        return hits


def identify_roi_from_aparc(pos, elc_line, elc_length, lut, aseg_data, approx=4, nei_dimensions=None,
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

    def find_neighboring_regions(pos, elc_length, elc_line, aseg_data, lut, approx, dimensions, excludes):
        compiled_excludes = re.compile('|'.join(excludes))
        _region_is_excluded = partial(region_is_excluded, compiled_excludes=compiled_excludes)
        neighb = calc_neighbors(pos, elc_length + approx, dimensions)
        dists = np.min(cdist(elc_line, neighb), 0)
        neighb = neighb[np.where(dists<approx)]
        regions = []
        for nei in neighb:
            nei_regions = set()
            round_neis = round_coords(nei)
            for round_nei in round_neis:
                cx, cy, cz = map(int, round_nei)
                d_type = aseg_data[cx, cy, cz]
                label_index = np.where(lut['index'] == d_type)[0][0]
                region = lut['label'][label_index]
                if not _region_is_excluded(region):
                    nei_regions.add(region)
            for region in nei_regions:
                regions.append(region)

        # regions = exclude_regions(regions, excludes)
        cnt = Counter(regions)
        regions, hits = [], []
        for region, count in cnt.items():
            regions.append(region)
            hits.append(count)
        return regions, hits
        # return np.unique(regions).tolist()

    def round_coords(pos):
        rounds = [[np.floor(pos[d]), np.ceil(pos[d])] for d in range(3)]
        coords = list(product(*rounds))
        return coords

    def to_ras(points, round_coo=False):
        RAS_AFF = np.array([[-1, 0, 0, 128],
            [0, 0, -1, 128],
            [0, 1, 0, 128],
            [0, 0, 0, 1]])
        ras = [np.dot(RAS_AFF, np.append(p, 1))[:3] for p in points]
        if round_coo:
            ras = [np.around(p) for p in ras]
        return ras

    if subcortical_only:
        excludes.append('ctx')

    ras_pos = to_ras([pos])[0]
    ras_elc_line = to_ras(elc_line)
    return find_neighboring_regions(ras_pos, elc_length, ras_elc_line, aseg_data, lut, approx, nei_dimensions, excludes)


def import_freesurfer_lut(subjects_dir, fs_lut=''):
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


def exclude_regions(regions, excludes):
    if (excludes):
        excluded = compile('|'.join(excludes))
        regions = [x for x in regions if not excluded.search(x)]
    return regions


def region_is_excluded(region, compiled_excludes):
    if isinstance(region, np.bytes_):
        region = region.astype(str)
    return not compiled_excludes.search(region) is None


def calc_neighbors(pos, approx=None, dimensions=None, calc_bins=False):
    if not approx is None:
        sz = int(np.around(approx * 2 + (2 if calc_bins else 1)))
        sx = sy = sz
    elif not dimensions is None:
        sx, sy, sz = dimensions
    else:
        raise Exception('approx and dimensions are None!')
        logging.error('calc_neighbors: approx and dimensions are None!')

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


def get_electrodes(subject, bipolar=False, elecs_dir='', delimiter=','):
    if elecs_dir=='':
        elecs_dir = get_electrodes_dir()
    elec_file = op.join(elecs_dir, '{}.csv'.format(subject))
    data = np.genfromtxt(elec_file, dtype=str, delimiter=delimiter)
    data = fix_str_items_in_csv(data)
    # Check if the electrodes coordinates has a header
    # if isinstance(data[0, 1:], np.ndarray) and data[0, 1:].dtype.kind == 'S':
    #     data = np.delete(data, (0), axis=0)
    #     print('First line in the electrodes RAS coordinates is a header')
    # else:
    try:
        header = data[0, 1:].astype(float)
    except:
        data = np.delete(data, (0), axis=0)
        print('First line in the electrodes RAS coordinates is a header')

    # if not isinstance(data[:, 1:], np.ndarray):
    pos = data[:, 1:].astype(float)
    dists = []
    if bipolar:
        pos_biploar, names = [], []
        for index in range(data.shape[0]-1):
            elc_group1, elc_num1 = elec_group_number(data[index, 0])
            elc_group2, elc_num12 = elec_group_number(data[index+1, 0])
            if elc_group1 == elc_group2:
            # if data[index+1, 0][:3] == data[index, 0][:3]:
                elec_name = '{}-{}'.format(data[index+1, 0],data[index, 0])
                names.append(elec_name)
                pos_biploar.append(pos[index] + (pos[index+1]-pos[index])/2)
                dists.append(np.linalg.norm(pos[index+1]-pos[index]))
        pos = np.array(pos_biploar)
    else:
        names = data[:, 0]
        dists = [np.linalg.norm(p2-p1) for p1,p2 in zip(pos[:-1], pos[1:])]
        # Add the distance for the last electrode
        dists.append(0.)
    names = np.array([name.strip() for name in names])
    if not len(names) == len(pos) == len(dists):
        logging.error('get_electrodes ({}): not len(names)==len(pos)==len(dists)!'.format(subject))
        raise Exception('get_electrodes: not len(names)==len(pos)==len(dists)!')

    return names, pos, dists


def fix_str_items_in_csv(csv):
    lines = []
    for line in csv:
        fix_line = list(map(lambda x: str(x).replace('"', ''), line))
        if not np.all([len(v) == 0 for v in fix_line[1:]]):
            lines.append(fix_line)
    return np.array(lines)


def get_electrodes_dir():
    curr_dir = op.dirname(op.realpath(__file__))
    elec_dir = op.join(op.split(curr_dir)[0], 'electrodes')
    utils.make_dir(elec_dir)
    return elec_dir


def write_results_to_csv(results, atlas, elecs_dir='', post_fix='',
        write_only_cortical=False, write_only_subcortical=False):

    if elecs_dir=='':
        elecs_dir = get_electrodes_dir()

    cortical_rois, subcortical_rois = [], []
    for elecs in results.values():
        for elc in elecs:
            cortical_rois.extend(elc['cortical_rois'])
            subcortical_rois.extend(elc['subcortical_rois'])
    cortical_rois = list(np.unique(cortical_rois))
    subcortical_rois = list(np.unique(subcortical_rois))

    for subject, elecs in results.items():
        write_values(elecs, ['electrode'] + cortical_rois + subcortical_rois + ['approx', 'elc_length'],
            [cortical_rois, subcortical_rois],
            ['cortical_rois','subcortical_rois'], ['cortical_probs', 'subcortical_probs'],
            op.join(elecs_dir, '{}_{}_electrodes_all_rois{}.csv'.format(subject, atlas, post_fix)))

        if write_only_cortical:
            write_values(elecs, ['electrode'] + cortical_rois, [cortical_rois],['cortical_rois'], ['cortical_probs'],
                op.join(elecs_dir, '{}_{}_electrodes_cortical_rois{}.csv'.format(subject, atlas, post_fix)))

        if write_only_subcortical:
            write_values(elecs, ['electrode']  + subcortical_rois, [subcortical_rois],
                ['subcortical_rois'], ['subcortical_probs'],
                op.join(elecs_dir, '{}_{}_electrodes_subcortical_rois{}.csv'.format(subject, atlas, post_fix)))


def write_values(elecs, header, rois_arr, rois_names, probs_names, file_name):
    with open(file_name, 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerow(header)
        print('Writing {} with header length of {}'.format(file_name, len(header)))
        for elc in elecs:
            values = [elc['name']]
            for rois, rois_field, prob_field in zip(rois_arr, rois_names, probs_names):
                for col, roi in enumerate(rois):
                    if roi in elc[rois_field]:
                        index = elc[rois_field].index(roi)
                        values.append(str(elc[prob_field][index]))
                    else:
                        values.append(0.)
            values.extend([elc['approx'], elc['elc_length']])
            writer.writerow(values)


# def add_labels_per_electrodes_probabilities(subject, elecs_dir='', post_fix=''):
#     if elecs_dir == '':
#         elecs_dir = get_electrodes_dir()
#     csv_fname = op.join(elecs_dir, '{}_{}_electrodes_all_rois{}.csv'.format(subject, atlas, post_fix))
#     np.genfromtxt(csv_fname)


def get_electrodes_orientation(elecs_names, elecs_pos, bipolar):
    elcs_oris = []
    for elc_name, elc_pos in zip(elecs_names, elecs_pos):
        if bipolar_electrodes:
            elc_group, elc_num1, elc_num2 = elec_group_number(elc_name, True)
            next_elc = '{}{}-{}{}'.format(elc_group, elc_num2 + 1, elc_group, elc_num1 + 1)
        else:
            elc_group, elc_num = elec_group_number(elc_name)
            next_elc = '{}{}'.format(elc_group, elc_num+1)
        ori = 1
        if next_elc not in elecs_names:
            if bipolar_electrodes:
                next_elc = '{}{}-{}{}'.format(elc_group, elc_num1, elc_group, elc_num1-1)
            else:
                next_elc = '{}{}'.format(elc_group, elc_num-1)
            ori = -1
        next_elc_index = np.where(elecs_names==next_elc)[0][0]
        next_elc_pos = elecs_pos[next_elc_index]
        dist = np.linalg.norm(next_elc_pos-elc_pos)
        elc_ori = ori * (next_elc_pos-elc_pos) / dist # norm(elc_ori)=1mm
        elcs_oris.append(elc_ori)
        # print(elc_name, elc_pos, next_elc, next_elc_pos, elc_line(1))
    return elcs_oris


def elec_group_number(elec_name, bipolar=False):
    if bipolar:
        elec_name2, elec_name1 = elec_name.split('-')
        group, num1 = elec_group_number(elec_name1, False)
        _, num2 = elec_group_number(elec_name2, False)
        return group, num1, num2
    else:
        ind = np.where([int(s.isdigit()) for s in elec_name])[-1][0]
        num = int(elec_name[ind:])
        group = elec_name[:ind]
        return group, num


def get_subjects():
    files = glob.glob(op.join(get_electrodes_dir(), '*.csv'))
    names = set()
    for full_file_name in files:
        file_name = op.split(full_file_name)[1]
        if '_' not in file_name:
            names.add(op.splitext(file_name)[0])
    return names


def get_all_subjects(subjects_dir, prefix, exclude_substr):
    subjects = []
    folders = [utils.namebase(fol) for fol in utils.get_subfolders(subjects_dir)]
    for subject_fol in folders:
        if subject_fol[:len(prefix)].lower() == prefix and exclude_substr not in subject_fol:
            subjects.append(subject_fol)
    return subjects


def check_for_annot_file(subject, subjects_dir, atlas, fsaverage='fsaverage5c', overwrite_labels=False,
        overwrite_annot=False, read_labels_from_annotation=False, solve_labels_collisions=False, n_jobs=6):
    annot_file = op.join(subjects_dir, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))
    if read_labels_from_annotation and op.isfile(annot_file.format(hemi='rh')) and \
            op.isfile(annot_file.format(hemi='lh')):
        # Nothing to do, read the labels from an existing annotation file
        return
    if overwrite_labels or overwrite_annot or not op.isfile(annot_file.format(hemi='rh')) or not \
            op.isfile(annot_file.format(hemi='lh')):
        morph_labels_from_fsaverage(subject, subjects_dir, atlas, n_jobs=n_jobs, fsaverage=fsaverage, overwrite=overwrite_labels)
        if solve_labels_collisions:
            backup_labels_fol = '{}_before_solve_collision'.format(atlas, fsaverage)
            lu.solve_labels_collision(subject, subjects_dir, atlas, backup_labels_fol, n_jobs)
        # lu.backup_annotation_files(subject, subjects_dir, atlas)
        # labels_to_annot(subject, subjects_dir, atlas, overwrite=overwrite_annot)


def morph_labels_from_fsaverage(subject, subjects_dir='', aparc_name='aparc250', fs_labels_fol='', sub_labels_fol='', n_jobs=6, fsaverage='fsaverage', overwrite=False):
    if subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    subject_dir = op.join(subjects_dir, subject)
    labels_fol = op.join(subjects_dir, fsaverage, 'label', aparc_name) if fs_labels_fol=='' else fs_labels_fol
    sub_labels_fol = op.join(subject_dir, 'label', aparc_name) if sub_labels_fol=='' else sub_labels_fol
    utils.make_dir(sub_labels_fol)
    if overwrite:
        utils.delete_folder_files(sub_labels_fol)
    labels_fnames = glob.glob(op.join(labels_fol, '*.label'))
    labels_fnames_chunks = utils.chunks(labels_fnames, len(labels_fnames) / n_jobs)
    params_chunks = [(labels_fnames_chunk, subject, subjects_dir, sub_labels_fol, fsaverage, overwrite) \
              for labels_fnames_chunk in labels_fnames_chunks]
    utils.run_parallel(_morph_labels_parallel, params_chunks, n_jobs)


def _morph_labels_parallel(params_chunks):
    labels_fnames_chunk, subject, subjects_dir, sub_labels_fol, fsaverage, overwrite = params_chunks
    for label_fname in labels_fnames_chunk:
        local_label_name = op.join(sub_labels_fol, '{}.label'.format(op.splitext(op.split(label_fname)[1])[0]))
        if op.isfile(local_label_name) and overwrite:
            os.remove(local_label_name)
        if not op.isfile(local_label_name) or overwrite:
            fs_label = mne.read_label(label_fname)
            fs_label.values.fill(1.0)
            sub_label = fs_label.morph(fsaverage, subject, grade=None, n_jobs=n_jobs, subjects_dir=subjects_dir)
            sub_label.save(local_label_name)


# def labels_to_annot(subject, subjects_dir='', aparc_name='aparc250', labels_fol='', overwrite=True):
#     if subjects_dir=='':
#         subjects_dir = os.environ['SUBJECTS_DIR']
#     subject_dir = op.join(subjects_dir, subject)
#     labels_fol = op.join(subject_dir, 'label', aparc_name) if labels_fol=='' else labels_fol
#     labels = []
#     for label_file in glob.glob(op.join(labels_fol, '*.label')):
#         # print label_file
#         try:
#             label = mne.read_label(label_file)
#             labels.append(label)
#         except:
#             print('error reading the label!')
#             logging.error('labels_to_annot ({}): {}'.format(subject, traceback.format_exc()))
#             print(traceback.format_exc())
#
#     mne.write_labels_to_annot(subject=subject, labels=labels, parc=aparc_name, overwrite=overwrite,
#                               subjects_dir=subjects_dir, surface='pial')


# def get_all_labels_and_segmentations(subject, atlas):
#     percs, segs = [], []
#     all_segs = import_freesurfer_lut()['label']
#     excludes=['Unknown', 'unknown', 'Cerebral-Cortex', 'ctx']
#     compiled_excludes = re.compile('|'.join(excludes))
#     _region_is_excluded = partial(region_is_excluded, compiled_excludes=compiled_excludes)
#     for seg in all_segs:
#         if not _region_is_excluded(seg):
#             segs.append(seg)
#
#     for hemi in ['rh', 'lh']:
#         annot_file = op.join(subjects_dir, subject, 'label', '{}.{}.annot'.format(hemi, atlas))
#         labels = mne.read_labels_from_annot(subject, surf_name='pial', annot_fname=annot_file)
#         for label in labels:
#             percs.append(label.name)
#
#     return percs, segs


def prepare_local_subjects_folder(neccesary_files, subject, remote_subject_dir, local_subjects_dir, print_traceback=False):
    local_subject_dir = op.join(local_subjects_dir, subject)
    for fol, files in neccesary_files.items():
        if not op.isdir(op.join(local_subject_dir, fol)):
            os.makedirs(op.join(local_subject_dir, fol))
        for file_name in files:
            try:
                if not op.isfile(op.join(local_subject_dir, fol, file_name)):
                    shutil.copyfile(op.join(remote_subject_dir, fol, file_name),
                                op.join(local_subject_dir, fol, file_name))
            except:
                logging.error('{}: {}'.format(subject, traceback.format_exc()))
                if print_traceback:
                    print(traceback.format_exc())
    all_files_exists = True
    for fol, files in neccesary_files.items():
        for file_name in files:
            if not op.isfile(op.join(local_subject_dir, fol, file_name)):
                print("The file {} doesn't exist in the local subjects folder!!!".format(file_name))
                all_files_exists = False
    if not all_files_exists:
        raise Exception('Not all files exist in the local subject folder!!!')
        logging.error('{}: {}'.format(subject, 'Not all files exist in the local subject folder!!!'))


def check_for_necessary_files(subjects_dir, subject, neccesary_files):
    remote_subject_dir = build_remote_subject_dir(remote_subject_dir_template, subject)
    prepare_local_subjects_folder(neccesary_files, subject, remote_subject_dir, subjects_dir,
        print_traceback=True)
    elecs_dir = get_electrodes_dir()
    elec_file = op.join(elecs_dir, '{}.csv'.format(subject))
    if not op.isfile(elec_file):
        copy_electrodes_file(subjects_dir, subject, elec_file)


def copy_electrodes_file(subjects_dir, subject, elec_file):
    subject_elec_fname = op.join(subjects_dir, subject, 'electrodes', '{}_RAS.csv'.format(subject))
    if not op.isfile(subject_elec_fname):
        rename_and_convert_electrodes_file(subject, subjects_dir)
    if op.isfile(subject_elec_fname):
        shutil.copyfile(subject_elec_fname, elec_file)
    else:
        raise Exception('{}: Electrodes file does not exist! {}'.format(subject, subject_elec_fname))


def rename_and_convert_electrodes_file(subject, subjects_dir):
    subject_elec_fname_pattern = op.join(subjects_dir, subject, 'electrodes', '{subject}_RAS.{postfix}')
    subject_elec_fname_csv_upper = subject_elec_fname_pattern.format(subject=subject.upper(), postfix='csv')
    subject_elec_fname_csv = subject_elec_fname_pattern.format(subject=subject, postfix='csv')
    subject_elec_fname_xlsx_upper = subject_elec_fname_pattern.format(subject=subject.upper(), postfix='xlsx')
    subject_elec_fname_xlsx = subject_elec_fname_pattern.format(subject=subject, postfix='xlsx')

    if op.isfile(subject_elec_fname_csv_upper):
        os.rename(subject_elec_fname_csv_upper, subject_elec_fname_csv)
    elif op.isfile(subject_elec_fname_xlsx_upper):
        os.rename(subject_elec_fname_xlsx_upper, subject_elec_fname_xlsx)
    if op.isfile(subject_elec_fname_xlsx) and not op.isfile(subject_elec_fname_csv):
        utils.csv_from_excel(subject_elec_fname_xlsx, subject_elec_fname_csv)


# def find_electrodes_closets_label(subject, labels, bipolar_electrodes):
#     elecs_names, elecs_pos, elecs_dists = get_electrodes(subject, bipolar_electrodes)
#     pia_verts = {}
#     for hemi in ['rh', 'lh']:
#         pia_verts[hemi], _ = nib.freesurfer.read_geometry(
#             op.join(subjects_dir, subject, 'surf', '{}.pial'.format(hemi)))
#     pia = np.vstack((pia_verts['lh'], pia_verts['rh']))


def run_for_all_subjects(subjects, atlas, error_radius, elc_length, subjects_dir, template_brain='fsaverage',
        bipolar_electrodes=False, neccesary_files=None, remote_subject_dir_template='', output_files_post_fix='',
        overwrite=False, overwrite_annotation=False, overwrite_labels=False, write_only_cortical=False, write_only_subcortical=False,
        strech_to_dist=False, enlarge_if_no_hit=False, only_check_files=False, overwrite_labels_pkl=False,
        overwrite_csv=False, read_labels_from_annotation=False, solve_labels_collisions=False, n_jobs=6):

    ok_subjects, bad_subjects = [], []
    results = {}
    for subject in subjects:
        output_file = op.join(get_electrodes_dir(), '{}_{}_electrodes_all_rois{}.csv'.format(subject, atlas, output_files_post_fix))
        if not op.isfile(output_file) or overwrite_csv:
            try:
                results_fname = op.join(get_electrodes_dir(), '{}_{}_electrodes_all_rois{}.pkl'.format(
                    subject, atlas, output_files_post_fix))
                if op.isfile(results_fname) and not overwrite:
                    elecs = utils.load(results_fname)
                else:
                    print('****************** {} ******************'.format(subject))
                    check_for_necessary_files(subjects_dir, subject, neccesary_files)
                    check_for_annot_file(subject=subject, subjects_dir=subjects_dir, atlas=atlas, fsaverage=template_brain,
                        overwrite_labels=overwrite_labels, overwrite_annot=overwrite_annotation, n_jobs=n_jobs,
                        read_labels_from_annotation=read_labels_from_annotation,
                        solve_labels_collisions=solve_labels_collisions)
                    if only_check_files:
                        continue
                    elecs_names, elecs_pos, elecs_dists = get_electrodes(subject, bipolar_electrodes)
                    elcs_ori = get_electrodes_orientation(elecs_names, elecs_pos, bipolar_electrodes)
                    labels = read_labels_vertices(subjects_dir, subject, atlas, read_labels_from_annotation,
                        overwrite_labels_pkl, n_jobs)
                    elecs = identify_roi_from_atlas(labels, elecs_names, elecs_pos, elcs_ori,
                        atlas=atlas, approx=error_radius, elc_length=elc_length, elecs_dists=elecs_dists,
                        strech_to_dist=strech_to_dist, enlarge_if_no_hit=enlarge_if_no_hit,
                        bipolar_electrodes=bipolar_electrodes, subjects_dir=subjects_dir, subject=subject,
                        aseg_atlas=False, n_jobs=n_jobs)
                    utils.save(elecs, results_fname)
                results[subject] = elecs
                ok_subjects.append(subject)
            except:
                bad_subjects.append(subject)
                logging.error('{}: {}'.format(subject, traceback.format_exc()))
                print(traceback.format_exc())

    # Write the results for all the subjects at once, to have a common labeling
    write_results_to_csv(results, atlas, post_fix=output_files_post_fix,
        write_only_cortical=write_only_cortical, write_only_subcortical=write_only_subcortical)

    print('ok subjects:')
    print(ok_subjects)
    print('bad_subjects:')
    print(bad_subjects)


def add_colors_to_probs(subjects, atlas, output_files_post_fix):
    for subject in subjects:
        results_fname = op.join(get_electrodes_dir(), '{}_{}_electrodes_all_rois{}.pkl'.format(
            subject, atlas, output_files_post_fix))
        elecs = utils.load(results_fname)
        for elc in elecs:
            elc['subcortical_colors'] = cu.arr_to_colors(elc['subcortical_probs'], colors_map='YlOrRd')
            elc['cortical_colors'] = cu.arr_to_colors(elc['cortical_probs'], colors_map='YlOrRd')
        utils.save(elecs, results_fname)


def remove_white_matter_and_normalize(elc):
    no_white_inds = [ind for ind, label in enumerate(elc['subcortical_rois']) if label not in
                     ['Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter']]
    subcortical_probs_norm = elc['subcortical_probs'][no_white_inds]
    subcortical_probs_norm *= 1/sum(subcortical_probs_norm)
    subcortical_rois_norm = elc['subcortical_rois'][no_white_inds]
    return subcortical_probs_norm, subcortical_rois_norm


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
    subjects_dir = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
    freesurfer_home = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
    blender_dir = utils.get_link_dir(LINKS_DIR, 'mmvt')
    os.environ['SUBJECTS_DIR'] = subjects_dir
    os.environ['FREESURFER_HOME'] = freesurfer_home
    atlas = 'arc_april2016' # 'aparc.DKTatlas40' # 'laus250'
    neccesary_files = {'mri': ['aseg.mgz'], 'surf': ['rh.pial', 'lh.pial', 'rh.sphere.reg', 'lh.sphere.reg', 'lh.white', 'rh.white']}
    remote_subject_dir_template = {'template':'/space/huygens/1/users/mia/subjects/{subject}_SurferOutput', 'func': lambda x: x.upper()}
    template_brain = 'fsaverage5c'
    subjects = ['mg78'] # set(get_all_subjects(subjects_dir, 'mg', '_')) - set(['mg63', 'mg94']) # get_subjects()
    error_radius = 3
    elc_length = 4
    strech_to_dist = True # If bipolar, strech to the neighbours
    enlarge_if_no_hit = True
    only_check_files = False
    overwrite = True
    overwrite_annotation = False
    overwrite_labels = False
    write_only_cortical = False
    write_only_subcortical = False
    overwrite_labels_pkl = True
    overwrite_csv = True
    read_labels_from_annotation = False
    solve_labels_collisions = False
    cpu_num = utils.cpu_count()
    if cpu_num <= 2:
        n_jobs = cpu_num
    else:
        n_jobs = cpu_num - 2
    print('n_jobs: {}'.format(n_jobs))
    logging.basicConfig(filename='errors.log',level=logging.ERROR)

    for bipolar_electrodes in [False, True]:
        output_files_post_fix = '_cigar_r_{}_l_{}{}{}'.format(error_radius, elc_length,
            '_bipolar' if bipolar_electrodes else '', '_stretch' if strech_to_dist and bipolar_electrodes else '')
        run_for_all_subjects(subjects, atlas, error_radius, elc_length,
            subjects_dir, template_brain, bipolar_electrodes, neccesary_files,
            remote_subject_dir_template, output_files_post_fix, overwrite, overwrite_annotation, overwrite_labels,
            write_only_cortical, write_only_subcortical, strech_to_dist, enlarge_if_no_hit, only_check_files,
            overwrite_labels_pkl=overwrite_labels_pkl, overwrite_csv=overwrite_csv,
            read_labels_from_annotation=read_labels_from_annotation, solve_labels_collisions=solve_labels_collisions,
            n_jobs=n_jobs)
        add_colors_to_probs(subjects, atlas, output_files_post_fix)

    print('finish!')