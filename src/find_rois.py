import csv
import glob
import logging
import os
import os.path as op
import re
import shutil
import traceback
from collections import Counter, defaultdict
from functools import partial
from itertools import product

import mne
import nibabel as nib
import numpy as np
from mne.surface import read_surface
from scipy.spatial.distance import cdist

from src import colors_utils as cu
from src import freesurfer_utils as fu
from src import labels_utils as lu
from src import utils

LINKS_DIR = utils.get_links_dir()
DEPTH, GRID = range(2)
EXISTING_FREESURFER_ANNOTATIONS = ['aparc.DKTatlas40.annot', 'aparc.annot', 'aparc.a2009s.annot']


def identify_roi_from_atlas(labels, elecs_names, elecs_pos, elecs_ori=None, approx=4, elc_length=1,
                            elecs_dists=None, elecs_types=None, strech_to_dist=False, enlarge_if_no_hit=False,
                            bipolar_electrodes=False, subjects_dir=None, subject=None, excludes=None, n_jobs=6,
                            nei_dimensions=None, aseg_atlas=True):

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
            logging.warning("{} doesnot exist!".format(aseg_atlas_fname))
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
            logging.error('!!!!! Error in loading aseg file !!!!! ')
            logging.error('!!!!! No subcortical labels !!!!!')
            aseg_data = None

    lut = fu.import_freesurfer_lut(subjects_dir, lut_fname)

    # load the surfaces and annotation
    # uses the pial surface, this change is pushed to MNE python
    pia_verts = {}
    for hemi in ['rh', 'lh']:
        pia_verts[hemi], _ = nib.freesurfer.read_geometry(
            op.join(subjects_dir, subject, 'surf', '{}.pial'.format(hemi)))
    # pia = np.vstack((pia_verts['lh'], pia_verts['rh']))
    len_lh_pia = len(pia_verts['lh'])

    elecs = []
    if elecs_ori is None:
        elecs_ori = [None] * len(elecs_pos)
    if elecs_types is None:
        elecs_types = [DEPTH] * len (elecs_pos)

    elecs_data = list(enumerate(zip(elecs_pos, elecs_names, elecs_ori, elecs_dists, elecs_types)))
    N = len(elecs_data)
    elecs_data_chunks = utils.chunks(elecs_data, len(elecs_data) / n_jobs)
    params = [(elecs_data_chunk, subject, subjects_dir, labels, aseg_data, lut, pia_verts, len_lh_pia, approx,
               elc_length, nei_dimensions, strech_to_dist, enlarge_if_no_hit, bipolar_electrodes, excludes, N) for
              elecs_data_chunk in elecs_data_chunks]
    print('run with {} jobs'.format(n_jobs))
    results = utils.run_parallel(_find_elecs_roi_parallel, params, n_jobs)
    for results_chunk in results:
        for elec_name, regions, regions_hits, subcortical_regions, subcortical_hits, approx, elc_length,\
                elec_hemi_vertices, elec_hemi_vertices_dists, hemi in results_chunk:
            regions_probs = np.hstack((regions_hits, subcortical_hits)) / float(
                np.sum(regions_hits) + np.sum(subcortical_hits))
            if not np.allclose([np.sum(regions_probs)],[1.0]):
                logging.warning('Warning!!! {}: sum(regions_probs) = {}!'.format(elec_name, sum(regions_probs)))
            elecs.append({'name': elec_name, 'cortical_rois': regions, 'subcortical_rois': subcortical_regions,
                'cortical_probs': regions_probs[:len(regions)],
                'subcortical_probs': regions_probs[len(regions):], 'approx': approx, 'elc_length': elc_length,
                'cortical_indices': elec_hemi_vertices, 'cortical_indices_dists':elec_hemi_vertices_dists,
                'hemi': hemi})
    return elecs


def _find_elecs_roi_parallel(params):
    results = []
    elecs_data_chunk, subject, subjects_dir, labels, aseg_data, lut, pia_verts, len_lh_pia, approx, elc_length,\
        nei_dimensions, strech_to_dist, enlarge_if_no_hit, bipolar_electrodes, excludes, N = params
    for elc_num, (elec_pos, elec_name, elc_ori, elc_dist, elc_type) in elecs_data_chunk:
        # if elec_name != 'GR12':
        #     continue
        print('{}: {} / {}'.format(elec_name, elc_num, N))
        regions, regions_hits, subcortical_regions, subcortical_hits, approx, elc_length, elec_hemi_vertices, \
                elec_hemi_vertices_dists, hemi = \
            identify_roi_from_atlas_per_electrode(labels, elec_pos, pia_verts, len_lh_pia, lut,
                aseg_data, elec_name, approx, elc_length, nei_dimensions, elc_ori, elc_dist, elc_type, strech_to_dist,
                enlarge_if_no_hit, bipolar_electrodes, subjects_dir, subject, excludes, n_jobs=1)
        results.append((elec_name, regions, regions_hits, subcortical_regions, subcortical_hits, approx, elc_length,
                        elec_hemi_vertices, elec_hemi_vertices_dists, hemi))
    return results


def identify_roi_from_atlas_per_electrode(labels, pos, pia_verts, len_lh_pia, lut, aseg_data, elc_name,
    approx=4, elc_length=1, nei_dimensions=None, elc_ori=None, elc_dist=0, elc_type=DEPTH, strech_to_dist=False,
    enlarge_if_no_hit=False, bipolar_electrodes=False, subjects_dir=None, subject=None, excludes=None, n_jobs=1):
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

    if excludes is None:
        excludes = ['Unknown', 'unknown', 'Cerebral-Cortex', 'corpuscallosum', 'WM-hypointensities']

    # find closest vertex
    pia = np.vstack((pia_verts['lh'], pia_verts['rh']))
    closest_vert = np.argmin(cdist(pia, [pos]))

    # we force the label to only contact one hemisphere even if it is
    # beyond the extent of the medial surface
    hemi_str = 'lh' if closest_vert<len_lh_pia else 'rh'
    # hemi_code = 0 if hemi_str=='lh' else 1

    if hemi_str == 'rh':
        closest_vert -= len_lh_pia

    # todo: send the verts to the function
    surf_fname = op.join(subjects_dir, subject, 'surf', hemi_str + '.pial')
    verts, _ = read_surface(surf_fname)
    # closest_vert_pos = verts[closest_vert]

    if elc_type == GRID:
        elc_dist = 0
        elc_length = 0
        elc_ori = None

    we_have_a_hit = False
    if strech_to_dist and bipolar_electrodes and elc_length < elc_dist:
        elc_length = elc_dist
    while not we_have_a_hit:
        # grow the area of surface surrounding the vertex
        # radius_label, = mne.grow_labels(subject, closest_vert, approx, hemi_code,
        #     subjects_dir=subjects_dir, surface='pial')

        # bins = calc_neighbors(closest_vert_pos, approx + elc_length, nei_dimensions, calc_bins=True)
        bins = calc_neighbors(pos, approx + elc_length, nei_dimensions, calc_bins=True)
        elc_line = get_elec_line(pos, elc_ori, elc_length)

        hemi_verts_dists = np.min(cdist(elc_line, pia_verts[hemi_str]), 0)
        elec_hemi_vertices_mask = hemi_verts_dists < approx
        hemi_vertices_indices = np.arange(len(pia_verts[hemi_str]))
        elec_hemi_vertices = hemi_vertices_indices[elec_hemi_vertices_mask]
        elec_hemi_vertices_dists = hemi_verts_dists[elec_hemi_vertices_mask]

        compiled_excludes = re.compile('|'.join(excludes))
        _region_are_excluded = partial(fu.region_are_excluded, compiled_excludes=compiled_excludes)
        regions, regions_hits = calc_hits(labels, hemi_str, verts, elc_line, bins, approx, _region_are_excluded)
        subcortical_regions, subcortical_hits = identify_roi_from_aparc(pos, elc_line, elc_length, lut, aseg_data,
            approx=approx, nei_dimensions=nei_dimensions, subcortical_only=True, excludes=excludes)

        we_have_a_hit = do_we_have_a_hit(regions, subcortical_regions)
        if not we_have_a_hit and enlarge_if_no_hit:
            approx += .5
            if elc_type == DEPTH:
                elc_length += 1
            elif elc_type == GRID:
                logging.warning('Grid electrode ({}) without a cortical hit?!?! Trying a bigger cigar'.format(elc_name))
            print('No hit! Recalculate with a bigger cigar')

    return regions, regions_hits, subcortical_regions, subcortical_hits, approx, elc_length,\
           elec_hemi_vertices, elec_hemi_vertices_dists, hemi_str


def do_we_have_a_hit(regions, subcortical_regions):
    if len(regions) == 0 and len(subcortical_regions) == 0:
        we_have_a_hit = False
    else:
        we_have_a_hit = not electrode_is_only_in_white_matter(regions, subcortical_regions)
    return we_have_a_hit


def get_elec_line(elec_pos, elec_ori, elec_length, points_number=100):
    if not elec_ori is None:
        elc_line = [elec_pos + elec_ori * t for t in np.linspace(-elec_length / 2.0, elec_length / 2.0, points_number)]
    else:
        elc_line = [elec_pos]
    return elc_line


def calc_hits(labels, hemi_str, surf_verts, elc_line, bins, approx, _region_are_excluded):
    labels_with_hits, labels_hits = [], []
    # res.append(('', 0))
    for label in labels:
        label = utils.Bag(label)
        # parcel = mne.read_label(parcel_file)
        if label.hemi != hemi_str:
            continue
        elif _region_are_excluded(str(label.name)):
            continue
        else:
            hits = calc_hits_in_neighbors_from_line(elc_line, surf_verts[label.vertices], bins, approx)
        if hits > 0:
            labels_with_hits.append(str(label.name))
            labels_hits.app(hits)
        # res.append((str(label.name), hits))
    # return res
    return labels_with_hits, labels_hits


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
    return len(regions) == 0 and len(subcortical_regions) == 1 and \
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
        dists = calc_dists_between_line_and_bin_centers(line, bin_edges, hist)
        hits = len(np.where((hist > 0) & (dists<approx))[0])
        return hits


def calc_dists_between_line_and_bin_centers(line, bin_edges, hist):
    hist_bin_centers_list = [bin_edges[d][:-1] + (bin_edges[d][1:] - bin_edges[d][:-1]) / 2.
                             for d in range(len(bin_edges))]
    bin_centers = list(product(*hist_bin_centers_list))
    dists = np.min(cdist(line, bin_centers), 0).reshape(hist.shape)
    return dists


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
        _region_are_excluded = partial(fu.region_are_excluded, compiled_excludes=compiled_excludes)
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
                if not _region_are_excluded(region):
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

    if aseg_data is None:
        return [], []

    if subcortical_only:
        excludes.append('ctx')

    ras_pos = to_ras([pos])[0]
    ras_elc_line = to_ras(elc_line)
    return find_neighboring_regions(ras_pos, elc_length, ras_elc_line, aseg_data, lut, approx, nei_dimensions, excludes)


def exclude_regions(regions, excludes):
    if (excludes):
        excluded = compile('|'.join(excludes))
        regions = [x for x in regions if not excluded.search(x)]
    return regions


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


def grid_or_depth(data):
    pos = data[:, 1:].astype(float)
    dists = defaultdict(list)
    group_type = {}
    electrodes_group_type = [None] * pos.shape[0]
    for index in range(data.shape[0] - 1):
        elc_group1, _ = elec_group_number(data[index, 0])
        elc_group2, _ = elec_group_number(data[index + 1, 0])
        if elc_group1 == elc_group2:
            dists[elc_group1].append(np.linalg.norm(pos[index + 1] - pos[index]))
    for group, group_dists in dists.items():
        #todo: not sure this is the best way to check it. Strip with 1xN will be mistaken as a depth
        if np.max(group_dists) > 2 * np.median(group_dists):
            group_type[group] = GRID
        else:
            group_type[group] = DEPTH
    for index in range(data.shape[0]):
        elc_group, _ = elec_group_number(data[index, 0])
        electrodes_group_type[index] = group_type[elc_group]
    return np.array(electrodes_group_type)


def get_electrodes_from_file(pos_fname, bipolar):
    f = np.load(pos_fname)
    if 'pos' not in f or 'names' not in f:
        raise Exception('electrodes posistions file was given, but without the fields "pos" or "names"!')
    pos, names = f['pos'], f['names']
    if not bipolar:
        dists = [np.linalg.norm(p2 - p1) for p1, p2 in zip(pos[:-1], pos[1:])]
        # Add the distance for the last electrode
        dists.append(0.)
    else:
        for ind in range(len(names) - 1):
            group1, _, _ = elec_group_number(names[ind], True)
            group2, _, _ = elec_group_number(names[ind], True)
            names[ind + 1]


def raise_err(err_msg):
    logging.error(err_msg)
    raise Exception(err_msg)


def get_electrodes(subject, bipolar, elecs_dir='', delimiter=',', pos_fname=''):
    if pos_fname != '':
        f = np.load(pos_fname)
        if 'pos' not in f or 'names' not in f or 'dists' not in f:
            raise_err('electrodes positions file was given, but without the fields' +
                '"pos", "names", "dists" or "electrodes_types!')
        if f['bipolar'] != bipolar:
            raise_err('electrodes positions file was given, but its bipolarity is {} '.format(f['bipolar']) + \
                'while you set the bipolarity to {}!'.format(bipolar))
        return f['names'], f['pos'], f['dists'], f['electrodes_types']

    check_for_electrodes_coordinates_file(subject)
    if elecs_dir=='':
        elecs_dir = get_electrodes_dir()
    elec_file = op.join(elecs_dir, '{}.csv'.format(subject))
    data = np.genfromtxt(elec_file, dtype=str, delimiter=delimiter)
    data = fix_str_items_in_csv(data)
    # Check if the electrodes coordinates has a header
    try:
        header = data[0, 1:].astype(float)
    except:
        data = np.delete(data, (0), axis=0)
        print('First line in the electrodes RAS coordinates is a header')

    electrodes_types = grid_or_depth(data)
    # print([(n, elec_group_number(n), t) for n, t in zip(data[:, 0], electrodes_group_type)])
    if bipolar:
        depth_data = data[electrodes_types == DEPTH, :]
        pos = depth_data[:, 1:4].astype(float)
        pos_depth, names_depth, dists_depth = [], [], []
        for index in range(depth_data.shape[0]-1):
            elc_group1, elc_num1 = elec_group_number(depth_data[index, 0])
            elc_group2, elc_num12 = elec_group_number(depth_data[index+1, 0])
            if elc_group1 == elc_group2:
                elec_name = '{}-{}'.format(depth_data[index+1, 0],depth_data[index, 0])
                names_depth.append(elec_name)
                pos_depth.append(pos[index] + (pos[index+1]-pos[index])/2)
                dists_depth.append(np.linalg.norm(pos[index+1]-pos[index]))
        # There is no point in calculating bipolar for grid electrodes
        grid_data = data[electrodes_types == GRID, :]
        names_grid, _, pos_grid = get_names_dists_non_bipolar_electrodes(grid_data)
        names = np.concatenate((names_depth, names_grid))
        # Put zeros as dists for the grid electrodes
        dists = np.concatenate((np.array(dists_depth), np.zeros((len(names_grid)))))
        pos = utils.vstack(pos_depth, pos_grid)
        electrodes_types = [DEPTH] * len(names_depth) + [GRID] * len(names_grid)
    else:
        names, dists, pos = get_names_dists_non_bipolar_electrodes(data)
    # names = np.array([name.strip() for name in names])
    if not len(names) == len(pos) == len(dists) == len(electrodes_types):
        logging.error('get_electrodes ({}): not len(names)==len(pos)==len(dists)!'.format(subject))
        raise Exception('get_electrodes: not len(names)==len(pos)==len(dists)!')

    return names, pos, dists, electrodes_types


def get_names_dists_non_bipolar_electrodes(data):
    names = data[:, 0]
    pos = data[:, 1:4].astype(float)
    dists = [np.linalg.norm(p2 - p1) for p1, p2 in zip(pos[:-1], pos[1:])]
    # Add the distance for the last electrode
    dists.append(0.)
    return names, dists, pos


def fix_str_items_in_csv(csv):
    lines = []
    for line in csv:
        if 'ref' in line[0].lower() or len(re.findall('\d', line[0])) == 0:
            continue
        fix_line = list(map(lambda x: str(x).replace('"', ''), line))
        if not np.all([len(v) == 0 for v in fix_line[1:]]):
            fix_line[0] = fix_line[0].strip()
            lines.append(fix_line)
    return np.array(lines)


def get_electrodes_dir():
    curr_dir = op.dirname(op.realpath(__file__))
    elec_dir = op.join(op.split(curr_dir)[0], 'electrodes')
    utils.make_dir(elec_dir)
    return elec_dir


def write_results_to_csv(results, results_fname_csv_temlate, args, bipolar_electrodes):
    if args.write_all_labels:
        utils.make_dir(utils.get_resources_fol())
        cortical_rois = lu.read_labels(args.subject[0], args.subjects_dir, args.atlas, only_names=True,
            output_fname=op.join(utils.get_resources_fol(), '{}_corticals.txt'.format(args.atlas)), n_jobs=args.n_jobs)
        input_fname = output_fname = op.join(utils.get_resources_fol(), 'subcorticals.txt')
        subcortical_rois, subcortical_rois_header = fu.get_subcortical_regions(args.excludes, output_fname, input_fname,
              ['Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter'])
    else:
        cortical_rois, subcortical_rois = [], []
        for elecs in results.values():
            for elc in elecs:
                cortical_rois.extend(elc['cortical_rois'])
                subcortical_rois.extend(elc['subcortical_rois'])
        cortical_rois = list(np.unique(cortical_rois))
        subcortical_rois = list(np.unique(subcortical_rois))
        subcortical_rois_header = subcortical_rois

    for subject, elecs in results.items():
        results_fname_csv = results_fname_csv_temlate.format(subject=subject)
        write_values(elecs, results_fname_csv,
            ['electrode'] + cortical_rois + subcortical_rois_header + ['approx', 'elc_length'],
            [cortical_rois, subcortical_rois],
            ['cortical_rois','subcortical_rois'], ['cortical_probs', 'subcortical_probs'], args, bipolar_electrodes)

        if args.write_only_cortical:
            write_values(elecs, results_fname_csv.replace('electrodes', 'cortical_electrodes'),
                         ['electrode'] + cortical_rois, [cortical_rois],['cortical_rois'], ['cortical_probs'],
                         args, bipolar_electrodes)

        if args.write_only_subcortical:
            write_values(elecs, results_fname_csv.replace('electrodes', 'subcortical_electrodes'),
                ['electrode']  + subcortical_rois_header, [subcortical_rois],
                ['subcortical_rois'], ['subcortical_probs'], args, bipolar_electrodes)


def write_values(elecs, results_fname, header, rois_arr, rois_names, probs_names, args, bipolar_electrodes=False):
    try:
        with open(results_fname, 'w') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(header)
            print('Writing {} with header length of {}'.format(results_fname, len(header)))
            for elc in elecs:
                elc_name = elc['name']
                if args.write_compact_bipolar and bipolar_electrodes and '-' in elc_name:
                    elc_group, elc_num1, elc_num2 = elec_group_number(elc_name, True)
                    elc_name = '{}{}{}'.format(elc_group, elc_num1, elc_num2)
                values = [elc_name]
                for rois, rois_field, prob_field in zip(rois_arr, rois_names, probs_names):
                    for col, roi in enumerate(rois):
                        if roi in elc[rois_field]:
                            index = elc[rois_field].index(roi)
                            values.append(str(elc[prob_field][index]))
                        else:
                            values.append(0.)
                values.extend([elc['approx'], elc['elc_length']])
                writer.writerow(values)
    except:
        logging.error('write_values to {}: {}'.format(results_fname, traceback.format_exc()))
        print(traceback.format_exc())


# def add_labels_per_electrodes_probabilities(subject, elecs_dir='', post_fix=''):
#     if elecs_dir == '':
#         elecs_dir = get_electrodes_dir()
#     csv_fname = op.join(elecs_dir, '{}_{}_electrodes_all_rois{}.csv'.format(subject, atlas, post_fix))
#     np.genfromtxt(csv_fname)


def get_electrodes_orientation(elecs_names, elecs_pos, bipolar, elecs_types, elecs_oris_fname=''):
    if elecs_oris_fname != '':
        f = np.load(elecs_oris_fname)
        if 'electrodes_oris' not in f:
            logging.error('elecs oris fname was given, but without the "electrodes_oris" field!')
            raise Exception('elecs oris fname was given, but without the "electrodes_oris" field!')
        elcs_oris = f['electrodes_oris']
        if elcs_oris.shape[0] != len(elecs_names):
            logging.error('elecs oris fname was given, but with a wrong number of orientations!')
            raise Exception('elecs oris fname was given, but with a wrong number of orientations!')
        return elcs_oris

    elcs_oris = np.zeros((len(elecs_names), 3))
    for index, (elc_name, elc_pos, elc_type) in enumerate(zip(elecs_names, elecs_pos, elecs_types)):
        if elecs_types[index] == DEPTH:
            if bipolar:
                elc_group, elc_num1, elc_num2 = elec_group_number(elc_name, True)
                next_elc = '{}{}-{}{}'.format(elc_group, elc_num2 + 1, elc_group, elc_num1 + 1)
            else:
                elc_group, elc_num = elec_group_number(elc_name)
                next_elc = '{}{}'.format(elc_group, elc_num+1)
            ori = 1
            if next_elc not in elecs_names:
                if bipolar:
                    next_elc = '{}{}-{}{}'.format(elc_group, elc_num1, elc_group, elc_num1-1)
                else:
                    next_elc = '{}{}'.format(elc_group, elc_num-1)
                ori = -1
            next_elc_index = np.where(elecs_names==next_elc)[0][0]
            next_elc_pos = elecs_pos[next_elc_index]
            dist = np.linalg.norm(next_elc_pos-elc_pos)
            elcs_oris[index] = ori * (next_elc_pos - elc_pos) / dist # norm(elc_ori)=1mm
        # print(elc_name, elc_pos, next_elc, next_elc_pos, elc_line(1))
    return elcs_oris


def elec_group_number(elec_name, bipolar=False):
    if bipolar:
        elec_name2, elec_name1 = elec_name.split('-')
        group, num1 = elec_group_number(elec_name1, False)
        _, num2 = elec_group_number(elec_name2, False)
        return group, num1, num2
    else:
        # ind = np.where([int(s.isdigit()) for s in elec_name])[-1][0]
        # num = int(elec_name[ind:])
        elec_name = elec_name.strip()
        num = int(re.sub('\D', ',', elec_name).split(',')[-1])
        # group = elec_name[:ind]
        group = elec_name[:elec_name.rfind(str(num))]
        # print('name: {}, group: {}, num: {}'.format(elec_name, group, num))
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
        overwrite_annot=False, read_labels_from_annotation=False, solve_labels_collisions=False,
        freesurfer_home='', n_jobs=6):
    annot_file = op.join(subjects_dir, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))
    if '{}.annot'.format(atlas) in EXISTING_FREESURFER_ANNOTATIONS and \
            (not op.isfile(annot_file.format(hemi='rh')) or not op.isfile(annot_file.format(hemi='lh'))):
        overwrite_annot = False
        solve_labels_collisions = False
        if freesurfer_home == '':
            freesurfer_home = os.environ.get('FREESURFER_HOME', '')
        if freesurfer_home == '':
            raise Exception('There are no annotation file for {}, please source freesurfer and run again'.format(atlas))
        fu.create_freesurfer_annotation_file(subject, atlas, subjects_dir, freesurfer_home)
    if read_labels_from_annotation and op.isfile(annot_file.format(hemi='rh')) and \
            op.isfile(annot_file.format(hemi='lh')):
        # Nothing to do, read the labels from an existing annotation file
        return
    if overwrite_labels or overwrite_annot or not op.isfile(annot_file.format(hemi='rh')) or not \
            op.isfile(annot_file.format(hemi='lh')):
        morph_labels_from_fsaverage(subject, subjects_dir, atlas, n_jobs=n_jobs, fsaverage=fsaverage,
                                    overwrite=overwrite_labels)
        if solve_labels_collisions:
            backup_labels_fol = '{}_before_solve_collision'.format(atlas, fsaverage)
            lu.solve_labels_collision(subject, subjects_dir, atlas, backup_labels_fol, n_jobs)
        # lu.backup_annotation_files(subject, subjects_dir, atlas)
        # labels_to_annot(subject, subjects_dir, atlas, overwrite=overwrite_annot)


def morph_labels_from_fsaverage(subject, subjects_dir='', aparc_name='aparc250', fs_labels_fol='', sub_labels_fol='',
                                n_jobs=6, fsaverage='fsaverage', overwrite=False):
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
                logging.error("The file {} doesn't exist in the local subjects folder!!!".format(file_name))
                all_files_exists = False
    copy_electrodes_ras_file(subject, local_subject_dir, remote_subject_dir)
    if not all_files_exists:
        raise Exception('Not all files exist in the local subject folder!!!')
        logging.error('{}: {}'.format(subject, 'Not all files exist in the local subject folder!!!'))


def copy_electrodes_ras_file(subject, local_subject_dir, remote_subject_dir):
    # local_file_found = False
    for local_ras_fname, remote_ras_fname in zip(
            [op.join(local_subject_dir, 'electrodes', '{}_RAS.xlsx'.format(subject.upper())),
             op.join(local_subject_dir, 'electrodes', '{}_RAS.xlsx'.format(subject))],
            [op.join(remote_subject_dir, '{}_RAS.xlsx'.format(subject.upper())),
             op.join(remote_subject_dir, '{}_RAS.xlsx'.format(subject))]):
        # local_ras_fname = op.join(local_subject_dir, 'electrodes', '{}_RAS.xlsx'.format(subject.upper()))
        # remote_ras_fname = op.join(remote_subject_dir, '{}_RAS.xlsx'.format(subject.upper()))
        if not op.isfile(local_ras_fname) and op.isfile(remote_ras_fname):
            shutil.copyfile(remote_ras_fname, local_ras_fname)
        # local_file_found = local_file_found or op.isfile(local_ras_fname)
    # if not local_file_found and not op.isfile():
    #     raise Exception("Can't find electrodes RAS coordinates! {}".format(local_ras_fname))
    #     logging.error("Can't find electrodes RAS coordinates! {}".format(local_ras_fname))


def check_for_necessary_files(subjects_dir, subject, neccesary_files, remote_subject_dir_template):
    remote_subject_dir = build_remote_subject_dir(remote_subject_dir_template, subject)
    prepare_local_subjects_folder(neccesary_files, subject, remote_subject_dir, subjects_dir,
        print_traceback=True)


def check_for_electrodes_coordinates_file(subject):
    elecs_dir = get_electrodes_dir()
    elec_file = op.join(elecs_dir, '{}.csv'.format(subject))
    # if not op.isfile(elec_file) or op.getsize(elec_file) == 0:
    copy_electrodes_file(subjects_dir, subject, elec_file)


def copy_electrodes_file(subjects_dir, subject, elec_file):
    subject_elec_fname = op.join(subjects_dir, subject, 'electrodes', '{}_RAS.csv'.format(subject))
    if not op.isfile(subject_elec_fname) or op.getsize(subject_elec_fname) == 0:
       rename_and_convert_electrodes_file(subject, subjects_dir)
    if op.isfile(subject_elec_fname):
        if op.isfile(elec_file):
            os.remove(elec_file)
        shutil.copyfile(subject_elec_fname, elec_file)
    else:
        raise Exception('{}: Electrodes file does not exist! {}'.format(subject, subject_elec_fname))


def rename_and_convert_electrodes_file(subject, subjects_dir):
    subject_elec_fname_no_ras_pattern = op.join(subjects_dir, subject, 'electrodes', '{subject}.{postfix}')
    subject_elec_fname_pattern = op.join(subjects_dir, subject, 'electrodes', '{subject}_RAS.{postfix}')
    subject_elec_fname_csv = subject_elec_fname_pattern.format(subject=subject, postfix='csv')
    subject_elec_fname_xlsx = subject_elec_fname_pattern.format(subject=subject, postfix='xlsx')

    utils.rename_files([subject_elec_fname_no_ras_pattern.format(subject=subject, postfix='xlsx'),
                        subject_elec_fname_no_ras_pattern.format(subject=subject.upper(), postfix='xlsx'),
                        subject_elec_fname_no_ras_pattern.format(subject=subject, postfix='xls'),
                        subject_elec_fname_no_ras_pattern.format(subject=subject.upper(), postfix='xls')],
                       subject_elec_fname_pattern.format(subject=subject, postfix='xlsx'))
    utils.rename_files([subject_elec_fname_pattern.format(subject=subject.upper(), postfix='csv')],
                       subject_elec_fname_csv)
    utils.rename_files([subject_elec_fname_pattern.format(subject=subject.upper(), postfix='xlsx')],
                       subject_elec_fname_xlsx)
    if op.isfile(subject_elec_fname_xlsx) and \
                    (not op.isfile(subject_elec_fname_csv) or op.getsize(subject_elec_fname_csv) == 0):
        utils.csv_from_excel(subject_elec_fname_xlsx, subject_elec_fname_csv)


def run_for_all_subjects(subjects, atlas, subjects_dir, bipolar_electrodes, neccesary_files,
                         remote_subject_dir_template, output_files_post_fix, args, freesurfer_home):
    ok_subjects, bad_subjects = [], []
    results = {}
    for subject in subjects:
        print('****************** {} ******************'.format(subject))
        logging.info('****************** {} bipolar {}, {}******************'.format(subject, bipolar_electrodes, utils.now()))
        results_fname_csv_template = op.join(get_electrodes_dir(), '{}_{}_electrodes{}.csv'.format(
            '{subject}', atlas, output_files_post_fix))
        results_fname_csv = results_fname_csv_template.format(subject=subject)
        results_fname_pkl = results_fname_csv.replace('csv', 'pkl')
        if not op.isfile(results_fname_csv) or args.overwrite_csv:
            try:
                if op.isfile(results_fname_pkl) and not args.overwrite:
                    elecs = utils.load(results_fname_pkl)
                elif 'all' in args.function:
                    check_for_necessary_files(subjects_dir, subject, neccesary_files, remote_subject_dir_template)
                    check_for_annot_file(subject, subjects_dir, atlas, args.template_brain, args.overwrite_labels,
                        args.overwrite_annotation, args.read_labels_from_annotation, args.solve_labels_collisions,
                        freesurfer_home, n_jobs=args.n_jobs)
                    if args.only_check_files:
                        continue
                    elecs_names, elecs_pos, elecs_dists, elecs_types = get_electrodes(
                        subject, bipolar_electrodes, pos_fname=args.pos_fname)
                    elcs_ori = get_electrodes_orientation(
                        elecs_names, elecs_pos, bipolar_electrodes, elecs_types, elecs_oris_fname=args.pos_fname)
                    labels = read_labels_vertices(subjects_dir, subject, atlas, args.read_labels_from_annotation,
                        args.overwrite_labels_pkl, args.n_jobs)
                    elecs = identify_roi_from_atlas(
                        labels, elecs_names, elecs_pos, elcs_ori, args.error_radius, args.elc_length,
                        elecs_dists, elecs_types, args.strech_to_dist, args.enlarge_if_no_hit,
                        bipolar_electrodes, subjects_dir, subject, args.excludes, args.n_jobs)
                    utils.save(elecs, results_fname_pkl)
                if au.should_run('add_colors_to_probs', args):
                    add_colors_to_probs(subject, atlas, results_fname_pkl)
                results[subject] = elecs
                ok_subjects.append(subject)
            except:
                bad_subjects.append(subject)
                logging.error('{}: {}'.format(subject, traceback.format_exc()))
                print(traceback.format_exc())

    # Write the results for all the subjects at once, to have a common labeling
    write_results_to_csv(results, results_fname_csv_template, args, bipolar_electrodes)

    if ok_subjects:
        print('ok subjects:')
        print(ok_subjects)
    if bad_subjects:
        print('bad_subjects:')
        print(bad_subjects)
        logging.error('bad_subjects:')
        logging.error(bad_subjects)


def add_colors_to_probs(subject, atlas, results_fname):
    # results_fname = op.join(get_electrodes_dir(), '{}_{}_electrodes{}.pkl'.format(
    #     subject, atlas, output_files_post_fix))
    if op.isfile(results_fname):
        elecs = utils.load(results_fname)
        for elc in elecs:
            elc['subcortical_colors'] = cu.arr_to_colors(elc['subcortical_probs'], colors_map='YlOrRd')
            elc['cortical_colors'] = cu.arr_to_colors(elc['cortical_probs'], colors_map='YlOrRd')
        utils.save(elecs, results_fname)
    else:
        print("!!! Can't find the probabilities file !!!")


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
    import argparse
    from src import args_utils as au

    subjects_dir = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
    freesurfer_home = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
    blender_dir = utils.get_link_dir(LINKS_DIR, 'mmvt')
    os.environ['SUBJECTS_DIR'] = subjects_dir
    os.environ['FREESURFER_HOME'] = freesurfer_home

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas40')
    parser.add_argument('-f', '--function', help='function name', required=False, default='all')
    parser.add_argument('-b', '--bipolar', help='bipolar electrodes', required=False, default='1,0', type=au.bool_arr_type)
    parser.add_argument('--error_radius', help='error radius', required=False, default=3)
    parser.add_argument('--elc_length', help='elc length', required=False, default=4)
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    parser.add_argument('--template_brain', help='template brain', required=False, default='fsaverage5c')
    parser.add_argument('--strech_to_dist', help='strech_to_dist', required=False, default=1, type=au.is_true)
    parser.add_argument('--enlarge_if_no_hit', help='enlarge_if_no_hit', required=False, default=1, type=au.is_true)
    parser.add_argument('--only_check_files', help='only_check_files', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite', help='overwrite', required=False, default=1, type=au.is_true)
    parser.add_argument('--overwrite_annotation', help='overwrite_annotation', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_labels', help='overwrite_labels', required=False, default=0, type=au.is_true)
    parser.add_argument('--write_only_cortical', help='write_only_cortical', required=False, default=0, type=au.is_true)
    parser.add_argument('--write_only_subcortical', help='write_only_subcortical', required=False, default=0, type=au.is_true)
    parser.add_argument('--write_all_labels', help='Write all the labels', required=False, default=1, type=au.is_true)
    parser.add_argument('--overwrite_labels_pkl', help='overwrite_labels_pkl', required=False, default=1, type=au.is_true)
    parser.add_argument('--overwrite_csv', help='overwrite_csv', required=False, default=1, type=au.is_true)
    parser.add_argument('--read_labels_from_annotation', help='read_labels_from_annotation', required=False, default=1, type=au.is_true)
    parser.add_argument('--solve_labels_collisions', help='solve_labels_collisions', required=False, default=0, type=au.is_true)
    parser.add_argument('--remote_subject_dir_template', help='remote_subject_dir_template', required=False)
    parser.add_argument('--pos_fname', help='electrodes positions fname', required=False, default='')
    parser.add_argument('--output_postfix', help='output_postfix', required=False, default='')
    parser.add_argument('--write_compact_bipolar', help='write x23 instead x3-x2', required=False, default=0, type=au.is_true)
    parser.add_argument('--excludes', help='excluded labels', required=False, type=au.str_arr_type,
                        default='Unknown,unknown,Cerebral-Cortex,corpuscallosum,WM-hypointensities,Ventricle')

    args = utils.Bag(au.parse_parser(parser))
    args.n_jobs = utils.get_n_jobs(args.n_jobs)
    args.subjects_dir = subjects_dir
    print(args)

    subjects, atlas = args['subject'], args['atlas'] # 'arc_april2016' # 'aparc.DKTatlas40' # 'laus250'
    os.environ['SUBJECT'] = subjects[0]

    neccesary_files = {'mri': ['aseg.mgz'], 'surf': ['rh.pial', 'lh.pial', 'rh.sphere.reg', 'lh.sphere.reg', 'lh.white', 'rh.white', 'lh.smoothwm', 'rh.smoothwm']}
    remote_subject_dir_template = {'template':'/home/ieluuser/links/subjects/{subject}_SurferOutput', 'func': lambda x: x.upper()}
    # if subject == 'all':
    #     subjects = set(get_all_subjects(subjects_dir, 'mg', '_')) - set(['mg63', 'mg94']) # get_subjects()
    # else:
    #     subjects = [subject]
    logging.basicConfig(filename='log.log',level=logging.DEBUG)
    for bipolar in args['bipolar']:
        output_files_post_fix = '_cigar_r_{}_l_{}{}{}{}'.format(args['error_radius'], args['elc_length'],
            '_bipolar' if bipolar else '', '_not_stretch' if not args['strech_to_dist'] and bipolar else '',
            args.output_postfix)
        run_for_all_subjects(subjects, atlas, subjects_dir, bipolar, neccesary_files, remote_subject_dir_template,
                         output_files_post_fix, args, freesurfer_home)
    print('finish!')
