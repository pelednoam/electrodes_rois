import argparse
import csv
import getpass
import glob
import logging
import os
import os.path as op
import re
import shutil
import traceback
from collections import Counter, defaultdict, OrderedDict
from functools import partial
from itertools import product

import mne
import nibabel as nib
import numpy as np
from mne.surface import read_surface
from scipy.spatial.distance import cdist

from find_rois import args_utils as au
from find_rois import colors_utils as cu
from find_rois import freesurfer_utils as fu
from find_rois import labels_utils as lu
from find_rois import utils
from find_rois.snap_grid_to_pial import snap_electrodes_to_surface

LINKS_DIR = utils.get_links_dir()
ELECTRODES_TYPES = ('depth', 'grid', 'strip', 'microgrid', 'neuroport')
DEPTH, GRID = range(2)
EXISTING_FREESURFER_ANNOTATIONS = ['aparc.DKTatlas40.annot', 'aparc.annot', 'aparc.a2009s.annot']


def identify_roi_from_atlas(atlas, labels, elecs_names, elecs_pos, elecs_ori=None, approx=4, elc_length=1,
                            elecs_dists=None, elecs_types=None, strech_to_dist=False, enlarge_if_no_hit=False,
                            bipolar=False, subjects_dir=None, subject=None, excludes=None,
                            specific_elec='', nei_dimensions=None, aseg_atlas=True, aseg_data=None, lut=None,
                            pia_verts=None, n_jobs=6):

    if subjects_dir is None or subjects_dir == '':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject == '':
        subject = os.environ['SUBJECT']

    # get the segmentation file
    if aseg_data is None:
        aseg_fname = op.join(subjects_dir, subject, 'mri', 'aseg.mgz')
        asegf = aseg_fname
        aseg_atlas_fname = op.join(subjects_dir, subject, 'mri', '{}+aseg.mgz'.format(atlas))
        lut_atlast_fname = op.join(subjects_dir, subject, 'mri', '{}ColorLUT.txt'.format(atlas))
        lut_fname = ''
        if aseg_atlas:
            if op.isfile(aseg_atlas_fname):
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

    if lut is None:
        lut = fu.import_freesurfer_lut(lut_fname)

    if pia_verts is None:
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
               elc_length, nei_dimensions, strech_to_dist, enlarge_if_no_hit, bipolar, excludes,
               specific_elec, N) for elecs_data_chunk in elecs_data_chunks]
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
        nei_dimensions, strech_to_dist, enlarge_if_no_hit, bipolar, excludes, specific_elec, N = params
    for elc_num, (elec_pos, elec_name, elc_ori, elc_dist, elc_type) in elecs_data_chunk:
        if specific_elec != '' and elec_name != specific_elec:
            continue
        print('{}: {} / {}'.format(elec_name, elc_num, N))
        regions, regions_hits, subcortical_regions, subcortical_hits, approx_after_strech, elc_length, elec_hemi_vertices, \
                elec_hemi_vertices_dists, hemi = \
            identify_roi_from_atlas_per_electrode(labels, elec_pos, pia_verts, len_lh_pia, lut,
                aseg_data, elec_name, approx, elc_length, nei_dimensions, elc_ori, elc_dist, elc_type, strech_to_dist,
                enlarge_if_no_hit, bipolar, subjects_dir, subject, excludes, n_jobs=1)
        results.append((elec_name, regions, regions_hits, subcortical_regions, subcortical_hits, approx_after_strech, elc_length,
                        elec_hemi_vertices, elec_hemi_vertices_dists, hemi))
    return results


def identify_roi_from_atlas_per_electrode(labels, pos, pia_verts, len_lh_pia, lut, aseg_data, elc_name,
    approx=4, elc_length=1, nei_dimensions=None, elc_ori=None, elc_dist=0, elc_type=DEPTH, strech_to_dist=False,
    enlarge_if_no_hit=False, bipolar=False, subjects_dir=None, subject=None, excludes=None, n_jobs=1):
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
        excludes = ['Unknown', 'unknown', 'Cerebral-Cortex', 'corpuscallosum', 'WM-hypointensities', 'Ventricle']
    compiled_excludes = re.compile('|'.join(excludes))
    _region_are_excluded = partial(fu.region_are_excluded, compiled_excludes=compiled_excludes)

    # find closest vertex
    pia = np.vstack((pia_verts['lh'], pia_verts['rh']))
    closest_vert = np.argmin(cdist(pia, [pos]))

    # we force the label to only contact one hemisphere even if it is
    # beyond the extent of the medial surface
    hemi_str = 'lh' if closest_vert<len_lh_pia else 'rh'
    hemi_code = 0 if hemi_str=='lh' else 1

    if hemi_str == 'rh':
        closest_vert -= len_lh_pia

    # todo: send the verts to the function
    surf_fname = op.join(subjects_dir, subject, 'surf', hemi_str + '.pial')
    verts, _ = read_surface(surf_fname)
    # closest_vert_pos = verts[closest_vert]

    # if elc_type == GRID:
    #     elc_dist = 0
    #     elc_length = 0
    #     elc_ori = None

    we_have_a_hit = False
    if strech_to_dist and bipolar and elc_length < elc_dist:
        elc_length = elc_dist
    loop_ind = 0
    while not we_have_a_hit and loop_ind < 10:
        if elc_type == GRID:
            regions, regions_hits = [], []
            # grow the area of surface surrounding the vertex
            radius_label, = mne.grow_labels(subject, closest_vert, approx, hemi_code,
                subjects_dir=subjects_dir, surface='pial')
            # inter_labels, inter_labels_tups = [], []
            # total_inter_verts = 0
            for label in labels:
                label = utils.Bag(label)
                if label.hemi != radius_label.hemi:
                    continue
                overlapped_vertices = np.intersect1d(radius_label.vertices, label.vertices)
                # overlapped_vertices = set(radius_label.vertices) & set(label['vertices'])
                if len(overlapped_vertices) > 0 and not _region_are_excluded(str(label.name)):
                    regions.append(label.name)
                    regions_hits.append(len(overlapped_vertices))
                    # total_inter_verts += len(overlapped_vertices)
                    # inter_labels_tups.append((len(overlapped_vertices), label['name']))
            # inter_labels_tups = sorted(inter_labels_tups)[::-1]
            hemi_verts_dists = cdist([verts[closest_vert]], pia_verts[hemi_str])[0]
            elec_hemi_vertices_mask = hemi_verts_dists < approx
            subcortical_regions, subcortical_hits = [], []
            we_have_a_hit = True
        else:
            # bins = calc_neighbors(closest_vert_pos, approx + elc_length, nei_dimensions, calc_bins=True)
            bins = calc_neighbors(pos, approx + elc_length, nei_dimensions, calc_bins=True)
            elc_line = get_elec_line(pos, elc_ori, elc_length)
            hemi_verts_dists = np.min(cdist(elc_line, pia_verts[hemi_str]), 0)
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
            print('{}: No hit! Recalculate with a bigger cigar ({})'.format(elc_name, loop_ind))
            loop_ind += 1

    elec_hemi_vertices_mask = hemi_verts_dists < approx
    hemi_vertices_indices = np.arange(len(pia_verts[hemi_str]))
    elec_hemi_vertices = hemi_vertices_indices[elec_hemi_vertices_mask]
    elec_hemi_vertices_dists = hemi_verts_dists[elec_hemi_vertices_mask]

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
            labels_hits.append(hits)
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
    return len(regions) == 0 and \
           len(set(subcortical_regions) - set(['Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter'])) == 0

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
        neighb = neighb[np.where(dists < approx)]
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
        logging.error('calc_neighbors: approx and dimensions are None!')
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


def grid_or_depth(data, electrods_type=None):
    pos = data[:, 1:4].astype(float)
    electrodes_types = [None] * pos.shape[0]

    if electrods_type is not None:
        print('All the electrodes are {}'.format('grid' if electrods_type == GRID else 'depth'))
        for index in range(data.shape[0]):
            electrodes_types[index] = electrods_type
        return np.array(electrodes_types), None

    if data.shape[1] > 4:
        if len(set(data[:, 4]) - set(ELECTRODES_TYPES)) > 0:
            raise Exception('In column 5 the only permitted values are {}'.format(ELECTRODES_TYPES))
        else:
            for ind, elc_type in enumerate(data[:, 4]):
                electrodes_types[ind] = GRID if elc_type in ['grid', 'strip'] else DEPTH
            return np.array(electrodes_types), data[:, 4]

    dists = defaultdict(list)
    group_type = {}
    for index in range(data.shape[0] - 1):
        elc_group1, _ = elec_group_number(data[index, 0])
        elc_group2, _ = elec_group_number(data[index + 1, 0])
        if elc_group1 == elc_group2:
            dists[elc_group1].append(np.linalg.norm(pos[index + 1] - pos[index]))
    for group, group_dists in dists.items():
        #todo: not sure this is the best way to check it. Strip with 1xN will be mistaken as a depth
        if np.max(group_dists) > 2 * np.percentile(group_dists, 25):
            group_type[group] = GRID
        else:
            group_type[group] = DEPTH
        print('group {} is {}'.format(group, 'grid' if group_type[group] == GRID else 'depth'))
    for index in range(data.shape[0]):
        elc_group, _ = elec_group_number(data[index, 0])
        electrodes_types[index] = group_type[elc_group]

    return np.array(electrodes_types), None


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


def get_electrodes_types_set(subject, args):
    subject_elecs_dir = op.join(args.subjects_dir, subject, 'electrodes')
    data = read_electrodes_xls(subject, subject_elecs_dir, args)
    if data.shape[1] > 4:
        return set(data[:, 4])
    return []


def get_electrodes(subject, bipolar, args):
    if args.pos_fname != '':
        f = np.load(args.pos_fname)
        if 'pos' not in f or 'names' not in f or 'dists' not in f:
            raise_err('electrodes positions file was given, but without the fields' +
                '"pos", "names", "dists" or "electrodes_types!')
        if f['bipolar'] != bipolar:
            raise_err('electrodes positions file was given, but its bipolarity is {} '.format(f['bipolar']) + \
                'while you set the bipolarity to {}!'.format(bipolar))
        return f['names'], f['pos'], f['dists'], f['electrodes_types'], None

    subject_elecs_dir = op.join(args.subjects_dir, subject, 'electrodes')
    utils.make_dir(subject_elecs_dir)
    if args.elecs_dir == '':
        args.elecs_dir = get_electrodes_dir()

    data = read_electrodes_xls(subject, subject_elecs_dir, args)
    # Check if the electrodes coordinates has a header
    try:
        header = data[0, 1:4].astype(float)
    except:
        data = np.delete(data, (0), axis=0)
        print('First line in the electrodes RAS coordinates is a header')

    electrodes_types, electrodes_types_names = grid_or_depth(data, args.electrodes_type)
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
        names_grid, _, pos_grid = get_names_dists_non_bipolar(grid_data)
        names = np.concatenate((names_depth, names_grid))
        # Put zeros as dists for the grid electrodes
        dists = np.concatenate((np.array(dists_depth), np.zeros((len(names_grid)))))
        pos = utils.vstack(pos_depth, pos_grid)
        electrodes_types = [DEPTH] * len(names_depth) + [GRID] * len(names_grid)
        electrodes_types_names = ['depth'] * len(names_depth)
        for name in names[len(names_depth):]:
            electrodes_types_names.append(data[np.where(data[:, 0] == name)[0], -1][0])
    else:
        names, dists, pos = get_names_dists_non_bipolar(data)
    # names = np.array([name.strip() for name in names])
    if not len(names) == len(pos) == len(dists) == len(electrodes_types):
        logging.error('get_electrodes ({}): not len(names)==len(pos)==len(dists)!'.format(subject))
        raise Exception('get_electrodes: not len(names)==len(pos)==len(dists)!')

    return names, pos, dists, np.array(electrodes_types), electrodes_types_names


def read_electrodes_xls(subject, subject_elecs_dir, args):
    if args.snap:
        elec_file = op.join(subject_elecs_dir, '{}_snap_RAS.csv'.format(subject))
        if op.isfile(elec_file):
            print('Reading snap RAS coordinates file')
            data = np.genfromtxt(elec_file, dtype=str, delimiter=args.csv_delimiter)
            return data
        else:
            print('snap is True but there is no snap csv file! {}'.format(elec_file))

    file_exist = rename_and_convert_electrodes_file(subject, subject_elecs_dir)
    if not file_exist:
        mmvt_elecs_dir = utils.make_dir(op.join(args.mmvt_dir, subject, 'electrodes'))
        file_exist = rename_and_convert_electrodes_file(subject, mmvt_elecs_dir)
        if file_exist:
            shutil.copy(op.join(mmvt_elecs_dir, '{}_RAS.csv'.format(subject)),
                        op.join(subject_elecs_dir, '{}_RAS.csv'.format(subject)))
    if not op.isfile(op.join(args.elecs_dir, '{}_RAS.csv'.format(subject))) and not file_exist:
        raise Exception('No coordinates csv file for {}!'.format(subject))
    if not op.isfile(op.join(subject_elecs_dir, '{}_RAS.csv'.format(subject))) and op.isfile(
            op.join(args.elecs_dir, '{}_RAS.csv'.format(subject))):
        shutil.copy(op.join(args.elecs_dir, '{}_RAS.csv'.format(subject)),
                    op.join(subject_elecs_dir, '{}_RAS.csv'.format(subject)))
    check_for_electrodes_coordinates_file(subject, args.subjects_dir, args.elecs_dir)
    elec_file = op.join(args.elecs_dir, '{}.csv'.format(subject))
    data = np.genfromtxt(elec_file, dtype=str, delimiter=args.csv_delimiter)
    data = fix_str_items_in_csv(data)
    return data


def get_names_dists_non_bipolar(data):
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
        if not np.all([len(v) == 0 for v in fix_line[1:]]) and np.all([utils.is_float(x) for x in fix_line[1:4]]):
            fix_line[0] = fix_line[0].strip()
            lines.append(fix_line)
        else:
            print('csv: ignoring the following line: {}'.format(line))
    return np.array(lines)


def get_electrodes_dir():
    curr_dir = op.dirname(op.realpath(__file__))
    elec_dir = op.join(op.split(curr_dir)[0], 'electrodes')
    utils.make_dir(elec_dir)
    return elec_dir


def write_results_to_csv(results, elecs_types, args):
    if args.write_all_labels:
        utils.make_dir(utils.get_resources_fol())
        cortical_rois = lu.read_labels(
            args.subject[0], args.subjects_dir, args.atlas, only_names=True,
            output_fname=op.join(utils.get_resources_fol(), '{}_corticals.txt'.format(args.atlas)),
            remove_unknown=True, n_jobs=args.n_jobs)
        input_fname = output_fname = op.join(utils.get_resources_fol(), 'subcorticals.txt')
        subcortical_rois, subcortical_rois_header = fu.get_subcortical_regions(args.excludes, output_fname, input_fname,
              ['Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter'])
    else:
        cortical_rois, subcortical_rois = [], []
        for bipolar in results.keys():
            for elecs in results[bipolar].values():
                for elc in elecs:
                    cortical_rois.extend(elc['cortical_rois'])
                    subcortical_rois.extend(elc['subcortical_rois'])

        cortical_rois = list(np.unique(cortical_rois))
        subcortical_rois = list(np.unique(subcortical_rois))
        subcortical_rois_header = subcortical_rois

    if args.write_compact_subcorticals:
        subcortical_rois = ['{}-lh'.format(sub_roi[len('Left-'):].lower()) if 'Left' in sub_roi else
                            '{}-lh'.format(sub_roi[len('Right-'):].lower()) if 'Right' in sub_roi else
                            sub_roi.lower() for sub_roi in subcortical_rois]
    for bipolar in results.keys():
        electrodes_summation, labels_types = None, None
        for subject, elecs in results[bipolar].items():
            results_fname_csv = get_output_csv_fname(subject, bipolar, args)
            header = ['electrode'] + cortical_rois + subcortical_rois_header + ['approx', 'elc_length']
            file_values = write_values(elecs, elecs_types[subject], results_fname_csv, header,
                [cortical_rois, subcortical_rois],
                ['cortical_rois','subcortical_rois'], ['cortical_probs', 'subcortical_probs'], args, bipolar)
            if labels_types is None:
                labels_types = np.array([0] * len(cortical_rois) + [1] * len(subcortical_rois))
            electrodes_summation = np.mean(file_values, 0) if electrodes_summation is None else \
                electrodes_summation + np.mean(file_values, 0)
            most_probable_rois_and_electrodes(subject, elecs, results_fname_csv, elecs_types, bipolar, args)
            if args.write_only_cortical:
                write_values(elecs, elecs_types[subject], results_fname_csv.replace('electrodes', 'cortical_electrodes'),
                             ['electrode'] + cortical_rois, [cortical_rois],['cortical_rois'], ['cortical_probs'],
                             args, bipolar)

            if args.write_only_subcortical:
                write_values(elecs, elecs_types[subject], results_fname_csv.replace('electrodes', 'subcortical_electrodes'),
                    ['electrode']  + subcortical_rois_header, [subcortical_rois],
                    ['subcortical_rois'], ['subcortical_probs'], args, bipolar)

            mmvt_csv_fname = op.join(args.mmvt_dir, subject, 'electrodes', op.basename(results_fname_csv))
            if args.overwrite_mmvt and op.isfile(mmvt_csv_fname):
                os.remove(mmvt_csv_fname)
            if args.overwrite_mmvt or not op.isfile(mmvt_csv_fname):
                shutil.copy(results_fname_csv, mmvt_csv_fname)

        save_no_zeros_labels(results, bipolar, electrodes_summation, header, labels_types)


def save_no_zeros_labels(results, bipolar, electrodes_summation, header, labels_types):
    non_zero_indices = np.where(electrodes_summation > 0)[0]
    header = np.array(header)[1:-2]
    non_zero_header = header[non_zero_indices]
    non_zeros_labels_types = labels_types[non_zero_indices]
    non_zeros_rois = non_zero_header[non_zeros_labels_types == 0]
    non_zeros_subs = non_zero_header[non_zeros_labels_types == 1]
    subjects = sorted([str(name) for name in results[bipolar].keys()])
    np.savetxt(op.join(get_electrodes_dir(), '{}_{}no_zero_rois.csv'.format(
        '_'.join(subjects), 'bipolar_' if bipolar else '')), non_zeros_rois, fmt='%s')
    np.savetxt(op.join(get_electrodes_dir(), '{}_{}_no_zero_subs.csv'.format(
        '_'.join(subjects), 'bipolar_' if bipolar else '')),non_zeros_subs, fmt='%s')


def write_values(elecs, elecs_types, results_fname, header, rois_arr, rois_names, probs_names, args, bipolar=False):
    file_data = []
    try:
        with open(results_fname, 'w') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(header)
            print('Writing {} with header length of {}'.format(results_fname, len(header)))
            for elc, elc_type in zip(elecs, elecs_types):
                elc_name = elc['name']
                if args.write_compact_bipolar:
                    elc_name = get_compact_bipolar_elc_name(elc_name, bipolar, elc_type)
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
                file_data.append(values[1:-2])
    except:
        logging.error('write_values to {}: {}'.format(results_fname, traceback.format_exc()))
        print(traceback.format_exc())

    return np.array(file_data).astype(np.float)


def most_probable_rois_and_electrodes(subject, elecs, results_fname_csv, elecs_types, bipolar, args):
    mp_rois = get_most_probable_rois(elecs)  # Most probable ROIs for each electrode
    mp_roi_elec = OrderedDict()
    for mp_elec, mp_roi, mp_prob in mp_rois:
        if mp_roi not in mp_roi_elec:
            mp_roi_elec[mp_roi] = []
        if mp_elec not in [e['name'] for e in elecs]:
            print("sdf")
        mp_roi_elec[mp_roi].append(mp_elec)
    ms_rois_fname_csv = '{}_mprois{}'.format(*op.splitext(results_fname_csv))
    ms_rois_elecs_fname_csv = '{}_rois_electrodes{}'.format(*op.splitext(results_fname_csv))
    write_rois_electordes(mp_roi_elec, ms_rois_elecs_fname_csv, elecs, elecs_types[subject], bipolar, args)
    write_most_probable_rois(mp_rois, ms_rois_fname_csv, elecs_types[subject], bipolar, args)


def write_most_probable_rois(mp_rois, results_fname, elecs_types, bipolar, args):
    try:
        with open(results_fname, 'w') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(['electrode', 'most probable roi', 'probability'])
            for (elc_name, ms_roi, prob), elc_type in zip(mp_rois, elecs_types):
                if args.write_compact_bipolar:
                    elc_name = get_compact_bipolar_elc_name(elc_name, bipolar, elc_type)
                writer.writerow([elc_name, ms_roi, prob])
    except:
        logging.error('write_values to {}: {}'.format(results_fname, traceback.format_exc()))
        print(traceback.format_exc())


def write_rois_electordes(rois_elecs, results_fname, elecs, elecs_types, bipolar, args):
    try:
        elecs_types_dic = {elec['name']: elec_type for elec, elec_type in zip(elecs, elecs_types)}
        with open(results_fname, 'w') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(['roi', 'electrodes'])
            for roi_name, electrodes in rois_elecs.items():
                if args.write_compact_bipolar:
                    electrodes = [get_compact_bipolar_elc_name(elc_name, bipolar, elecs_types_dic[elc_name]) for
                                  elc_name in electrodes]
                writer.writerow([roi_name, *electrodes])
    except:
        print(traceback.format_exc())
        logging.error('write_values to {}: {}'.format(results_fname, traceback.format_exc()))



def get_compact_bipolar_elc_name(elc_name, bipolar, elc_type):
    if bipolar and '-' in elc_name:
        elc_group, elc_num1, elc_num2 = elec_group_number(elc_name, True)
        elc_name = '{}.{}{}'.format(elc_group, elc_num1, elc_num2)
    if elc_type == GRID:
        elc_group, elc_num = elec_group_number(elc_name, False)
        elc_name = '{}.{}'.format(elc_group, elc_num)
    return elc_name



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
                if elc_num2 > elc_num1:
                    next_elc = '{}{}-{}{}'.format(elc_group, elc_num2 + 1, elc_group, elc_num1 + 1)
                else:
                    next_elc = '{}{}-{}{}'.format(elc_group, elc_num1 + 1, elc_group, elc_num2 + 1)
            else:
                elc_group, elc_num = elec_group_number(elc_name)
                next_elc = '{}{}'.format(elc_group, elc_num+1)
            ori = 1
            if next_elc not in elecs_names:
                if bipolar:
                    if elc_num2 > elc_num1:
                        next_elc = '{}{}-{}{}'.format(elc_group, elc_num2 - 1, elc_group, elc_num1 - 1)
                    else:
                        next_elc = '{}{}-{}{}'.format(elc_group, elc_num1 - 1, elc_group, elc_num2 - 1)
                else:
                    next_elc = '{}{}'.format(elc_group, elc_num - 1)
                ori = -1
            if next_elc not in elecs_names:
                print("{} doesn't seem to be depth, changing the type to grid".format(elc_name))
                elecs_types[index] = GRID
            elif next_elc == elc_name:
                raise Exception('next_elc ({}) == elc_name ({}) !!!'.format(elc_name, next_elc))
            else:
                next_elc_index = np.where(elecs_names == next_elc)[0][0]
                next_elc_pos = elecs_pos[next_elc_index]
                dist = np.linalg.norm(next_elc_pos-elc_pos)
                elcs_oris[index] = ori * (next_elc_pos - elc_pos) / dist # norm(elc_ori)=1mm
        # print(elc_name, elc_pos, next_elc, next_elc_pos, elc_line(1))
    return elcs_oris


def elec_group_number(elec_name, bipolar=False):
    if bipolar:
        elec_name2, elec_name1 = utils.split_bipolar_name(elec_name)
        group, num1 = elec_group_number(elec_name1, False)
        _, num2 = elec_group_number(elec_name2, False)
        return group, num1, num2
    else:
        # ind = np.where([int(s.isdigit()) for s in elec_name])[-1][0]
        # num = int(elec_name[ind:])
        elec_name = elec_name.strip()
        num = int(utils.find_elec_num(elec_name))
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


def check_for_annot_file(subject, args):
    annot_file = op.join(args.subjects_dir, subject, 'label', '{}.{}.annot'.format('{hemi}', args.atlas))
    if '{}.annot'.format(args.atlas) in EXISTING_FREESURFER_ANNOTATIONS and \
            (not op.isfile(annot_file.format(hemi='rh')) or not op.isfile(annot_file.format(hemi='lh'))):
        args.overwrite_annotation = False
        args.solve_labels_collisions = False
        if args.freesurfer_home == '':
            raise Exception('There are no annotation file for {}, please source freesurfer and run again'.format(args.atlas))
        fu.create_freesurfer_annotation_file(subject, args.atlas, args.subjects_dir, args.freesurfer_home)
    if args.read_labels_from_annotation and op.isfile(annot_file.format(hemi='rh')) and \
            op.isfile(annot_file.format(hemi='lh')):
        # Nothing to do, read the labels from an existing annotation file
        return
    if args.overwrite_labels or args.overwrite_annotation or not op.isfile(annot_file.format(hemi='rh')) or not \
            op.isfile(annot_file.format(hemi='lh')):
        morph_labels_from_fsaverage(subject, args.subjects_dir, args.atlas, n_jobs=args.n_jobs,
                                    fsaverage=args.fsaverage, overwrite=args.overwrite_labels)
        if args.solve_labels_collisions:
            backup_labels_fol = '{}_before_solve_collision'.format(args.atlas, args.fsaverage)
            lu.solve_labels_collision(subject, args.subjects_dir, args.atlas, backup_labels_fol, args.n_jobs)
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
            sub_label = fs_label.morph(fsaverage, subject, grade=None, n_jobs=1, subjects_dir=subjects_dir)
            sub_label.save(local_label_name)


# def prepare_local_subjects_folder(neccesary_files, subject, remote_subject_dir, local_subjects_dir, print_traceback=False):
#     local_subject_dir = op.join(local_subjects_dir, subject)
#     for fol, files in neccesary_files.items():
#         if not op.isdir(op.join(local_subject_dir, fol)):
#             os.makedirs(op.join(local_subject_dir, fol))
#         for file_name in files:
#             try:
#                 if not op.isfile(op.join(local_subject_dir, fol, file_name)):
#                     shutil.copyfile(op.join(remote_subject_dir, fol, file_name),
#                                 op.join(local_subject_dir, fol, file_name))
#             except:
#                 logging.error('{}: {}'.format(subject, traceback.format_exc()))
#                 if print_traceback:
#                     print(traceback.format_exc())
#     all_files_exists = True
#     for fol, files in neccesary_files.items():
#         for file_name in files:
#             if not op.isfile(op.join(local_subject_dir, fol, file_name)):
#                 logging.error("The file {} doesn't exist in the local subjects folder!!!".format(file_name))
#                 all_files_exists = False
#     copy_electrodes_ras_file(subject, local_subject_dir, remote_subject_dir)
#     if not all_files_exists:
#         raise Exception('Not all files exist in the local subject folder!!!')
#         logging.error('{}: {}'.format(subject, 'Not all files exist in the local subject folder!!!'))


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


def check_for_necessary_files(subject, args, sftp_password=''):
    if '{subject}' in args.remote_subject_dir:
        remote_subject_dir = build_remote_subject_dir(subject, args.remote_subject_dir, args.remote_subject_dir_func)
    else:
        remote_subject_dir = args.remote_subject_dir
    all_files_exist = utils.prepare_local_subjects_folder(args.neccesary_files, subject, remote_subject_dir,
        args.subjects_dir, args, sftp_password, print_traceback=True)
    if not all_files_exist:
        print('Not all files exist in the local subject folder!!!')
        return False
    else:
        return True
    # prepare_local_subjects_folder(neccesary_files, subject, remote_subject_dir, subjects_dir,
    #     print_traceback=True)


def check_for_electrodes_coordinates_file(subject, subjects_dir, electrodes_dir):
    elec_file = op.join(electrodes_dir, '{}.csv'.format(subject))
    # if not op.isfile(elec_file) or op.getsize(elec_file) == 0:
    copy_electrodes_file(subjects_dir, subject, elec_file)


def copy_electrodes_file(subjects_dir, subject, elec_file):
    elecs_fol = op.join(subjects_dir, subject, 'electrodes')
    subject_elec_fname = op.join(elecs_fol, '{}_RAS.csv'.format(subject))
    if not op.isfile(subject_elec_fname) or op.getsize(subject_elec_fname) == 0:
        rename_and_convert_electrodes_file(subject, elecs_fol)
    if op.isfile(subject_elec_fname):
        if op.isfile(elec_file):
            os.remove(elec_file)
        shutil.copyfile(subject_elec_fname, elec_file)
    else:
        raise Exception('{}: Electrodes file does not exist! {}'.format(subject, subject_elec_fname))


def rename_and_convert_electrodes_file(subject, electrodes_fol):
    subject_elec_fname_no_ras_pattern = op.join(electrodes_fol, '{subject}.{postfix}')
    subject_elec_fname_pattern = op.join(electrodes_fol, '{subject}_RAS.{postfix}')
    subject_elec_fname_csv = subject_elec_fname_pattern.format(subject=subject, postfix='csv')
    subject_elec_fname_xlsx = subject_elec_fname_pattern.format(subject=subject, postfix='xlsx')

    subject_upper = subject[:2].upper() + subject[2:]
    files = [patt.format(subject=sub, postfix=post) for patt, sub, post in product(
        [subject_elec_fname_no_ras_pattern, subject_elec_fname_pattern], [subject, subject_upper], ['xls', 'xlsx'])]
    utils.rename_files(files, subject_elec_fname_xlsx)
    utils.rename_files([subject_elec_fname_pattern.format(subject=subject_upper, postfix='csv'),
                        subject_elec_fname_pattern.format(subject=subject, postfix='csv')],
                       subject_elec_fname_csv)
    if op.isfile(subject_elec_fname_xlsx) and \
                    (not op.isfile(subject_elec_fname_csv) or op.getsize(subject_elec_fname_csv) == 0):
        utils.csv_from_excel(subject_elec_fname_xlsx, subject_elec_fname_csv, subject)
    return op.isfile(subject_elec_fname_csv)


def check_if_files_exist(args):
    return np.all([utils.check_if_all_neccesary_files_exist(
        subject, args.neccesary_files, op.join(args.subjects_dir, subject)) for subject in args.subject])


def get_output_csv_fname(subject, bipolar, args):
    return op.join(get_electrodes_dir(), args.output_template.format(
        subject=subject, atlas=args.atlas,
        error_radius=args.error_radius, elec_length=args.elc_length,
        bipolar='_bipolar' if bipolar else '',
        stretch='_not_stretch' if not args.strech_to_dist and bipolar else '',
        postfix=args.output_postfix)) + '.csv'


def snap(subject, elecs_names, elecs_pos, elecs_types, subjects_dir):
    grids_inds = np.where(elecs_types == GRID)[0]
    if len(grids_inds) == 0:
        return
    groups_inds = defaultdict(list)
    for grid_ind in grids_inds:
        elec_group, elec_num = elec_group_number(elecs_names[grid_ind], False)
        groups_inds[elec_group].append(grid_ind)
    for elec_group, group_inds in groups_inds.items():
        group_inds = np.array(group_inds)
        snap_electrodes_to_surface(subject, elecs_pos[group_inds], elec_group, subjects_dir)


def read_snap_electrodes(subject, elecs_names, elecs_pos, elecs_types_names, subjects_dir):
    snap_grids = glob.glob(op.join(subjects_dir, subject, 'electrodes', '*_snap_electrodes.npz'))
    if elecs_types_names is None:
        elecs_types_names = [''] * len(elecs_names)
    for snap_grid_fname in snap_grids:
        grid_name = utils.namebase(snap_grid_fname).split('_')[0]
        grid = np.load(snap_grid_fname)
        grid_pos = grid['snapped_electrodes'] #''snapped_electrodes_pial']
        elcs_inds = [elc_ind for elc_ind, elc_name in enumerate(elecs_names) if elc_name.startswith(grid_name) and \
                     utils.is_int(utils.find_elec_num(elc_name))]
        if len(elcs_inds) != grid_pos.shape[0]:
            raise Exception('read_snap_electrodes: Wonrg number of snapped electrodes indices!')
        elecs_pos[elcs_inds] = grid_pos
    with open(op.join(subjects_dir, subject, 'electrodes', '{}_snap_RAS.csv'.format(subject)), 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for elecs_name, elec_pos, elec_type in zip(elecs_names, elecs_pos, elecs_types_names):
            csv_writer.writerow([elecs_name.replace('elec_unsorted', ''), *elec_pos, elec_type])
    return elecs_pos


def run_for_all_subjects(args):
    ok_subjects, bad_subjects = [], []
    results = defaultdict(dict)
    all_elecs_types = {}
    logging.basicConfig(filename='log.log',level=logging.DEBUG)
    if not args.only_check_files:
        for subject in args.subject:
            utils.make_dir(op.join(args.subjects_dir, subject, 'electrodes'))
            rename_and_convert_electrodes_file(subject, op.join(args.subjects_dir, subject, 'electrodes'))
            if not op.isfile(op.join(args.subjects_dir, subject, 'electrodes', '{}_RAS.csv'.format(subject))) and op.isfile(
                    op.join(get_electrodes_dir(), '{}.csv'.format(subject))):
                shutil.copyfile(op.join(get_electrodes_dir(), '{}.csv'.format(subject)),
                                op.join(args.subjects_dir, subject, 'electrodes', '{}_RAS.csv'.format(subject)))
    all_files_exist = check_if_files_exist(args)
    if args.sftp and not all_files_exist:
        sftp_password = getpass.getpass('Please enter the sftp password for {}: '.format(args.sftp_username))
    else:
        sftp_password = ''
    write_results = args.function != 'get_electrodes_types'
    if args.function == 'get_electrodes_types':
        args.bipolar = [False]
    for subject, bipolar in product(args.subject, args.bipolar):
        os.environ['SUBJECT'] = subject
        results_fname_csv = get_output_csv_fname(subject, bipolar, args)
        results_fname_pkl = results_fname_csv.replace('csv', 'pkl')
        try:
            if args.function == 'get_electrodes_types':
                electrodes_types_set = get_electrodes_types_set(subject, args)
                print('{}: {}'.format(subject, electrodes_types_set))
                continue
            print('****************** {} ******************'.format(subject))
            logging.info('****************** {} bipolar {}, {}******************'.format(subject, bipolar, utils.now()))
            if op.isfile(results_fname_pkl) and not args.overwrite:
                print('Loading data file from {}'.format(results_fname_pkl))
                elecs = utils.load(results_fname_pkl)
                _, _, _, elecs_types, _ = get_electrodes(subject, bipolar, args)
                all_elecs_types[subject] = elecs_types
            else:
                all_files_exist = check_for_necessary_files(subject, args, sftp_password)
                if not all_files_exist:
                    bad_subjects.append(subject)
                    continue
                check_for_annot_file(subject, args)
                if args.only_check_files:
                    continue
                elecs_names, elecs_pos, elecs_dists, elecs_types, _ = get_electrodes(
                    subject, bipolar, args)
                if 'snap_grid_to_pial' in args.function:
                    snap(subject, elecs_names, elecs_pos, elecs_types, args.subjects_dir)
                    continue
                if 'read_snap_electrodes' in args.function or 'snap_grid_to_pial' in args.function:
                    _elecs_names, _elecs_pos, _, _elecs_types, _ = get_electrodes(
                        subject, False, args)
                    read_snap_electrodes(subject, _elecs_names, _elecs_pos, _, args.subjects_dir)
                    continue
                all_elecs_types[subject] = elecs_types
                elcs_ori = get_electrodes_orientation(
                    elecs_names, elecs_pos, bipolar, elecs_types, elecs_oris_fname=args.pos_fname)
                labels = read_labels_vertices(args.subjects_dir, subject, args.atlas, args.read_labels_from_annotation,
                    args.overwrite_labels_pkl, args.n_jobs)
                elecs = identify_roi_from_atlas(
                    args.atlas, labels, elecs_names, elecs_pos, elcs_ori, args.error_radius, args.elc_length,
                    elecs_dists, elecs_types, args.strech_to_dist, args.enlarge_if_no_hit,
                    bipolar, args.subjects_dir, subject, args.excludes, args.specific_elec, n_jobs=args.n_jobs)
                if args.specific_elec != '':
                    continue
                utils.save(elecs, results_fname_pkl)
            # if au.should_run('add_colors_to_probs', args):
            #     add_colors_to_probs(subject, args.atlas, results_fname_pkl)
            results[bipolar][subject] = elecs
            ok_subjects.append(subject)
            if op.isdir(args.mmvt_dir):
                utils.make_dir(op.join(args.mmvt_dir, subject, 'electrodes'))
                mmvt_pkl_fname = op.join(args.mmvt_dir, subject, 'electrodes', op.basename(results_fname_pkl))
                if args.overwrite_mmvt and op.isfile(mmvt_pkl_fname):
                    os.remove(mmvt_pkl_fname)
                if args.overwrite_mmvt or not op.isfile(mmvt_pkl_fname):
                    shutil.copy(results_fname_pkl, mmvt_pkl_fname)
        except:
            bad_subjects.append(subject)
            logging.error('{}: {}'.format(subject, traceback.format_exc()))
            print(traceback.format_exc())

    # Write the results for all the subjects at once, to have a common labeling
    if write_results and (not op.isfile(results_fname_csv) or args.overwrite_csv):
        write_results_to_csv(results, all_elecs_types, args)

    if ok_subjects:
        print('ok subjects:')
        print(ok_subjects)
    if bad_subjects:
        print('bad_subjects:')
        print(bad_subjects)
        logging.error('bad_subjects:')
        logging.error(bad_subjects)
    print('finish!')


def add_colors_to_probs(subject, atlas, results_fname):
    # results_fname = op.join(get_electrodes_dir(), '{}_{}_electrodes{}.pkl'.format(
    #     subject, atlas, output_files_postfix))
    try:
        if op.isfile(results_fname):
            elecs = utils.load(results_fname)
            for elc in elecs:
                elc['subcortical_colors'] = cu.arr_to_colors(elc['subcortical_probs'], colors_map='YlOrRd')
                elc['cortical_colors'] = cu.arr_to_colors(elc['cortical_probs'], colors_map='YlOrRd')
            utils.save(elecs, results_fname)
        else:
            print("!!! Can't find the probabilities file !!!")
    except:
        print("Can't calc probs colors!")


def remove_white_matter_and_normalize(elc):
    no_white_inds = [ind for ind, label in enumerate(elc['subcortical_rois']) if label not in
                     ['Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter']]
    subcortical_probs_norm = elc['subcortical_probs'][no_white_inds]
    subcortical_probs_norm *= 1/sum(subcortical_probs_norm)
    subcortical_rois_norm = elc['subcortical_rois'][no_white_inds]
    return subcortical_probs_norm, subcortical_rois_norm


def get_most_probable_rois(elecs_probs, laterally=True, p_threshold=0):
    elecs_probs = [elec for elec in elecs_probs if elec['name'] == 'LOF2-LOF1']
    probable_rois = [(elec['name'], *get_most_probable_roi([*elec['cortical_probs'], *elec['subcortical_probs']],
        [*elec['cortical_rois'], *elec['subcortical_rois']], p_threshold)) for elec in elecs_probs]
    if laterally:
        return probable_rois
    else:
        return utils.get_hemi_indifferent_rois(probable_rois)


def get_most_probable_roi(probs, rois, p_threshold):
    probs_rois = sorted([(p, r) for p, r in zip(probs, rois)])[::-1]
    if len(probs_rois) == 0:
        roi = ''
        prob = 0.0
    elif len(probs_rois) == 1 and 'white' in probs_rois[0][1].lower():
        roi = probs_rois[0][1]
        prob = 1.0
    elif 'white' in probs_rois[0][1].lower():
        if 'white' in probs_rois[1][1].lower():
            if len(probs_rois) > 2 and probs_rois[2][0] > p_threshold:
                roi = probs_rois[2][1]
                prob = probs_rois[2][0]
            else:
                roi = probs_rois[0][1]
                prob = probs_rois[0][0]
        else:
            if probs_rois[1][0] > p_threshold:
                roi = probs_rois[1][1]
                prob = probs_rois[1][0]
            else:
                roi = probs_rois[0][1]
                prob = probs_rois[0][0]
    else:
        roi = probs_rois[0][1]
        prob = probs_rois[0][0]
    return roi, prob



def build_remote_subject_dir(subject, remote_subject_dir, remote_subject_dir_func):
    if isinstance(remote_subject_dir, dict):
        if 'func' in remote_subject_dir:
            template_val = remote_subject_dir['func'](subject)
            remote_subject_dir = remote_subject_dir['template'].format(subject=template_val)
        else:
            remote_subject_dir = remote_subject_dir['template'].format(subject=subject)
    else:
        if remote_subject_dir_func != '':
            if remote_subject_dir_func == 'upper':
                subject = subject.upper()
        remote_subject_dir = remote_subject_dir.format(subject=subject)

    return remote_subject_dir


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas40')
    parser.add_argument('-f', '--function', help='function name', required=False, default='all')
    parser.add_argument('-b', '--bipolar', help='bipolar electrodes', required=False, default='1,0', type=au.bool_arr_type)
    parser.add_argument('--error_radius', help='error radius', required=False, default=3)
    parser.add_argument('--elc_length', help='elc length', required=False, default=4)
    parser.add_argument('--snap', help='read electrodes snap RAS coordinates file', required=False, default=0, type=au.is_true)
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    parser.add_argument('--template_brain', help='template brain', required=False, default='fsaverage5c')
    parser.add_argument('--strech_to_dist', help='strech_to_dist', required=False, default=1, type=au.is_true)
    parser.add_argument('--enlarge_if_no_hit', help='enlarge_if_no_hit', required=False, default=1, type=au.is_true)
    parser.add_argument('--output_template', help='output template', required=False,
        default='{subject}_{atlas}_electrodes_cigar_r_{error_radius}_l_{elec_length}{bipolar}{stretch}{postfix}')
    parser.add_argument('--only_check_files', help='only_check_files', required=False, default=0, type=au.is_true)
    parser.add_argument('--fsaverage', help='fsaverage template brain', required=False, default='fsaverage')
    parser.add_argument('--overwrite', help='overwrite', required=False, default=1, type=au.is_true)
    parser.add_argument('--overwrite_annotation', help='overwrite_annotation', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_labels', help='overwrite_labels', required=False, default=0, type=au.is_true)
    parser.add_argument('--write_only_cortical', help='write_only_cortical', required=False, default=0, type=au.is_true)
    parser.add_argument('--write_only_subcortical', help='write_only_subcortical', required=False, default=0, type=au.is_true)
    parser.add_argument('--write_all_labels', help='Write all the labels', required=False, default=1, type=au.is_true)
    parser.add_argument('--overwrite_labels_pkl', help='overwrite_labels_pkl', required=False, default=1, type=au.is_true)
    parser.add_argument('--overwrite_csv', help='overwrite_csv', required=False, default=1, type=au.is_true)
    parser.add_argument('--overwrite_mmvt', help='overwrite_mmvt', required=False, default=1, type=au.is_true)
    parser.add_argument('--read_labels_from_annotation', help='read_labels_from_annotation', required=False, default=1, type=au.is_true)
    parser.add_argument('--solve_labels_collisions', help='solve_labels_collisions', required=False, default=0, type=au.is_true)
    parser.add_argument('--remote_subject_dir', help='remote_subject_dir', required=False, default='')
    parser.add_argument('--remote_subject_dir_func', help='remote_subject_dir_func', required=False, default='')
    parser.add_argument('--pos_fname', help='electrodes positions fname', required=False, default='')
    parser.add_argument('--elecs_dir', help='electrodes positions folder', required=False, default='')
    parser.add_argument('--output_postfix', help='output_postfix', required=False, default='')
    parser.add_argument('--write_compact_bipolar', help='write x.23 instead x3-x2', required=False, default=0, type=au.is_true)
    parser.add_argument('--write_compact_subcorticals', help='change subcorticals names as xxx-rh/lh', required=False, default=0, type=au.is_true)
    parser.add_argument('--csv_delimiter', help='ras csv delimiter', required=False, default=',')
    parser.add_argument('--excludes', help='excluded labels', required=False, type=au.str_arr_type,
        default='Unknown,unknown,Cerebral-Cortex,corpuscallosum,WM-hypointensities,Ventricle,Inf-Lat-Vent,choroid-plexus,CC,CSF,VentralDC')
    parser.add_argument('--exclude_white', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--specific_elec', help='run on only one electrodes', required=False, default='')
    parser.add_argument('--sftp', help='copy subjects files over sftp', required=False, default=0, type=au.is_true)
    parser.add_argument('--sftp_username', help='sftp username', required=False, default='')
    parser.add_argument('--sftp_domain', help='sftp domain', required=False, default='')
    parser.add_argument('--electrodes_type', help='', required=False, default=None)


    args = utils.Bag(au.parse_parser(parser, argv))
    args.n_jobs = utils.get_n_jobs(args.n_jobs)
    args.subjects_dir = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
    args.freesurfer_home = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
    if args.freesurfer_home == '':
        args.freesurfer_home = os.environ.get('FREESURFER_HOME', '')
    else:
        os.environ['FREESURFER_HOME'] = args.freesurfer_home
    args.mmvt_dir = utils.get_link_dir(LINKS_DIR, 'mmvt')
    os.environ['SUBJECTS_DIR'] = args.subjects_dir
    args.neccesary_files = {
        'mri': ['aseg.mgz'],
        'surf': ['rh.pial', 'lh.pial', 'rh.sphere.reg', 'lh.sphere.reg', 'lh.white', 'rh.white',
                 'lh.smoothwm', 'rh.smoothwm']}
    fu.extend_subcorticals_excludes(args.excludes, include_white=args.exclude_white)
        # 'electrodes': ['{subject}_RAS.csv']}
    # print(args)
    return args


if __name__ == '__main__':
    # remote_subject_dir = {'template':'/home/ieluuser/links/subjects/{subject}_SurferOutput', 'func': lambda x: x.upper()}
    # if subject == 'all':
    #     subjects = set(get_all_subjects(subjects_dir, 'mg', '_')) - set(['mg63', 'mg94']) # get_subjects()
    # else:
    #     subjects = [subject]
    args = get_args()
    run_for_all_subjects(args)
