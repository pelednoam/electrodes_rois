import numpy as np
import os
from functools import partial
import re
import nibabel as nib
from collections import Counter
from scipy.spatial.distance import cdist
import mne
from mne.surface import read_surface

def identify_roi_from_atlas(elecs_names, elecs_pos, approx=4, atlas=None, subjects_dir=None,
    subject=None):

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

    elecs = {}
    for elec_pos, elec_name in zip(elecs_pos, elecs_names):
        regions, regions_hits, subcortical_regions, subcortical_hits = \
            identify_roi_from_atlas_per_electrode(elec_pos, pia, len_lh_pia,
                parcels, lut, aseg_header, approx, subjects_dir, subject)
        regions_hits, subcortical_hits = np.array(regions_hits), np.array(subcortical_hits)
        regions_probs = np.hstack((regions_hits, subcortical_hits)) / float(np.sum(regions_hits) + np.sum(subcortical_hits))
        elecs[elec_name] = {'cortical_rois': regions, 'subcortical_rois': subcortical_regions,
            'cortical_probs': regions_probs[:len(regions)],
            'subcortical_probs': regions_probs[len(regions):]}

    return elecs


def identify_roi_from_atlas_per_electrode(pos, pia, len_lh_pia, parcels, lut, aseg_header,
    approx=4, subjects_dir=None, subject=None):
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

    # grow the area of surface surrounding the vertex
    radius_label, = mne.grow_labels(subject, closest_vert, approx, hemi_code,
        subjects_dir=subjects_dir, surface='pial')

    surf_fname = os.path.join(subjects_dir, subject, 'surf', hemi_str + '.pial')
    verts, _ = read_surface(surf_fname)
    closest_vert_pos = verts[closest_vert]
    bins = calc_neighbors(closest_vert_pos, approx, True)
    bins_centers = calc_neighbors(closest_vert_pos, approx, False)

    excludes=['white', 'WM', 'Unknown', 'White', 'unknown', 'Cerebral-Cortex']
    compiled_excludes = re.compile('|'.join(excludes))
    _region_is_excluded = partial(region_is_excluded, compiled_excludes=compiled_excludes)

    regions, regions_hits = [], []
    for parcel in parcels[hemi_str]:
        if _region_is_excluded(str(parcel.name)):
            continue
        intersect_verts = np.intersect1d(parcel.vertices, radius_label.vertices)
        if len(intersect_verts)>0:
            hits = calc_hits_in_neighbors(pos, verts[intersect_verts], bins, bins_centers, approx)
            regions_hits.append(hits)
            #force convert from unicode
            regions.append(str(parcel.name))

    # print('regions')
    # print([(n,h) for n,h in zip(regions, regions_hits)])

    subcortical_regions, subcortical_hits = identify_roi_from_aparc(pos, lut, aseg_header, approx=approx,
        subjects_dir=subjects_dir, subject=subject, subcortical_only=True,
        excludes=excludes)
    # print('subcortical_regions')
    # print([(n,h) for n,h in zip(subcortical_regions, subcortical_hits)])
    # regions = exclude_regions(regions, excludes)
    return regions, regions_hits, subcortical_regions, subcortical_hits


def calc_hits_in_neighbors(pos, points, neighb, bins_centers, approx):
    bins = [np.sort(np.unique(neighb[:, idim])) for idim in range(points.shape[1])]
    hist, binedges = np.histogramdd(points, bins=bins, normed=False)
    dists = cdist(bins_centers, [pos])
    r = int(round(len(bins_centers)**(1./3)))
    dists = dists.reshape((r,r,r))
    inds = np.where((hist>0) & (dists<approx))

    from itertools import product
    # indices = product([range(len(binedges[d])) for d in len(bins_centers)])
    # for i,j,k in indices:
    #     if cdist([bins_centers[]])
    return len(inds[0])


def identify_roi_from_aparc(pos, lut, aseg_header, approx=4, subjects_dir=None, subject=None,
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

    def find_neighboring_regions(pos, aseg_header, lut, approx, excludes):
        compiled_excludes = re.compile('|'.join(excludes))
        _region_is_excluded = partial(region_is_excluded, compiled_excludes=compiled_excludes)
        neighb = calc_neighbors(pos, approx)
        dists = cdist([[0,0,0]], neighb)[0]
        neighb = neighb[np.where(dists<approx)]
        regions = []
        for p in xrange(neighb.shape[0]):
            cx, cy, cz = map(int, neighb[p,:])
            d_type = aseg_header[cx, cy, cz]
            label_index = np.where(lut['index']==d_type)[0][0]
            region = lut['label'][label_index]
            if not _region_is_excluded(region):
                regions.append(region)

        # regions = exclude_regions(regions, excludes)
        cnt = Counter(regions)
        regions, hits = [], []
        for region, count in cnt.iteritems():
            regions.append(region)
            hits.append(count)
        return regions, hits
        # return np.unique(regions).tolist()

    if subcortical_only:
        excludes.append('ctx')


    RAS_AFF = np.array([[-1, 0, 0, 128],
        [0, 0, -1, 128],
        [0, 1, 0, 128],
        [0, 0, 0, 1]])
    ras_pos = np.around(np.dot(RAS_AFF, np.append(pos, 1)))[:3]

    return find_neighboring_regions(ras_pos, aseg_header, lut, approx, excludes)


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

def calc_neighbors(pos, approx, calc_bins=False):
    spot_sz = int(np.around(approx * 2 + (2 if calc_bins else 1)))
    x, y, z = np.meshgrid(range(spot_sz), range(spot_sz), range(spot_sz))

    # approx is in units of millimeters as long as we use the RAS space
    # segmentation
    neighb = np.vstack((np.reshape(x, (1, spot_sz ** 3)),
        np.reshape(y, (1, spot_sz ** 3)),
        np.reshape(z, (1, spot_sz ** 3)))).T - approx

    if calc_bins:
        neighb = neighb.astype(float)
        neighb -= 0.5

    return pos + neighb


# def check_dists(neighb, r):
#     dd = neighb.reshape((r,r,r,3)).transpose([1,0,2,3])
#     d = np.zeros((r,r,r))
#     for x in range(dd.shape[0]):
#         for y in range(dd.shape[1]):
#             for z in range(dd.shape[2]):
#                 d[x,y,z] = cdist([dd[x,y,z]], [[0,0,0]])[0]
#
#     dd2 = neighb.reshape((r,r,r,3))
#     d2 = np.zeros((r,r,r))
#     for x in range(dd2.shape[0]):
#         for y in range(dd2.shape[1]):
#             for z in range(dd2.shape[2]):
#                 d2[x,y,z] = cdist([dd2[x,y,z]], [[0,0,0]])[0]
#


def get_electrodes(subject, elecs_dir='', delimiter=','):
    if elecs_dir=='':
        elecs_dir = get_electrodes_dir()
    elec_file = os.path.join(elecs_dir, '{}.csv'.format(subject))
    data = np.genfromtxt(elec_file, dtype=str, delimiter=delimiter)
    pos = data[1:, 1:].astype(float)
    names = data[1:, 0]
    return names, pos


def get_electrodes_dir():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    elec_dir = os.path.join(os.path.split(curr_dir)[0], 'electrodes')
    return elec_dir


def write_results_to_csv(subject, elecs, parcelation='aparc250', elecs_dir=''):
    if elecs_dir=='':
        elecs_dir = get_electrodes_dir()

    cortical_rois, subcortical_rois = [], []
    for elc in elecs.itervalues():
        cortical_rois.extend(elc['cortical_rois'])
        subcortical_rois.extend(elc['subcortical_rois'])
    cortical_rois = list(np.unique(cortical_rois))
    subcortical_rois = list(np.unique(subcortical_rois))

    header = ['electrode'] + cortical_rois + subcortical_rois
    values = insert_values(elecs, header, [cortical_rois, subcortical_rois],
        ['cortical_rois','subcortical_rois'], ['cortical_probs', 'subcortical_probs'])
    np.savetxt(os.path.join(elecs_dir, '{}_{}_electrodes_all_rois.csv'.format(subject, parcelation)),
               values, fmt="%s")

    header = ['electrode'] + cortical_rois
    values = insert_values(elecs, header, [cortical_rois],['cortical_rois'], ['cortical_probs'])
    np.savetxt(os.path.join(elecs_dir, '{}_{}_electrodes_cortical_rois.csv'.format(subject, parcelation)),
               values, fmt="%s")

    header = ['electrode']  + subcortical_rois
    values = insert_values(elecs, header, [subcortical_rois],
        ['subcortical_rois'], ['subcortical_probs'])
    np.savetxt(os.path.join(elecs_dir, '{}_{}_electrodes_subcortical_rois.csv'.format(subject, parcelation)),
               values, fmt="%s")


def insert_values(elecs, header, rois_arr, rois_names, probs_names):
    cols_num = sum(map(len, rois_arr)) + 1
    values = np.empty((len(elecs) + 1, cols_num), dtype=str)
    values[0, :] = header
    for line, (name, elc) in enumerate(elecs.iteritems()):
        values[line + 1, 0] = name
        for rois, rois_field, prob_field in zip(rois_arr, rois_names, probs_names):
            for col, roi in enumerate(rois):
                if roi in elc[rois_field]:
                    index = elc[rois_field].index(roi)
                    values[line + 1, col+1] = str(elc[prob_field][index])
    return values


if __name__ == '__main__':
    subject = 'mg63'
    subject = 'mg78'
    subjects_dir = '/home/noam/subjects' # os.environ.get('SUBJECTS_DIR')
    parcellation = 'aparc250'
    error_radius = 4

    elecs_names, elecs_pos = get_electrodes(subject)
    elecs = identify_roi_from_atlas(elecs_names, elecs_pos,
        atlas=parcellation, approx=error_radius,
        subjects_dir=subjects_dir, subject=subject)
    write_results_to_csv(subject, elecs)
