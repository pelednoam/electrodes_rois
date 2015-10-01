import numpy as np
import os
import nibabel as nib
from scipy.spatial.distance import cdist

def identify_roi_from_atlas(pos, approx=4, atlas=None, subjects_dir=None,
    subject=None):

    for elec_pos in pos:
        rois = identify_roi_from_atlas_per_electrode(elec_pos, approx, atlas, subjects_dir, subject)
        print rois
        break


def identify_roi_from_atlas_per_electrode(pos, approx=4, atlas=None, subjects_dir=None,
    subject=None):
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
    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    # conceptually, we should grow the closest vertex around this electrode
    # probably following snapping but the code for this function is not
    # altered either way

    # load the surfaces and annotation
    # uses the pial surface, this change is pushed to MNE python

    lh_pia, _ = nib.freesurfer.read_geometry(
        os.path.join(subjects_dir, subject, 'surf', 'lh.pial'))

    rh_pia, _ = nib.freesurfer.read_geometry(
        os.path.join(subjects_dir, subject, 'surf', 'rh.pial'))

    pia = np.vstack((lh_pia, rh_pia))

    # find closest vertex
    closest_vert = np.argmin(cdist(pia, [pos]))

    # grow the area of surface surrounding the vertex
    import mne

    # we force the label to only contact one hemisphere even if it is
    # beyond the extent of the medial surface
    hemi_str = 'lh' if closest_vert<len(lh_pia) else 'rh'
    hemi_code = 0 if hemi_str=='lh' else 1

    if hemi_str == 'rh':
        closest_vert -= len(lh_pia)

    radius_label, = mne.grow_labels(subject, closest_vert, approx, hemi_code,
        subjects_dir=subjects_dir, surface='pial')

    from mne.surface import read_surface
    surf_fname = os.path.join(subjects_dir, subject, 'surf', hemi_str + '.pial')
    verts, _ = read_surface(surf_fname)
    closest_vert_pos = verts[closest_vert]
    import matplotlib.pyplot as plt
    plt.hist(cdist([closest_vert_pos], radius_label.pos).ravel())
    plt.show()

    parcels = mne.read_labels_from_annot(subject, parc=atlas, hemi=hemi_str,
        subjects_dir=subjects_dir, surf_name='pial')

    excludes=['white', 'WM', 'Unknown', 'White', 'unknown', 'Cerebral-Cortex']
    regions = []
    for parcel in parcels:
        if len(np.intersect1d(parcel.vertices, radius_label.vertices))>0:
            #force convert from unicode
            regions.append(str(parcel.name))

    subcortical_regions = identify_roi_from_aparc(pos, approx=approx,
        subjects_dir=subjects_dir, subject=subject, subcortical_only=True,
        excludes=excludes)

    regions.extend(subcortical_regions)
    regions = exclude_regions(regions, excludes)
    return regions


def identify_roi_from_aparc( pos, approx=4, subjects_dir=None, subject=None,
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
    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    def find_neighboring_regions(pos, aseg, region, approx, excludes):
        asegd = aseg.get_data()
        asegh = aseg.get_header()

        spot_sz = int(np.around(approx * 2 + 1))
        x, y, z = np.meshgrid(range(spot_sz), range(spot_sz), range(spot_sz))

        # approx is in units of millimeters as long as we use the RAS space
        # segmentation
        neighb = np.vstack((np.reshape(x, (1, spot_sz ** 3)),
            np.reshape(y, (1, spot_sz ** 3)),
            np.reshape(z, (1, spot_sz ** 3)))).T - approx

        regions = []
        dists = cdist([pos], pos+neighb)[0]
        neighb = neighb[np.where(dists<approx)]
        for p in xrange(neighb.shape[0]):
            cx, cy, cz = (pos[0]+neighb[p,0], pos[1]+neighb[p,1],
                pos[2]+neighb[p,2])
            d_type = asegd[cx, cy, cz]
            label_index = region['index'].index(d_type)
            regions.append(region['label'][label_index])

        regions = exclude_regions(regions, excludes)
        return np.unique(regions).tolist()

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

        idx = []
        label = []
        rgba = np.empty((0, 4))

        with open(fs_lut, 'r') as f:
            for l in f:
                if len(l) <= 1 or l[0] == '#' or l[0] == '\r':
                    continue
                (t0, t1, t2, t3, t4, t5) = [t(s) for t, s in
                        zip((int, str, int, int, int, int), l.split())]

                idx.append(t0)
                label.append(t1)
                rgba = np.vstack((rgba, np.array([t2, t3, t4, t5])))

        return idx, label, rgba

    # get the segmentation file
    asegf = os.path.join( subjects_dir, subject, 'mri', 'aseg.mgz')# 'aparc+aseg.mgz' )
    aseg = nib.load(asegf)

    # get the aseg LUT file
    lut = import_freesurfer_lut()
    lut = {'index':lut[0], 'label':lut[1], 'RGBA':lut[2]}

    if subcortical_only:
        excludes.append('ctx')


    RAS_AFF = np.array([[-1, 0, 0, 128],
        [0, 0, -1, 128],
        [0, 1, 0, 128],
        [0, 0, 0, 1]])
    ras_pos = np.around(np.dot(RAS_AFF, np.append(pos, 1)))[:3]

    return find_neighboring_regions(ras_pos, aseg, lut, approx, excludes)


def exclude_regions(regions, excludes):
    from re import compile

    if (excludes):
        excluded = compile('|'.join(excludes))
        regions = [x for x in regions if not excluded.search(x)]
    return regions

def get_electrodes_positions(subject, delimiter=','):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    elec_dir = os.path.join(os.path.split(curr_dir)[0], 'electrodes')
    elec_file = os.path.join(elec_dir, '{}.csv'.format(subject))
    data = np.genfromtxt(elec_file, dtype=str, delimiter=delimiter)
    pos = data[1:, 1:].astype(float)
    names = data[1:, 0]
    return names, pos


if __name__ == '__main__':
    subject = 'mg78'
    subjects_dir = '/home/noam/subjects' # os.environ.get('SUBJECTS_DIR')
    parcellation = 'aparc250'
    error_radius = 4
    names, pos = get_electrodes_positions(subject)

    roi_hits = identify_roi_from_atlas(pos,
        atlas = parcellation,
        approx = error_radius,
        subjects_dir = subjects_dir,
        subject = subject)

