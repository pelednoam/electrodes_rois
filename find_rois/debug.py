import os.path as op
from find_rois import main
from find_rois import utils

LINKS_DIR = utils.get_links_dir()


def test(subject, elc_name, approx, elc_length):
    mmvt_dir = utils.get_link_dir(LINKS_DIR, 'mmvt')
    debug_fname = op.join(mmvt_dir, subject, 'electrodes', '{}_{}_{}.pkl'.format(
        elc_name, approx, elc_length))
    (elc_name, pos, elc_length, labels, hemi_str, verts, elc_line, bins, approx, _region_are_excluded,
     lut, aseg_data, approx, nei_dimensions, excludes, hit_only_cortex) = utils.load(debug_fname)
    regions, regions_hits = main.calc_hits(labels, hemi_str, verts, elc_line, bins, approx, _region_are_excluded)
    subcortical_regions, subcortical_hits = main.identify_roi_from_aparc(
        pos, elc_line, elc_length, lut, aseg_data, approx=approx, nei_dimensions=nei_dimensions,
        subcortical_only=True, excludes=excludes)
    we_have_a_hit = main.do_we_have_a_hit(regions, subcortical_regions, hit_only_cortex)

if __name__ == '__main__':
    test()