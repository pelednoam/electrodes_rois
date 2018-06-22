import argparse

from find_rois import find_rois
from find_rois import utils
from find_rois import args_utils as au

# 'mg72,mg76,mg83,mg85,mg88'
def snap_grid_to_pial_use_sftp(subject, args):
    argv = ['-s', subject, '--sftp', '1', '--sftp_username', 'npeled',
            '--sftp_domain', 'door.nmr.mgh.harvard.edu',
            '--remote_subject_dir', '/space/thibault/1/users/npeled/remote_subjects/{subject}',
            '--write_compact_bipolar', '1', '--overwrite', '0',
            '-f', 'snap_grid_to_pial']
    args = find_rois.get_args(argv)
    find_rois.run_for_all_subjects(args)


def darpa(subject, args):
    subject = subject[:2].upper() + subject[2:]
    argv = ['-s', subject, '-a', args.atlas,
            '--remote_subject_dir', '/space/huygens/1/users/kara/{subject}_SurferOutput']
            # '--remote_subject_dir_func', 'upper']
    args = find_rois.get_args(argv)
    find_rois.run_for_all_subjects(args)


def darpa_sftp(subject, args):
    darpa_subject = subject[:2].upper() + subject[2:]
    argv = ['-s', subject, '-a', args.atlas,
            '--sftp', '1', '--sftp_username', 'npeled',
            '--sftp_domain', 'door.nmr.mgh.harvard.edu',
            '--remote_subject_dir', '/space/huygens/1/users/kara/{}_SurferOutput'.format(darpa_subject)]
    args = find_rois.get_args(argv)
    find_rois.run_for_all_subjects(args)


def sftp(subject, args):
    argv = ['-s', subject, '-a', args.atlas,
            '--sftp', '1', '--sftp_username', 'npeled',
            '--sftp_domain', 'door.nmr.mgh.harvard.edu',
            '--remote_subject_dir', '/autofs/space/thibault_001/users/npeled/subjects/{}'.format(subject)]
    args = find_rois.get_args(argv)
    find_rois.run_for_all_subjects(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Electrodes labeling')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas40')
    parser.add_argument('-f', '--function', help='function name', required=True)
    args = utils.Bag(au.parse_parser(parser))
    for subject in args.subject:
        locals()[args.function](subject, args)