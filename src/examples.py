import argparse

from src import find_rois
from src import utils
from src import args_utils as au

# 'mg72,mg76,mg83,mg85,mg88'
def use_sftp(subject, args):
    argv = ['-s', subject, '--sftp', '1', '--sftp_username', 'npeled',
            '--sftp_domain', 'door.nmr.mgh.harvard.edu',
            '--remote_subject_dir_template', '/autofs/cluster/neuromind/npeled/subjects/{subject}',
            '--elecs_dir', '/home/noam/subjects/RAScoordinatessnapped',
            '--write_compact_bipolar', '1', '--overwrite', '0']
    args = find_rois.get_args(argv)
    find_rois.run_for_all_subjects(args)


def darpa(subject, args):
    subject = subject[:2].upper() + subject[2:]
    argv = ['-s', subject, '-a', args.atlas,
            '--remote_subject_dir_template', '/space/huygens/1/users/kara/{subject}_SurferOutput']
            # '--remote_subject_dir_func', 'upper']
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