from src import find_rois


def use_sftp():
    argv = ['-s', 'mg72,mg76,mg83,mg85,mg88', '--sftp', '1', '--sftp_username', 'npeled',
            '--sftp_domain', 'door.nmr.mgh.harvard.edu',
            '--remote_subject_dir_template', '/autofs/cluster/neuromind/npeled/subjects/{subject}',
            '--elecs_dir', '/home/noam/subjects/RAScoordinatessnapped',
            '--write_compact_bipolar', '1', '--overwrite', '0']
    args = find_rois.get_args(argv)
    find_rois.run_for_all_subjects(args)


if __name__ == '__main__':
    use_sftp()