# from surfer import Brain
import csv
import glob
import os
import os.path as op
import shutil

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

from src import utils


SUBJECTS_DIR = '/homes/5/npeled/space3/subjects'
COPY_FROM = '/homes/5/npeled/space3/Downloads/for_noam'
COPY_FROM_2 = '/homes/5/npeled/space3/Downloads/SurfExtras'


def copy_some_files():
    neccesary_files = {'mri': ['aseg.mgz'], 'surf': ['rh.pial', 'lh.pial', 'rh.sphere.reg', 'lh.sphere.reg', 'lh.white', 'rh.white']}
    files = glob.glob(os.path.join(COPY_FROM, '*.*'))
    for full_file_name in files:
        file = os.path.split(full_file_name)[1]
        subject, file_name = file.split('_')
        subject = subject.lower()
        sub_folder = 'mri' if file_name in neccesary_files['mri'] else 'surf'
        mkdirs(os.path.join(SUBJECTS_DIR, subject, sub_folder))
        local_file = os.path.join(SUBJECTS_DIR, subject, sub_folder, file_name)
        if os.path.isfile(local_file):
            print('{} already exist'.format(local_file))
        else:
            print('copying {} to {}'.format(full_file_name, local_file))
            shutil.copyfile(full_file_name, local_file)


def copy_some_files2():
    neccesary_files = {'mri': ['aseg.mgz'], 'surf': ['rh.pial', 'lh.pial', 'rh.sphere.reg', 'lh.sphere.reg', 'lh.white', 'rh.white']}

    for subject_folder in get_subfolders(COPY_FROM_2):
        folder = os.path.split(subject_folder)[1]
        subject = folder[:-4]
        subject = subject.lower()
        for full_file_name in glob.glob(os.path.join(COPY_FROM_2, subject_folder, '*.*')):
            file_name = os.path.split(full_file_name)[1]
            sub_folder = 'mri' if file_name in neccesary_files['mri'] else 'surf'
            mkdirs(os.path.join(SUBJECTS_DIR, subject, sub_folder))
            local_file = os.path.join(SUBJECTS_DIR, subject, sub_folder, file_name)
            if os.path.isfile(local_file):
                print('{} already exist'.format(local_file))
            else:
                print('copying {} to {}'.format(full_file_name, local_file))
                shutil.copyfile(full_file_name, local_file)


def get_subfolders(fol):
    return [os.path.join(fol,subfol) for subfol in os.listdir(fol) if os.path.isdir(os.path.join(fol,subfol))]


def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass


def electrodes_npz_to_csv(npz_file, csv_file):
    d = np.load(npz_file)
    names, pos = d['names'], d['pos']
    with open(csv_file, 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerow(['', 'R', 'A', 'S'])
        for name, pos in zip(names, pos):
            values = [name]
            values.extend(pos)
            writer.writerow(values)


def check_cigar(elc_length, r):
    import itertools
    pos = np.array([21.5,   3.,   9.])
    next_pos = np.array([26.5,   3.,   9.])
    dist = np.linalg.norm(next_pos-pos)
    elc_ori = (next_pos-pos) / dist # norm(elc_ori)=1mm
    elc_line = np.array([pos + elc_ori*t for t in np.linspace(-elc_length/2.0, elc_length/2.0, 100)])
    points_axis = [np.arange(15, 35, 0.5), np.arange(-5, 13, 0.5), np.arange(0, 20, 0.5)]
    points = np.array(list(itertools.product(*points_axis)))
    # N = 1000000
    # points = np.zeros((N, 3))
    # points[:, 0] = np.random.uniform(0.0, 40.0, (N,))
    # points[:, 1] = np.random.uniform(-20.0, 20.0, (N,))
    # points[:, 2] = np.random.uniform(-20.0, 20.0, (N,))
    dists = np.min(cdist(elc_line, points), 0)
    inside = dists <= r
    outside = dists > r
    print(sum(inside))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(elc_line[:, 0], elc_line[:, 1], elc_line[:, 2], c='k')
    # ax.scatter(points[outside, 0], points[outside, 1], points[outside, 2], c='blue', edgecolors='none', alpha=0.1)
    ax.scatter(points[inside, 0], points[inside, 1], points[inside, 2], c='red', edgecolors='none', alpha=0.3)
    plt.axis('off')
    plt.show()


def check_labels(subject, hemi, atlas):
    brain = Brain(subject, hemi, 'pial', subjects_dir=SUBJECTS_DIR)
    labels = glob.glob(os.path.join(SUBJECTS_DIR, subject, 'label', atlas, '*.label'))
    for label_fname in labels:
        if label_fname[-8:-6] == hemi:
            brain.add_label(label_fname, borders=True)
    brain.save_image(os.path.join(SUBJECTS_DIR, subject, 'label', '{}_{}_labels.png'.format(atlas, hemi)))


def compare_between_files(csv1, csv2):
    data1 = np.genfromtxt(csv1, delimiter=',', dtype=str)
    data2 = np.genfromtxt(csv2, delimiter=',', dtype=str)
    for line_ind, (line1, line2) in enumerate(zip(data1, data2)):
        for col_ind, (val1, val2) in enumerate(zip(line1, line2)):
            try:
                fval1, fval2 = float(val1), float(val2)
                same = abs(fval2-fval1) < 1e-2
            except ValueError:
                same = val1 == val2
            if not same:
                print('diff in line {} col {}: {}, {}'.format(line_ind, col_ind, val1, val2))


def find_subjects_with_electrodes_in(areas, areas_name, threshold=0):
    files = glob.glob(op.join('../electrodes/*_aparc.DKTatlas40_electrodes_cigar_r_3_l_4.pkl'))
    areas = [area.lower() for area in areas]
    from collections import defaultdict
    electrodes = defaultdict(list)
    str = ''
    subjects_num = 0
    for fname in files:
        subject = utils.namebase(fname).split('_')[0]
        electrodes[subject] = defaultdict(list)
        if not subject.lower().startswith('mg'):
            continue
        x = utils.load(fname)
        elcs_str = ''
        area_electrodes = 0
        for elc in x:
            for sub, prob in zip(elc['subcortical_rois'], elc['subcortical_probs']):
                if any([area in sub.lower() for area in areas]) and prob > threshold:
                    electrodes[subject][elc['name']].append((sub, prob))
            if len(electrodes[subject][elc['name']]) == 0:
                continue
            area_electrodes += 1
            elcs_str += '{}: {}, '.format(elc['name'], ','.join(['{} ({:.2f})'.format(sub, prob) for sub, prob in electrodes[subject][elc['name']]]))
            # for sub, prob in electrodes[subject][elc['name']]:
            #     elcs_str += '({}, {})'.format(sub, prob)
        if area_electrodes == 0:
            continue
        subjects_num += 1
        str += '{}, {} electrodes, {}\n'.format(subject, area_electrodes, elcs_str)

    str = '{} subjects:\n{}'.format(subjects_num, str)
    with open(op.join('..', 'electrodes', '{}.csv'.format(areas_name)), 'w') as file:
        file.write(str)
    print(str)



if __name__ == '__main__':
    # copy_some_files()
    # electrodes_npz_to_csv('/homes/5/npeled/space3/subjects/mg79/electrodes/electrodes_positions.npz',
    #     '/homes/5/npeled/space3/subjects/mg79/electrodes/mg79_RAS.csv')
    # check_cigar(4, 2)
    # copy_some_files2()
    # check_labels('mg96', 'rh', 'laus250')
    # for subject in ['mg72','mg83','mg85','mg88']:
    #     print('****** {} ******'.format(subject))
    #     compare_between_files('/home/noam/code/electrodes_rois/electrodes/{}_aparc.DKTatlas40_electrodes_cigar_r_3_l_4_bipolar.csv'.format(subject),
    #                       '/home/noam/Desktop/{}_aparc.DKTatlas40_electrodes_cigar_r_3_l_4_bipolar.csv'.format(subject))
    striatal = ['basal-ganglia', 'basal-nucleus', 'pallidum', 'caudate', 'putamen']
    find_subjects_with_electrodes_in(striatal, 'striatal', 0)
    print('finish!')