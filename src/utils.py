import os
import shutil
import numpy as np
import multiprocessing
try:
    import cPickle as pickle
except:
    import pickle


def get_link_dir(links_dir, link_name, var_name, default_val='', throw_exception=False):
    val = os.path.join(links_dir, link_name)
    if not os.path.isdir(val) and default_val != '':
        val = default_val
    if not os.path.isdir(val):
        val = os.environ.get(var_name, '')
    if not os.path.isdir(val):
        if throw_exception:
            raise Exception('No {} dir!'.format(link_name))
        else:
            print('No {} dir!'.format(link_name))
    return val


def get_links_dir():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    proj_dir = os.path.split(curr_dir)[0]
    links_dir = os.path.join(proj_dir, 'links')
    return links_dir


def delete_folder_files(fol):
    if os.path.isdir(fol):
        shutil.rmtree(fol)
    os.makedirs(fol)


def plot_3d_scatter(X, names=None, labels=None, classifier=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    plt.show()


def read_ply_file(ply_file):
    with open(ply_file, 'r') as f:
        lines = f.readlines()
        verts_num = int(lines[2].split(' ')[-1])
        faces_num = int(lines[6].split(' ')[-1])
        verts_lines = lines[9:9 + verts_num]
        faces_lines = lines[9 + verts_num:]
        verts = np.array([map(float, l.strip().split(' ')) for l in verts_lines])
        faces = np.array([map(int, l.strip().split(' ')) for l in faces_lines])[:,1:]
    return verts, faces


def namebase(file_name):
    return os.path.splitext(os.path.basename(file_name))[0]


def run_parallel(func, params, njobs=1):
    if njobs == 1:
        results = [func(p) for p in params]
    else:
        pool = multiprocessing.Pool(processes=njobs)
        results = pool.map(func, params)
        pool.close()
    return results


def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]


def make_dir(fol):
    if not os.path.isdir(fol):
        os.makedirs(fol)
    return fol


def get_subfolders(fol):
    return [os.path.join(fol,subfol) for subfol in os.listdir(fol) if os.path.isdir(os.path.join(fol,subfol))]


def csv_from_excel(xlsx_fname, csv_fname):
    import xlrd
    import csv
    wb = xlrd.open_workbook(xlsx_fname)
    sh = wb.sheet_by_name('Sheet1')
    csv_file = open(csv_fname, 'wb')
    wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)

    for rownum in xrange(sh.nrows):
        wr.writerow([str(val) for val in sh.row_values(rownum)])

    csv_file.close()


def save(obj, fname):
    with open(fname, 'wb') as fp:
        # protocol=2 so we'll be able to load in python 2.7
        pickle.dump(obj, fp)


def load(fname):
    with open(fname, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


class Bag( dict ):
    """ a dict with d.key short for d["key"]
        d = Bag( k=v ... / **dict / dict.items() / [(k,v) ...] )  just like dict
    """
        # aka Dotdict

    def __init__(self, *args, **kwargs):
        dict.__init__( self, *args, **kwargs )
        self.__dict__ = self

    def __getnewargs__(self):  # for cPickle.dump( d, file, protocol=-1)
        return tuple(self)
