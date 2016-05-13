import os
import shutil
import numpy as np
import multiprocessing
import traceback
from functools import partial
import subprocess
try:
    import cPickle as pickle
except:
    import pickle


def get_link_dir(links_dir, link_name, var_name='', default_val='', throw_exception=False):
    val = os.path.join(links_dir, link_name)
    # check if this is a windows folder shortcup
    if os.path.isfile('{}.lnk'.format(val)):
        val = read_windows_dir_shortcut('{}.lnk'.format(val))
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
    parent_dir = os.path.split(proj_dir)[0]
    links_dir = os.path.join(parent_dir, 'links')
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
    n = int(max(1, n))
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
    try:
        wb = xlrd.open_workbook(xlsx_fname)
        sh = wb.sheet_by_name('Sheet1')
        csv_file = open(csv_fname, 'wb')
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)

        for rownum in range(sh.nrows):
            # wr.writerow([str(val).encode('utf_8') for val in sh.row_values(rownum)])
            csv_file.write(b','.join([str(val).encode('utf_8') for val in sh.row_values(rownum)]) + b'\n')

        csv_file.close()
    except:
        print('Error converting excel to csv!')
        print(traceback.format_exc())


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


# From http://stackoverflow.com/a/28952464/1060738
def read_windows_dir_shortcut(dir_path):
    import struct
    try:
        with open(dir_path, 'rb') as stream:
            content = stream.read()

            # skip first 20 bytes (HeaderSize and LinkCLSID)
            # read the LinkFlags structure (4 bytes)
            lflags = struct.unpack('I', content[0x14:0x18])[0]
            position = 0x18

            # if the HasLinkTargetIDList bit is set then skip the stored IDList
            # structure and header
            if (lflags & 0x01) == 1:
                position = struct.unpack('H', content[0x4C:0x4E])[0] + 0x4E

            last_pos = position
            position += 0x04

            # get how long the file information is (LinkInfoSize)
            length = struct.unpack('I', content[last_pos:position])[0]

            # skip 12 bytes (LinkInfoHeaderSize, LinkInfoFlags, and VolumeIDOffset)
            position += 0x0C

            # go to the LocalBasePath position
            lbpos = struct.unpack('I', content[position:position+0x04])[0]
            position = last_pos + lbpos

            # read the string at the given position of the determined length
            size= (length + last_pos) - position - 0x02
            temp = struct.unpack('c' * size, content[position:position+size])
            target = ''.join([chr(ord(a)) for a in temp])
    except:
        # could not read the file
        target = None

    return target


def cpu_count():
    return multiprocessing.cpu_count()


def fix_bin_str_in_arr(arr):
    return [s.astype(str) if isinstance(s, np.bytes_) else s for s in arr]


def get_n_jobs(n_jobs):
    cpu_num = multiprocessing.cpu_count()
    n_jobs = int(n_jobs)
    if n_jobs > cpu_num:
        n_jobs = cpu_num
    elif n_jobs < 0:
        n_jobs = cpu_num + n_jobs
    return n_jobs


# def bool_arr_type(var): return var
# def str_arr_type(var): return var
#
#
# def parse_parser(parser):
#     in_args = vars(parser.parse_args())
#     args = {}
#     for val in parser._option_string_actions.values():
#         if val.type is bool:
#             args[val.dest] = is_true(in_args[val.dest])
#         elif val.type is str_arr_type:
#             args[val.dest] = get_args_list(in_args, val.dest)
#         elif val.type is bool_arr_type:
#             args[val.dest] = get_args_list(in_args, val.dest, is_true)
#         elif val.dest in in_args:
#             args[val.dest] = in_args[val.dest]
#     return args
#
#
# def get_args_list(args, key, var_type=None):
#     if ',' in args[key]:
#         ret = args[key].split(',')
#     elif len(args[key]) == 0:
#         ret = []
#     else:
#         ret = [args[key]]
#     if var_type:
#         ret = list(map(var_type, ret))
#     return ret
#
#
# def is_true(val):
#     if isinstance(val, str):
#         if val.lower() == 'true':
#             return True
#         elif val.lower() == 'false':
#             return False
#         elif val.isnumeric():
#             return bool(int(val))
#         else:
#             raise Exception('Wrong value for boolean variable')
#     else:
#         return bool(val)
#

def partial_run_script(vars, more_vars=None, print_only=False):
    return partial(_run_script_wrapper, vars=vars, print_only=print_only)


def _run_script_wrapper(cmd, vars, print_only=False, **kwargs):
    for k,v in kwargs.items():
        vars[k] = v
    print(cmd.format(**vars))
    if not print_only:
        run_script(cmd.format(**vars))


def run_script(cmd, verbose=False):
    if verbose:
        print('running: {}'.format(cmd))
    output = subprocess.check_output('{} | tee /dev/stderr'.format(cmd),
                                     shell=True)
    print(output)
    return output

