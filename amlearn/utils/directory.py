import os
import lockfile
import shutil

__author__ = "Qi Wang"
__email__ = "qiwang.mse@gmail.com"


def copy_path(source_path, sink_path):
    shutil.copytree(source_path, sink_path)


def delete_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def create_path(path, overwrite=False, merge=False, verbosity=0):
    if os.path.exists(path):
        if overwrite:
            delete_path(path)
            if verbosity > 0:
                print("Delete previous path {}.".format(path))
        elif merge:
            if verbosity > 0:
                print("Path {} exists, just write files here.".format(path))
        else:
            raise FileExistsError("path {} already exists.".format(path))
    else:
        os.makedirs(path)
    if verbosity > 0:
        print("Create path {} successful.".format(path))


def auto_rename_file(file):
    pass


def write_file(file, message, mode='w'):
    lock_file = "{}.lock".format(file)
    with lockfile.LockFile(lock_file):
        with open(file, mode) as wf:
            wf.write(message)


def read_file(file, mode='r'):
    with open(file, mode) as rf:
        lines = rf.readlines()
    return lines

