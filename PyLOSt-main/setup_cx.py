# coding: utf-8
# /*#########################################################################
# Copyright (C) 2018 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
import sys
if "build_exe" not in sys.argv:
    print("Usage:")
    print("python setup_cx.py build_exe")
    sys.exit(0)

version = "0.3"

from cx_Freeze import setup, Executable, hooks
import os
import sys, glob
import shutil


import PyLOSt
import PyQt5
import numpy
import collections
import scipy
import h5py
import ctypes
import silx
import sqlobject
import pkg_resources

# special modules are included completely, with their data files
special_modules = [PyLOSt,silx, h5py,
                   ctypes, collections,
                   numpy,
                   scipy,
                   sqlobject,
                   pkg_resources]
try:
    import pysqlite2
    special_modules.append(pysqlite2)
    print("pysqlite2 found")
except:
    pass

try:
    import sqlite2
    special_modules.append(sqlite2)
    print("sqlite2 found")
except:
    pass

try:
    import sqlite3
    special_modules.append(sqlite3)
    print("sqlite3 found")
except:
    pass

try:
    import sqlite
    special_modules.append(sqlite)
    print("sqlite found")
except:
    pass

special_modules_dir = [os.path.dirname(mod.__file__) for mod in special_modules if mod is not None]
include_files = [(dir_, os.path.basename(dir_)) for dir_ in special_modules_dir]

# TODO: check if this line is still necessary
# include_files.append(("qtconffile", "qt.conf"))

# TODO: future dependency
# include_files.append((SILX_DATA_DIR,  os.path.join('silx', 'resources')))

packages = ["PyLOSt"]  # ["silx"]

includes = []

def dummy(*var, **kw):
    return

hooks.load_tkinter = dummy


# modules excluded from library.zip
excludes = ["tcl", "tk", "OpenGL", "scipy",
            "numpy", "pkg_resources", "DateTime"]

build_options = {
    "packages": packages,
    "includes": includes,
    "include_files": include_files,
    "excludes": excludes, }
    #"compressed": True, }

install_options = {}

PyMcaDir = os.path.dirname(PyLOSt.__file__)
exec_list = {"PyLOSt": os.path.join(".", "scripts","pylost.py"),
            }
if os.path.exists("build"):
    shutil.rmtree("build")
print("creating directory")
os.mkdir("build")
tmpDir = os.path.join("build", "tmp")
print("Creating temporary directory <%s>" % tmpDir) 
os.mkdir(tmpDir)

for f in list(exec_list.keys()):
    infile = open(exec_list[f], "r")
    outname = os.path.join(tmpDir, os.path.basename(exec_list[f]))
    outfile = open(outname, "w")
    outfile.write("import os\n")
    outfile.write("import ctypes\n")
    magic = 'os.environ["PATH"] += os.path.dirname(os.path.dirname(ctypes.__file__))\n'
    outfile.write(magic)
    for line in infile:
        outfile.write(line)
    outfile.close()
    infile.close()
    exec_list[f] = outname

executables = []
for key in exec_list:
    icon = None
    # this allows to map a different icon to each executable
    if sys.platform.startswith('win'):
        if key in ["PyMcaMain", "QStackWidget"]:
            icon = os.path.join(os.path.dirname(__file__), "icons", "PyMca.ico")
    executables.append(Executable(exec_list[key],
                                  base="Console" if sys.platform == 'win32' else None,
                                  icon=icon))

setup(name='pylost',
      version=version,
      description="PyLOSt %s" % version,
      options=dict(build_exe=build_options,
                   install_exe=install_options),
      executables=executables)

# cleanup
if 1:
    filesToRemove = ["MSVCP140.dll", "python37.dll"]
    work0 = []
    work1 = []
    for root, directory, files in os.walk("build"):
        for fname in files:
            if fname in filesToRemove:
                work0.append(os.path.join(root, fname))
        for dire in directory:
            if dire == "__pycache__":
                work1.append(os.path.join(root, dire))


    for item in work0[2:]:
        os.remove(item)

    for item in work1:
        shutil.rmtree(item)


exe_win_dir = "exe.win-amd64-%d.%d" % (sys.version_info[0], sys.version_info[1])

if 1:
    # remove duplicated modules
    import shutil
    destinationDir = os.path.join(".", "build",exe_win_dir, "lib")
    for dirname in special_modules_dir:
        destination = os.path.join(destinationDir, os.path.basename(dirname))
        if os.path.exists(destination):
            print("Deleting %s" % destination)
            shutil.rmtree(destination)
            print("Deleted")

    # remove unwanted directories
    for dirname in ["PyMca5", "fisx"]:
        destination = os.path.join(destinationDir, dirname)
        if os.path.exists(destination):
            print("Deleting %s" % destination)
            shutil.rmtree(destination)
            print("Deleted")

    # remove Qt binary files (qml and translation might be needed)
    for item in ["bin", "qml", "translations"]:
        destination = os.path.join(destinationDir,
                               "PyQt5","Qt",item)
        if os.path.exists(destination):
            print("Deleting %s" % destination)
            shutil.rmtree(destination)
            print("Deleted")
        else:
            print("NOT DELETING ", destination)

if 1:
    # replace excessively big files
    destinationDir = os.path.join(".", "build",exe_win_dir)
    safe_replacement = [os.path.dirname(mod.__file__) \
                        for mod in [h5py, numpy] \
                        if mod is not None]
    for dirname in safe_replacement:
        destination = os.path.join(destinationDir, os.path.basename(dirname))
        if os.path.exists(destination):
            print("Deleting %s" % destination)
            shutil.rmtree(destination)
            print("Deleted")
    for dirname in safe_replacement:
        destination = os.path.join(destinationDir, os.path.basename(dirname))
        print("Copying %s" % destination)
        shutil.copytree(dirname, destination)

if 1:
    # add sql pyd file if found
    try:
        import _sqlite3
        ffile = _sqlite3.__file__
        destinationDir = os.path.join(".", "build",exe_win_dir)
        destination = os.path.join(destinationDir, os.path.basename(ffile))
        print("Copying %s" % destination)
        shutil.copyfile(ffile, destination)
    except ImportError:
        pass

    # add top level .py modules
    destinationDir = os.path.join(".", "build",exe_win_dir, "lib")
    for ffile in glob.glob(os.path.join(os.path.dirname(os.path.dirname(ctypes.__file__)), "*.py")):
        destination = os.path.join(destinationDir, os.path.basename(ffile))
        print("Copying %s" % destination)
        shutil.copyfile(ffile, destination)
        
nsis = os.path.join("\Program Files (x86)", "NSIS", "makensis.exe")
if sys.platform.startswith("win") and os.path.exists(nsis):
    # check if we can perform the packaging
    outFile = "nsisscript.nsi"
    f = open("nsisscript.nsi.in", "r")
    content = f.readlines()
    f.close()
    if os.path.exists(outFile):
        os.remove(outFile)
    pymcaexe = "pylost%s-win64.exe" % version
    if os.path.exists(pymcaexe):
        os.remove(pymcaexe)
    frozenDir = os.path.join(".", "build", exe_win_dir)
    f = open(outFile, "w")
    for line in content:
        if "__VERSION__" in line:
            line = line.replace("__VERSION__", version)
        if "__SOURCE_DIRECTORY__" in line:
            line = line.replace("__SOURCE_DIRECTORY__", frozenDir)
        f.write(line)
    f.close()
    cmd = '"%s" %s' % (nsis, outFile)
    print(cmd)
    os.system(cmd)
