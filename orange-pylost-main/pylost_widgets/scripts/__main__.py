# coding=utf-8
## Module util_scripts has add_script_path, del_script_path, list_paths, import_paths
# For all these functions, param_name can be added to add other script
# E.g.1. add_script_path(dir_path, param_name='FILE_FORMAT_PATH')   # default param_name is 'GENERAL_SCRIPT_PATH'
from pylost_widgets.util.util_scripts import *
import os

param_name = 'GENERAL_SCRIPT_PATH'
dir_path = os.path.dirname(os.path.realpath(__file__))
var = input('Enter diretory type (1 = File format scripts, all other values = General script):')
if var == '1':
    param_name = 'FILE_FORMAT_PATH'
var = input('Action (1 = Add path, 2 = Delete path, 3 = Add path and upload scripts, 4 = Delete path and uploaded scripts, all other values = List paths):')
if var == '1':
    add_script_path(dir_path, param_name)
elif var == '2':
    del_script_path(dir_path, param_name)
elif var == '3':
    add_script_path(dir_path, param_name)
    upload_scripts(dir_path, param_name)
elif var == '4':
    del_script_path(dir_path, param_name)
    del_upload_scripts(dir_path, param_name)
list_paths(param_name)
