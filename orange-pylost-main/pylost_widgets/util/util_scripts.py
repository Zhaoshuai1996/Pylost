# coding=utf-8
import datetime
import importlib
import os
import pkgutil

from PyLOSt.databases.gs_table_classes import ConfigParams, connectDB


def add_script_path(path, param_name='GENERAL_SCRIPT_PATH', description='', ctype='D'):
    """
    Add a new script directory. The path is saved to ConfigParams table in sql database.

    :param path: New path
    :type path: str
    :param param_name: Name of parameter, default 'GENERAL_SCRIPT_PATH'
    :type param_name: str
    :param description: Description of path
    :type description: str
    :param ctype: Type of parameter, 'D' for discreet (default) and 'C' for continuous
    :type ctype: str
    """
    conn = connectDB()
    try:
        qpaths = ConfigParams.selectBy(paramName=param_name, paramValue=path)
        if qpaths.count() > 0:
            conn.close()
            raise Exception('Path already exists in database')
        if description == '':
            if param_name == 'FILE_FORMAT_PATH':
                description = 'User file format scripts path'
            else:
                description = 'User scripts path'
        ConfigParams(paramName=param_name, paramDesc=description, paramType=ctype, paramValue=path,
                     dateCreated=datetime.datetime.today().strftime('%Y-%m-%d'))
    except Exception as e:
        print(e)
    conn.close()


def del_script_path(path, param_name='GENERAL_SCRIPT_PATH'):
    """
    Delete a directory from script paths.

    :param path: Directory path to delete
    :type path: str
    :param param_name: Name of parameter, default 'GENERAL_SCRIPT_PATH'
    :type param_name: str
    """
    conn = connectDB()
    try:
        if path is not None and path != '':
            qpaths = ConfigParams.selectBy(paramName=param_name, paramValue=path)
        else:
            qpaths = ConfigParams.selectBy(paramName=param_name)
        for qp in qpaths:
            qp.destroySelf()
    except Exception as e:
        print(e)
    conn.close()


def list_paths(param_name='GENERAL_SCRIPT_PATH'):
    """
    List all the script directory paths

    :param param_name: Name of parameter, default 'GENERAL_SCRIPT_PATH'
    :type param_name: str
    """
    conn = connectDB()
    try:
        qpaths = ConfigParams.selectBy(paramName=param_name)
        for qp in qpaths:
            print('{}={}'.format(qp.paramName, qp.paramValue))
    except Exception as e:
        print(e)
    conn.close()


def import_paths(param_name='GENERAL_SCRIPT_PATH'):
    """
    Import all modules present in script paths.

    :param param_name: Name of parameter, default 'GENERAL_SCRIPT_PATH'
    :type param_name: str
    """
    conn = connectDB()
    try:
        pkg_paths = []
        qpaths = ConfigParams.selectBy(paramName=param_name)
        for qp in qpaths:
            path = qp.paramValue
            if os.path.exists(path):
                pkg_paths.append(path)
            else:
                pkg = importlib.import_module(path, __package__)
                pkg_paths.append(os.path.dirname(pkg.__file__))

        for (module_loader, module_name, ispkg) in pkgutil.walk_packages(pkg_paths):
            if not ispkg:
                if module_name != '__main__':
                    try:
                        module_loader.find_module(module_name).load_module(module_name)
                    except Exception as e:
                        print(e)
            else:
                module_loader.find_module(module_name).load_module(module_name)
    except Exception as e:
        print(e)
    finally:
        if conn is not None:
            conn.close()


# TODO: Uploading scritps to gitlab. Requires login?
def upload_scripts(path, param_name='GENERAL_SCRIPT_PATH'):
    """
    TODO: Upload scripts to orange pylost.

    :param path: Path to scripts directory
    :type path: str
    :param param_name: Type of script, e.g. 'GENERAL_SCRIPT_PATH', 'FILE_FORMAT_PATH'
    :type param_name: str
    """
    conn = connectDB()
    try:
        import shutil
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if param_name == 'GENERAL_SCRIPT_PATH':
            dir_path = os.path.join(dir_path, '..', 'scripts', 'general')
        elif param_name == 'FILE_FORMAT_PATH':
            dir_path = os.path.join(dir_path, '..', 'scripts', 'file_formats')
        else:
            raise Exception('Please enter a valid script type, e.g. GENERAL_SCRIPT_PATH')

        pkg_paths = []
        if path is None:
            qpaths = ConfigParams.selectBy(paramName=param_name)
            for qp in qpaths:
                path = qp.paramValue
                if os.path.exists(path):
                    pkg_paths.append(path)
        else:
            pkg_paths = [path]

        for (module_loader, module_name, ispkg) in pkgutil.walk_packages(pkg_paths):
            if not ispkg:
                if module_name != '__main__':
                    src_path = module_loader.find_module(module_name).path
                    dst_path = os.path.join(dir_path, module_name.replace('.', os.sep) + '.py')
                    print('Adding script {}'.format(dst_path))
                    shutil.copy2(src_path, dst_path)
            else:
                dst_dir = os.path.join(dir_path, module_name.replace('.', os.sep))
                if not os.path.exists(dst_dir):
                    print('Creating {}'.format(dst_dir))
                    os.makedirs(dst_dir)
                open(os.path.join(dst_dir, '__init__.py'), 'a').close()
    except Exception as e:
        print(e)
    finally:
        if conn is not None:
            conn.close()


def del_upload_scripts(path, param_name='GENERAL_SCRIPT_PATH'):
    """
    TODO: Delete uploaded scripts.

    :param path: Path to scripts directory
    :type path: str
    :param param_name: Type of script, e.g. 'GENERAL_SCRIPT_PATH', 'FILE_FORMAT_PATH'
    :type param_name: str
    """
    conn = connectDB()
    try:
        import shutil
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if param_name == 'GENERAL_SCRIPT_PATH':
            dir_path = os.path.join(dir_path, '..', 'scripts', 'general')
        elif param_name == 'FILE_FORMAT_PATH':
            dir_path = os.path.join(dir_path, '..', 'scripts', 'file_formats')
        else:
            raise Exception('Please enter a valid script type, e.g. GENERAL_SCRIPT_PATH')

        pkg_paths = []
        if path is None:
            qpaths = ConfigParams.selectBy(paramName=param_name)
            for qp in qpaths:
                path = qp.paramValue
                if os.path.exists(path):
                    pkg_paths.append(path)
        else:
            pkg_paths = [path]

        for (module_loader, module_name, ispkg) in pkgutil.walk_packages(pkg_paths):
            if not ispkg:
                if module_name != '__main__':
                    dst_path = os.path.join(dir_path, module_name.replace('.', os.sep) + '.py')
                    if os.path.exists(dst_path):
                        print('Deleting script {}'.format(dst_path))
                        os.remove(dst_path)
            else:
                dst_dir = os.path.join(dir_path, module_name.replace('.', os.sep))
                if os.path.exists(dst_dir):
                    print('Deleting package {}'.format(dst_dir))
                    shutil.rmtree(dst_dir)
    except Exception as e:
        print(e)
    finally:
        if conn is not None:
            conn.close()
