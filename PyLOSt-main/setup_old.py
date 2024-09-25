from setuptools import setup, find_packages

setup(
        name='PyLOSt',
        version='0.3',
        packages=find_packages(exclude=[]),
        package_data={
                  'PyLOSt':['*.ui'],
                  'PyLOSt.util':['*.ui'],
                  'PyLOSt.databases':['*.db'],
            },
        install_requires=['sqlobject', 'numpy', 'h5py', 'PyQt5', 'silx', 'matplotlib', 'scipy', 'astropy']
      )