from setuptools import setup, find_packages

setup(
    name='PyLOSt',
    version='0.3',
    packages=find_packages(exclude=[]),
    package_data={
              'PyLOSt':['*.ui'],
              'PyLOSt.ui':['*.ui'],
              'PyLOSt.ui.settings':['*.ui'],
              'PyLOSt.ui.simulations':['*.ui'],
              'PyLOSt.databases':['*.db'],
              'PyLOSt.data_in.esrf':['*.txt'],
        },
    install_requires=['setuptools', 'line_profiler', 'sqlobject'],
  )
