from setuptools import setup, find_packages
# Install from sources:
#     # install
#     python -m pip install .
#     # or install using links (developer)
#     python -m pip install -e . --no-deps --no-binary :all:
setup(
    name="orange-pylost",
    version="1.0",
#    packages=find_packages(include=['pylost_widgets'], exclude=[]),
    package_data={"pylost_widgets": ["config/*.yml", "icons/*.svg", "icons/*.png", "icons/*.ico", "gui/*.ui"],
                  },
    exclude_package_data={"doc": ["*.py"]},
    install_requires=['setuptools', 'line_profiler', 'scikit-image', 'h5py', 'astropy', 'silx', 'pyFAI', 'sympy', 'torch', 'orange3'],
    entry_points={"orange.widgets": ("orange-pylost = pylost_widgets.widgets", "pylost-model = pylost_widgets.learning")},
)
