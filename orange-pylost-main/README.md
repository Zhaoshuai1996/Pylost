Implementaion of Python Large Optic Stitching software with orange data mining platform.

**Installation**
- Install Orange software (https://orangedatamining.com/download/)
- Install git, if not already present (https://git-scm.com/downloads)
- Run the following command in 'Orange command prompt'

    `pip install git+https://gitlab.esrf.fr/moonpics_stitching_2018/orange-pylost.git`

- Start Orange shortcut (or with command orange-canvas)

**Update with following**
- `pip uninstall orange-pylost`
- `pip uninstall pylost`
- `pip install git+https://gitlab.esrf.fr/moonpics_stitching_2018/orange-pylost.git`

**For development of custom widgets**
- `git clone https://gitlab.esrf.fr/moonpics_stitching_2018/orange-pylost.git`
- `git clone https://gitlab.esrf.fr/moonpics_stitching_2018/PyLOSt.git`
- In orange-pylost directory, using Orange command prompt run 

    `python -m pip install -e . --no-deps --no-binary :all:`


