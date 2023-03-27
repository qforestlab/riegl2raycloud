## Usage:

``python riegl2raycloud.py -p /path/to/RiSCANfolder``

Optional parameters:
- ``--debug``: also save individual .ply files
- ``-r/--resolution``: perform downsampling to given resolution in cm

First generate .DAT files for project using Registration > Multiple SOP Export > Export complete 4x4 matrix as *.DA in RiSCAN PRO.

Tested using python 3.11 and Ubuntu 22.04 (WSL), using RiSCAN PRO 2.15.
"Killed" message means that the process ran out of memory.

## TODO's
- Atempt to decrease memory usage further
- Support for .riproject folders from vz400
