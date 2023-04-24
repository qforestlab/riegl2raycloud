## Usage:
1. generate .DAT files for project using Registration > Multiple SOP Export > Export complete 4x4 matrix as *.DA in RiSCAN PRO.

2. Run script using:

``python riegl2raycloud.py -p /path/to/RiSCANfolder``

Optional parameters:
- ``--debug``: also save individual .ply files
- ``-r/--resolution``: perform downsampling to given resolution in cm
- ``-b/--edgebuffer``: buffer in meter around edge of plot (default = 5)
- ``-t/--tilebuffer``: overlap between/buffer around each tile in meter (default = 2)
- ``-s/--tilesize``: tile size in meter(default = 20)
- ``--exact-size``: if True, uses the exact tile size and ends up with smaller tiles with leftover part of plot at end. If False, rounds tile size in each dimension to nearest size that produces identical rectangular tiles (default = False)



Tested using python 3.11 and Ubuntu 22.04 (WSL), using RiSCAN PRO 2.15.
"Killed" message means that the process ran out of memory.

## TODO's
- Support for .riproject folders from vz400
