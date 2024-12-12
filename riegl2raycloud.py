import os
import argparse
import glob
# import csv
import json
import time

import matplotlib.pyplot as plt
import pdal
import open3d as o3d
import numpy as np

from boundingrectangle import boundingrectangle
from tile import tile_from_corner_points

def read_dat(matrix_dir):
    """
    Reads all transformation matrix files from the specified directory.

    Parameters
    ----------
    matrix_dir : str
        Path to the directory containing matrix files.

    Returns
    -------
    dict
        A dictionary where keys are scan position names and values are 2D numpy arrays of transformation matrices.
    """
    pos_dict = {}
    for datfile in glob.glob(os.path.join(matrix_dir, "*")):
        scanpos = os.path.splitext(os.path.basename(datfile))[0]
        with open(datfile) as f:
            mat = []
            for line in f.readlines():
                mat_row = [float(el) for el in line.split(' ')]
                mat.append(mat_row)
            pos_dict[scanpos] = np.array(mat)
    return pos_dict

def scanposfromdat(matrix):
    """
    Extracts the scan position from the last column of the given transformation matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        A 4x4 transformation matrix.

    Returns
    -------
    numpy.ndarray
        A 1D array representing the scan position [x, y, z].
    """
    return matrix[:-1,-1]

def plot_dat_positions(positions, visualisation: bool = False, out_dir = None):
    """
    Plot the scan positions from the matrix files.
    
    Parameters
    ----------
    positions : dict
        Dictionary of scan positions, where keys are names and values are transformation matrices.
    visualisation : bool, optional
        If True, displays the plot.
    out_dir : str, optional
        Directory where the plot image will be saved, if specified.

    """
    plt.scatter([scanposfromdat(el)[0] for el in positions.values()], 
                [scanposfromdat(el)[1] for el in positions.values()])
    for txt in positions:
        plt.annotate(txt, (scanposfromdat(positions[txt])[0], scanposfromdat(positions[txt])[1] + 1))
    if out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(os.path.join(out_dir, "PlotScanpositions.png"))
    if visualisation:
        plt.show()
    return

def get_pipeline(rxp_file, args):
    """
    Constructs a PDAL pipeline for processing .rxp files.

    Parameters
    ----------
    rxp_file : str
        Path to the .rxp file.
    args : argparse.Namespace
        Parsed command-line arguments containing pipeline options.

    Returns
    -------
    pdal.Pipeline
        A configured PDAL pipeline instance.
    """
    cmds = []
    read = {"type":"readers.rxp",
            "filename": rxp_file,
            "sync_to_pps": "false",
            "reflectance_as_intensity": "false"}

    cmds.append(read)
    # option to add PDAL filters here
    ## add filters
    dev_filter = {"type":"filters.range", 
                    "limits":"Deviation[0:{}]".format(args.deviation)}
    cmds.append(dev_filter)    

    refl_filter = {"type":"filters.range", 
                    "limits":"Reflectance[{}:{}]".format(*args.reflectance)}
    cmds.append(refl_filter)

    dmp = json.dumps(cmds)
    pipeline = pdal.Pipeline(dmp)

    return pipeline

def transform_rxp(rxp_array, matrix):
    """
    Returns numpy array with xyz coords from rxp file transformed using .dat matrix

    Parameters
    ----------
    rxp_array
        array of tuples containing rxp data points
    matrix
        4x4 transformation matrix
    """

    xyz = rxp_array[['X', 'Y', 'Z']]
    # need to do this to convert tuples to arrays, there might be a faster way with .view or something
    xyz_np = np.array(xyz.tolist())

    # append extra row of ones to make transform work
    extra_dim = np.ones((xyz_np.shape[0],1))
    xyz_np = np.hstack((xyz_np, extra_dim))
    # transpose
    xyz_np = np.transpose(xyz_np)
    # perform transformation
    xyz_np = np.matmul(matrix, xyz_np)
    # retranspose to get nx4 array again
    xyz_np = np.transpose(xyz_np)
    # remove final column of 1's
    xyz_np = xyz_np[:,:-1]

    return xyz_np

def read_rxps(scanpos_dirs, pos_dict, args): 
    """
    Reads and processes rxp files from all scans
    Returns generator with scanpos, points

    Parameters
    ----------
    scanpos_dirs
        a list of all ScanPos directories
    pos_dict
        a dictionary with transformation matrices for all scan positions
    args : argparse.Namespace
        Parsed command-line arguments containing pipeline options.

    Yields
    ------
    tuple
        A tuple (scanpos_label, transformed_points), 
        where `scanpos_label` is the scan position label
        and `transformed_points` is a numpy array of transformed points.
    """
    for scan_dir in sorted(scanpos_dirs):
        try:
            base, scan = os.path.split(scan_dir)
            try:
                rxp = sorted(glob.glob(os.path.join(base, scan, 'scans' if 'SCNPOS' in scan else '', '??????_??????.rxp')))[-1]
                print(rxp)
            except:
                print(f"Error: Cannot find {os.path.join(base, scan, '??????_??????.rxp')}")
                return
        except:
            print(f"Error in 'base, scan = os.path.split({scan_dir})' ")
            return

        # get and execute pdal pipeline
        pipeline = get_pipeline(rxp, args)
        pipeline.execute()
        
        # register all raw scans (.rxp files) into a unified coordinate system
        # using the SOP matrix for each scan
        scanposlabel = scan.split('.')[0]
        points = transform_rxp(pipeline.arrays[0], pos_dict[scanposlabel])
        
        # yield makes the function a generator
        # generator is memory-efficient as it allows processing one item at a time without loading everything into memory at once
        yield scanposlabel, points
        del points # don't know if this actually does something but anyways

def appendray(points, scanpos, time) -> o3d.t.geometry.PointCloud:
    """
    Returns o3d Pointcloud with scanposition coordinates saved in normal field and appended time field
    
    Parameters
    ----------
    points
        np array of points
    scanpos
        np array of xyz of scanpos
    time
        time value of scan pos
    
    Returns
    -------
    o3d.t.geometry.PointCloud
        An Open3D point cloud object.
    """
    
    device = o3d.core.Device("CPU:0")
    dtype_f32 = o3d.core.float32
    dtype_f64 = o3d.core.float64

    pcd = o3d.t.geometry.PointCloud(device)

    pcd.point.positions = o3d.core.Tensor(points, dtype_f32, device)
    n_points = len(points)

    # append scanpos
    scanpos = np.repeat([scanpos], n_points, axis=0)
    pcd.point.normals = o3d.core.Tensor( scanpos, dtype_f32, device)

    # append time
    time = np.repeat([[time]], n_points, axis=0)
    pcd.point.time = o3d.core.Tensor( time, dtype_f64, device)
    return pcd

def pc2rc(pos_dict, out_dir, args):
    """
    Converts pointclouds into rayclouds, and merges into one big .ply conforming to raycloudtools formatting
    Also crops the pointcloud to bounding rectangle with buffer

    Parameters
    ----------
    pos_dict
        dictionary with scanner positions: {"ScanPosXX" -> [x, y, z]}
    out_dir
        path to output directory
    args
        Parsed command-line arguments containing processing options.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # get bounding rectangle of scanpositions
    positions = [scanposfromdat(el)[:2] for el in pos_dict.values()]

    if len(positions) < 4:
        print("Less then 4 positions found, not cropping with bounding rectangle")
    else:
        area, bbox_xy_corners = boundingrectangle(positions, buffer=args.edgebuffer, out_dir=out_dir)

    # first create 03d pointcloud, then get bounding box from this pointcloud
    # no direct way of getting bbox in current o3d
    # also can't use inf because o3d complains, so use LARGE_Z (update when trees get larger then 1 million metres!)
    LARGE_Z = 1000000

    top_corners = np.hstack((bbox_xy_corners, np.asarray([[LARGE_Z]]*4)))
    bot_corners = np.hstack((bbox_xy_corners, np.asarray([[-LARGE_Z]]*4)))

    corners_pc = o3d.t.geometry.PointCloud(o3d.core.Tensor(np.vstack((top_corners, bot_corners))))

    crop_bbox = corners_pc.get_oriented_bounding_box()

    merged = None
    
    # get all scanpos directories
    scanpos_dirs = sorted(glob.glob(os.path.join(args.project, f'{args.prefix}*')))
    if len(scanpos_dirs) == 0: raise Exception('no scan positions found')
    
    # call generator function for rxps
    for scanposlabel, points in read_rxps(scanpos_dirs, pos_dict, args):
        print(f"Processing position {scanposlabel}")
        pcd = appendray(points, scanposfromdat(pos_dict[scanposlabel]), 1)
        # crop using bbox
        print("Cropping")
        pcd = pcd.crop(crop_bbox)
        # downsampling is done before merging, as otherwise normals (aka rays) are averaged in voxel
        if (args.resolution):
            print("Downsampling")
            pcd = pcd.voxel_down_sample(args.resolution)
        print("Merging")
        # ugly code but for some reason o3d errors when appending to empty pointcloud
        if merged is None:
            # copy constructor
            merged = o3d.t.geometry.PointCloud(pcd)
        else:
            merged = merged.append(pcd)
        if(args.debug):
            # for debugging: also write single point clouds
            pos_dir = os.path.join(out_dir, "pos")
            if not os.path.exists(pos_dir):
                os.makedirs(pos_dir)
            filename = os.path.join(pos_dir, f'{scanposlabel}_raycloud.ply')
            o3d.t.io.write_point_cloud(filename, pcd, write_ascii=False, 
                                       compressed=False, print_progress=False)
        del pcd, points
    
    print("Tiling merged point cloud")
    # Tile based on corners forming rectangle
    tile_out_dir = os.path.join(out_dir, "tiled")
    if not os.path.exists(tile_out_dir):
        os.makedirs(tile_out_dir)

    tiles = tile_from_corner_points(bbox_xy_corners, merged, size=args.tilesize, 
                                    buffer=args.tilebuffer, exact_size=args.exact_tiles, 
                                    visualization=False, out_dir=tile_out_dir)

    for i, tile in enumerate(tiles):
        o3d.t.io.write_point_cloud(os.path.join(tile_out_dir,"Tile"+str(i)+".ply"), tile, 
                                   write_ascii=False, compressed=False, print_progress=False)
    
    #write merged pointcloud
    print("Writing merged pointcloud")
    o3d.t.io.write_point_cloud(os.path.join(out_dir, "merged_raycloud.ply"), merged, 
                               write_ascii=False, compressed=False, print_progress=False)
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", type=str, required=True,
                        help="Path to folder containing ScanPosXXX dirs") 
    parser.add_argument("-m", "--matrix", type=str, required=True,
                        help='Path to folder containing sop matrix files') # new argument
    parser.add_argument("-o", "--odir", type=str, default='.',
                        help='Path to output dir') # new argument
    parser.add_argument('--deviation', type=float, default=15, 
                        help='deviation filter') # new argument
    parser.add_argument('--reflectance', type=float, nargs=2, default=[-20, 5], 
                        help='reflectance filter') # new argument
    parser.add_argument("--prefix", type=str, default='ScanPos',
                        help='scanpos dir name prefix, default:ScanPos') # new argument
    parser.add_argument("--debug", action='store_true',
                        help='use for debugging, will write individual ScanPos rayclouds') 
    parser.add_argument("-r", "--resolution", type=float, default=0.02,
                        help='Voxel size for downsampling, default:0.02 m')
    # tile related
    parser.add_argument("-b", "--edgebuffer", type=int, default=5,
                        help='Buffer around edge of plot, default:5 m')
    parser.add_argument("-t", "--tilebuffer", type=int, default=2,
                        help='overlap between / buffer around each tile, default:2 m')
    parser.add_argument("-s", "--tilesize", type=int, default=20,
                        help='The length of the side of each square tile, default:20 m')
    parser.add_argument("--exact_tiles", action='store_true')
    print("")

    args = parser.parse_args()

    if not os.path.exists(args.project):
        print("couldnt find project folder")
        os._exit(1)

    pos_dict = read_dat(args.matrix)

    print("Converting raycloud")
    plot_dat_positions(pos_dict, out_dir=args.odir)

    t = time.process_time()
    pc2rc(pos_dict, args.odir, args)
    t2 = time.process_time()
    # print(f"Converted pointclouds to rayclouds in {(t2 - t):.2f} seconds")
    total_seconds = t2 - t
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(f"Converted pointclouds to rayclouds in {hours} h {minutes} min {seconds} s")

    return


if __name__ == "__main__":
    main()
