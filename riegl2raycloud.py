import os
import argparse
import glob
import csv
import json
import time

import matplotlib.pyplot as plt
import pdal
import open3d as o3d
import numpy as np

from boundingrectangle import boundingrectangle
from tile import tile_from_corner_points

OUT_DIR = "out"

def read_csv(file):
    pos_dict = {}
    with open(file) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            pos_dict[row[0]] = [float(i) for i in row[1:]]
    return pos_dict

def read_dat(folder):
    pos_dict = {}
    for datfile in glob.glob(os.path.join(folder, "*.DAT")):
        scanpos = os.path.splitext(os.path.basename(datfile))[0]
        with open(datfile) as f:
            mat = []
            for line in f.readlines():
                mat_row = [float(el) for el in line.split(' ')]
                mat.append(mat_row)
            pos_dict[scanpos] = np.array(mat)
    return pos_dict

def plot_dat_positions(positions, visualisation: bool = False, out_dir = None):
    plt.scatter([scanposfromdat(el)[0] for el in positions.values()], [scanposfromdat(el)[1] for el in positions.values()])
    for txt in positions:
        plt.annotate(txt, (scanposfromdat(positions[txt])[0], scanposfromdat(positions[txt])[1] + 1))
    if out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(os.path.join(out_dir, "PlotScanpositions.png"))
    if visualisation:
        plt.show()
    return

def get_pipeline(rxp_file):
    cmds = []
    read = {"type":"readers.rxp",
                "filename": rxp_file,
                "sync_to_pps": "false",
                "reflectance_as_intensity": "false"}

    cmds.append(read)
    # option to add PDAL filters here

    dmp = json.dumps(cmds)
    pipeline = pdal.Pipeline(dmp)

    return pipeline

def transform_rxp(rxp_array, matrix):
    """
    Returns numpy array with xyz's of rxp file transformed using .dat matrix

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
    #perform transformation
    xyz_np = np.matmul(matrix, xyz_np)
    #retranspose to get nx4 array again
    xyz_np = np.transpose(xyz_np)
    # remove final column of 1's
    xyz_np = xyz_np[:,:-1]

    return xyz_np

def read_rxps(project, pos_dict): 
    """
    Reads all rxp files in a project folder
    Returns generator with scanpos, points

    Parameters
    ----------
    project
        .RISCAN folder
    pos_dict
        dictionary with .DAT matrices
    """
    for scanpos in sorted(os.listdir(os.path.join(project, 'SCANS'))): # TODO: temp slice for laptop memory reasons
        if scanpos not in pos_dict:
            print(f"Can't read rxp {scanpos} as not present in pos_dict, make sure .DAT files are generated before running (skipping)")
            continue
        rxp = glob.glob(os.path.join(project, 'SCANS', scanpos, '**/*_*.rxp'))
        # remove all the residual files
        rxp = [el for el in rxp if not 'residual' in el]

        if len(rxp) != 1:
            print(f"Error for scanpos {scanpos}: rxp file not found (list = {rxp})")
            continue

        # get and execute pdal pipeline
        pipeline = get_pipeline(rxp[0])
        pipeline.execute()
        
        # transform using DAT files into 1 coordinate system
        points = transform_rxp(pipeline.arrays[0], pos_dict[scanpos])
        
        yield scanpos, points
        del points # don't know if this actually does something but anyways

def scanposfromdat(matrix):
    return matrix[:-1,-1]

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
    # call generator function for rxps
    for scanpos, points in read_rxps(args.project, pos_dict):
        print(f"Processing position {scanpos}")
        pcd = appendray(points, scanposfromdat(pos_dict[scanpos]), 1)
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
            pos_dir = out_dir + "/pos/"
            if not os.path.exists(pos_dir):
                os.makedirs(pos_dir)
            filename = pos_dir+scanpos+"_raycloud.ply"
            o3d.t.io.write_point_cloud( filename, pcd, write_ascii=False, compressed=False, print_progress=False)
        del pcd, points
    
    print("Tiling merged point cloud")
    # Tile based on corners forming rectangle
    tile_out_dir = os.path.join(out_dir, "tiled")
    if not os.path.exists(tile_out_dir):
        os.makedirs(tile_out_dir)

    tiles = tile_from_corner_points(bbox_xy_corners, merged, size=args.tilesize, buffer=args.tilebuffer, exact_size=args.exact_tiles, visualization=False, out_dir=tile_out_dir)

    for i, tile in enumerate(tiles):
        o3d.t.io.write_point_cloud(os.path.join(tile_out_dir,"Tile"+str(i)+".ply"), tile, write_ascii=False, compressed=False, print_progress=False)
    
    #write merged pointcloud
    print("Writing merged pointcloud")
    o3d.t.io.write_point_cloud(os.path.join(out_dir, "merged_raycloud.ply"), merged, write_ascii=False, compressed=False, print_progress=False)
    return

# @profile
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("-r", "--resolution", type=float)
    # tile related
    parser.add_argument("-b", "--edgebuffer", type=int, default=5)
    parser.add_argument("-t", "--tilebuffer", type=int, default=2)
    parser.add_argument("-s", "--tilesize", type=int, default=20)
    parser.add_argument("--exact_tiles", action='store_true')
    print("")

    args = parser.parse_args()

    if not os.path.exists(args.project):
        print("couldnt find folder")
        os._exit(1)

    pos_dict = read_dat(args.project)

    # get output folder from project name
    if (args.project.endswith("/")):
        args.project = args.project[:len(args.project) -1]
    out_dir = os.path.join("out", os.path.splitext(os.path.basename(args.project))[0])

    print("Converting raycloud")
    plot_dat_positions(pos_dict, out_dir=out_dir)

    t = time.process_time()
    pc2rc(pos_dict, out_dir, args)
    t2 = time.process_time()
    print(f"Converted pointclouds to rayclouds in {(t2 - t):.2f} seconds")

    return


if __name__ == "__main__":
    main()
