import os
import argparse
import glob
import csv
import json
import time
import re

import matplotlib.pyplot as plt
import pdal
import open3d as o3d
import numpy as np

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

def visualize(positions):
    plt.scatter([el[0] for el in positions.values()], [el[1] for el in positions.values()])
    for txt in positions:
        plt.annotate(txt, (positions[txt][0], positions[txt][1] + 1))
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
    pipeline.execute()

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
    Returns dict{ scanpos: array([[x,y,z], ..) }

    Parameters
    ----------
    project
        .RISCAN folder
    """
    dct = {}
    for scanpos in sorted(os.listdir(os.path.join(project, 'SCANS'))): # TODO: temp slice, runs out of memory otherwise
        if scanpos not in pos_dict:
            print(f"Can't read rxp {scanpos} as not present in pos_dict, make sure .DAT files are generated before running (skipping)")
            continue
        rxp = glob.glob(os.path.join(project, 'SCANS', scanpos, '**/*_*.rxp'))
        # remove all the residual files
        rxp = [el for el in rxp if not 'residual' in el]

        if len(rxp) != 1:
            print(f"Error for scanpos {scanpos}: rxp file not found (list = {rxp})")
            continue
        pipeline = get_pipeline(rxp[0])
        
        points = transform_rxp(pipeline.arrays[0], pos_dict[scanpos])
        dct[scanpos] = points

    return dct

def scanposfromdat(matrix):
    return matrix[:-1,-1]

def appendray(points, scanpos, time):
    """"
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

def merge_pcs(pcs):
    # set out to pc[0] first because merging empty pc gives error
    out = pcs[0]
    for pc in pcs[1:]:
        # add to combined pc
        out = out.append(pc)
    return out

def pc2rc(pos_dict, point_dict, out_dir, args):
    """
    Converts pointclouds into rayclouds, and merges into one big .ply conforming to raycloudtools formatting

    Parameters
    ----------
    pos_dict
        dictionary with scanner positions: {"ScanPosXX" -> [x, y, z]}
    point_dict
        dictionary with converted rxp pointclouds: {"ScanPosXX" -> [[x, y, z, ...], ...]}
    out_dir
        path to output directory
    args
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    pcs = []
    for pos in pos_dict:
        if pos not in point_dict:
            # print(f"Warning: points of scan position {pos} not found. Skipping")
            continue
        pcd = appendray(point_dict[pos], scanposfromdat(pos_dict[pos]), 1)
        # downsampling is done before merging, as otherwise normals (aka rays) are averaged in voxel
        if (args.resolution):
            pcd = pcd.voxel_down_sample(args.resolution)
        pcs.append(pcd)
        if(args.debug):
            # for debugging: also write single point clouds
            pos_dir = out_dir + "/pos/"
            if not os.path.exists(pos_dir):
                os.makedirs(pos_dir)
            filename = pos_dir+pos+"_raycloud.ply"
            o3d.t.io.write_point_cloud( filename, pcd, write_ascii=False, compressed=False, print_progress=False)
    
    # downsample and merge point clouds
    out_pc = merge_pcs(pcs)
    o3d.t.io.write_point_cloud(out_dir + "merged_raycloud.ply", out_pc, write_ascii=False, compressed=False, print_progress=False)
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("-r", "--resolution", type=float)
    print("")

    args = parser.parse_args()

    if not os.path.exists(args.project):
        print("couldnt find folder")
        os._exit(1)

    pos_dict = read_dat(args.project)
    
    # change if using dat files
    # extension = "CSV"
    # result = glob.glob(args.project + '*.{}'.format(extension))

    # if len(result) > 1 and extension == '.csv':
    #     print("more then one csv file found, aborting")
    #     os._exit(1)

    # # change if using dat files
    # pos_dict = read_csv(result[0])


    print("Reading and converting rxp files")
    t = time.process_time()
    point_dict = read_rxps(args.project, pos_dict)
    t2 = time.process_time()
    print(f"Read and converted rxps in {(t2 - t):.2f} seconds")


    # get output folder from project name
    if (args.project[-1].endswith("/")):
        args.project = args.project[:len(args.project) -1]
    out_dir = "out/" + os.path.splitext(os.path.basename(args.project))[0] + "/"

    if (args.resolution):
        print(f"Downsampling to resolution {args.resolution:.2f} and converting pointclouds to rayclouds")
    else:
        print("Converting raycloud without downsampling")

    t = time.process_time()
    pc2rc(pos_dict, point_dict, out_dir, args)
    t2 = time.process_time()
    print(f"Converted pointclouds to rayclouds in {(t2 - t):.2f} seconds")

    return


if __name__ == "__main__":
    main()