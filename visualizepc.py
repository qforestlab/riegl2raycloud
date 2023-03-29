import open3d as o3d
from typing import Union
import argparse
import os


def visualizeo3d(pc: Union[str, o3d.geometry.PointCloud, o3d.t.geometry.PointCloud]):
    """
    Visualizes pointcloud
    Parameters
    ----------
    pc
        May be string with path to pointcloud, or o3d.(t) PointCloud object
    """

    # check type
    if isinstance(pc, str):
        if not os.path.exists(pc):
            print(f"Can't find file at location {pc}")
            return
        pcd = o3d.io.read_point_cloud(pc)
    elif isinstance(pc, o3d.geometry.PointCloud) or isinstance(pc, o3d.t.geometry.PointCloud):
        pcd = pc

    # Visualize point clouds
    o3d.visualization.draw_geometries([pcd])

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pointcloud", type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.pointcloud):
        print("couldnt find pointcloud")
        os._exit(1)
    
    visualizeo3d(args.pointcloud)