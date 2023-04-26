from typing import Union
import os
import math as m
import numpy as np
import matplotlib.pyplot as plt

import open3d as o3d

from boundingrectangle import boundingrectangle


def tile_from_corner_points(corners, pc: Union[str, o3d.geometry.PointCloud, o3d.t.geometry.PointCloud], size: int = 10, buffer:int = None, exact_size: bool = False, visualization: bool = False, out_dir: str = None):
    """
    Tiles pointcloud (on x,y) into tiles of given size, given xy coordinates of bounding rectangle
    The bool exact_size indicates wether exact given size is used: 
        if True exact squares will be made and any left over parts will be smaller tiles.
        if False the closest length to divide each side into exact squares will be used.
    Parameters
    ----------
    corners
        2D np array with 4 xy coordinates
    pc
        May be string with path to pointcloud, or o3d.(t) PointCloud object
    size
        Size of one side of tile in metres 
    (optional) buffer
        Buffer in meters to add to each tile border
    (optional) exact_size
        Wether to use exact given tile size
    (optional) visualization
        Visualization of tiles
    (optional) out_dir
        out_dir to save visualization of tiles
    """
    if isinstance(pc, str):
        if not os.path.exists(pc):
            print(f"Can't find file at location {pc}")
            return
        pcd = o3d.io.read_point_cloud(pc)
    elif isinstance(pc, o3d.geometry.PointCloud) or isinstance(pc, o3d.t.geometry.PointCloud):
        pcd = pc
    else:
        print(f"Can't read pointcloud {pc}")
        return
    if not buffer:
        buffer = 0
    # get xy matrix to rotate rectangle
    y_min_crnr = corners[np.argmin(corners, axis=0)[1]]
    x_max_crnr = corners[np.argmax(corners, axis=0)[0]]
    theta = m.atan2((x_max_crnr[1]-y_min_crnr[1] ), (x_max_crnr[0] - y_min_crnr[0]))
    rot_matr_2D = np.array([[m.cos(theta), -m.sin(theta)], [m.sin(theta), m.cos(theta)]])

    # rotated xy bounding box
    rotated_corners = np.matmul(corners, rot_matr_2D)

    # tile rotated bounding box
    tiles = create_tiles_aligned_corners(rotated_corners, size=size, buffer=buffer, exact=exact_size)
    inv_rotation = rot_matr_2D.T

    tiled_pcs = []

    rotated_tiles = []

    for tile in tiles: 
        # rotate tile back
        rotated_tile = np.matmul(tile, inv_rotation)
        rotated_tiles.append(rotated_tile)

        # create bounding box from rotated tile
        LARGE_Z = 1000000

        top_corners = np.hstack((rotated_tile, np.asarray([[LARGE_Z]]*4)))
        bot_corners = np.hstack((rotated_tile, np.asarray([[-LARGE_Z]]*4)))

        if isinstance(pcd, o3d.geometry.PointCloud):
            tile_corners_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack((top_corners, bot_corners))))
        elif isinstance(pcd, o3d.t.geometry.PointCloud):
            tile_corners_pc = o3d.t.geometry.PointCloud(o3d.core.Tensor(np.vstack((top_corners, bot_corners))))
        else:
            return None

        tile_crop_bbox = tile_corners_pc.get_oriented_bounding_box()

        cropped_pc = pcd.crop(tile_crop_bbox)

        tiled_pcs.append(cropped_pc)
    
    if (visualization or out_dir) :
        visualize_rectangles([corners, *tiles, rotated_corners, *rotated_tiles], visualization=visualization, out_dir=out_dir)

    return tiled_pcs

def create_tiles_aligned_corners(corner_points, size, buffer, exact):
    """
    Tiles rectangle with tiles of given size and given overlap
    NOTE: assumes corner_points are aligned with xy
    
    The bool exact_size indicates wether exact given size is used: 
        if True exact squares will be made and any left over parts will be smaller tiles.
        if False the closest length to divide each side into exact squares will be used.

    Returns
    ----------
    tiles:
        list of xy corner points in format [[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]]
    """

    # get min and max x and y
    max = np.amax(corner_points, axis=0)[:2]
    min = np.amin(corner_points, axis=0)[:2]

    lengths = max - min

    tiles = [] # format: each tile: np.array of [[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]]
    
    if exact:
        # slice into tiles of exact given length, then append smaller leftover tiles
        n_tiles = np.ceil((lengths - buffer) / ( size - buffer ))
        # bottom left corners
        x_crnrs = [(min[0] + i*(size-buffer)) for i in range(int(n_tiles[0]))]
        y_crnrs = [(min[1] + i*(size-buffer)) for i in range(int(n_tiles[1]))]
        for i in range(len(x_crnrs)-1):
            for j in range(len(y_crnrs)-1):
                tiles.append([[x_crnrs[i],y_crnrs[j]], [x_crnrs[i],y_crnrs[j+1]],[x_crnrs[i+1],y_crnrs[j]],[x_crnrs[i+1],y_crnrs[j+1]]])
        # append all small boxes left on y side
        for i in range(len(x_crnrs)-1):
            tiles.append([[x_crnrs[i],y_crnrs[-1]], [x_crnrs[i],max[1]],[x_crnrs[i+1],y_crnrs[-1]],[x_crnrs[i+1], max[1]]])
        # append all small boxes left on x side
        for j in range(len(y_crnrs)-1):
            tiles.append([[x_crnrs[-1],y_crnrs[j]], [x_crnrs[-1],y_crnrs[j+1]],[max[0],y_crnrs[j]],[max[0],y_crnrs[j+1]]])
        # append smallest box with xy leftover
        tiles.append([[x_crnrs[-1],y_crnrs[-1]], [x_crnrs[-1],max[1]],[max[0],y_crnrs[-1]],[max[0],max[1]]])
    
    else:
        # use size where we get exact same rectangles for each tile
        # recalculate size of tiles in each direction
        n_tiles = (lengths - buffer) / ( size - buffer )
        # can't divide into 0 tiles
        n_tiles = np.where(n_tiles == 0, 1, n_tiles)
        sizes = (lengths - buffer) / np.round(n_tiles) + buffer
        print(f"Actual sizes used: x: {sizes[0]}, y: {sizes[1]}")
        # bottom left corners
        x_crnrs = [(min[0] + i*(sizes[0]-buffer)) for i in range(int(np.round(n_tiles[0])))]
        y_crnrs = [(min[1] + i*(sizes[1]-buffer)) for i in range(int(np.round(n_tiles[1])))]
        for i in range(len(x_crnrs)):
            for j in range(len(y_crnrs)):
                tiles.append([[x_crnrs[i],y_crnrs[j]], [x_crnrs[i],y_crnrs[j] + sizes[1]],[x_crnrs[i] + sizes[0],y_crnrs[j]],[x_crnrs[i] + sizes[0],y_crnrs[j] + sizes[1]]])

    return tiles

def visualize_rectangles(rectangles, visualization:bool = False, out_dir: str = None):
    plt.figure()
    plt.axis('equal')
    colors = ['r', 'b', 'g', 'k', 'c', 'm', 'y']
    for j, rectangle in enumerate(rectangles):
        for point in rectangle:
            plt.plot(point[0], point[1], 'ro')
        # create edges: sort based on distance with one of points, two closest points are neighbours
        anchor = rectangle[0]
        sorted_corners = sorted(rectangle[1:], key = lambda x : m.sqrt((x[0]-anchor[0])**2 + (x[1]-anchor[1])**2))
        # connect anchor with 2 closest points
        plt.plot([anchor[0], sorted_corners[0][0]], [anchor[1], sorted_corners[0][1]], '-', color=colors[j % len(colors)])
        plt.plot([anchor[0], sorted_corners[1][0]], [anchor[1], sorted_corners[1][1]], '-', color=colors[j % len(colors)])
        # connect farthest point with 2 other points
        plt.plot([sorted_corners[2][0], sorted_corners[0][0]], [sorted_corners[2][1], sorted_corners[0][1]], '-', color=colors[j % len(colors)])
        plt.plot([sorted_corners[2][0], sorted_corners[1][0]], [sorted_corners[2][1], sorted_corners[1][1]], '-', color=colors[j % len(colors)])

        min_x, min_y = np.amin(rectangle, axis=0)
        max_x, max_y = np.amax(rectangle, axis=0)

        plt.annotate("T" + str(j), xy= ((max_x + min_x)/2, (max_y + min_y)/2))
    if out_dir:
        if not os.path.exists(out_dir):
                print("Cant find location {out_dir} to save Tiles")
        else:
            plt.savefig(os.path.join(out_dir,"Tiles.png"))
    if visualization:
        plt.show()