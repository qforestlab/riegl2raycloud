import open3d as o3d
import numpy as np
device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32
dtype64 = o3d.core.float64
pcd = o3d.t.geometry.PointCloud(device)
# Array with x, y, z positions
positions = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
pcd.point.positions = o3d.core.Tensor( [[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype64, device)
n_points = len(positions) 
# Single scan position coordinates
scanpos = [[5, 5, 5]]
scanpos = np.repeat(scanpos, n_points, axis=0)
pcd.point.normals = o3d.core.Tensor( scanpos, dtype, device)
# Random time
time = [[0]]
time = np.repeat(time, n_points, axis=0)
pcd.point.time = o3d.core.Tensor( time, dtype, device) # Save point cloud to file
filename = 'pcl_with_pos_as_normal.ply'
o3d.t.io.write_point_cloud( filename, pcd, write_ascii=True, compressed=False, print_progress=False, )