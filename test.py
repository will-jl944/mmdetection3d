import numpy as np
import open3d

viewer = open3d.visualization.Visualizer()
viewer.create_window()
pcd = open3d.io.read_point_cloud('/Users/lujiafeng/workspace/dev/1623773905.3447_d919397e-264e-4730-90a9-bf662abf0262/velodyne_points/1622593775514184560-NOCOMP.pcd')
viewer.add_geometry(pcd)
opt = viewer.get_render_option()
opt.show_coordinate_frame = True
opt.background_color = np.asarray([0.5, 0.5, 0.5])
viewer.run()
