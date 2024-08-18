from geomdl import BSpline
from geomdl.visualization import VisMPL
from geomdl import multi, operations
from geomdl import exchange
from geomdl import fitting
from matplotlib import cm
import numpy as np
import os

# Control points
ctrlpts = [[[-105.0, -105.0, -10.0],  [-105.0, -75.0, -10.0], [-105.0, -45.0, -5.0], [-105.0, -15.0, 0.0], [-105.0, 15.0, 0.0], [-105.0, 45.0, -5.0], [-105.0, 75.0, -10.0], [-105.0, 105.0, -10.0]],
           [[-75.0,  -105.0, -10.0],  [-75.0,  -75.0, -10.0], [-75.0,  -45.0, -5.0], [-75.0,  -15.0, 0.0], [-75.0,  15.0, 0.0], [-75.0,  45.0, -5.0], [-75.0,  75.0, -10.0], [-75.0,  105.0, -10.0]],
           [[-45.0,  -105.0, -8.0],   [-45.0,  -75.0, -8.0],  [-45.0,  -45.0,  4.0], [-45.0,  -15.0, 4.0], [-45.0,  15.0, 4.0], [-45.0,  45.0,  4.0], [-45.0,  75.0, -8.0],  [-45.0,  105.0, -8.0]],
           [[-15.0,  -105.0, -3.0],   [-15.0,  -75.0,  3.0],  [-15.0,  -45.0,  3.0], [-15.0,  -15.0, 8.0], [-15.0,  15.0, 8.0], [-15.0,  45.0,  3.0], [-15.0,  75.0, -5.0],  [-15.0,  105.0, -5.0]],
           [[ 15.0,  -105.0, -3.0],   [ 15.0,  -75.0, -3.0],  [ 15.0,  -45.0,  2.0], [ 15.0,  -15.0, 8.0], [ 15.0,  15.0, 8.0], [ 15.0,  45.0,  2.0], [ 15.0,  75.0, -3.0],  [ 15.0,  105.0, -3.0]],
           [[ 45.0,  -105.0, -8.0],   [ 45.0,  -75.0, -8.0],  [ 45.0,  -45.0,  4.0], [ 45.0,  -15.0, 4.0], [ 45.0,  15.0, 4.0], [ 45.0,  45.0,  4.0], [ 45.0,  75.0, -8.0],  [ 45.0,  105.0, -8.0]],
           [[ 75.0,  -105.0, -10.0],  [ 75.0,  -75.0, -5.0],  [ 75.0,  -45.0, -5.0], [ 75.0,  -15.0, 2.0], [ 75.0,  15.0, 2.0], [ 75.0,  45.0, -5.0], [ 75.0,  75.0, -10.0], [ 75.0,  105.0, -10.0]],
           [[ 105.0, -105.0, -10.0],  [ 105.0, -75.0, -5.0],  [ 105.0, -45.0, -5.0], [ 105.0, -15.0, 2.0], [ 105.0, 15.0, 2.0], [ 105.0, 45.0, -5.0], [ 105.0, 75.0, -10.0], [ 105.0, 105.0, -10.0]]]

surf_1 = BSpline.Surface() # Create a BSpline surface

# Set degrees
surf_1.degree_u = 3
surf_1.degree_v = 3

surf_1.ctrlpts2d = ctrlpts # Set control points

# Set knot vectors
surf_1.knotvector_u = [0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
surf_1.knotvector_v = [0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]

surf_2 = fitting.approximate_surface([300, 0, 0] + np.array(ctrlpts).reshape(64,3), size_u=8, size_v=8,
                        degree_u=3, degree_v=3, centripetal=False, ctrlpts_size_u=6, ctrlpts_size_v=6)

surf_3 = fitting.approximate_surface([300, -150, 0] + np.array(surf_2.ctrlpts).reshape(36,3), size_u=6, size_v=6,
                        degree_u=4, degree_v=4, centripetal=False, ctrlpts_size_u=5, ctrlpts_size_v=5)

surf_trans_2 = BSpline.Surface()
ctrlpts_size_u_trans_2 = surf_2.ctrlpts_size_u
ctrlpts_size_v_trans_2 = surf_2.ctrlpts_size_v

surf_trans_2.degree_u = surf_2.degree_u
surf_trans_2.degree_v = surf_2.degree_v
surf_trans_2.ctrlpts_size_u = ctrlpts_size_u_trans_2
surf_trans_2.ctrlpts_size_v = ctrlpts_size_v_trans_2
surf_trans_2.knotvector_u = surf_2.knotvector_u
surf_trans_2.knotvector_v = surf_2.knotvector_v

ctrlpts_trans_2 = np.zeros((ctrlpts_size_u_trans_2, ctrlpts_size_v_trans_2, 3))
for i_1 in range(0, ctrlpts_size_u_trans_2, 1):
        for j_1 in range(0, ctrlpts_size_v_trans_2, 1):
            ctrlpts_trans_2[i_1][j_1][:] = np.array(surf_3.derivatives(i_1 / (ctrlpts_size_u_trans_2 - 1),
                                                          j_1 / (ctrlpts_size_v_trans_2 - 1))).reshape(-1)

ctrlpts_trans_2 = [-300, -150, 0] + ctrlpts_trans_2
surf_trans_2.ctrlpts2d = ctrlpts_trans_2.tolist()

surf_trans_1 = BSpline.Surface()
ctrlpts_size_u_trans_1 = surf_1.ctrlpts_size_u
ctrlpts_size_v_trans_1 = surf_1.ctrlpts_size_v

surf_trans_1.degree_u = surf_1.degree_u
surf_trans_1.degree_v = surf_1.degree_v
surf_trans_1.ctrlpts_size_u = ctrlpts_size_u_trans_1
surf_trans_1.ctrlpts_size_v = ctrlpts_size_v_trans_1
surf_trans_1.knotvector_u = surf_1.knotvector_u
surf_trans_1.knotvector_v = surf_1.knotvector_v

ctrlpts_trans_1 = np.zeros((ctrlpts_size_u_trans_1, ctrlpts_size_v_trans_1, 3))
for i_2 in range(0, ctrlpts_size_u_trans_1, 1):
        for j_2 in range(0, ctrlpts_size_v_trans_1, 1):
            ctrlpts_trans_1[i_2][j_2][:] = np.array(surf_trans_2.derivatives(i_2 / (ctrlpts_size_u_trans_1 - 1),
                                                          j_2 / (ctrlpts_size_v_trans_1 - 1))).reshape(-1)

ctrlpts_trans_1 = [-300, 0, 0] + ctrlpts_trans_1
surf_trans_1.ctrlpts2d = ctrlpts_trans_1.tolist()

surf_list = [surf_1, surf_2, surf_3, surf_trans_2, surf_trans_1]
for surf in surf_list:
    surf.delta = 0.03 # Set evaluation delta

surface_container = multi.SurfaceContainer(surf_list) # Add surface list to the container
surface_container.vis = VisMPL.VisSurface()
surface_container.render()

if not os.path.exists('data_multi_resolution'):
        os.mkdir('data_multi_resolution')

exchange.export_json(surface_container, 'data_multi_resolution/figure_multi_resolution.json')

os.system('start ""/d "D:\\CODE_AMRTO" /wait "json2on.exe" "data_multi_resolution/figure_multi_resolution.json"')

for surf in surf_list:

    surf.evaluate() # Evaluate surface points

    # Plot the control points grid and the evaluated surface
    surf.vis = VisMPL.VisSurface()
    surf.render(colormap=cm.cool)