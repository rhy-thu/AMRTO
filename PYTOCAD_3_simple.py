# Programmed by: Hongyuan Ren
# School of Aerospace Engineering, Tsinghua University, Beijing, 100084
# Development Time: 2024/01/01
# Python 3.8
import time
import os
import numpy as np  # Version: 1.22.2
from geomdl import exchange  # geomdl Version: 5.3.1 (NURBS-Python)
from geomdl import multi
from geomdl import fitting
from joblib import Parallel, delayed  # Version: 1.4.2
from numba import jit  # Version: 0.56.0

class Node:
    # Define the node class
    def __init__(self, coordinates, index):
        # The constructor or initialization method of class, self represents the instance of the class.
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.z = coordinates[2]
        # The index position in the node list
        self.index = index
        # map_point records the coordinates after mapping to the unit square.
        self.map_point = [0, 0]

    def set_map_point(self, coordinate):
        self.map_point = coordinate

@jit()
def calc_length(node1_x,node1_y,node1_z,node2_x,node2_y,node2_z):
    # Calculate the distance between two points
    return np.sqrt((node1_x - node2_x) * (node1_x - node2_x) + (node1_y - node2_y) * (node1_y - node2_y) +
                     (node1_z - node2_z) * (node1_z - node2_z))

@jit()
def is_in_triangle(map_points_a, map_points_b, map_points_c, p):
    # Determine whether the point p is in the triangle.
    a, b, c = map_points_a, map_points_b, map_points_c
    ab = [b[0] - a[0], b[1] - a[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    ac = [c[0] - a[0], c[1] - a[1]]
    s_abc = abs((ab[0] * bc[1] - ab[1] * bc[0]) / 2)
    if s_abc == 0.0:
        return False, 0, 0, 0
    else:
        bp = [p[0] - b[0], p[1] - b[1]]
        cp = [p[0] - c[0], p[1] - c[1]]
        s_abp = abs((ab[0] * bp[1] - ab[1] * bp[0]) / 2)
        s_acp = abs((ac[0] * cp[1] - ac[1] * cp[0]) / 2)
        s_bcp = abs((bc[0] * cp[1] - bc[1] * cp[0]) / 2)
        sum_other = s_abp + s_acp + s_bcp
        return abs(s_abc - sum_other) <= 0.005 * abs(sum_other), s_bcp / s_abc, s_acp / s_abc, s_abp / s_abc

@jit()
def construct_grid(d_u, d_v):
    mesh_quad_index = list()
    for i_0 in range(d_u):
        for j_0 in range(d_v):
            mesh_quad_index.append((i_0 * (d_v + 1) + j_0,
                                    (i_0 + 1) * (d_v + 1) + j_0,
                                    (i_0 + 1) * (d_v + 1) + j_0 + 1,
                                    i_0 * (d_v + 1) + j_0 + 1))

    mesh_point = list()
    for i in [item / d_u for item in range(d_u + 1)]:
        for j in [item / d_v for item in range(d_v + 1)]:
            mesh_point.append((j, i))

    return mesh_point, mesh_quad_index

def write_3dm():
    os.system('start ""/d "D:\\CODE_AMRTO" /wait "json2on.exe" "data_3/SINGLE_HM_NURBS_surface_total.json"')

def quad_obj_load_vt_result(obj_path):
    points, pts_vt, quad_index, quad_vt, simplify_points = [], [], [], [], []
    with open(obj_path) as file:
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append([float(strs[1]), float(strs[2]), float(strs[3])])
            if strs[0] == "vt":
                pts_vt.append([float(strs[1]), float(strs[2])])
            if strs[0] == "f":
                quad_index_temp_list = [-1E5, -1E5, -1E5, -1E5]
                quad_vt_temp = [-1E5, -1E5, -1E5, -1E5]
                for i in range(1, len(strs), 1):
                    quad_index_temp_list[i - 1] = int(
                        strs[i].split("/")[0]) - 1  # Python count starts from 0 instead of 1
                    quad_vt_temp[i - 1] = int(strs[i].split("/")[1]) - 1  # Python count starts from 0 instead of 1
                quad_index.append(quad_index_temp_list)
                quad_vt.append(quad_vt_temp)
    quad_index, quad_vt, points, pts_vt = np.array(quad_index), np.array(quad_vt), np.array(points), np.array(pts_vt)

    return points, pts_vt, quad_index, quad_vt

def load_obj_vt(mesh_vertices, pts_vt):
    lines_vertices = list(mesh_vertices)
    # the nodes(points) data
    nodes = list()

    for index, line in enumerate(lines_vertices):
        node = Node([item for item in line], index)  # Node type defined in Data_structure
        node.set_map_point(pts_vt[index])
        nodes.append(node)

    return nodes

def vt_adjust(simple_vertices, simple_vertices_result, pts_vt_result, simple_patch_index_result, quad_vt_result,
              k_sample, d_min, d_max, k_type_gemodl, NURBS_degree_u, NURBS_degree_v, num_ctrlpts_min):
    pts_vt = -1E3 * np.ones((len(simple_vertices), 2))
    dict_pt = {}

    for i0 in range(0, len(simple_vertices), 1):
        ver_temp = simple_vertices[i0]
        dict_pt[i0] = np.where((simple_vertices_result == ver_temp).all(1))[0]
        # Point cloud registration, the key is the index value of the point in simple_vertices, and the value is the
        # index value of the point in simple_vertices_result. In order to improve the computational efficiency, == is
        # used for registration here, but it should be noted that if the calculation accuracy is not enough, the point
        # cloud registration is performed because the spatial distance between two points is less than a
        # given small value (such as 1e-9).
    for i0 in range(0, len(simple_vertices), 1):
        j0 = dict_pt[i0]
        pt_array_temp = []
        for k in quad_vt_result[np.where(simple_patch_index_result == j0)]:
            pt_array_temp.append(pts_vt_result[k])
        pt_array_temp_unique = np.unique(pt_array_temp, axis=0)
        if len(pt_array_temp_unique) == 1:
            pts_correct = pt_array_temp_unique
        else:
            special_patch = simple_patch_index_result[np.where(simple_patch_index_result == j0)[0]]
            for m0 in range(0, len(special_patch), 1):
                if special_patch[m0][0] in dict_pt.values() and special_patch[m0][1] in dict_pt.values() and \
                        special_patch[m0][2] in dict_pt.values() and special_patch[m0][3] in dict_pt.values():
                    pts_correct = pt_array_temp[m0]
                    break
        pts_vt[i0] = pts_correct
    u_max_bound = max(pts_vt[:, 0])
    u_min_bound = min(pts_vt[:, 0])
    v_max_bound = max(pts_vt[:, 1])
    v_min_bound = min(pts_vt[:, 1])
    for i0 in range(0, len(pts_vt), 1):
        pts_vt[i0] = [min(1.0, max((pts_vt[i0][0] - u_min_bound) / (u_max_bound - u_min_bound), 0.0)),
                      min(1.0, max((pts_vt[i0][1] - v_min_bound) / (v_max_bound - v_min_bound), 0.0))]
    d_u = int(k_sample * 200 * (v_max_bound - v_min_bound))
    d_v = int(k_sample * 200 * (u_max_bound - u_min_bound))
    d_u, d_v = min(d_max, d_u), min(d_max, d_v)
    d_u, d_v = max(d_min, d_u), max(d_min, d_v)
    num_ctrlpts_u = min(int(k_type_gemodl * 200 * (v_max_bound - v_min_bound)), int(0.8 * d_u) - 1 - NURBS_degree_u)
    num_ctrlpts_v = min(int(k_type_gemodl * 200 * (u_max_bound - u_min_bound)), int(0.8 * d_v) - 1 - NURBS_degree_v)
    num_ctrlpts_u, num_ctrlpts_v = max(num_ctrlpts_min, NURBS_degree_u + 1, num_ctrlpts_u), \
                                   max(num_ctrlpts_min, NURBS_degree_v + 1, num_ctrlpts_v)

    return [pts_vt, d_u, d_v, num_ctrlpts_u, num_ctrlpts_v]

def map_to_space(nodes, triangles, d_u, d_v):
    # The mesh points are inversely mapped back to the space surface
    quad_mesh, mesh_quad_index = construct_grid(d_u, d_v)
    mesh_points_3d = dict()
    # Find the triangle corresponding to the mesh point and its coordinates
    for index, mesh_point in enumerate(quad_mesh):
        for triangle in triangles:
            a, b, c = triangle
            map_points_a, map_points_b, map_points_c = nodes[a].map_point, nodes[b].map_point, nodes[c].map_point
            if mesh_point[0] < min(map_points_a[0],map_points_b[0],map_points_c[0]) - 0.01 or mesh_point[0] > max(map_points_a[0],map_points_b[0],map_points_c[0]) + 0.01:
                continue
            elif mesh_point[1] < min(map_points_a[1],map_points_b[1],map_points_c[1]) - 0.01 or mesh_point[1] > max(map_points_a[1],map_points_b[1],map_points_c[1]) + 0.01:
                continue
            else:
                res, p1, p2, p3 = is_in_triangle(map_points_a, map_points_b, map_points_c, mesh_point)
                if res:
                    point_3d_x = nodes[a].x * p1 + nodes[b].x * p2 + nodes[c].x * p3
                    point_3d_y = nodes[a].y * p1 + nodes[b].y * p2 + nodes[c].y * p3
                    point_3d_z = nodes[a].z * p1 + nodes[b].z * p2 + nodes[c].z * p3
                    mesh_points_3d[index] = [point_3d_x,point_3d_y,point_3d_z]
                    break

    return mesh_points_3d, mesh_quad_index

def quad_obj_trans_load_vt(obj_path):
    points_quad = []
    quad_index = []
    with open(obj_path) as file:
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points_quad.append(np.array([float(strs[1]), float(strs[2]), float(strs[3])]))
            if strs[0] == "f":
                quad_index_temp_list = [-1E5, -1E5, -1E5, -1E5]
                for i in range(1, len(strs), 1):
                    quad_index_temp_list[i - 1] = int(strs[i].split("/")[0])
                quad_index.append(quad_index_temp_list)
    quad_index_array = np.array(quad_index)
    tri_index_improve = []
    if (type(quad_index_array).__name__ == 'list'):
        quad_index_array = np.array(quad_index_array)

    for i in range(0, len(quad_index_array), 1):
        quad_index_temp = quad_index_array[i]
        tri_index_improve.append([int(quad_index_temp[0]), int(quad_index_temp[1]), int(quad_index_temp[2])])
        tri_index_improve.append([int(quad_index_temp[0]), int(quad_index_temp[2]), int(quad_index_temp[3])])


    tri_index_improve = np.array(tri_index_improve)
    tri_index_improve = tri_index_improve - 1
    quad_index, points = tri_index_improve, points_quad
    index_vertices_effective = np.array(quad_index).flatten().tolist()
    index_vertices_effective = list(set(index_vertices_effective))
    num_index_vertices_effective = len(index_vertices_effective)
    simple_patch_index = np.zeros((len(quad_index), 3))
    simple_vertices = np.zeros((num_index_vertices_effective, 3))
    dict_index_node = {}  # Transformation dict for node indexes,the key is the old index of the node and the value is the new index

    for i in range(0, num_index_vertices_effective, 1):
        dict_index_node[index_vertices_effective[i]] = i
        simple_vertices[i, :] = points[index_vertices_effective[i]]

    for j in range(0, len(quad_index), 1):
        simple_patch_index[j, 0:3] = [dict_index_node[quad_index[j, 0]], dict_index_node[quad_index[j, 1]],
                                      dict_index_node[quad_index[j, 2]]]
    simple_patch_index = simple_patch_index.astype(int)

    return simple_vertices, simple_patch_index

def CAD_reconstruction(number_quad, simple_vertices_result, pts_vt_result, simple_patch_index_result, quad_vt_result):
    k_type_gemodl = 1.2  # (kt) The coefficient of the number of control points in the single parameter direction
    # during NURBS fitting, the recommended range: [0.1, 0.7]. The larger the value, the more the number of control
    # points during fitting, and the higher the degree of surface bending.
    k_sample = 1  # (ks) The specified sampling rate coefficient when the remeshing is performed by the
    # mapping method. This value should not be too small, and different patches will be adaptively sampled.
    d_min = 10  # (mi) The minimum value of the specified sampling rate when the remeshing is performed
    # by the  mapping method. This value cannot be too low, the recommended range: [10,100].The greater the value is,
    # the higher the fitting accuracy relative to the original surface is, and the longer the time is.
    d_max = 2000  # (ma) The maximum value of the specified sampling rate when the mapping method is remeshing

    num_ctrlpts_min = 6  # (ct) The minimum number of control points in the single parameter direction during NURBS fitting
    NURBS_degree_u = 3  # (de) The order of NURBS patches along the u-parameter direction
    NURBS_degree_v = 3  # (de) The order of NURBS patches along the v-parameter direction
    file_path = 'data_3'  # The relative path of reading and storing files
    print(number_quad, end='\n')

    file_name = file_path + '/output_quad/quad_' + str(int(number_quad)) + '.obj'
    # The quad mesh model is used as input.
    [simple_vertices, simple_patch_index] = quad_obj_trans_load_vt(file_name)

    [pts_vt, d_v, d_u, num_ctrlpts_u, num_ctrlpts_v] = vt_adjust(simple_vertices,
                                simple_vertices_result, pts_vt_result, simple_patch_index_result,
                                quad_vt_result, k_sample, d_min, d_max, k_type_gemodl,
                                NURBS_degree_u, NURBS_degree_v, num_ctrlpts_min)

    nodes = load_obj_vt(simple_vertices, pts_vt)

    mesh_points_3d, mesh_quad_index = map_to_space(nodes, simple_patch_index, d_v, d_u)
    X_NURBS = np.zeros((len(mesh_points_3d), 3))

    for key in mesh_points_3d:
        X_NURBS[key] = mesh_points_3d[key]

    size_u, size_v = d_u + 1, d_v + 1

    NURBS_patch = fitting.approximate_surface(X_NURBS, size_v, size_u, NURBS_degree_v, NURBS_degree_u,
                                              centripetal=False, ctrlpts_size_u=num_ctrlpts_u, ctrlpts_size_v=num_ctrlpts_v)
    # ctrlpts_size_u: The number of control points in the direction of u parameter

    return NURBS_patch

if __name__ == '__main__':
    time_begin = time.time()
    number_quad_list = list(range(0, 67, 1))  # The scope of the reconstructed model
    number_quad_list = np.unique(number_quad_list)
    file_name_result = 'data_3/result_cantilever_deformed_quad_rhino.obj'  # A file containing texture coordinates and packaging textures derived from the generalized motorcycle graph method
    [simple_vertices_result, pts_vt_result, simple_patch_index_result, quad_vt_result] = \
        quad_obj_load_vt_result(file_name_result)
    surface = list(Parallel(n_jobs=24)(delayed(CAD_reconstruction)(i,simple_vertices_result,
        pts_vt_result, simple_patch_index_result, quad_vt_result) for i in number_quad_list))  # Parallel
    # The process of parallelization will result in an error in the order of printing results. In order to
    # obtain accurate print results, it is possible to set the n_jobs value to 1 (although this is not recommended).
    surface_total = multi.SurfaceContainer(surface)
    surface_total.delta = 0.04
    exchange.export_json(surface_total, 'data_3/SINGLE_HM_NURBS_surface_total.json')
    write_3dm()

    time_end = time.time()
    print('The total time consumption is ', f'{(time_end - time_begin):.2f}', 'seconds.')