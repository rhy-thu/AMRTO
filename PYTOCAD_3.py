# Programmed by: Hongyuan Ren
# School of Aerospace Engineering, Tsinghua University, Beijing, 100084
# Development Time: 2024/01/01
# Python 3.8
import time
import open3d as o3d  # Version: 0.17.0
import os
import numpy as np  # Version: 1.22.2
import matplotlib.pyplot as plt  # Version: 3.5.1
import math
from geomdl import exchange  # geomdl Version: 5.3.1 (NURBS-Python)
from geomdl import multi
from geomdl.visualization import VisMPL
from geomdl import fitting
from pyinstrument import Profiler  # Version: 4.5.3
from joblib import Parallel, delayed  # Version: 1.4.2
from numba import jit  # Version: 0.56.0

class Node:
    # Define the node class
    def __init__(self, coordinates, index):
        # The constructor or initialization method of class, self represents the instance of the class.
        # nodal coordinate
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.z = coordinates[2]
        # The index position in the node list
        self.index = index
        # Adjacent nodes, key is the index of adjacent nodes, and value is the index of two or one nodes connected by
        # the node and key.
        self.connection = dict()
        self.path = dict()
        # Sequence is the order of traversing adjacent nodes.
        self.sequence = None
        # Angle records the cot value of the diagonal angle of the edge composed of node and adjacent node,
        # that is, 1/tan(angle).
        self.angle = dict()
        # map_point records the coordinates after mapping to the unit square.
        self.map_point = (0, 0)

    def add_connection(self, index, opposite):
        if index not in self.connection.keys():
            self.connection[index] = list()
        self.connection[index].append(opposite)

    def get_num_of_neighbors(self):  # Get the number of adjacent nodes
        return len(self.connection)

    def set_path(self, path_name, step):
        if path_name not in self.path.keys():
            self.path[path_name] = list()
        self.path[path_name].append(step)

    def set_map_point(self, coordinate):
        self.map_point = coordinate

@jit()
def quad_obj_load(obj_path):
    points = []
    quad_index = []
    with open(obj_path) as file:
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")  # The data read
            if strs[0] == "v":
                points.append(np.array([float(strs[1]), float(strs[2]), float(strs[3])]))
            if strs[0] == "f":
                quad_index_temp_list = [-1E5, -1E5, -1E5, -1E5]
                for i in range(1, len(strs), 1):
                    quad_index_temp_list[i - 1] = int(strs[i].split("/")[0])
                quad_index.append(quad_index_temp_list)

    return points, np.array(quad_index)

@jit()
def triangle_area_3d(cor_1, cor_2, cor_3):
    # Calculate the area of a spatial triangle with vertices cor_1, cor_2, cor_3.
    len_a = math.sqrt((cor_1[0] - cor_2[0]) ** 2 + (cor_1[1] - cor_2[1]) ** 2 + (cor_1[2] - cor_2[2]) ** 2)
    len_b = math.sqrt((cor_1[0] - cor_3[0]) ** 2 + (cor_1[1] - cor_3[1]) ** 2 + (cor_1[2] - cor_3[2]) ** 2)
    len_c = math.sqrt((cor_3[0] - cor_2[0]) ** 2 + (cor_3[1] - cor_2[1]) ** 2 + (cor_3[2] - cor_2[2]) ** 2)
    p0 = (len_a + len_b + len_c) / 2

    return [math.sqrt(abs(p0 * (p0 - len_a) * (p0 - len_b) * (p0 - len_c))), len_a, len_b, len_c]

@jit()
def triangle_area_2d(a, b, c):
    # Calculates the area of a plane triangle with vertices a, b, c.
    ab = (b[0] - a[0], b[1] - a[1])
    bc = (c[0] - b[0], c[1] - b[1])

    return abs((ab[0] * bc[1] - ab[1] * bc[0]) / 2)

@jit()
def calc_length(node1_x,node1_y,node1_z,node2_x,node2_y,node2_z):
    # Calculate the distance between two points

    return math.sqrt((node1_x - node2_x) * (node1_x - node2_x) + (node1_y - node2_y) * (node1_y - node2_y) +
                     (node1_z - node2_z) * (node1_z - node2_z))

@jit()
def is_in_triangle(map_point_a, map_point_b, map_point_c, p):
    # Determine whether the point p is in the triangle.
    a, b, c = map_point_a, map_point_b, map_point_c
    ab = (b[0] - a[0], b[1] - a[1])
    bc = (c[0] - b[0], c[1] - b[1])
    ac = (c[0] - a[0], c[1] - a[1])
    bp = (p[0] - b[0], p[1] - b[1])
    cp = (p[0] - c[0], p[1] - c[1])
    s_abc = abs((ab[0] * bc[1] - ab[1] * bc[0]) / 2)
    if s_abc == 0.0:
        return False, 0, 0, 0
    else:
        s_abp = abs((ab[0] * bp[1] - ab[1] * bp[0]) / 2)
        s_acp = abs((ac[0] * cp[1] - ac[1] * cp[0]) / 2)
        s_bcp = abs((bc[0] * cp[1] - bc[1] * cp[0]) / 2)
        sum_other = s_abp + s_acp + s_bcp
        return abs(s_abc - sum_other) <= 0.005 * abs(sum_other), s_bcp / s_abc, s_acp / s_abc, s_abp / s_abc

@jit()
def construct_grid(d_u, d_v, k_construct_grid):
    mesh_quad_index = list()
    for i_0 in range(d_u):
        for j_0 in range(d_v):
            mesh_quad_index.append((i_0 * (d_v + 1) + j_0,
                                    (i_0 + 1) * (d_v + 1) + j_0,
                                    (i_0 + 1) * (d_v + 1) + j_0 + 1,
                                    i_0 * (d_v + 1) + j_0 + 1))
    if k_construct_grid == 1:
        mesh_point = list()
        for i in [item / d_u for item in range(d_u + 1)]:
            for j in [item / d_v for item in range(d_v + 1)]:
                mesh_point.append((j, i))
    elif k_construct_grid == 2:
        mesh_point = list()
        for i in [0.02 + 0.96 * (item) / (d_u) for item in range(d_u + 1)]:
            for j in [0.02 + 0.96 * (item) / (d_v) for item in range(d_v + 1)]:
                mesh_point.append((j, i))
    else:
        mesh_point = list()
        for i in [(item + 1) / (d_u + 2) for item in range(d_u + 1)]:
            for j in [(item + 1) / (d_v + 2) for item in range(d_v + 1)]:
                mesh_point.append((j, i))

    return mesh_point, mesh_quad_index

def map_to_complex_boundary(nodes, sequence, path):
    # Inversely maps the boundary points back to the layout
    map_point = dict()
    map_point[sequence[0]] = (nodes[path[0]].x, nodes[path[0]].y, nodes[path[0]].z)
    map_point[sequence[-1]] = (nodes[path[-1]].x, nodes[path[-1]].y, nodes[path[-1]].z)
    s = list()
    s.insert(0, calc_length(nodes[path[0]].x, nodes[path[0]].y, nodes[path[0]].z,
                            nodes[path[1]].x, nodes[path[1]].y, nodes[path[1]].z))
    for i in range(1, len(path) - 1):
        s.append(s[i - 1] + calc_length(nodes[path[i]].x, nodes[path[i]].y, nodes[path[i]].z,
                            nodes[path[i + 1]].x, nodes[path[i + 1]].y, nodes[path[i + 1]].z))

    for i in range(1, len(sequence) - 1):
        for j in range(len(s)):
            if s[j] / s[-1] > (i / (len(sequence) - 1)):
                length = calc_length(nodes[path[j]].x, nodes[path[j]].y, nodes[path[j]].z,
                            nodes[path[j + 1]].x, nodes[path[j + 1]].y, nodes[path[j + 1]].z)
                a = s[-1] * i / (len(sequence) - 1)
                if j > 0:
                    a -= s[j - 1]
                lam = a / length
                map_point[sequence[i]] = (nodes[path[j]].x * (1 - lam) + nodes[path[j + 1]].x * lam,
                                          nodes[path[j]].y * (1 - lam) + nodes[path[j + 1]].y * lam,
                                          nodes[path[j]].z * (1 - lam) + nodes[path[j + 1]].z * lam)
                break

    return map_point

def get_num_of_neighbors(nodes):
    for i in range(0, len(nodes), 1):
        num_neigh = nodes[i].get_num_of_neighbors()

def cal_boundary_length(points, triangles):
    # Calculate the number and total length of the edges on the boundary (edge) of a single patch.
    edges, num_edge, boundary_points = [], [], []
    inside_triangles, one_boundary_triangles, two_boundary_triangles, three_boundary_triangles = [], [], [], []
    boundary_length = 0
    for k0 in range(0, len(triangles), 1):
        edge0 = sorted([triangles[k0][0], triangles[k0][1]])
        edge1 = sorted([triangles[k0][0], triangles[k0][2]])
        edge2 = sorted([triangles[k0][1], triangles[k0][2]])
        if edge0 not in edges:
            edges.append(edge0)
            num_edge.append(1)
        else:
            num_edge[edges.index(edge0)] += 1
        if edge1 not in edges:
            edges.append(edge1)
            num_edge.append(1)
        else:
            num_edge[edges.index(edge1)] += 1
        if edge2 not in edges:
            edges.append(edge2)
            num_edge.append(1)
        else:
            num_edge[edges.index(edge2)] += 1

    for i in range(len(num_edge)):
        if num_edge[i] == 1:
            edge_temp = [int(edges[i][0]), int(edges[i][1])]
            boundary_points += edge_temp
            boundary_length += math.sqrt(
                (points[edge_temp[0]][0] - points[edge_temp[1]][0]) * (points[edge_temp[0]][0] -
                                                                       points[edge_temp[1]][0]) + (
                            points[edge_temp[0]][1] - points[edge_temp[1]][1]) *
                (points[edge_temp[0]][1] - points[edge_temp[1]][1]) + (points[edge_temp[0]][2] -
                                                                       points[edge_temp[1]][2]) * (
                            points[edge_temp[0]][2] - points[edge_temp[1]][2]))
    boundary_points_unique = np.unique(boundary_points)
    num_boundary_items = len(boundary_points_unique)
    # Prevent the same side of the triangle in the same patch from being counted multiple times
    for j0 in range(0, len(triangles), 1):
        tri_points_boundary = [triangles[j0][0] in boundary_points_unique, triangles[j0][1] in boundary_points_unique,
                               triangles[j0][2] in boundary_points_unique]
        num_tri_points_boundary = np.count_nonzero(tri_points_boundary)
        # The vertex of the triangular patch is the number of boundary points of the quadrilateral patch.(0~3)
        inside_triangles = triangles

    return [boundary_points_unique, inside_triangles, boundary_points]

def boundary_find(index_multi_region_temp, edge_boundary_temp):
    edge_boundary_mat_temp = (np.array(edge_boundary_temp)).reshape(-1, 2)
    edge_bou_order_temp = list()
    edge_bou_order_temp.append(index_multi_region_temp)
    k_judge = 0
    while (k_judge == 0):
        m_0 = np.argwhere(edge_bou_order_temp[-1] == edge_boundary_mat_temp)[:, 0]
        if len(m_0) > 0:
            m_1 = np.array(edge_boundary_mat_temp[m_0]).flatten().tolist()
            m_1 = np.unique(m_1)
            k_count = 0
            for ele in m_1:
                if ele not in edge_bou_order_temp:
                    edge_bou_order_temp.append(ele)
                    break
                else:
                    k_count += 1
            if k_count >= len(m_1):
                k_judge = 1

    if len(edge_bou_order_temp) != len(edge_boundary_mat_temp):
        print(len(edge_bou_order_temp), len(edge_boundary_mat_temp),
              'The region boundary is not completely traversed. The structural topology may be complex',
              ' and not isomorphic to the disk topology.')

    return edge_bou_order_temp

def write_3dm():
    os.system('start ""/d "D:\\CODE_AMRTO" /wait "json2on.exe" "data_3/SINGLE_HM_NURBS_surface_total.json"')

def Visualization_NURBS(surface_total, k_delta=0.01):
    surface_total.delta = k_delta
    vis_config = VisMPL.VisConfig(bbox=True, figure_size=[10, 6], axes_equal=True)
    surface_total.vis = VisMPL.VisSurface(vis_config)
    surface_total.render()
    """* ``ctrlpts`` (bool): Control points polygon/grid visibility. *Default: True*
    * ``evalpts`` (bool): Curve/surface points visibility. *Default: True*
    * ``bbox`` (bool): Bounding box visibility. *Default: False*
    * ``legend`` (bool): Figure legend visibility. *Default: True*
    * ``axes`` (bool): Axes and figure grid visibility. *Default: True*
    * ``labels`` (bool): Axis labels visibility. *Default: True*
    * ``trims`` (bool): Trim curves visibility. *Default: True*
    * ``axes_equal`` (bool): Enables or disables equal aspect ratio for the axes. *Default: True*
    * ``figure_size`` (list): Size of the figure in (x, y). *Default: [10, 8]*
    * ``figure_dpi`` (int): Resolution of the figure in DPI. *Default: 96*
    * ``trim_size`` (int): Size of the trim curves. *Default: 20*
    * ``alpha`` (float): Opacity of the evaluated points. *Default: 1.0*
    self.dtype = np.float
    self.display_ctrlpts = kwargs.get('ctrlpts', True)
    self.display_evalpts = kwargs.get('evalpts', True)
    self.display_bbox = kwargs.get('bbox', False)
    self.display_legend = kwargs.get('legend', True)
    self.display_axes = kwargs.get('axes', True)
    self.display_labels = kwargs.get('labels', True)
    self.display_trims = kwargs.get('trims', True)
    self.axes_equal = kwargs.get('axes_equal', True)
    self.figure_size = kwargs.get('figure_size', [10, 8])
    self.figure_dpi = kwargs.get('figure_dpi', 96)
    self.trim_size = kwargs.get('trim_size', 20)
    self.alpha = kwargs.get('alpha', 1.0)
    self.figure_image_filename = "temp-figure.png" 
    """

def NURBS_Boundary_gemodl_3D(type_value_points, degree_v, degree_u, size_v, size_u, num_ctrlpts_u,
                             num_ctrlpts_v, type_gemodl=1, k_delta=0.01):
    if type_gemodl == 1:
        surface = fitting.interpolate_surface(type_value_points, size_u, size_v, degree_u, degree_v)
        # interpolating fitting
    elif type_gemodl == 2:
        surface = fitting.approximate_surface(type_value_points, size_u, size_v, degree_u, degree_v, centripetal=False,
                                              ctrlpts_size_u=num_ctrlpts_u, ctrlpts_size_v=num_ctrlpts_v)
        # ctrlpts_size_u: The number of control points in the direction of u parameter
    elif type_gemodl == 3:
        surface = fitting.approximate_surface(type_value_points, size_u, size_v, degree_u, degree_v, centripetal=True)
        # centripetal: Centripetal parameterization method, the default value is False
    elif type_gemodl == 4:
        surface = fitting.approximate_surface(type_value_points, size_u, size_v, degree_u, degree_v, centripetal=True,
                                              ctrlpts_size_u=num_ctrlpts_u, ctrlpts_size_v=num_ctrlpts_v)
    surface.delta = k_delta

    return surface

def NURBSFIT_single(judge_num_NURBS, num_patches, X, size_u, size_v, num_ctrlpts_u,
                    num_ctrlpts_v, NURBS_degree_u=3, NURBS_degree_v=3, type_gemodl=1, k_delta=0.01,
                    export_single_json=1, export_single_obj=1, export_total_json=1, export_file_total_3dm=1,
                    decompose=0):
    surface = NURBS_Boundary_gemodl_3D(X, NURBS_degree_u, NURBS_degree_v, size_u, size_v, num_ctrlpts_u,
                                       num_ctrlpts_v, type_gemodl, k_delta)
    if decompose == 0:
        pass
    elif decompose == 1:
        surface = geomdl.operations.decompose_surface(surface, decompose_dir='u')
    elif decompose == 2:
        surface = geomdl.operations.decompose_surface(surface, decompose_dir='v')
    elif decompose == 3:
        surface = geomdl.operations.decompose_surface(surface, decompose_dir='uv')

    if export_single_json == 1:
        surface.delta = 0.04
        exchange.export_json(surface, 'data_3/SINGLE_HM_NURBS_surface_' + str(judge_num_NURBS) + '.json')
    elif export_single_json == 2:
        surface.delta = k_delta
        exchange.export_json(surface, 'data_3/SINGLE_HM_NURBS_surface_' + str(judge_num_NURBS) + '.json')
    if export_single_obj == 1:
        surface.delta = 0.04
        exchange.export_obj(surface, 'data_3/SINGLE_HM_NURBS_surface_obj_' + str(judge_num_NURBS) + '.obj')
    elif export_single_obj == 2:
        surface.delta = k_delta
        exchange.export_obj(surface, 'data_3/SINGLE_HM_NURBS_surface_obj_' + str(judge_num_NURBS) + '.obj')

    return surface

def adjust_corner_order(pts_index, boundary):
    # Sorts the four corners of the input so that they are: saddle_node_1,saddle_node_2,max_end,min_end,
    # If saddle_node_1 is taken as the starting point, along the boundary path,
    # the order of the corner points is:saddle_node_1,max_end,saddle_node_2,min_end
    # max_end-----(sequences_each_bou[1])-----saddle_node_2
    #       |                                          |
    # (sequences_each_bou[0])               (sequences_each_bou[2])
    #      |                                          |
    # saddle_node_1-----(sequences_each_bou[3])-----min_end
    pts_index_list = [boundary.index(pts_index[1]), boundary.index(pts_index[2]), boundary.index(pts_index[3])]
    corner_points_index_new = [pts_index[0], pts_index[pts_index_list.index(min(pts_index_list)) + 1],
                               pts_index[pts_index_list.index(max(pts_index_list)) + 1]]
    middle_index = [i for i in pts_index if i not in corner_points_index_new]
    corner_points_index_new.insert(1, middle_index[0])

    return corner_points_index_new

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

@jit()
def quad_obj_load_vt(number_quad, obj_path, input_type, k_save_vt_obj_result):
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
                if input_type == 2:
                    quad_index_temp_list = [-1E5, -1E5, -1E5, -1E5]
                    quad_vt_temp = [-1E5, -1E5, -1E5, -1E5]
                else:
                    quad_index_temp_list = [-1E5, -1E5, -1E5]
                    quad_vt_temp = [-1E5, -1E5, -1E5]
                for i in range(1, len(strs), 1):
                    quad_index_temp_list[i - 1] = int(
                        strs[i].split("/")[0]) - 1  # Python count starts from 0 instead of 1
                    quad_vt_temp[i - 1] = int(strs[i].split("/")[1]) - 1  # Python count starts from 0 instead of 1
                quad_index.append(quad_index_temp_list)
                quad_vt.append(quad_vt_temp)
    quad_index, quad_vt, points, pts_vt = np.array(quad_index), np.array(quad_vt), np.array(points), np.array(pts_vt)
    index_vertices_effective = np.array(quad_index).flatten().tolist()
    index_vertices_effective = list(set(index_vertices_effective))
    num_index_vertices_effective = len(index_vertices_effective)
    if input_type == 2:
        simple_patch_index = np.zeros((len(quad_index), 4))
    else:
        simple_patch_index = np.zeros((len(quad_index), 3))
    simple_vertices = np.zeros((num_index_vertices_effective, 3))
    dict_index_node = {}  # Transformation dict for node indexes,the key is the old index of the node and the value is the new index
    for i in range(0, num_index_vertices_effective, 1):
        dict_index_node[index_vertices_effective[i]] = i
        simple_vertices[i, :] = points[index_vertices_effective[i]]
    if input_type == 2:
        quad_vt = np.array([i_k for i_k in range(0, 4 * len(quad_index), 1)]).reshape(-1, 4)
        for j in range(0, len(quad_index), 1):
            simple_patch_index[j, 0:4] = [dict_index_node[quad_index[j, 0]], dict_index_node[quad_index[j, 1]],
                                          dict_index_node[quad_index[j, 2]], dict_index_node[quad_index[j, 3]]]
    else:
        for j in range(0, len(quad_index), 1):
            simple_patch_index[j, 0:3] = [dict_index_node[quad_index[j, 0]], dict_index_node[quad_index[j, 1]],
                                          dict_index_node[quad_index[j, 2]]]

    simple_patch_index = simple_patch_index.astype(int)
    file_name_simple = []
    if k_save_vt_obj_result == 1:
        if input_type == 2:
            if not os.path.exists('data_simple_vt_quad'):
                os.mkdir('data_simple_vt_quad')
            file_name_simple = 'data_simple_vt_quad/quad_simple_vt_' + str(number_quad) + '.obj'
            thefile = open(file_name_simple, 'w')
        else:
            if not os.path.exists('data_simple_vt_tri'):
                os.mkdir('data_simple_vt_tri')
            file_name_simple = 'data_simple_vt_tri/tri_simple_vt_' + str(number_quad) + '.obj'
            thefile = open(file_name_simple, 'w')
        for item in simple_vertices:
            thefile.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))
        for item in pts_vt:
            thefile.write("vt {0} {1}\n".format(item[0], item[1]))

        faces = np.concatenate((simple_patch_index, quad_vt), axis=1)
        faces = faces + 1
        if input_type == 2:
            for item in faces:
                thefile.write("f {0}/{1} {2}/{3} {4}/{5} {6}/{7}\n".format(item[0],
                                                                           item[4], item[1], item[5], item[2], item[6],
                                                                           item[3], item[7]))
        else:
            for item in faces:
                thefile.write("f {0}/{1} {2}/{3} {4}/{5}\n".format(item[0],
                                                                   item[3], item[1], item[4], item[2], item[5]))
        thefile.close()
    return simple_vertices, pts_vt, simple_patch_index, quad_vt, file_name_simple

def load_obj_vt(mesh_vertices, mesh_triangles, pts_vt, quad_vt):
    lines_vertices = list(mesh_vertices)
    mesh_vertices = np.asarray(mesh_vertices)
    len_mesh_vertices = len(mesh_vertices)
    len_mesh_triangles = len(mesh_triangles)
    # the nodes(points) data
    nodes = list()
    for index, line in enumerate(lines_vertices):
        node = Node([item for item in line], index)  # Node type defined in Data_structure
        nodes.append(node)
    # the edges data
    edges = list()
    lines_triangles = list(mesh_triangles)
    for line in lines_triangles:
        indexes = [item for item in line]
        edges.append((indexes[0], indexes[1]))
        edges.append((indexes[0], indexes[2]))
        edges.append((indexes[1], indexes[2]))
        for i in range(len(indexes)):
            # Adjacent nodes, key is the coordinates of adjacent nodes, value is the two or one node of the connection
            # between the node and the key.
            nodes[indexes[i]].add_connection(indexes[(i + 1) % 3], indexes[(i - 1) % 3])
            nodes[indexes[i]].add_connection(indexes[(i - 1) % 3], indexes[(i + 1) % 3])
    # Texture coordinates (parameter coordinates)
    u_max_bound = max(pts_vt[:, 0])
    u_min_bound = min(pts_vt[:, 0])
    v_max_bound = max(pts_vt[:, 1])
    v_min_bound = min(pts_vt[:, 1])
    corner_points_index = [-1E5, -1E5, -1E5, -1E5]
    for i0 in range(0, len(pts_vt), 1):
        pts_vt[i0] = [min(1.0, max((pts_vt[i0][0] - u_min_bound) / (u_max_bound - u_min_bound), 0.0)),
                      min(1.0, max((pts_vt[i0][1] - v_min_bound) / (v_max_bound - v_min_bound), 0.0))]
        if pts_vt[i0][0] == 0.0 and pts_vt[i0][1] == 0.0:
            corner_points_index[0] = i0
        if pts_vt[i0][0] == 1.0 and pts_vt[i0][1] == 0.0:
            corner_points_index[3] = i0
        if pts_vt[i0][0] == 0.0 and pts_vt[i0][1] == 1.0:
            corner_points_index[2] = i0
        if pts_vt[i0][0] == 1.0 and pts_vt[i0][1] == 1.0:
            corner_points_index[1] = i0

    if min(corner_points_index) < 0:
        for i0 in range(0, len(pts_vt), 1):
            if pts_vt[i0][0] <= 0.01 and pts_vt[i0][1] <= 0.01:
                corner_points_index[0] = i0
            if pts_vt[i0][0] >= 0.99 and pts_vt[i0][1] <= 0.01:
                corner_points_index[3] = i0
            if pts_vt[i0][0] <= 0.01 and pts_vt[i0][1] >= 0.99:
                corner_points_index[2] = i0
            if pts_vt[i0][0] >= 0.99 and pts_vt[i0][1] >= 0.99:
                corner_points_index[1] = i0
    for node in nodes:
        index = node.index
        node.set_map_point(pts_vt[index])
    ###list_mesh_triangles = mesh_triangles.flatten().tolist()
    ###list_quad_vt = quad_vt.flatten().tolist()
    return nodes, edges, len_mesh_vertices, len_mesh_triangles, mesh_vertices, mesh_triangles, corner_points_index

def sequences_find(nodes, node1, node2, max_end, min_end, boundary):
    sequences = list()
    point_01, point_02, point_03, point_04 = node1.index, max_end, node2.index, min_end
    sequences.append(boundary[:boundary.index(point_02) + 1])
    sequences.append(boundary[boundary.index(point_02):boundary.index(point_03) + 1])
    sequences.append(boundary[boundary.index(point_03):boundary.index(point_04) + 1])
    bou_node1_min = boundary[boundary.index(point_04):] + boundary[:boundary.index(point_01)] + [point_01]
    bou_node1_min.reverse()
    sequences.append(bou_node1_min)
    return sequences

def path_and_plot(nodes, patch, edges_total, patch_nodes, k_print_total):
    #  Map the layout to a unit square
    if k_print_total != 0:
        print('--Map the layout to a unit square...--')
    # The boundary and internal data are the indexes of nodes.
    node1, node2, max_end, min_end, boundary, insides, triangles = patch
    sequences = sequences_find(nodes, node1, node2, max_end, min_end, boundary)

    return sequences

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

    return [simple_vertices, pts_vt, d_u, d_v, num_ctrlpts_u, num_ctrlpts_v]

def map_to_space(nodes, patch, d_u, d_v, path, k_print_total, k_construct_grid):
    # The mesh points are inversely mapped back to the space surface
    quad_mesh, mesh_quad_index = construct_grid(d_u, d_v, k_construct_grid)
    node1, node2, max_end, min_end, boundary, insides, triangles = patch
    point_triangle_dict = dict()
    # Find the triangle corresponding to the mesh point and its coordinates
    for index, mesh_point in enumerate(quad_mesh):
        for triangle in triangles:
            a, b, c = triangle
            map_point_a, map_point_b, map_point_c = nodes[a].map_point, nodes[b].map_point, nodes[c].map_point
            res, p1, p2, p3 = is_in_triangle(map_point_a, map_point_b, map_point_c, mesh_point)
            if res:
                point_triangle_dict[index] = (triangle, p1, p2, p3)
                break

    mesh_points_3d = dict()
    for index, value in point_triangle_dict.items():
        triangle, p1, p2, p3 = value
        triangle = list(triangle)
        point_3d_x = nodes[triangle[0]].x * p1 + nodes[triangle[1]].x * p2 + nodes[triangle[2]].x * p3
        point_3d_y = nodes[triangle[0]].y * p1 + nodes[triangle[1]].y * p2 + nodes[triangle[2]].y * p3
        point_3d_z = nodes[triangle[0]].z * p1 + nodes[triangle[1]].z * p2 + nodes[triangle[2]].z * p3
        mesh_points_3d[index] = (point_3d_x, point_3d_y, point_3d_z)

    sequence = list()
    for i in range(4):
        sequence.append(list())
    for i in range(0, d_v + 1, 1):
        sequence[0].append(i)
    for i in range(0, d_u * (d_v + 1) + 1, d_v + 1):
        sequence[1].append(i)
    for i in range(d_u * (d_v + 1), (d_u + 1) * (d_v + 1)):
        sequence[2].append(i)
    for i in range((d_u + 1) * (d_v + 1) - 1, d_u - 1, -d_v - 1):
        sequence[3].append(i)
    path0, path1, path2, path3 = path[0], path[1], path[2], path[3]

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
    exist_tri = 0
    if (type(quad_index_array).__name__ == 'list'):
        quad_index_array = np.array(quad_index_array)
    if quad_index_array.__contains__(-1E5):
        print('ERROR!!! Imported quads obj file has patches that are not quads.')
        exist_tri = 1

    for i in range(0, len(quad_index_array), 1):
        quad_index_temp = quad_index_array[i]
        if exist_tri == 1:
            if quad_index_temp.__contains__(-1E5):
                tri_index_improve.append(
                    [int(quad_index_temp[0]), int(quad_index_temp[1]), int(quad_index_temp[2])])
        else:
            [S_abc, length, length, length] = triangle_area_3d(points_quad[int(quad_index_temp[0]) - 1],
                                                               points_quad[int(quad_index_temp[1]) - 1],
                                                               points_quad[int(quad_index_temp[2]) - 1])
            [S_acd, length, length, length] = triangle_area_3d(points_quad[int(quad_index_temp[0]) - 1],
                                                               points_quad[int(quad_index_temp[2]) - 1],
                                                               points_quad[int(quad_index_temp[3]) - 1])
            [S_abd, length, length, length] = triangle_area_3d(points_quad[int(quad_index_temp[0]) - 1],
                                                               points_quad[int(quad_index_temp[1]) - 1],
                                                               points_quad[int(quad_index_temp[3]) - 1])
            [S_bcd, length, length, length] = triangle_area_3d(points_quad[int(quad_index_temp[1]) - 1],
                                                               points_quad[int(quad_index_temp[2]) - 1],
                                                               points_quad[int(quad_index_temp[3]) - 1])
            if (S_abc - S_acd) ** 2 - (S_abd - S_bcd) ** 2 < 0:
                tri_index_improve.append(
                    [int(quad_index_temp[0]), int(quad_index_temp[1]), int(quad_index_temp[2])])
                tri_index_improve.append(
                    [int(quad_index_temp[0]), int(quad_index_temp[2]), int(quad_index_temp[3])])
            else:
                tri_index_improve.append(
                    [int(quad_index_temp[0]), int(quad_index_temp[1]), int(quad_index_temp[3])])
                tri_index_improve.append(
                    [int(quad_index_temp[1]), int(quad_index_temp[2]), int(quad_index_temp[3])])

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

def CAD_reconstruction(number_quad):
    k_delta = 0.05  # (kd) NURBS visualization and obj derived smoothness. When this value is closer to 0,
    # the higher the degree of subdivision, the smoother obj, and the longer the time. The value range is ( 0.0,1.0 ),
    # and the recommended value is 0.05.
    k_type_gemodl = 1.2  # (kt) The coefficient of the number of control points in the single parameter direction
    # during NURBS fitting, the recommended range: [0.1, 0.7]. The larger the value, the more the number of control
    # points during fitting, and the higher the degree of surface bending.
    k_sample = 1  # (ks) The specified sampling rate coefficient when the remeshing is performed by the
    # mapping method. This value should not be too small, and different patches will be adaptively sampled.
    d_min = 10  # (mi) The minimum value of the specified sampling rate when the remeshing is performed
    # by the  mapping method. This value cannot be too low, the recommended range: [10,100].The greater the value is,
    # the higher the fitting accuracy relative to the original surface is, and the longer the time is.
    d_max = 2000  # (ma) The maximum value of the specified sampling rate when the mapping method is remeshing

    k_construct_grid = 1  # (cg) 1: Normal sampling in parameter domain sampling; 2: Sampling from 0.02 to 0.98 in
    # the parameter domain, preserving the splicing space.;else: When sampling the parameter domain, only the internal
    # points are sampled, and the sampling grid does not appear on the boundary.
    num_ctrlpts_min = 6  # (ct) The minimum number of control points in the single parameter direction during NURBS fitting
    export_total_json = 2  # (et) 1: Exports the json file of the overall result with delta = 0.04.;
    # 2: The json file of the overall result is exported with the set k_delta. The closer k_delta is to 0, the longer
    # the export time is.; else: json file is not exported
    export_single_json = 0  # (es) 1: Export the json file of each single patch with delta = 0.04.;
    # 2: The json file of each single patch is exported with the set k_delta. The closer the k_delta is to 0, the longer
    # the export time is.; else: The json file of each single patch is not exported.
    export_single_obj = 0  # (es) 1: Export the obj file of each single patch with delta = 0.04.;
    # 2: The obj file of each single patch is exported with the set k_delta. The closer the k_delta is to 0, the longer
    # the export time is.; else: The obj file of each single patch is not exported.

    k_input_quad = 1  # (kiq) 1: Taking the quadrilateral patch obj as the input, the triangular patch splitting
    # based on the state is carried out.; else: Taking the original triangle as input (output_tri/tri_.obj)
    NURBS_degree_u = 3  # (de) The order of NURBS patches along the u-parameter direction
    NURBS_degree_v = 3  # (de) The order of NURBS patches along the v-parameter direction
    type_gemodl = 2  # (ty) fitting mode: 1: Accurate interpolation fitting;
    # 2: Least square approximation fitting of the specified number of control points (recommended);
    # 3: Centripetal parametric least squares approximation fitting without specifying the number of control points;
    # 4: Approximate Centripetal Parameterized Least Squares Fitting of the Number of Control Points;
    k_print_total = 0  # (kp) 0: Do not print all results; else: print
    k_save_vt_obj_result = 0
    export_file_total_3dm = 1  # (em) 1: Export the 3dm file of the result, which requires that export_total_json must
    # be 1 or 2 to be successfully exported.; else: Do not export
    decompose = 0  # (dec) 0: Undivided NURBS surface; 1: The NURBS surface is segmented along the
    # direction of u parameter.; 2: The NURBS surface is segmented along the direction of v parameter.;
    # 3: The NURBS surface is segmented along the direction of u and v parameters.;

    file_path = 'data_3'  # The relative path of reading and storing files
    file_name_result = 'data_3/result_cantilever_deformed_quad_rhino.obj'  # A file containing texture coordinates and packaging textures derived from the generalized motorcycle graph method

    if not os.path.exists('data_3'):
        os.mkdir('data_3')

    if k_print_total != 0:
        print('\n------The index value is', number_quad, '------')
    else:
        print(number_quad, end='\n')

    if k_input_quad == 1:
        file_name = file_path + '/output_quad/quad_' + str(int(number_quad)) + '.obj'
        # The quad mesh model is used as input.
        [simple_vertices, simple_patch_index] = quad_obj_trans_load_vt(file_name)
        input_type = 0
        pts_vt, quad_vt, file_name_simple = [], [], []
    else:
        file_name = file_path + '/output_tri/tri_' + str(int(number_quad)) + '.obj'
        # The triangle mesh model is used as input.
        input_type = 0
        [simple_vertices, pts_vt, simple_patch_index, quad_vt, file_name_simple] = quad_obj_load_vt(
            number_quad, file_name, input_type, k_save_vt_obj_result)

    [simple_vertices_result, pts_vt_result, simple_patch_index_result, quad_vt_result] = \
        quad_obj_load_vt_result(file_name_result)
    [simple_vertices, pts_vt, d_v, d_u, num_ctrlpts_u, num_ctrlpts_v] = vt_adjust(simple_vertices,
                                simple_vertices_result, pts_vt_result, simple_patch_index_result,
                                quad_vt_result, k_sample, d_min, d_max, k_type_gemodl,
                                NURBS_degree_u, NURBS_degree_v, num_ctrlpts_min)

    nodes, edges, num_vertices_tri_mesh, num_triangles_tri_mesh, mesh_ver, mesh_tri, corner_points_index \
        = load_obj_vt(simple_vertices, simple_patch_index, pts_vt, quad_vt)
    edges = np.sort(edges, axis=1)
    edges_total = np.unique(edges, axis=0)
    [boundary_temp, insides, edge_boundary_temp] = cal_boundary_length(mesh_ver, mesh_tri)
    boundary = boundary_find(int(corner_points_index[0]), edge_boundary_temp)
    insides = np.unique(insides)
    insides = [k for k in insides if k not in boundary]
    corner_points_index = adjust_corner_order(corner_points_index, boundary)

    if 0: # 0: Not drawing; 1: Drawing
        mesh_plot = o3d.geometry.TriangleMesh()
        mesh_plot.vertices = o3d.utility.Vector3dVector(mesh_ver)
        mesh_plot.triangles = o3d.utility.Vector3iVector(mesh_tri)
        mesh_plot.compute_vertex_normals()
        mesh_plot.triangle_normals = o3d.utility.Vector3dVector([])
        test_points_show = corner_points_index
        points_show_boundary, colors_show_boundary = [], []
        for k in test_points_show:  # The index value of the test point to be displayed
            points_show_boundary.append([mesh_ver[k][0], mesh_ver[k][1], mesh_ver[k][2]])
            colors_show_boundary.append([0, 0, 0])  # balck
        pcd_show_boundary = o3d.geometry.PointCloud()  # 3d point cloud
        pcd_show_boundary.points = o3d.utility.Vector3dVector(
            points_show_boundary)  # point3D Convert it to open3d point cloud format
        pcd_show_boundary.colors = o3d.utility.Vector3dVector(colors_show_boundary)
        vis_new = o3d.visualization.Visualizer()
        vis_new.create_window(window_name='Harmonic Mapping')  # Create a window
        vis_new.add_geometry(pcd_show_boundary)  # Add point cloud
        vis_new.add_geometry(mesh_plot)  # Add a mesh file
        render_option: o3d.visualization.RenderOption = vis_new.get_render_option()
        # Setting Point Cloud Rendering Parameters
        render_option.background_color = np.array([1, 1, 1])  # Set the background color (here is white)
        render_option.point_size = 25  # Set the size of the rendering point
        render_option.show_coordinate_frame = False  # If set to True, add a coordinate frame
        render_option.point_show_normal = False  # If set to True, the point normal is visualized.
        render_option.mesh_show_wireframe = True  # If set to True, the grid wireframe is visualized.
        render_option.mesh_show_back_face = True  # The back of the mesh triangle is visualized
        render_option.line_width = 20
        vis_new.run()
        vis_new.destroy_window()

    patch = nodes[corner_points_index[0]], nodes[corner_points_index[1]], corner_points_index[2], \
            corner_points_index[3], boundary, insides, mesh_tri
    patch_nodes_temp = [k for k in mesh_tri]
    patch_nodes = np.unique(patch_nodes_temp)
    path = path_and_plot(nodes, patch, edges_total, patch_nodes, k_print_total)
    mesh_points_3d, mesh_quad_index = map_to_space(nodes, patch, d_v, d_u, path,
                                                   k_print_total, k_construct_grid)
    X_NURBS = np.zeros((len(mesh_points_3d), 3))
    key_list = []
    for key0 in mesh_points_3d:
        key_list.append(key0)
    for u in range(0, len(mesh_points_3d), 1):
        if u not in key_list:
            print('u not in key_list', u, end=' ')

    if k_print_total != 0:
        print('num_ctrlpts_u =', num_ctrlpts_v, 'num_ctrlpts_v =', num_ctrlpts_u, 'd_u =', d_u, 'd_v =', d_v)

    for key in mesh_points_3d:
        X_NURBS[key] = mesh_points_3d[key]

    size_u, size_v = d_u + 1, d_v + 1

    NURBS_patch = NURBSFIT_single(number_quad, len(number_quad_list),
                                  X_NURBS, size_u, size_v, num_ctrlpts_u, num_ctrlpts_v,
                                  NURBS_degree_u, NURBS_degree_v, type_gemodl, k_delta,
                                  export_single_json, export_single_obj, export_total_json, export_file_total_3dm,
                                  decompose)

    return NURBS_patch

if __name__ == '__main__':

    profiler = Profiler()
    profiler.start()  # Analysis of each part of the code time-consuming
    time_begin = time.time()
    number_quad_list = list(range(0, 67, 1))  # The scope of the reconstructed model

    visualization = 0  # (vi) 0: No results are displayed; 1: Shows the final NURBS surface fitting results
    k_delta = 0.05
    export_total_json = 1
    export_file_total_3dm = 1
    number_quad_list = np.unique(number_quad_list)

    surface = list(Parallel(n_jobs=48)(delayed(CAD_reconstruction)(i) for i in number_quad_list))  # Parallel
    # The process of parallelization will result in an error in the order of printing results. In order to
    # obtain accurate print results, it is possible to set the n_jobs value to 1 (although this is not recommended).
    surface_total = multi.SurfaceContainer(surface)

    if visualization == 0:
        pass
    elif visualization == 1:
        Visualization_NURBS(surface_total, k_delta)  # Shows the final NURBS surface fitting results
    else:
        plt.show()

    if export_total_json == 1:
        surface_total.delta = 0.04
        exchange.export_json(surface_total, 'data_3/SINGLE_HM_NURBS_surface_total.json')
    elif export_total_json == 2:
        surface_total.delta = k_delta
        exchange.export_json(surface_total, 'data_3/SINGLE_HM_NURBS_surface_total.json')
    if export_file_total_3dm == 1:
        if export_total_json == 1 or 2:
            write_3dm()

    time_end = time.time()
    print('The total time consumption is ', f'{(time_end - time_begin):.2f}', 'seconds.')
    profiler.stop()
    profiler.print()