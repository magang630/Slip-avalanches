# -*- coding: cp936 -*-
# using NetworkX for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks
# Usage: python NetworkFeatureEmbedding.py
# Reference：
# [1] https://networkx.github.io/documentation/stable/
# [2] Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), 
# [2] L. da F. Costa, F. A. Rodrigues, G. Travieso, et al. Characterization of complex networks: A survey of measurements[J]. Advances in Physics, 2007, 56(1):167-242.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
from sys import argv,exit
import os
import pickle
import pyvoro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import sparse
from sklearn.preprocessing import RobustScaler
# from networkx.algorithms import approximation
# from scipy import stats
# from scipy.spatial import Delaunay
# from scipy.spatial import ConvexHull
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      

def rightTrim(input, suffix):
    if (input.find(suffix) == -1):
        input = input + suffix
    return input


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("/")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        print(path + ' 创建成功')
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


def dict_initiation(key):
    key = list(key)
    value = [0 for i in range(len(key))]
    return dict(zip(key, value))

#----------------------------------------------------------------------
# unit normal vector of plane defined by points a, b, and c
#
def unit_normal(a, b, c):
    x = det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
    y = det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
    z = det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)

#----------------------------------------------------------------------
# dot product of vectors a and b
#
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

#----------------------------------------------------------------------
# cross product of vectors a and b
#
def cross(a, b):
    x = a[1]*b[2] - a[2]*b[1]
    y = a[2]*b[0] - a[0]*b[2]
    z = a[0]*b[1] - a[1]*b[0]
    return (x, y, z)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Distance of two particles
#
def dist(pt_1, pt_2):
    pt_1 = np.array((pt_1[0], pt_1[1], pt_1[2]))
    pt_2 = np.array((pt_2[0], pt_2[1], pt_2[2]))
    return np.linalg.norm(pt_1-pt_2)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Distance of two vertexes
#
def vertex_distance(a, b):
    return np.sqrt((a[0] - b[0])**2.0 + (a[1] - b[1])**2.0 + (a[2] - b[2])**2.0)

#----------------------------------------------------------------------
# area of polygon poly
#
def PolygonArea(poly):
    if len(poly) < 3: # not a plane - no area
        return 0

    total = [0, 0, 0]
    for i in range(len(poly)):
        vi1 = poly[i]
        if i is len(poly)-1:
            vi2 = poly[0]
        else:
            vi2 = poly[i+1]
        prod = cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)

#----------------------------------------------------------------------
# find area of polygon from xyz coordinates
# https://stackoverflow.com/questions/12642256/python-find-area-of-polygon-from-xyz-coordinates/12643315
# determinant of matrix a
#
def det(a):
    return a[0][0]*a[1][1]*a[2][2] + a[0][1]*a[1][2]*a[2][0] + a[0][2]*a[1][0]*a[2][1] - a[0][2]*a[1][1]*a[2][0] - a[0][1]*a[1][0]*a[2][2] - a[0][0]*a[1][2]*a[2][1]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Volume of convex hull with QHull from SciPy
#
def convex_hull_volume(pts):
    ch = ConvexHull(pts)
    dt = Delaunay(pts[ch.vertices])
    tets = dt.points[dt.simplices]
    return np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],
                                     tets[:, 2], tets[:, 3]))

def convex_hull_volume_bis(pts):
    ch = ConvexHull(pts)

    simplices = np.column_stack((np.repeat(ch.vertices[0], ch.nsimplex),
                                 ch.simplices))
    tets = ch.points[simplices]
    return np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],
                                     tets[:, 2], tets[:, 3]))

def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d)))/6


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def GraphEmbedding(case, test_id, d50, time_step, scenario, strain_window, connectivity):

    file_path = os.path.pardir + '/' + case

    # dump files
    dump_path = file_path + '/test-' + str(test_id) + '/Particle potential'
    list_dir = os.listdir(dump_path)
    dump_frame = []
    file_prefix = 'Particle potential-'
    file_suffix = '.dump'
    prefix_len = len(file_prefix)
    suffix_len = len(file_suffix)
    for file in list_dir:
        dump_frame.append(int(file[prefix_len:][:-suffix_len]))
    dump_frame = np.array(sorted(dump_frame))
    dump_time = (dump_frame - np.min(dump_frame))*time_step
    frame_time = dict(zip(dump_frame, dump_time))

    start_frame = np.min(dump_frame)
    end_frame = np.max(dump_frame)
    frame_interval = (end_frame - start_frame)/scenario
    frame_list = np.arange(start_frame, end_frame, frame_interval)
    frame_list = np.append(frame_list, end_frame)
    frame_list = frame_list.astype(int)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 循环开始，提取每一步数据
    #
    output_path = file_path + '/test-' + str(test_id) + '/feature_embedding'
    dynamics_path = file_path + '/test-' + str(test_id) + '/strain window-' + str(round(strain_window, 3)) + '/particle dynamics'
    potential_path = file_path + '/test-' + str(test_id) + '/Particle potential'
    mkdir(output_path)
    mkdir(dynamics_path)
    mkdir(potential_path)

    # Create an empty graph with no nodes and no edges.
    allframe_networks = [[] for i in range(len(frame_list))]
    columns = ['Press', 'Shear_stress', 'Potential', 'Um', 'Vm', 'Nonaffine_um', 'Shear_strain',
               'Volumetric_strain', 'D2min', 'Temperature']
    df_allframe_node_features = pd.DataFrame(columns=columns)
    for idx, frame in enumerate(frame_list):
        if idx == 0: continue

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Taking the whole dataset
        #
        print(60 * '*')
        print('The %d th frame' %frame)
        print(60 * '*')

        particle_info = open(dynamics_path + '/Particle dynamics-' + str(frame) + '.dump', 'r')
        alllines = particle_info.readlines()
        lines = alllines[9:]
        particle_info.close()
        for i in range(len(lines)):
            if (lines[i] == '\n'): del lines[i]
        Par_id                = np.array([int(line.strip().split(' ')[0]) for line in lines])             # 字段以逗号分隔，这里取得是第1列
        Par_type              = np.array([int(line.strip().split(' ')[1]) for line in lines])           # 字段以逗号分隔，这里取得是第2列
        Par_radius            = np.array([float(line.strip().split(' ')[2]) for line in lines])       # 字段以逗号分隔，这里取得是第3列
        Par_xcor              = np.array([float(line.strip().split(' ')[3]) for line in lines])         # 字段以逗号分隔，这里取得是第4列
        Par_ycor              = np.array([float(line.strip().split(' ')[4]) for line in lines])         # 字段以逗号分隔，这里取得是第5列
        Par_zcor              = np.array([float(line.strip().split(' ')[5]) for line in lines])         # 字段以逗号分隔，这里取得是第6列
        Par_um                = np.array([float(line.strip().split(' ')[9]) for line in lines])           # 字段以逗号分隔，这里取得是第10列
        Par_nonaffine_um      = np.array([float(line.strip().split(' ')[13]) for line in lines]) # 字段以逗号分隔，这里取得是第14列
        Par_temperature       = np.array([float(line.strip().split(' ')[14]) for line in lines])  # 字段以逗号分隔，这里取得是第15列
        Par_D2min             = np.array([float(line.strip().split(' ')[15]) for line in lines])        # 字段以逗号分隔，这里取得是第16列
        Par_volumetric_strain = np.array([float(line.strip().split(' ')[16]) for line in lines])  # 字段以逗号分隔，这里取得是第17列
        Par_shear_strain      = np.array([float(line.strip().split(' ')[17]) for line in lines])  # 字段以逗号分隔，这里取得是第18列

        Par_id = Par_id[Par_type == 1]
        Par_radius = Par_radius[Par_type == 1]
        Par_xcor = Par_xcor[Par_type == 1]
        Par_ycor = Par_ycor[Par_type == 1]
        Par_zcor = Par_zcor[Par_type == 1]
        Par_um = Par_um[Par_type == 1]
        Par_D2min = Par_D2min[Par_type == 1]
        Par_shear_strain = Par_shear_strain[Par_type == 1]
        Par_volumetric_strain = Par_volumetric_strain[Par_type == 1]
        Par_nonaffine_um = Par_nonaffine_um[Par_type == 1]
        Par_temperature = Par_temperature[Par_type == 1]
        Par_diameter = 2.0*Par_radius
        Par_num = np.size(Par_id)

        particle_info = open(potential_path + '/Particle potential-' + str(frame) + '.dump', 'r')
        alllines = particle_info.readlines()
        lines = alllines[9:]
        particle_info.close()
        for i in range(len(lines)):
            if (lines[i] == '\n'): del lines[i]
        Par_id2             = np.array([int(line.strip().split(' ')[0]) for line in lines])     # 字段以逗号分隔，这里取得是第1列
        Par_type2           = np.array([int(line.strip().split(' ')[1]) for line in lines])     # 字段以逗号分隔，这里取得是第2列
        Par_vm              = np.array([float(line.strip().split(' ')[6]) for line in lines])   # 字段以逗号分隔，这里取得是第3列
        Par_press           = np.array([float(line.strip().split(' ')[7]) for line in lines])   # 字段以逗号分隔，这里取得是第3列
        Par_deviator_stress = np.array([float(line.strip().split(' ')[8]) for line in lines])   # 字段以逗号分隔，这里取得是第3列
        Par_snn             = np.array([float(line.strip().split(' ')[9]) for line in lines])   # 字段以逗号分隔，这里取得是第3列
        Par_sxy             = np.array([float(line.strip().split(' ')[10]) for line in lines])  # 字段以逗号分隔，这里取得是第3列
        Par_penergy         = np.array([float(line.strip().split(' ')[11]) for line in lines])  # 字段以逗号分隔，这里取得是第3列

        Par_id2 = Par_id2[Par_type2 == 1]
        Par_vm = Par_vm[Par_type2 == 1]
        Par_press = Par_press[Par_type2 == 1]
        Par_deviator_stress = Par_deviator_stress[Par_type2 == 1]
        Par_snn = Par_snn[Par_type2 == 1]
        Par_sxy = Par_sxy[Par_type2 == 1]
        Par_penergy = Par_penergy[Par_type2 == 1]

        # Par_radius = dict(zip(Par_id, Par_radius))
        # Par_xcor = dict(zip(Par_id, Par_xcor))
        # Par_ycor = dict(zip(Par_id, Par_ycor))
        # Par_zcor = dict(zip(Par_id, Par_zcor))
        # Par_D2min = dict(zip(Par_id, Par_D2min))
        # Par_shear_strain = dict(zip(Par_id, Par_shear_strain))
        # Par_volumetric_strain = dict(zip(Par_id, Par_volumetric_strain))
        # Par_nonaffine_um = dict(zip(Par_id, Par_nonaffine_um))
        # Par_temperature = dict(zip(Par_id, Par_temperature))
        # Par_vm = dict(zip(Par_id2, Par_vm))
        # Par_press = dict(zip(Par_id2, Par_press))
        # Par_deviator_stress = dict(zip(Par_id2, Par_deviator_stress))
        # Par_penergy = dict(zip(Par_id2, Par_penergy))

        Par_press[Par_press < 0] = 0
        Par_deviator_stress[Par_deviator_stress < 0] = 0
        Par_snn[Par_snn < 0] = 0
        Par_sxy[Par_sxy < 0] = 0
        node_features = {'Press': Par_press,
                         'Shear_stress': Par_deviator_stress,
                         'Sxy': Par_sxy,
                         'Potential': Par_penergy,
                         'Vm': Par_vm,
                         'Um': Par_um,
                         'Nonaffine_um': Par_nonaffine_um,
                         'Shear_strain': Par_shear_strain,
                         'Volumetric_strain': Par_volumetric_strain,
                         'D2min': Par_D2min}
        df_node_features = pd.DataFrame(node_features, index=[frame for i in range(Par_num)])
        df_allframe_node_features = pd.concat([df_allframe_node_features, df_node_features], axis=0)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2. Creating a graph (constructing graphs node-by-node or edge-by-edge)
        #

        # neighbor particles in the current configuration
        # Call the voro++ library
        Par_coord = np.stack((Par_xcor, Par_ycor, Par_zcor), axis=1)
        domain_xmin, domain_xmax = 0, 0.04
        domain_zmin, domain_zmax = 0, 0.02
        # domain_xmin, domain_xmax = np.min(Par_coord[:, 0] - Par_radius), np.max(Par_coord[:, 0] + Par_radius)
        domain_ymin, domain_ymax = np.min(Par_coord[:, 1] - Par_radius), np.max(Par_coord[:, 1] + Par_radius)
        # domain_zmin, domain_zmax = np.min(Par_coord[:, 2] - Par_radius), np.max(Par_coord[:, 2] + Par_radius)
        container = [[domain_xmin, domain_xmax], [domain_ymin, domain_ymax], [domain_zmin, domain_zmax]]
        vor = pyvoro.compute_voronoi(Par_coord, limits=container, dispersion=4*d50, radii=Par_radius, periodic=[True, False, True])
        vor_nearest_neighbors = [[] for i in range(Par_num)]
        for i, cell in enumerate(vor):
            faces = cell['faces']
            face_num = len(faces)
            adjacent_cell = []
            for j in range(face_num):
                if faces[j]['adjacent_cell'] >= 0: adjacent_cell.append(faces[j]['adjacent_cell'])
            vor_nearest_neighbors[i] = adjacent_cell

        for par_i, neighbor_par_i in enumerate(vor_nearest_neighbors):
            for par_j in neighbor_par_i:
                neighbor_par_j = vor_nearest_neighbors[par_j]
                if par_i in neighbor_par_j:
                    continue
                else:
                    vor_nearest_neighbors[par_i].remove(par_j)

        # Assign graph attributes when creating a new graph
        G = nx.Graph(series=frame)
        # The graph G can be grown in several ways. We add one node at a time and assign node attributes when adding a new node
        for i in range(Par_num):
            # G.add_node(i, diameter=2*Par_radius[i], xcor=Par_xcor[i], ycor=Par_ycor[i], zcor=Par_zcor[i],
            #            press=Par_press[i], shear_stress=Par_deviator_stress[i], potential=Par_penergy[i],
            #            um=Par_um[i], vm=Par_vm[i], nonaffine_um=Par_nonaffine_um[i],
            #            shear_strain=Par_shear_strain[i], volumetric_strain=Par_volumetric_strain[i],
            #            D2min=Par_D2min[i], temperature=Par_temperature[i])
            G.add_node(i)

            if connectivity == 'hard':
                # Hard connectivity
                neighbor_list = vor_nearest_neighbors[i]
                if i in neighbor_list: neighbor_list.remove(i)
                for j in neighbor_list:
                    # Add one edge at a time and add edge attributes using add_edge()
                    if j > i: G.add_edge(i, j)

            elif connectivity == 'soft':
                # Soft connectivity
                vertices_coord = np.array(vor[i]['vertices'])
                for k, face in enumerate(vor[i]['faces']):
                    if face['adjacent_cell'] < 0: continue
                    j = face['adjacent_cell']
                    neighbor_par_j = vor_nearest_neighbors[j]
                    if i not in neighbor_par_j: continue
                    vertices_list = face['vertices']
                    polygon = vertices_coord[vertices_list]
                    face_area = PolygonArea(polygon)
                    clearance = vertex_distance(Par_coord[i], Par_coord[j]) - Par_radius[i] - Par_radius[j]
                    diameter = 4.0*Par_radius[i]*Par_radius[j]/(Par_radius[i] + Par_radius[j])
                    # print("Weight  ", face_area/clearance/diameter, diameter/clearance)
                    # if face_area/clearance >= epsilon*diameter:
                    if clearance <= epsilon*diameter:
                        if j > i: G.add_edge(i, j)

        allframe_networks[idx] = G
        print(nx.info(G))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3. Node feature scaling
    #
    dmin, dmax = 0*d50, 2*d50
    par_diameter_scaled = (Par_diameter - dmin)/(dmax - dmin)
    df_par_diameter_scaled = pd.DataFrame(par_diameter_scaled, columns=['Diameter'], index=Par_id)

    robust_scaler = RobustScaler(quantile_range=(0.0, 99.0))
    allframe_node_features_scaled = robust_scaler.fit_transform(df_allframe_node_features)
    df_allframe_node_features_scaled = pd.DataFrame(allframe_node_features_scaled, columns=df_allframe_node_features.columns,
                                                    index=df_allframe_node_features.index)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. Attributed network embeeding
    #
    from karateclub import MUSAE
    from karateclub import SINE
    from karateclub import TENE
    from karateclub import TADW
    from karateclub import FSCNMF

    for idx, frame in enumerate(frame_list):
        if idx == 0: continue
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4.1 Node features
        #
        df_node_features_scaled = df_allframe_node_features_scaled.loc[frame]
        df_node_features_scaled.index = Par_id
        df_node_features = pd.merge(df_par_diameter_scaled, df_node_features_scaled, left_index=True, right_index=True)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4.2 Attributed network embeeding using karateclub
        #
        # transform numpy.matrix or array to scipy sparse matrix
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
        X = sparse.coo_matrix(df_node_features.values)

        model = MUSAE()
        model.fit(allframe_networks[idx], X)
        X_embeeding = model.get_embedding()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4.2 Save the results to dump files
        #
        fw = open(output_path + '/Attributed node embedding-' + str(frame) + '.dump', 'wb')
        pickle.dump(allframe_networks[idx], fw)
        pickle.dump(X_embeeding, fw)
        fw.close()

#==================================================================
# S T A R T
#
if __name__ == '__main__':
    
    file_path = None
    file_name = None
    case = 'shear-rate-2-press-1e6'
    test_id = 1
    d50 = 0.001
    time_step = 2e-8
    strain_window = 0.01
    scenario = 2
    connectivity = 'hard'
    epsilon = 100.0
    argList = argv
    argc = len(argList)
    i=0
    while (i < argc):
        if (argList[i][:2] == "-c"):
            i += 1
            case = str(argList[i])
        elif (argList[i][:2] == "-t"):
            i += 1
            test_id = int(argList[i])
        elif (argList[i][:2] == "-w"):
            i += 1
            strain_window = float(argList[i])
        elif (argList[i][:4] == "-sce"):
            i += 1
            scenario = int(argList[i])
        elif (argList[i][:2] == "-h"):            
            print(__doc__)
            exit(0)
        i += 1

    print(60 * '~')
    print("Running case:      %s" % case)
    print("Test id:           %d" % test_id)
    print("Strain window:     %.5f" % strain_window)
    print("Scenario:          %d" % scenario)
    print("Strain window:     %.5f" % strain_window)
    print("Connectivity:      %s" % connectivity)
    print(60 * '~')
    GraphEmbedding(case, test_id, d50, time_step, scenario, strain_window, connectivity)
