# -*- coding: cp936 -*-
'''
CalParNonaffinity.py
Script to calculate the nonaffine measure.Another local measure of nonaffinity is based on the fluctuations of particle
Script to calculate the nonaffine displacement in terms of the deviations from the global deformation.
Usage: python CalParNonaffineMeasure.py -test 1 -start 6400000 -strain 1.0 -delta_strain 0.01 -rate 0.5 -neighborhood 2
References:
[1] Chikkadi V, Schall P. Nonaffine measures of particle displacements in sheared colloidal glasses[J]. Physical Review E, 2012, 85(3): 031402.
[2] Goldenberg C, Tanguy A, Barrat J L. Particle displacements in the elastic deformation of amorphous materials: Local fluctuations vs. non-affine field[J]. EPL (Europhysics Letters), 2007, 80(1): 16003.
[3] Goldhirsch I, Goldenberg C. On the microscopic foundations of elasticity[J]. The European Physical Journal E: Soft Matter and Biological Physics, 2002, 9(3): 245-251.
[4] Guo, N., & Zhao, J. (2014). Local fluctuations and spatial correlations in granular flows under constant-volume quasistatic shear. Physical Review E - Statistical, Nonlinear, and Soft Matter Physics, 89(4), 1–16.
[5] Rognon, P., Miller, T., & Einav, I. (2015). A circulation-based method for detecting vortices in granular materials. Granular Matter, 17(2), 177–188.
'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sys import argv,exit
import os
import numpy as np
import pyvoro
import pickle
import networkx as nx
import scipy.stats as sts
# import pysal
# import boo
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def rightTrim(input,suffix):
    if (input.find(suffix) == -1):
        input = input + suffix
    return input

def mkdir(path):
    # 引入模块
    import os
 
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("/")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        print(path+' 创建成功')
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Distance of two vertexes
#
def vertex_distance(a, b):
    return np.sqrt((a[0] - b[0])**2.0 + (a[1] - b[1])**2.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A spatial CG function, which is a normalized non-negative function with a single maximum at R=0 and a characteristic width w (i.e. the CG scale)
# e.g. in 2D, a Gaussian coarse-graining function.
def CG_function(pt, pt_i, width):
    R = vertex_distance(pt, pt_i)
    width = np.float(width)
    #return np.exp(-(R/width)**2.0)/(np.pi*width**2.0)
    return np.exp(-0.5*(R/width)**2.0)/(np.sqrt(2.0*np.pi)*width)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculate angle (clockwise) between two points
#
def determinant(v, w):
    return v[0]*w[1] - v[1]*w[0]

def length(v):
    return np.sqrt(v[0]**2 + v[1]**2)

def dot_product(v, w):
    return v[0]*w[0] + v[1]*w[1]

def determinant(v, w):
    return v[0]*w[1] - v[1]*w[0]

def inner_angle(v, w):
    cosx = dot_product(v, w)/(length(v)*length(w))
    if cosx > 1.0: cosx = 1.0
    rad = np.arccos(cosx)  # in radians
    return rad*180/np.pi  # returns degrees

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector/np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check whether a particle is located inside the region of interest
#
def check_inside_region(Par_coord, boundary_gap):
    Par_num = Par_coord.shape[0]
    boundary_xgap = boundary_gap[0]
    boundary_ygap = boundary_gap[1]
    boundary_zgap = boundary_gap[2]
    Par_inside_band = np.zeros(Par_num).astype(bool)
    domain_xmin, domain_xmax = np.min(Par_coord[:, 0]), np.max(Par_coord[:, 0])
    domain_ymin, domain_ymax = np.min(Par_coord[:, 1]), np.max(Par_coord[:, 1])
    domain_zmin, domain_zmax = np.min(Par_coord[:, 2]), np.max(Par_coord[:, 2])
    region_xmin, region_xmax = domain_xmin + boundary_xgap, domain_xmax - boundary_xgap
    region_ymin, region_ymax = domain_ymin + boundary_ygap, domain_ymax - boundary_ygap
    region_zmin, region_zmax = domain_zmin + boundary_zgap, domain_zmax - boundary_zgap
    for i in range(Par_num):
        x_flag = region_xmin <= Par_coord[i, 0] and Par_coord[i, 0] <= region_xmax
        y_flag = region_ymin <= Par_coord[i, 1] and Par_coord[i, 1] <= region_ymax
        z_flag = region_zmin <= Par_coord[i, 2] and Par_coord[i, 2] <= region_zmax
        if x_flag*y_flag*z_flag:
            Par_inside_band[i] = True
        else:
            Par_inside_band[i] = False
    return Par_inside_band

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# array normalization
#
def array_normalization(array):
    return (array - np.min(array))/(np.mean(array) - np.min(array))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# q-Exponential function
#
def func(x, q, lamda):
    return (2-q)*lamda*(1-lamda*(1-q)*x)**(1/(1-q))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# non-gaussian parameter
#
def non_gaussian(array, dimension):
    if dimension == 1:
        Cn3 = 1.0/3.0
    elif dimension == 2:
        Cn3 = 1.0/2.0
    elif dimension == 3:
        Cn3 = 3.0/5.0
    return Cn3*np.mean(array**4.0)/np.power(np.mean(array**2.0),2.0) - 1.0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 求单个样本估算的标准误
#
def standard_error(sample):
    std=np.std(sample, ddof=0)
    standard_error=std/np.sqrt(len(sample))
    return standard_error

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author : 余欢
# Date : Dec 28, 2015    4:09:29 PM
# company : 南京师范大学--大数据实验室
# description : 清理异常值
# https://blog.csdn.net/redaihanyu/article/details/50421773
#
def is_outlier(points, threshold=2):
    """
    返回一个布尔型的数组，如果数据点是异常值返回True，反之，返回False。

    数据点的值不在阈值范围内将被定义为异常值
    阈值默认为3.5
    """
    # 转化为向量
    if len(points.shape) == 1:
        points = points[:,None]

    # 数组的中位数
    median = np.median(points, axis=0)

    # 计算方差
    diff = np.sum((points - median)**2, axis=-1)
    #标准差
    diff = np.sqrt(diff)
    # 中位数绝对偏差
    med_abs_deviation = np.median(diff)

    # compute modified Z-score
    # http://www.itl.nist.gov/div898/handbook/eda/section4/eda43.htm#Iglewicz
    modified_z_score = 0.6745*diff/med_abs_deviation

    # return a mask for each outlier
    return modified_z_score > threshold

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Find the n walk neighbor list of particle i
#
def neighborhood(G, node, step):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)
    neighbor_list = []
    for node, length in path_lengths.iteritems():
        if length <= step: neighbor_list.append(node)
    return neighbor_list

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
def CalParNonaffinity(case, test_id, d50, distance, neighborhood_size, shear_rate, time_step, scenario, strain_window):

    file_path = os.path.pardir + '/' + case
    # dump files
    dump_path = file_path + '/test-' + str(test_id) + '/post'
    list_dir = os.listdir(dump_path)
    dump_frame = []
    file_prefix = 'dump-'
    file_suffix = '.sample'
    prefix_len = len(file_prefix)
    suffix_len = len(file_suffix)
    for file in list_dir:
        dump_frame.append(int(file[prefix_len:][:-suffix_len]))
    dump_frame = sorted(dump_frame)
    dump_time = (np.array(dump_frame) - np.min(dump_frame))*time_step
    frame_time = dict(zip(dump_frame, dump_time))

    start_frame = np.min(dump_frame)
    end_frame = np.max(dump_frame)
    frame_interval = (end_frame - start_frame)/scenario
    frame_list = np.arange(start_frame, end_frame, frame_interval)
    frame_list = np.append(frame_list, end_frame)
    frame_list = frame_list.astype(int)

    time_window = strain_window/shear_rate
    frame_window = int(time_window/time_step)

    # reference frame specified by variable start_frame
    # particle_info = open(dump_path+'/dump-'+str(start_frame)+'.sample','r')
    # alllines = particle_info.readlines()
    # time_p= float(alllines[1])*time_step
    # lines = alllines[9:]
    # particle_info.close()
    # for i in range(len(lines)):
        # if (lines[i] == '\n'): del lines[i]
    # Par_id0   = np.array([int(line.strip().split(' ')[0]) for line in lines])    # 字段以逗号分隔，这里取得是第1列
    # Par_type0 = np.array([int(line.strip().split(' ')[1]) for line in lines])    # 字段以逗号分隔，这里取得是第2列
    # Par_radius0 = np.array([float(line.strip().split(' ')[2]) for line in lines])  # 字段以逗号分隔，这里取得是第3列
    # Par_xcor0 = np.array([float(line.strip().split(' ')[3]) for line in lines])  # 字段以逗号分隔，这里取得是第4列
    # Par_ycor0 = np.array([float(line.strip().split(' ')[4]) for line in lines])  # 字段以逗号分隔，这里取得是第5列
    # Par_zcor0 = np.array([float(line.strip().split(' ')[5]) for line in lines])  # 字段以逗号分隔，这里取得是第6列

    # sorted_indices = np.argsort(Par_id0)
    # Par_id = Par_id0[sorted_indices]
    # Par_type = Par_type0[sorted_indices]
    # Par_xcor = Par_xcor0[sorted_indices]
    # Par_ycor = Par_ycor0[sorted_indices]
    # Par_zcor = Par_zcor0[sorted_indices]
    # Par_radius = Par_radius0[sorted_indices]
    # Par_diameter = 2.0*Par_radius
    # Par_area = np.pi*Par_radius**2.0
    # Par_volume = 4./3.0*np.pi*Par_radius**3.0
    # Par_num = np.size(Par_id)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Array initialization
    #
    frame_msd = []
    frame_statistics = []
    frame_temperature = []
    domain_length, domain_depth = 0.04, 0.02

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 循环开始，提取每一步数据
    #
    output_path = file_path + '/test-' + str(test_id) + '/strain window-' + str(round(strain_window, 3)) + '/particle dynamics'
    mkdir(output_path)
    for idx, frame in enumerate(frame_list):
        if idx == 0: continue
        if (frame - frame_window) not in dump_frame: continue
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Taking the whole dataset
        #
        # Start point
        particle_info = open(dump_path+'/dump-'+str(frame - frame_window)+'.sample','r')
        alllines = particle_info.readlines()
        #frame =  int(alllines[1])
        #time = float(alllines[1])*time_step
        lines = alllines[9:]
        particle_info.close()
        for i in range(len(lines)):
            if (lines[i] == '\n'): del lines[i]
        Par_id0 = np.array([int(line.strip().split(' ')[0]) for line in lines])  # 字段以逗号分隔，这里取得是第1列
        Par_xcor0 = np.array([float(line.strip().split(' ')[3]) for line in lines])  # 字段以逗号分隔，这里取得是第4列
        Par_ycor0 = np.array([float(line.strip().split(' ')[4]) for line in lines])  # 字段以逗号分隔，这里取得是第5列
        Par_zcor0 = np.array([float(line.strip().split(' ')[5]) for line in lines])  # 字段以逗号分隔，这里取得是第6列

        sorted_indices = np.argsort(Par_id0)
        Par_xcor_p = Par_xcor0[sorted_indices]
        Par_ycor_p = Par_ycor0[sorted_indices]
        Par_zcor_p = Par_zcor0[sorted_indices]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # End point
        #
        particle_info = open(dump_path+'/dump-'+str(frame)+'.sample','r')
        alllines = particle_info.readlines()
        #frame =  int(alllines[1])
        #time = float(alllines[1])*time_step
        lines = alllines[9:]
        particle_info.close()
        for i in range(len(lines)):
            if (lines[i] == '\n'): del lines[i]
        Par_id0 = np.array([int(line.strip().split(' ')[0]) for line in lines])  # 字段以逗号分隔，这里取得是第1列
        Par_type0 = np.array([int(line.strip().split(' ')[1]) for line in lines])    # 字段以逗号分隔，这里取得是第2列
        Par_radius0 = np.array([float(line.strip().split(' ')[2]) for line in lines])  # 字段以逗号分隔，这里取得是第3列
        Par_xcor0 = np.array([float(line.strip().split(' ')[3]) for line in lines])  # 字段以逗号分隔，这里取得是第4列
        Par_ycor0 = np.array([float(line.strip().split(' ')[4]) for line in lines])  # 字段以逗号分隔，这里取得是第5列
        Par_zcor0 = np.array([float(line.strip().split(' ')[5]) for line in lines])  # 字段以逗号分隔，这里取得是第6列

        sorted_indices = np.argsort(Par_id0)
        Par_id = Par_id0[sorted_indices]
        Par_type = Par_type0[sorted_indices]
        Par_xcor_n = Par_xcor0[sorted_indices]
        Par_ycor_n = Par_ycor0[sorted_indices]
        Par_zcor_n = Par_zcor0[sorted_indices]
        Par_radius = Par_radius0[sorted_indices]
        Par_diameter = 2.0*Par_radius
        Par_area = np.pi*Par_radius**2.0
        Par_volume = 4./3.0*np.pi*Par_radius**3.0
        Par_num = np.size(Par_id)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # particles across the periodic boundary
        #
        Par_periodicity = np.zeros(Par_num).astype(bool)
        for i in range(Par_num):
            if ((Par_xcor_n[i] - Par_xcor_p[i]) >= 20*d50):
                Par_periodicity[i] = True
                Par_xcor_p[i] += domain_length
            elif ((Par_xcor_n[i] - Par_xcor_p[i]) < -20*d50):
                Par_periodicity[i] = True
                Par_xcor_p[i] -= domain_length
            elif ((Par_zcor_n[i] - Par_zcor_p[i]) >= 10*d50):
                Par_periodicity[i] = True
                Par_zcor_p[i] += domain_depth
            elif ((Par_zcor_n[i] - Par_zcor_p[i]) < -10*d50):
                Par_periodicity[i] = True
                Par_zcor_p[i] -= domain_depth
            else:
                Par_periodicity[i] = False

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # neighbor particles in the current configuration
        #
        Par_coord = np.stack((Par_xcor_n, Par_ycor_n, Par_zcor_n), axis=1)
        # Par_inside_region = check_inside_region(Par_coord, boundary_gap=[2*d50, 2*d50, 2*d50])
        # Par_num_region = np.sum(Par_inside_region == True)
        # Par_id_region = Par_id[Par_inside_region]

        if distance == 'euclidean':
            # Spatial Weights
            wthresh = float(neighborhood_size)*d50
            wid = pysal.weights.DistanceBand(Par_coord, threshold=wthresh, binary=True)
            # wid = pysal.weights.insert_diagonal(wid, np.zeros(wid.n))

            # k-nearest neighbor weights
            # The neighbors for a given observations can be defined using a k-nearest neighbor criterion
            # wknn3 = pysal.weights.KNN(Par_coord, k=3)
        
        if distance == 'manhattan':
            # Call the voro++ library
            domain_xmin, domain_xmax = 0.0, domain_length
            domain_zmin, domain_zmax = 0.0, domain_depth
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
            del vor

            # # Assign graph attributes when creating a new graph
            # G = nx.Graph(series=frame)
            # # The graph G can be grown in several ways. We add one node at a time and assign node attributes when adding a new node
            # for i in range(Par_num):
            #     G.add_node(i)
            #     neighbor_list = vor_nearest_neighbors[i]
            #     if i in neighbor_list: neighbor_list.remove(i)
            #     for j in neighbor_list:
            #         # Add one edge at a time and add edge attributes using add_edge()
            #         if j > i: G.add_edge(i, j)

        print(60*'*')
        print('Start frame:   %d' %(frame - frame_window))
        print('End frame:     %d' %frame)
        print('Frame interval:%d' %frame_window)
        print('Time window:   %.6f' %time_window)
        print('Strain window: %.6f' %strain_window)
        print(60*'*')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2.1 Array initialization
        #
        Par_D2min = np.zeros(Par_num)
        Par_volumetric_strain = np.zeros(Par_num)
        Par_shear_strain = np.zeros(Par_num)

        CG_ux = np.zeros(Par_num)
        CG_uy = np.zeros(Par_num)
        CG_uz = np.zeros(Par_num)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2.2 提取上两步单元形心坐标，并计算颗粒位移和速度矢量
        #
        Par_ux = Par_xcor_n - Par_xcor_p
        Par_uy = Par_ycor_n - Par_ycor_p
        Par_uz = Par_zcor_n - Par_zcor_p
        Par_um = np.linalg.norm(np.stack((Par_ux, Par_uy, Par_uz), axis=1), axis=1)

        Par_vx = Par_ux/time_window
        Par_vy = Par_uy/time_window
        Par_vz = Par_uz/time_window
        Par_vm = Par_um/time_window

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3. A local measure of nonaffinity is achieved by mapping the motion of nearest-neighbor particles on an affine transformation.
        #    This is done by identifying the nearest neighbors of each particle and determining the best affine tensor that transforms the
        #    nearest-neighbor vectors on the time vector δt.
        #    References:
        #    [1] Chikkadi V, Schall P. Nonaffine measures of particle displacements in sheared colloidal glasses[J]. Physical Review E, 2012, 85(3): 031402.
        #    [2] Nordstrom K N, Lim E, Harrington M, et al. Granular dynamics during impact[J]. Physical review letters, 2014, 112(22): 228002.
        #    [3] Guo N, Zhao J. Local fluctuations and spatial correlations in granular flows under constant-volume quasistatic shear[J]. Physical Review E, 2014, 89(4): 042208.
        for i in range(Par_num):
            if distance == 'euclidean':
                neighbor_list = wid.neighbors[i]
            if distance == 'manhattan':
                # neighbor_list = neighborhood(G, i, neighborhood_size)
                # By Yuxiong Zou
                neighbor_list_tem = [i]
                neighbor_list = []
                neighbor_mark = []
                for j in range(int(neighborhood_size)):
                    for k in neighbor_list_tem:
                        if k in neighbor_mark: continue
                        neighbor_mark.append(k)
                        neighbor_list.extend(vor_nearest_neighbors[k])
                    neighbor_list_tem = list(set(neighbor_list))
            # neighbor_list_tem = [i]
            # neighbor_list = []
            # for j in range(int(neighborhood_size)):
            #     exec('neighbor_list{} = {}'.format(j, []))
            # for j in range(int(neighborhood_size)):
            #     for k in neighbor_list_tem:
            #         exec('neighbor_list{}.extend({})'.format(j, vor_nearest_neighbors[k]))
            #     exec('neighbor_list_tem = neighbor_list{}'.format(j))
            #     exec('neighbor_list.extend(neighbor_list{})'.format(j))
            neighbor_list = list(set(neighbor_list))
            if i in neighbor_list: neighbor_list.remove(i)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 3.1 Calculate the affine transformation tensor
            #
            CG_volume = 0
            matrix_X = np.zeros([3, 3])
            matrix_Y = np.zeros([3, 3])
            matrix_affine = np.zeros([3, 3])
            for j in neighbor_list:
                CG_volume += Par_volume[j]
                pos_relative_p = np.array([Par_xcor_p[j], Par_ycor_p[j], Par_zcor_p[j]]) - np.array([Par_xcor_p[i], Par_ycor_p[i], Par_zcor_p[i]])
                pos_relative_n = np.array([Par_xcor_n[j], Par_ycor_n[j], Par_zcor_n[j]]) - np.array([Par_xcor_n[i], Par_ycor_n[i], Par_zcor_n[i]])

                matrix_X += np.outer(pos_relative_n, pos_relative_p)*Par_volume[j]
                matrix_Y += np.outer(pos_relative_p, pos_relative_p)*Par_volume[j]

            matrix_X = matrix_X/CG_volume
            matrix_Y = matrix_Y/CG_volume
            matrix_eye = np.eye(3)
            try:
                matrix_Y_inv = np.linalg.inv(matrix_Y)
            except:
                continue
            else:
                matrix_affine = np.dot(matrix_X, matrix_Y_inv) - matrix_eye

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 3.2 Calculate the local strain tensor, volumetric strain, and deviatoric strain
            #
            local_strain_tensor = -(matrix_affine + matrix_affine.T)/2
            evals, evecs = np.linalg.eig(local_strain_tensor)
            Par_volumetric_strain[i] = np.sum(evals)
            # Par_shear_strain[i] = np.max(evals) - np.min(evals)
            Par_shear_strain[i] = np.sqrt(2.0/9.0*((evals[0] - evals[1])**2.0 + (evals[1] - evals[2])**2.0 + (evals[0] - evals[2])**2.0))

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 3.3 Calculate the deviation from locally affine motion
            #
            for j in neighbor_list:
                pos_relative_p = np.array([Par_xcor_p[j], Par_ycor_p[j], Par_zcor_p[j]]) - np.array([Par_xcor_p[i], Par_ycor_p[i], Par_zcor_p[i]])
                pos_relative_n = np.array([Par_xcor_n[j], Par_ycor_n[j], Par_zcor_n[j]]) - np.array([Par_xcor_n[i], Par_ycor_n[i], Par_zcor_n[i]])
                # matrix_temp = (matrix_affine + matrix_eye)*np.matrix(pos_relative_p.reshape(-1, 1))
                matrix_temp = np.dot((matrix_affine + matrix_eye), pos_relative_p.reshape(-1, 1))
                vector_temp = np.squeeze(np.asarray(matrix_temp))
                vector = pos_relative_n - vector_temp
                Par_D2min[i] += np.sum(vector*vector)*Par_volume[j]
            # Par_D2min[i] /= len(neighbor_list)
            Par_D2min[i] /= CG_volume

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 3.4 Calculate the coarse grained nonaffine displacement.
            #
            CG_volume = 0
            neighbor_list.append(i)
            for j in neighbor_list:
                # distij = ((Par_xcor_p[i] - Par_xcor_p[j])/wthresh_x)**2.0 + ((Par_ycor_p[i] - Par_ycor_p[j])/wthresh_y)**2.0
                # if distij > 1.0: continue
                CG_volume += Par_volume[j]
                CG_ux[i] += Par_ux[j]*Par_volume[j]
                CG_uy[i] += Par_uy[j]*Par_volume[j]
                CG_uz[i] += Par_uz[j]*Par_volume[j]

            if CG_volume:
                CG_ux[i] /= CG_volume
                CG_uy[i] /= CG_volume
                CG_uz[i] /= CG_volume
            else:
                CG_ux[i], CG_uy[i], CG_uz[i] = 0.0, 0.0, 0.0

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4. Calculate the nonaffine displacement.
        #    The nonaffine displecement is obtained by subtracting the global displacement field from the individual particle displacements.
        #    It describes the deviation of the displacement of particle i from the dispalcement profile crresponding to the average of the deformation
        #    field in the shear direction.
        #    References:
        #    [1] Ma G, Regueiro R A, Zhou W, et al. Role of particle crushing on particle kinematics and shear banding in granular materials[J]. Acta Geotechnica, 1-18.
        #    [2] Chikkadi V, Schall P. Nonaffine measures of particle displacements in sheared colloidal glasses[J]. Physical Review E, 2012, 85(3): 031402.
        #    [3] Rognon, P., Miller, T., & Einav, I. (2015). A circulation-based method for detecting vortices in granular materials. Granular Matter, 17(2), 177–188.
        Par_nonaffine_ux = Par_ux - CG_ux
        Par_nonaffine_uy = Par_uy - CG_uy
        Par_nonaffine_uz = Par_uz - CG_uz
        Par_nonaffine_um = np.linalg.norm(np.stack((Par_nonaffine_ux, Par_nonaffine_uy, Par_nonaffine_uz), axis=1), axis=1)
        Par_nonaffine_vx = Par_nonaffine_ux/time_window
        Par_nonaffine_vy = Par_nonaffine_uy/time_window
        Par_nonaffine_vz = Par_nonaffine_uz/time_window
        Par_nonaffine_vm = Par_nonaffine_um/time_window

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4.2 Particle temperature is defined in analogous to the kinetic temperature in the kinetic theory of gases.
        #     The granular temperature measures the fluctuations in the translational velocity of the particles and can be defiend as Tk = <Vi*Vi>/D
        #     Where D is the dimension of the system, Vi is the fluctuation in the i-th component of the velocity of a particle, and <..> indicates the ensemble average.
        # reference:
        # [1] Campbell C S. Granular material flows–an overview[J]. Powder Technology, 2006, 162(3): 208-229.
        # [2] Tenneti S, Garg R, Hrenya C M, et al. Direct numerical simulation of gas–solid suspensions at moderate Reynolds number: quantifying the coupling between hydrodynamic forces and particle velocity fluctuations[J]. Powder Technology, 2010, 203(1): 57-69.
        # [2] Sun L, Wang S, Lu H, et al. Simulations of configurational and granular temperatures of particles using DEM in roller conveyor[J]. Powder Technology, 2014, 268: 436-445.
        # [3] Wang T, He Y, Yan S, et al. Rotation characteristic and granular temperature analysis in a bubbling fluidized bed of binary particles[J]. Particuology, 2015, 18: 76-88.
        # [4] Sun Q, Jin F, Wang G, et al. On granular elasticity[J]. Scientific reports, 2015, 5: 9652.
        # [5] Lee C H, Huang Z, Chiew Y M. A three-dimensional continuum model incorporating static and kinetic effects for granular flows with applications to collapse of a two-dimensional granular column[J]. Physics of Fluids, 2015, 27(11): 113303.
        # [6] Artoni R, Richard P. Effective wall friction in wall-bounded 3D dense granular flows[J]. Physical review letters, 2015, 115(15): 158001.
        # Par_temperature = (Par_nonaffine_vx**2.0 + Par_nonaffine_vy**2.0 + Par_nonaffine_vz**2.0)/3.0
        Par_temperature = Par_nonaffine_um**2.0

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 5. 数据归一化
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Par_ux /= d50
        Par_uy /= d50
        Par_um /= d50
        Par_nonaffine_ux /= d50
        Par_nonaffine_uy /= d50
        Par_nonaffine_um /= d50
        Par_temperature /= d50**2.0
        Par_D2min /= d50**2.0

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 6. 输出颗粒D^2、颗粒位移、非仿射位移和颗粒温度
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        myOutputFile = open(output_path + '/Particle dynamics-' + str(frame) + '.dump', 'w')
        myOutputFile.write("ITEM: TIMESTEP\n")
        myOutputFile.write("%-d\n" %frame)
        myOutputFile.write("ITEM: NUMBER OF ATOMS\n")
        myOutputFile.write("%-d\n" %Par_num)
        myOutputFile.write("ITEM: BOX BOUNDS pp ff pp\n")
        myOutputFile.write("0 0.04\n")
        myOutputFile.write("-0.005 0.025\n")
        myOutputFile.write("0 0.02\n")
        myOutputFile.write("ITEM: ATOMS id type radius x y z ux uy uz um nonaffine_ux nonaffine_uy nonaffine_uz nonaffine_um temperature D2min eps_v eps_s\n")
        for i in range(Par_num):
            myOutputFile.write("%d %d %.8f %.8f %.8f %.8f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n"
                               % (Par_id[i], Par_type[i], Par_radius[i],
                                  Par_xcor_n[i], Par_ycor_n[i], Par_zcor_n[i],
                                  Par_ux[i], Par_uy[i], Par_uz[i], Par_um[i],
                                  Par_nonaffine_ux[i], Par_nonaffine_uy[i], Par_nonaffine_uz[i], Par_nonaffine_um[i],
                                  Par_temperature[i], Par_D2min[i], Par_volumetric_strain[i], Par_shear_strain[i]))
        myOutputFile.close()

        # myOutputFile = open(output_path + '/Particle displacement-' + str(frame) + '.dump', 'w')
        # myOutputFile.write("ITEM: TIMESTEP\n")
        # myOutputFile.write("%-d\n" %frame)
        # myOutputFile.write("ITEM: NUMBER OF ATOMS\n")
        # myOutputFile.write("%-d\n" %Par_num)
        # myOutputFile.write("ITEM: BOX BOUNDS pp ff pp\n")
        # myOutputFile.write("0 0.04\n")
        # myOutputFile.write("-0.005 0.025\n")
        # myOutputFile.write("0 0.02\n")
        # myOutputFile.write("ITEM: ATOMS id type radius x y z ux uy uz um nonaffine_ux nonaffine_uy nonaffine_uz nonaffine_um temperature\n")
        # for i in range(Par_num):
        #     myOutputFile.write("%d %d %.8f %.8f %.8f %.8f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n"
        #                         % (Par_id[i], Par_type[i], Par_radius[i],
        #                            Par_xcor_n[i], Par_ycor_n[i], Par_zcor_n[i],
        #                            Par_ux[i], Par_uy[i], Par_uz[i], Par_um[i],
        #                            Par_nonaffine_ux[i], Par_nonaffine_uy[i], Par_nonaffine_uz[i],
        #                            Par_nonaffine_um[i], Par_temperature[i]))
        # myOutputFile.close()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 7 数据分析
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 7.1 particle measurements inside the region of interest
        #
        numerator_ux = np.sum(Par_nonaffine_ux**2.0)
        numerator_uy = np.sum(Par_nonaffine_uy**2.0)
        numerator_uz = np.sum(Par_nonaffine_uz**2.0)
        numerator_um = np.sum(Par_nonaffine_um**2.0)
        numerator_temperature = np.sum(Par_temperature**2.0)

        denominator_ux = np.sum(Par_nonaffine_ux**4.0)
        denominator_uy = np.sum(Par_nonaffine_uy**4.0)
        denominator_uz = np.sum(Par_nonaffine_uz**4.0)
        denominator_um = np.sum(Par_nonaffine_um**4.0)
        denominator_temperature = np.sum(Par_temperature[i]**4.0)

        # participation_ratio = numerator**2.0/(Par_num*denominator)
        pr_nonaffine_ux = numerator_ux**2.0/(Par_num*denominator_ux)
        pr_nonaffine_uy = numerator_uy**2.0/(Par_num*denominator_uy)
        pr_nonaffine_uz = numerator_uz**2.0/(Par_num*denominator_uz)
        pr_nonaffine_um = numerator_um**2.0/(Par_num*denominator_um)
        pr_temperature = numerator_temperature**2.0/(Par_num*denominator_temperature)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 7.2 Statistics on the particle displacements and particle fluctuations
        #     Refernences:
        #     [1] Chikkadi V, Schall P. Nonaffine measures of particle displacements in sheared colloidal glasses[J]. Physical Review E, 2012, 85(3): 031402.
        #     [2] Goldenberg C, Tanguy A, Barrat J L. P

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Mean squared particle displacements and particle rotations
        #
        mean_squared_displacement = np.mean(Par_um*Par_um)
        mean_squared_nonaffine_displacement = np.mean(Par_nonaffine_um*Par_nonaffine_um)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # System averaged granular temperature, weighted by particle mass/area
        #
        system_temperature = np.mean(Par_temperature)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 7.3 Statistics
        #
        mean_nonaffine_um = np.mean(Par_nonaffine_um)
        std_nonaffine_um = np.std(Par_nonaffine_um)
        ske_nonaffine_um = sts.skew(Par_nonaffine_um)
        kur_nonaffine_um = sts.kurtosis(Par_nonaffine_um)
        alpha_nonaffine_um = non_gaussian(Par_nonaffine_um, dimension=2)

        mean_nonaffine_ux = np.mean(Par_nonaffine_ux)
        std_nonaffine_ux = np.std(Par_nonaffine_ux)
        ske_nonaffine_ux = sts.skew(Par_nonaffine_ux)
        kur_nonaffine_ux = sts.kurtosis(Par_nonaffine_ux)
        alpha_nonaffine_ux = non_gaussian(Par_nonaffine_ux, dimension=2)

        mean_nonaffine_uy = np.mean(Par_nonaffine_uy)
        std_nonaffine_uy= np.std(Par_nonaffine_uy)
        ske_nonaffine_uy = sts.skew(Par_nonaffine_uy)
        kur_nonaffine_uy = sts.kurtosis(Par_nonaffine_uy)
        alpha_nonaffine_uy= non_gaussian(Par_nonaffine_uy, dimension=2)

        mean_nonaffine_uz = np.mean(Par_nonaffine_uz)
        std_nonaffine_uz= np.std(Par_nonaffine_uz)
        ske_nonaffine_uz= sts.skew(Par_nonaffine_uz)
        kur_nonaffine_uz = sts.kurtosis(Par_nonaffine_uz)
        alpha_nonaffine_uz= non_gaussian(Par_nonaffine_uz, dimension=2)

        frame_msd.append([mean_squared_displacement, mean_squared_nonaffine_displacement])
        frame_temperature.append([system_temperature, pr_temperature])
        frame_statistics.append([mean_nonaffine_um, std_nonaffine_um, ske_nonaffine_um, kur_nonaffine_um, alpha_nonaffine_um, pr_nonaffine_um,
                                 mean_nonaffine_ux, std_nonaffine_ux, ske_nonaffine_ux, kur_nonaffine_ux, alpha_nonaffine_ux, pr_nonaffine_ux,
                                 mean_nonaffine_uy, std_nonaffine_uy, ske_nonaffine_uy, kur_nonaffine_uy, alpha_nonaffine_uy, pr_nonaffine_uy,
                                 mean_nonaffine_uz, std_nonaffine_uz, ske_nonaffine_uz, kur_nonaffine_uz, alpha_nonaffine_uz, pr_nonaffine_uz])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 8. Output the results
    #
    myOutputFile1 = open(output_path + '/Particle displacement statistics.dat', 'w')
    myOutputFile1.write("~~~~~~~~~~~~~~~~~ MSD\n")
    myOutputFile1.write("time window,strain window,   MSD,         MSND\n")
    for i in range(len(frame_msd)):
        myOutputFile1.write("%10.8f, %10.8f, %16.14f, %16.14f\n"  %(time_window, strain_window, frame_msd[i][0], frame_msd[i][1]))

    myOutputFile1.write("~~~~~~~~~~~~~~~~~ Particle kinematics\n")
    myOutputFile1.write("time window,    strain window,              mean_um,           std_um,           ske_um,            kur_um,           alpha_um,          pr_um\n")
    for i in range(len(frame_statistics)):
        statistics = frame_statistics[i]
        myOutputFile1.write("%10.8f, %10.8f, %16.14f, %16.14f, %16.14f, %16.14f, %16.14f, %16.14f\n"
                            %(time_window, strain_window, statistics[0], statistics[1], statistics[2], statistics[3], statistics[4], statistics[5]))

    myOutputFile1.write("~~~~~~~~~~~~~~~~~ Particle kinematics in x direction\n")
    myOutputFile1.write("time window,    strain window,              mean_ux,           std_ux,           ske_ux,            kur_ux,           alpha_ux,          pr_ux\n")
    for i in range(len(frame_statistics)):
        statistics = frame_statistics[i]
        myOutputFile1.write("%10.8f, %10.8f, %16.14f, %16.14f, %16.14f, %16.14f, %16.14f, %16.14f\n"
                            %(time_window, strain_window, statistics[6], statistics[7], statistics[8], statistics[9], statistics[10], statistics[11]))

    myOutputFile1.write("~~~~~~~~~~~~~~~~~ Particle kinematics in y direction\n")
    myOutputFile1.write("time window,    strain window,              mean_uy,           std_uy,           ske_uy,            kur_uy,           alpha_uy,          pr_uy\n")
    for i in range(len(frame_statistics)):
        statistics = frame_statistics[i]
        myOutputFile1.write("%10.8f, %10.8f, %16.14f, %16.14f, %16.14f, %16.14f, %16.14f, %16.14f\n"
                            %(time_window, strain_window, statistics[12], statistics[13], statistics[14], statistics[15], statistics[16], statistics[17]))

    myOutputFile1.write("~~~~~~~~~~~~~~~~~ Particle kinematics in z direction\n")
    myOutputFile1.write("time window,    strain window,              mean_uz,           std_uz,           ske_uz,            kur_uz,           alpha_uz,          pr_uz\n")
    for i in range(len(frame_statistics)):
        statistics = frame_statistics[i]
        myOutputFile1.write("%10.8f, %10.8f, %16.14f, %16.14f, %16.14f, %16.14f, %16.14f, %16.14f\n"
                            %(time_window, strain_window, statistics[18], statistics[19], statistics[20], statistics[21], statistics[22], statistics[23]))

    myOutputFile1.write("~~~~~~~~~~~~~~~~~ Particle temperature statistics\n")
    myOutputFile1.write("time window,    strain window,                temp,            pr_temp\n")
    for i in range(len(frame_temperature)):
        myOutputFile1.write("%10.8f, %10.8f, %16.14f, %16.14f\n"
                            %(time_window, strain_window, frame_temperature[i][0], frame_temperature[i][1]))
    myOutputFile1.write("\n")
    myOutputFile1.close()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 9. Save results to dump files
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fw = open(output_path + '/Particle displacement statistics.dump','wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(frame_msd, fw)
    pickle.dump(frame_temperature, fw)
    pickle.dump(frame_statistics, fw)

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
    shear_strain = 5.0
    shear_rate = 2.0
    strain_window = 0.01
    scenario = 100
    neighborhood_size = 1
    distance = 'manhattan'
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
        elif (argList[i][:4] == "-str"):
            i += 1
            shear_strain = float(argList[i])
        elif (argList[i][:2] == "-w"):
            i += 1
            strain_window = float(argList[i])
        elif (argList[i][:4] == "-rat"):
            i += 1
            shear_rate = float(argList[i])
        elif (argList[i][:2] == "-n"):
            i += 1
            neighborhood_size = float(argList[i])
        elif (argList[i][:4] == "-dis"):
            i += 1
            distance = str(argList[i])
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
    print("Particle diameter: %.5f" % d50)
    print("Shear rate:        %.5f" % shear_rate)
    print("Shear strain:      %.5f" % shear_strain)
    print("Scenario:          %d" % scenario)
    print("Strain window:     %.5f" % strain_window)
    print("Neighborhood type: %s" % distance)
    print("Neighborhood size: %.5f" % neighborhood_size)
    print(60 * '~')
    CalParNonaffinity(case, test_id, d50, distance, neighborhood_size, shear_rate, time_step, scenario, strain_window)
