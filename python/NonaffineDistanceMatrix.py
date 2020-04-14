# -*- coding: cp936 -*-
'''
CalParNonaffinity.py
Script to calculate the nonaffine distance matrix
Script to calculate the nonaffine measure in terms of the deviations from the global deformation.
Usage: python NonaffineDistanceMatrix.py -test 1
References:
[1] Chikkadi V, Schall P. Nonaffine measures of particle displacements in sheared colloidal glasses[J]. Physical Review E, 2012, 85(3): 031402.
[2] Goldenberg C, Tanguy A, Barrat J L. Particle displacements in the elastic deformation of amorphous materials: Local fluctuations vs. non-affine field[J]. EPL (Europhysics Letters), 2007, 80(1): 16003.
[3] Goldhirsch I, Goldenberg C. On the microscopic foundations of elasticity[J]. The European Physical Journal E: Soft Matter and Biological Physics, 2002, 9(3): 245-251.
[4] Guo, N., & Zhao, J. (2014). Local fluctuations and spatial correlations in granular flows under constant-volume quasistatic shear. Physical Review E - Statistical, Nonlinear, and Soft Matter Physics, 89(4), 1–16.
[5] Rognon, P., Miller, T., & Einav, I. (2015). A circulation-based method for detecting vortices in granular materials. Granular Matter, 17(2), 177–188.
[6] Cao, P., Short, M.P., Yip, S., 2019. Potential energy landscape activations governing plastic flows in glass rheology. Proc. Natl. Acad. Sci. U. S. A. 116, 18790–18797. doi:10.1073/pnas.1907317116
'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sys import argv,exit
import os
import pyvoro
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numba as nb
# import pysal

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
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
# function for nice Latex style tick formatting
# copied from
# http://stackoverflow.com/questions/25983218/
# scientific-notation-colorbar-in-matplotlib
# output formating for colorbar in 2D plots
def fmt(x, pos):
  a, b = '{:.2e}'.format(x).split('e')
  b = int(b)
  return r'${} \times 10^{{{}}}$'.format(a, b)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# @nb.jit(nopython=True)
#
def CalD2min(Par_coord_n, Par_coord_p, Par_volume, vor_nearest_neighbors, domain_length, domain_depth):

    Par_num = len(Par_volume)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. particles across the periodic boundary
    #
    Par_periodicity = np.zeros(Par_num).astype(bool)
    for i in range(Par_num):
        if ((Par_coord_n[i, 0] - Par_coord_p[i, 0]) >= 20*d50):
            Par_periodicity[i] = True
            Par_coord_p[i, 0] += domain_length
        elif ((Par_coord_n[i, 0] - Par_coord_p[i, 0]) < -20*d50):
            Par_periodicity[i] = True
            Par_coord_p[i, 0] -= domain_length
        elif ((Par_coord_n[i, 2] - Par_coord_p[i, 2]) >= 10*d50):
            Par_periodicity[i] = True
            Par_coord_p[i, 2] += domain_depth
        elif ((Par_coord_n[i, 2] - Par_coord_p[i, 2]) < -10*d50):
            Par_periodicity[i] = True
            Par_coord_p[i, 2] -= domain_depth
        else:
            Par_periodicity[i] = False

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2 Array initialization
    #
    Par_D2min = np.zeros(Par_num)
    # Par_shear_strain = np.zeros(Par_num)
    # Par_volumetric_strain = np.zeros(Par_num)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3. D2min
    #    A local measure of nonaffinity is achieved by mapping the motion of nearest-neighbor particles on an affine transformation.
    #    This is done by identifying the nearest neighbors of each particle and determining the best affine tensor that transforms the
    #    nearest-neighbor vectors on the time vector δt.
    #    References:
    #    [1] Chikkadi V, Schall P. Nonaffine measures of particle displacements in sheared colloidal glasses[J]. Physical Review E, 2012, 85(3): 031402.
    #    [2] Nordstrom K N, Lim E, Harrington M, et al. Granular dynamics during impact[J]. Physical review letters, 2014, 112(22): 228002.
    #    [3] Guo N, Zhao J. Local fluctuations and spatial correlations in granular flows under constant-volume quasistatic shear[J]. Physical Review E, 2014, 89(4): 042208.
    for i in range(Par_num):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3.1 Neighbor list of particle i
        #

        # if distance == 'euclidean':
        #     neighbor_list = wid.neighbors[i]
        if distance == 'manhattan':
            neighbor_list_tem = [i]
            neighbor_list = []
            neighbor_mark = []
            for j in range(int(neighborhood_size)):
                for k in neighbor_list_tem:
                    if k in neighbor_mark: continue
                    neighbor_mark.append(k)
                    neighbor_list.extend(vor_nearest_neighbors[k])
                neighbor_list_tem = list(set(neighbor_list))
        neighbor_list = list(set(neighbor_list))
        if i in neighbor_list: neighbor_list.remove(i)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3.2 Calculate the affine transformation tensor
        #
        CG_volume = 0
        matrix_X = np.zeros([3, 3])
        matrix_Y = np.zeros([3, 3])
        matrix_affine = np.zeros([3, 3])
        for j in neighbor_list:
            CG_volume += Par_volume[j]
            pos_relative_p = Par_coord_p[j] - Par_coord_p[i]
            pos_relative_n = Par_coord_n[j] - Par_coord_n[i]

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
        # 3.3 Calculate the local strain tensor, volumetric strain, and deviatoric strain
        #
        # local_strain_tensor = -(matrix_affine + matrix_affine.T)/2
        # evals, evecs = np.linalg.eig(local_strain_tensor)
        # Par_volumetric_strain[i] = np.sum(evals)
        # Par_shear_strain[i] = np.max(evals) - np.min(evals)
        # Par_shear_strain[i] = np.sqrt(2.0/9.0*((evals[0] - evals[1])**2.0 + (evals[1] - evals[2])**2.0 + (evals[0] - evals[2])**2.0))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3.4 Calculate the deviation from locally affine motion
        #
        for j in neighbor_list:
            pos_relative_p = Par_coord_p[j] - Par_coord_p[i]
            pos_relative_n = Par_coord_n[j] - Par_coord_n[i]
            # matrix_temp = (matrix_affine + matrix_eye)*np.matrix(pos_relative_p.reshape(-1, 1))
            matrix_temp = np.dot((matrix_affine + matrix_eye), pos_relative_p.reshape(-1, 1))
            vector_temp = np.squeeze(np.asarray(matrix_temp))
            vector = pos_relative_n - vector_temp
            Par_D2min[i] += np.sum(vector*vector)*Par_volume[j]
        # Par_D2min[i] /= len(neighbor_list)
        Par_D2min[i] /= CG_volume

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. D2min normalized by mean particle diameter
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Par_D2min /= d50**2.0

    return Par_D2min

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
def CalNaDM(case, test_id, d50, distance, neighborhood_size, shear_rate, time_step, start_strain, end_strain, strain_interval):

    file_path = os.path.pardir + '/' + case
    output_path = file_path + '/test-' + str(test_id) + '/nonaffine distance matrix'
    mkdir(output_path)

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
    dump_frame = np.array(sorted(dump_frame))

    frame_list = dump_frame.astype(int)
    frame_time = (dump_frame - np.min(dump_frame))*time_step
    shear_strain = frame_time*shear_rate
    dict_frame_time = dict(zip(frame_list, frame_time))
    dict_frame_strain = dict(zip(frame_list, shear_strain))
    strain_window = np.array([2e-3, 4e-3, 1e-2, 2e-2, 4e-2])
    frame_window = int(round(strain_window/shear_rate/time_step))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Pandas DataFrame initialization
    #
    frame_range = (shear_strain >= start_strain) & (shear_strain < end_strain)
    strain_range = shear_strain[frame_range]
    frame_sub_list = frame_list[frame_range]
    df_ASNaD = pd.DataFrame(np.zeros([len(strain_range), len(strain_window)]), index=strain_range, columns=strain_window)
    df_NaDM = pd.DataFrame(np.zeros([len(strain_range), len(shear_strain)]), index=strain_range, columns=shear_strain)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Begin the loop
    #
    domain_length, domain_depth = 0.04, 0.02
    for idx, frame_n in enumerate(frame_sub_list):

        print(60*'*')
        print('Frame:        %d' % frame_n)
        print('Frame time:   %.5f' % dict_frame_time[frame_n])
        print('Shear strain: %.5f' % dict_frame_strain[frame_n])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Particle configuration at frame_n
        #
        particle_info = open(dump_path + '/dump-' + str(frame_n) + '.sample', 'r')
        alllines = particle_info.readlines()
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
        Par_volume = 4./3.0*np.pi*Par_radius**3.0
        Par_num = np.size(Par_id)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # neighbor particles in the current configuration
        #
        Par_coord_n = np.stack((Par_xcor_n, Par_ycor_n, Par_zcor_n), axis=1)
        # Par_inside_region = check_inside_region(Par_coord_n, boundary_gap=[2*d50, 2*d50, 2*d50])
        # Par_num_region = np.sum(Par_inside_region == True)
        # Par_id_region = Par_id[Par_inside_region]

        if distance == 'euclidean':
            # Spatial Weights
            wthresh = float(neighborhood_size)*d50
            wid = pysal.weights.DistanceBand(Par_coord_n, threshold=wthresh, binary=True)
            # wid = pysal.weights.insert_diagonal(wid, np.zeros(wid.n))

            # k-nearest neighbor weights
            # The neighbors for a given observations can be defined using a k-nearest neighbor criterion
            # wknn3 = pysal.weights.KNN(Par_coord_n, k=3)
        elif distance == 'manhattan':
            # Call the voro++ library
            domain_xmin, domain_xmax = 0.0, domain_length
            domain_zmin, domain_zmax = 0.0, domain_depth
            # domain_xmin, domain_xmax = np.min(Par_coord_n[:, 0] - Par_radius), np.max(Par_coord_n[:, 0] + Par_radius)
            domain_ymin, domain_ymax = np.min(Par_coord_n[:, 1] - Par_radius), np.max(Par_coord_n[:, 1] + Par_radius)
            # domain_zmin, domain_zmax = np.min(Par_coord_n[:, 2] - Par_radius), np.max(Par_coord_n[:, 2] + Par_radius)
            container = [[domain_xmin, domain_xmax], [domain_ymin, domain_ymax], [domain_zmin, domain_zmax]]
            vor = pyvoro.compute_voronoi(Par_coord_n, limits=container, dispersion=4*d50, radii=Par_radius, periodic=[True, False, True])
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

        for jdx, frame_shift in enumerate(frame_window):
            frame_bwd = int(frame_n - frame_shift/2)
            frame_fwd = int(frame_n + frame_shift/2)
            if (frame_bwd not in frame_list) or (frame_fwd not in frame_list): continue

            # Start point
            particle_info = open(dump_path + '/dump-' + str(frame_bwd) + '.sample', 'r')
            alllines = particle_info.readlines()
            lines = alllines[9:]
            particle_info.close()
            for i in range(len(lines)):
                if (lines[i] == '\n'): del lines[i]
            Par_id0 = np.array([int(line.strip().split(' ')[0]) for line in lines])  # 字段以逗号分隔，这里取得是第1列
            Par_xcor0 = np.array([float(line.strip().split(' ')[3]) for line in lines])  # 字段以逗号分隔，这里取得是第4列
            Par_ycor0 = np.array([float(line.strip().split(' ')[4]) for line in lines])  # 字段以逗号分隔，这里取得是第5列
            Par_zcor0 = np.array([float(line.strip().split(' ')[5]) for line in lines])  # 字段以逗号分隔，这里取得是第6列

            sorted_indices = np.argsort(Par_id0)
            Par_xcor_bwd = Par_xcor0[sorted_indices]
            Par_ycor_bwd = Par_ycor0[sorted_indices]
            Par_zcor_bwd = Par_zcor0[sorted_indices]

            # End point
            particle_info = open(dump_path + '/dump-' + str(frame_fwd) + '.sample', 'r')
            alllines = particle_info.readlines()
            lines = alllines[9:]
            particle_info.close()
            for i in range(len(lines)):
                if (lines[i] == '\n'): del lines[i]
            Par_id0 = np.array([int(line.strip().split(' ')[0]) for line in lines])  # 字段以逗号分隔，这里取得是第1列
            Par_xcor0 = np.array([float(line.strip().split(' ')[3]) for line in lines])  # 字段以逗号分隔，这里取得是第4列
            Par_ycor0 = np.array([float(line.strip().split(' ')[4]) for line in lines])  # 字段以逗号分隔，这里取得是第5列
            Par_zcor0 = np.array([float(line.strip().split(' ')[5]) for line in lines])  # 字段以逗号分隔，这里取得是第6列

            sorted_indices = np.argsort(Par_id0)
            Par_xcor_fwd = Par_xcor0[sorted_indices]
            Par_ycor_fwd = Par_ycor0[sorted_indices]
            Par_zcor_fwd = Par_zcor0[sorted_indices]

            Par_coord_bwd = np.stack((Par_xcor_bwd, Par_ycor_bwd, Par_zcor_bwd), axis=1)
            Par_coord_fwd = np.stack((Par_xcor_fwd, Par_ycor_fwd, Par_zcor_fwd), axis=1)
            Par_D2min = CalD2min(Par_coord_fwd, Par_coord_bwd, Par_volume, vor_nearest_neighbors, domain_length, domain_depth)
            df_ASNaD.iloc[idx, jdx] = np.sum(Par_D2min[Par_type == 1]*Par_volume[Par_type == 1])/np.sum(Par_volume[Par_type == 1])

            print('\n')
            print('Frame bwd:    %d' % frame_bwd)
            print('Frame fwd:    %d' % frame_fwd)
            print('D2min:        %.8f' % df_ASNaD.iloc[idx, jdx])

        for jdx, frame_p in enumerate(frame_list):
            if abs((frame_n - frame_p))*time_step*shear_rate > strain_interval:
                # df_NaDM.iloc[idx, jdx] = float('inf')
                # df_NaDM.iloc[jdx, idx] = float('inf')
                continue
            elif frame_n == frame_p: 
                continue

            particle_info = open(dump_path + '/dump-' + str(frame_p) + '.sample', 'r')
            alllines = particle_info.readlines()
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
            Par_coord_p = np.stack((Par_xcor_p, Par_ycor_p, Par_zcor_p), axis=1)
            
            if frame_n > frame_p:
                Par_D2min = CalD2min(Par_coord_n, Par_coord_p, Par_volume, vor_nearest_neighbors, domain_length, domain_depth)
            else:
                Par_D2min = CalD2min(Par_coord_p, Par_coord_n, Par_volume, vor_nearest_neighbors, domain_length, domain_depth)
            # df_NaDM.iloc[idx, jdx] = np.mean(Par_D2min[Par_type == 1])
            df_NaDM.iloc[idx, jdx] = np.sum(Par_D2min[Par_type == 1]*Par_volume[Par_type == 1])/np.sum(Par_volume[Par_type == 1])
            # df_NaDM.iloc[jdx, idx] = df_NaDM.iloc[idx, jdx]

            print('\n')
            print('Current frame:  %d' % frame_n)
            print('Previous frame: %d' % frame_p)
            print('D2min:          %.8f' % df_NaDM.iloc[idx, jdx])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save results to dump files
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fw = open(output_path + '/Nonaffine distance matrix-[' + str(lb) + '-' + str(ub) + '].dump','wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(df_ASNaD, fw)
    pickle.dump(df_NaDM, fw)
    fw.close()

def PostProcess(case, test_id, d50):

    file_path = os.path.pardir + '/' + case
    output_path = file_path + '/test-' + str(test_id) + '/nonaffine distance matrix'
    mkdir(output_path)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. Visualization setting
    #
    # sns.set_style('ticks')
    plt.style.use('seaborn-paper')
    my_palette = "bright"  # deep, muted, pastel, bright, dark, colorblind, Set3, husl, Paired
    sns.set_palette(my_palette)
    #不同类别用不同颜色和样式绘图
    # colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
    # colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    colors = sns.color_palette(my_palette, 10)
    markers = ['o', '^', 'v', 's', 'D', 'v', 'p', '*', '+']
    # markers = ['*', 'o', '^', 'v', 's', 'p', 'D', 'X']
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. Load dump file
    #
    fw = open(output_path + '/Nonaffine distance matrix.dump','rb')
    # Pickle dictionary using protocol 0.
    df_ASNaD = pickle.load(fw)
    df_NaDM = pickle.load(fw)
    fw.close()

    shear_strain = df_NaDM.index.values
    plot_range = (shear_strain >= 2.5) & (shear_strain <= 2.6)

    D2min_max = np.max(df_NaDM.values)
    for idx in range(len(df_NaDM)):
        df_NaDM.iloc[idx, idx] = 1e-9
    df_NaDM[df_NaDM == 0] = 10*D2min_max

    X, Y = np.meshgrid(df_NaDM.index, df_NaDM.columns)
    # https://www.stacknoob.com/s/4RHM9GKQSc8XhsfN5apy29
    fig, ax = plt.subplots()
    ax = plt.contourf(X, Y, df_NaDM.values, lvels=[0, D2min_max, 10], locator=ticker.MaxNLocator(100), aspect='auto', origin='lower')
    cbar = plt.colorbar(ax, orientation='vertical', format=ticker.FuncFormatter(fmt))

    # Automatic selection of levels works; setting the
    # log locator tells contourf to use a log scale:
    fig, ax = plt.subplots()
    # lev_exp = np.arange(np.floor(np.log10(1e-8) - 1), np.ceil(np.log10(D2min_max)+1))
    # levs = np.power(10, lev_exp)
    cs = ax.contourf(X, Y, df_NaDM.values, locator=ticker.LogLocator(), cmap=cm.PuBu_r)

    # Alternatively, you can manually set the levels
    # and the norm:
    # lev_exp = np.arange(np.floor(np.log10(z.min())-1),
    #                    np.ceil(np.log10(z.max())+1))
    # levs = np.power(10, lev_exp)
    # cs = ax.contourf(X, Y, z, levs, norm=colors.LogNorm())
    cbar = fig.colorbar(cs)

    plt.xlim([2.5, 2.6])
    plt.ylim([2.5, 2.6])
    plt.show()



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
    start_strain = 2.5
    end_strain = 2.6
    visualization = False
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
        elif (argList[i][:3] == "-lb"):
            i += 1
            start_strain = float(argList[i])
        elif (argList[i][:3] == "-ub"):
            i += 1
            end_strain = float(argList[i])
        elif (argList[i][:4] == "-ran"):
            i += 1
            strain_interval = float(argList[i])
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
    print("Neighborhood type: %s" % distance)
    print("Neighborhood size: %.5f" % neighborhood_size)
    print(60 * '~')
    if visualization:
        PostProcess(case, test_id, d50)
    else:
        CalNaDM(case, test_id, d50, distance, neighborhood_size, shear_rate, time_step, start_strain, end_strain, strain_interval)
