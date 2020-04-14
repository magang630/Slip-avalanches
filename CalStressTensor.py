# -*- coding: cp936 -*-
'''
CalStressTensor.py
Script to calculate the assembly averaged stress tensor based on contact force and branch vector.
Usage: python CalStressTensor.py -test 1
References:
    [1] Zhou, W., Liu, J., Ma, G., Yuan, W., Chang, X., 2016. Macroscopic and microscopic behaviors of granular materials under proportional strain path: a DEM study. Int. J. Numer. Anal. Methods Geomech. 40, 2450–2467.
    [2] Potyondy, D.O., Cundall, P.A., 2004. A bonded-particle model for rock. Int. J. Rock Mech. Min. Sci. 41, 1329–1364.
    [3] Zheng, W., Tannant, D.D., 2019. Influence of proppant fragmentation on fracture conductivity - Insights from three-dimensional discrete element modeling. J. Pet. Sci. Eng. 177, 1010–1023.
'''
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sys import argv, exit
import os
import numpy as np
# import pysal
import pyvoro
import scipy.stats as sts
import pickle
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Distance of two vertexes
#
def vertex_distance(a, b):
    return np.sqrt((a[0] - b[0])**2.0 + (a[1] - b[1])**2.0 + (a[2] - b[2])**2.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A spatial CG function, which is a normalized non-negative function with a single maximum at R=0 and a characteristic width w (i.e. the CG scale)
# e.g. in 2D, a Gaussian coarse-graining function.
def CG_function(pt, pt_i, width):
    R = vertex_distance(pt, pt_i)
    width = np.float(width)
    # return np.exp(-(R/width)**2.0)/(np.pi*width**2.0)
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Volume of convex hull with QHull from SciPy
#
def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d)))/6

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check whether a particle is located inside the region of interest
#
def check_inside_region(Par_coord, boundary_xgap, boundary_ygap, boundary_zgap):
    ParNum = Par_coord.shape[0]
    Par_inside_band = np.zeros(ParNum).astype(bool)
    domain_xmin, domain_xmax = np.min(Par_coord[:, 0]), np.max(Par_coord[:, 0])
    domain_ymin, domain_ymax = np.min(Par_coord[:, 1]), np.max(Par_coord[:, 1])
    domain_zmin, domain_zmax = np.min(Par_coord[:, 2]), np.max(Par_coord[:, 2])
    region_xmin, region_xmax = domain_xmin + boundary_xgap, domain_xmax - boundary_xgap
    region_ymin, region_ymax = domain_ymin + boundary_ygap, domain_ymax - boundary_ygap
    region_zmin, region_zmax = domain_zmin + boundary_zgap, domain_zmax - boundary_zgap
    for i in range(ParNum):
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
#
def CalStressTensor(case, test_id, d50, time_step, scenario):

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
        if file[:prefix_len] == file_prefix:
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
    # Array initialization
    #
    frame_stress = []
    frame_domain_volume = []
    frame_Par_penergy = {}

    youngsModulus = 25e9
    poissonsRatio = 0.25
    E = youngsModulus/(2*(1 - poissonsRatio**2))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 循环开始，提取第一步数据
    #
    output_path = file_path + '/test-' + str(test_id)
    force_path = file_path + '/test-' + str(test_id) + '/force'
    potential_path = file_path + '/test-' + str(test_id) + '/particle potential'
    mkdir(output_path)
    mkdir(force_path)
    mkdir(potential_path)
    for idx, frame in enumerate(frame_list):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Taking the whole dataset
        #
        force_info = open(force_path+'/dump-'+str(frame)+'.force','r')
        alllines = force_info.readlines()
        lines = alllines[9:]
        force_info.close()
        for i in range(len(lines)):
            if (lines[i] == '\n'): del lines[i]
        con_id        = np.array([int(line.strip().split(' ')[0]) for line in lines])    # 字段以逗号分隔，这里取得是第1列
        con_pari_xcor = np.array([float(line.strip().split(' ')[1]) for line in lines])  # 字段以逗号分隔，这里取得是第2列
        con_pari_ycor = np.array([float(line.strip().split(' ')[2]) for line in lines])  # 字段以逗号分隔，这里取得是第3列
        con_pari_zcor = np.array([float(line.strip().split(' ')[3]) for line in lines])  # 字段以逗号分隔，这里取得是第4列
        con_parj_xcor = np.array([float(line.strip().split(' ')[4]) for line in lines])  # 字段以逗号分隔，这里取得是第5列
        con_parj_ycor = np.array([float(line.strip().split(' ')[5]) for line in lines])  # 字段以逗号分隔，这里取得是第6列
        con_parj_zcor = np.array([float(line.strip().split(' ')[6]) for line in lines])  # 字段以逗号分隔，这里取得是第7列
        con_pari      = np.array([int(line.strip().split(' ')[7]) for line in lines])    # 字段以逗号分隔，这里取得是第8列
        con_parj      = np.array([int(line.strip().split(' ')[8]) for line in lines])    # 字段以逗号分隔，这里取得是第9列
        con_periodic  = np.array([int(line.strip().split(' ')[9]) for line in lines])    # 字段以逗号分隔，这里取得是第10列
        con_force_x   = np.array([float(line.strip().split(' ')[10]) for line in lines]) # 字段以逗号分隔，这里取得是第8列
        con_force_y   = np.array([float(line.strip().split(' ')[11]) for line in lines]) # 字段以逗号分隔，这里取得是第9列
        con_force_z   = np.array([float(line.strip().split(' ')[12]) for line in lines]) # 字段以逗号分隔，这里取得是第10列
        con_num = len(con_id)

        particle_info = open(dump_path + '/dump-' + str(frame) + '.sample', 'r')
        alllines = particle_info.readlines()
        lines = alllines[9:]
        particle_info.close()
        for i in range(len(lines)):
            if (lines[i] == '\n'): del lines[i]
        Par_id0 = np.array([int(line.strip().split(' ')[0]) for line in lines])       # 字段以逗号分隔，这里取得是第1列
        Par_type0 = np.array([int(line.strip().split(' ')[1]) for line in lines])     # 字段以逗号分隔，这里取得是第2列
        Par_radius0 = np.array([float(line.strip().split(' ')[2]) for line in lines]) # 字段以逗号分隔，这里取得是第3列
        Par_xcor0 = np.array([float(line.strip().split(' ')[3]) for line in lines])   # 字段以逗号分隔，这里取得是第4列
        Par_ycor0 = np.array([float(line.strip().split(' ')[4]) for line in lines])   # 字段以逗号分隔，这里取得是第5列
        Par_zcor0 = np.array([float(line.strip().split(' ')[5]) for line in lines])   # 字段以逗号分隔，这里取得是第6列
        Par_vx = np.array([float(line.strip().split(' ')[6]) for line in lines])     # 字段以逗号分隔，这里取得是第7列
        Par_vy = np.array([float(line.strip().split(' ')[7]) for line in lines])     # 字段以逗号分隔，这里取得是第8列
        Par_vz = np.array([float(line.strip().split(' ')[8]) for line in lines])     # 字段以逗号分隔，这里取得是第9列

        sorted_indices = np.argsort(Par_id0)
        Par_id = Par_id0[sorted_indices]
        Par_type = Par_type0[sorted_indices]
        Par_radius = Par_radius0[sorted_indices]
        Par_xcor = Par_xcor0[sorted_indices]
        Par_ycor = Par_ycor0[sorted_indices]
        Par_zcor = Par_zcor0[sorted_indices]

        # Par_id = Par_id[Par_type == 1]
        # Par_radius = Par_radius[Par_type == 1]
        # Par_xcor = Par_xcor[Par_type == 1]
        # Par_ycor = Par_ycor[Par_type == 1]
        # Par_zcor = Par_zcor[Par_type == 1]
        # Par_vx = Par_vx[Par_type == 1]
        # Par_vy = Par_vy[Par_type == 1]
        # Par_vz = Par_vz[Par_type == 1]
        Par_volume = 4./3.0*np.pi*Par_radius**3.0
        Par_vm = np.linalg.norm(np.stack((Par_vx, Par_vy, Par_vz), axis=1), axis=1)
        Par_num = np.size(Par_id)
        Par_volume_dict = dict(zip(Par_id, Par_volume))
        Par_radius_dict = dict(zip(Par_id, Par_radius))
        Par_type_dict = dict(zip(Par_id, Par_type))

        print(60*'*')
        print('Start frame:   %d' %frame)
        print(60*'*')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1.2 domain size
        #     Call the voro++ library
        #
        Par_coord = np.stack((Par_xcor, Par_ycor, Par_zcor), axis=1)
        # domain_volume = convex_hull_volume_bis(Par_coord)

        domain_length = 0.04
        domain_depth = 0.02
        # domain_height = np.max([np.max(con_pari_ycor), np.max(con_parj_ycor)]) - np.min([np.min(con_pari_ycor), np.min(con_parj_ycor)])
        # domain_length = np.max([np.max(con_pari_xcor), np.max(con_parj_xcor)]) - np.min([np.min(con_pari_xcor), np.min(con_parj_xcor)])
        # domain_height = np.max([np.max(con_pari_ycor), np.max(con_parj_ycor)]) - np.min([np.min(con_pari_ycor), np.min(con_parj_ycor)])
        # domain_depth  = np.max([np.max(con_pari_zcor), np.max(con_parj_zcor)]) - np.min([np.min(con_pari_zcor), np.min(con_parj_zcor)])
        # domain_volume = domain_length*domain_height*domain_depth

        domain_xmin, domain_xmax = 0, 0.04
        domain_zmin, domain_zmax = 0, 0.02
        # domain_xmin, domain_xmax = np.min(Par_coord[:, 0] - Par_radius), np.max(Par_coord[:, 0] + Par_radius)
        domain_ymin, domain_ymax = np.min(Par_coord[:, 1] - Par_radius), np.max(Par_coord[:, 1] + Par_radius)
        # domain_zmin, domain_zmax = np.min(Par_coord[:, 2] - Par_radius), np.max(Par_coord[:, 2] + Par_radius)
        container = [[domain_xmin, domain_xmax], [domain_ymin, domain_ymax], [domain_zmin, domain_zmax]]
        vor = pyvoro.compute_voronoi(Par_coord, limits=container, dispersion=4*d50, radii=Par_radius, periodic=[True, False, True])
        domain_volume = 0
        for i, cell in enumerate(vor):
            if Par_type[i] == 1: domain_volume += cell['volume']

        # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # # 1.3 fluctuating velocity
        # #
        # CG_vx = np.zeros(Par_num)
        # CG_vy = np.zeros(Par_num)
        # CG_vz = np.zeros(Par_num)
        #
        # # Spatial Weights
        # if idx % 10 == 0:
        #     weights_kernel = pysal.weights.DistanceBand(Par_coord, threshold=4.0*d50, binary=True)
        #     # weights_kernel = pysal.weights.insert_diagonal(weights_kernel)
        #
        # for i in range(Par_num):
        #     CG_volume = 0
        #     neighbor_list = weights_kernel.neighbors[i]
        #     for j in neighbor_list:
        #         # distij = ((Par_xcor_p[i] - Par_xcor_p[j])/wthresh_x)**2.0 + ((Par_ycor_p[i] - Par_ycor_p[j])/wthresh_y)**2.0
        #         # if distij > 1.0: continue
        #         CG_volume += Par_volume[j]
        #         CG_vx[i] += Par_volume[j]*Par_vx[j]
        #         CG_vy[i] += Par_volume[j]*Par_vy[j]
        #         CG_vz[i] += Par_volume[j]*Par_vz[j]
        #
        #     if CG_volume:
        #         CG_vx[i] /= CG_volume
        #         CG_vy[i] /= CG_volume
        #         CG_vz[i] /= CG_volume
        #     else:
        #         CG_vx[i], CG_vy[i], CG_vz[i] = 0.0, 0.0, 0.0
        #
        # Par_nonaffine_vx = Par_vx - CG_vx
        # Par_nonaffine_vy = Par_vy - CG_vy
        # Par_nonaffine_vz = Par_vz - CG_vz
        # Par_fluctuating_vel = np.array(zip(Par_nonaffine_vx, Par_nonaffine_vy, Par_nonaffine_vz))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2. Virial component
        #
        # 2.1 Pairwise interaction contribution: branch vector and contact force
        branch_vector_xcom = con_pari_xcor - con_parj_xcor
        branch_vector_ycom = con_pari_ycor - con_parj_ycor
        branch_vector_zcom = con_pari_zcor - con_parj_zcor
        branch_vector = np.stack((branch_vector_xcom, branch_vector_ycom, branch_vector_zcom), axis=1)
        contact_force = np.stack((con_force_x, con_force_y, con_force_z), axis=1)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2.2 virial stress tensor
        #
        Par_penergy = dict(zip(Par_id, np.zeros(Par_num)))
        Par_stress_tensor =dict(zip(Par_id, [np.zeros([3, 3]) for i in range(Par_num)]))
        virial_stress_tensor = np.zeros([3, 3])
        for c in range(con_num):
            Branch = branch_vector[c].reshape(3,1)
            # if con_periodic[c]:
            if Branch[0, 0] > 10*d50:
                Branch[0, 0] = Branch[0, 0] - domain_length
            if Branch[0, 0] < -10*d50:
                Branch[0, 0] = Branch[0, 0] + domain_length
            if Branch[2, 0] > 10*d50:
                Branch[2, 0] = Branch[2, 0] - domain_depth
            if Branch[2, 0] < -10*d50:
                Branch[2, 0] = Branch[2, 0] + domain_depth
            Force = contact_force[c].reshape(1, 3)
            R = Par_radius_dict[con_pari[c]]*Par_radius_dict[con_parj[c]]/(Par_radius_dict[con_pari[c]] + Par_radius_dict[con_parj[c]])
            F = np.dot(Force, Branch)/np.linalg.norm(Branch)
            F = np.linalg.norm(F)
            Par_penergy[con_pari[c]] += 4.0/15*E**(-2.0/3)*R**(-1.0/3)*(3.0/4*F)**(5/3)
            Par_penergy[con_parj[c]] += 4.0/15*E**(-2.0/3)*R**(-1.0/3)*(3.0/4*F)**(5/3)
            Par_stress_tensor[con_pari[c]] += np.outer(unit_vector(Branch), Force)
            Par_stress_tensor[con_parj[c]] += np.outer(unit_vector(Branch), Force)
            # if con_periodic[c]: continue
            if (Par_type_dict[con_pari[c]] == 2) and (Par_type_dict[con_parj[c]] == 2): continue
            virial_stress_tensor += np.outer(Branch, Force)
        virial_stress_tensor /= domain_volume

        Par_press = dict(zip(Par_id, np.zeros(Par_num)))
        Par_deviator_stress = dict(zip(Par_id, np.zeros(Par_num)))
        Par_sxy = dict(zip(Par_id, np.zeros(Par_num)))
        Par_snn = dict(zip(Par_id, np.zeros(Par_num)))
        assembly_stress_tensor = np.zeros([3, 3])
        for key, stress_tensor in Par_stress_tensor.items():
            stress_tensor = Par_radius_dict[key]/Par_volume_dict[key]*stress_tensor
            if Par_type_dict[key] == 1: assembly_stress_tensor += Par_volume_dict[key]*stress_tensor
            Par_snn[key] = (stress_tensor[0, 0] + stress_tensor[1, 1] + stress_tensor[2, 2])/3.0
            Par_sxy[key] = 0.5*(stress_tensor[0, 1] + stress_tensor[1, 0])
            eigenvalues, eigenvectors = np.linalg.eig(stress_tensor)
            Par_press[key] = np.sum(eigenvalues)/3.0
            Par_deviator_stress[key] = np.sqrt((eigenvalues[0] - eigenvalues[1])**2.0 + (eigenvalues[1] - eigenvalues[2])**2.0 +
                                               (eigenvalues[2] - eigenvalues[0])**2.0)/np.sqrt(2.0)
        assembly_stress_tensor /= domain_volume

        myOutputFile = open(potential_path + '/Particle potential-' + str(frame) + '.dump', 'w')
        myOutputFile.write("ITEM: TIMESTEP\n")
        myOutputFile.write("%-d\n" %frame)
        myOutputFile.write("ITEM: NUMBER OF ATOMS\n")
        myOutputFile.write("%-d\n" %len(Par_id[Par_type == 1]))
        myOutputFile.write("ITEM: BOX BOUNDS pp ff pp\n")
        myOutputFile.write("0 0.04\n")
        myOutputFile.write("-0.005 0.025\n")
        myOutputFile.write("0 0.02\n")
        myOutputFile.write("ITEM: ATOMS id type radius x y z vm p q snn sxy potential\n")
        for i in range(Par_num):
            if Par_type[i] != 1: continue
            myOutputFile.write("%d %d %.8f %.8f %.8f %.8f %.14f %.14f %.14f %.14f %.14f %.14f\n"
                                % (Par_id[i], Par_type[i], Par_radius[i],
                                   Par_xcor[i], Par_ycor[i], Par_zcor[i], Par_vm[i],
                                   Par_press[Par_id[i]], Par_deviator_stress[Par_id[i]],
                                   Par_snn[Par_id[i]], Par_sxy[Par_id[i]], Par_penergy[Par_id[i]]))
        myOutputFile.close()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3. kinetic energy contribution (temperature)
        #
        kinetic_stress_tensor = np.zeros([3,3])
        # for i in range(Par_num):
        #     fluctuating_vel = Par_fluctuating_vel[i].reshape(3,1)
        #     kinetic_stress_tensor += Par_mass[i]*(fluctuating_vel*Par_fluctuating_vel[i])
        # kinetic_stress_tensor /= domain_volume

        virial_stress_tensor = virial_stress_tensor + kinetic_stress_tensor
        sxx = virial_stress_tensor[0][0]
        syy = virial_stress_tensor[1][1]
        szz = virial_stress_tensor[2][2]
        sxy = 0.5*(virial_stress_tensor[0][1] + virial_stress_tensor[1][0])
        sxz = 0.5*(virial_stress_tensor[0][2] + virial_stress_tensor[2][0])
        syz = 0.5*(virial_stress_tensor[1][2] + virial_stress_tensor[2][1])

        sxx1 = assembly_stress_tensor[0][0]
        syy1 = assembly_stress_tensor[1][1]
        szz1 = assembly_stress_tensor[2][2]
        sxy1 = 0.5*(assembly_stress_tensor[0][1] + assembly_stress_tensor[1][0])
        sxz1 = 0.5*(assembly_stress_tensor[0][2] + assembly_stress_tensor[2][0])
        syz1 = 0.5*(assembly_stress_tensor[1][2] + assembly_stress_tensor[2][1])

        frame_stress.append([dump_frame[idx], dump_time[idx], sxx, syy, szz, sxy, sxz, syz, sxx1, syy1, szz1, sxy1, sxz1, syz1])
        frame_domain_volume.append([dump_frame[idx], dump_time[idx], domain_volume])

    myOutputFile = open(output_path + '/Particle assembly stress.dat', 'w')
    for i in range(len(frame_stress)):
        myOutputFile.write("%d, %12.4f, %12.4f, %12.4f, %12.4f, %12.4f, %12.4f, %12.4f, %12.4f, %12.4f, %12.4f, %12.4f, %12.4f, %12.4f\n"
                           %(frame_stress[i][0], frame_stress[i][1],
                             frame_stress[i][2], frame_stress[i][3], frame_stress[i][4],
                             frame_stress[i][5], frame_stress[i][6], frame_stress[i][7],
                             frame_stress[i][8], frame_stress[i][9], frame_stress[i][10],
                             frame_stress[i][11], frame_stress[i][12], frame_stress[i][13]))

    myOutputFile = open(output_path + '/Particle assembly volume.dat', 'w')
    for i in range(len(frame_domain_volume)):
        myOutputFile.write("%d, %12.10f, %12.10f\n"
                           %(frame_domain_volume[i][0], frame_domain_volume[i][1], frame_domain_volume[i][2]))

# ==================================================================
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
    scenario = 2
    argList = argv
    argc = len(argList)
    i = 0
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
        elif (argList[i][:4] == "-rat"):
            i += 1
            shear_rate = float(argList[i])
        elif (argList[i][:4] == "-sce"):
            i += 1
            scenario = int(argList[i])
        elif (argList[i][:2] == "-h"):
            print(__doc__)
            exit(0)
        i += 1

    print("Running case:      %s" % case)
    print("Test id:           %d" % test_id)
    print("Particle diameter: %.5f" % d50)
    print("Scenario:          %d" % scenario)
    CalStressTensor(case, test_id, d50, time_step, scenario)
