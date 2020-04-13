# -*- coding: cp936 -*-
'''
CalSpatialCorrelation.py
Usage: python CalSpatialCorrelation.py -test 1
References:
[1] Chikkadi V, Schall P. Nonaffine measures of particle displacements in sheared colloidal glasses[J]. Physical Review E, 2012, 85(3): 031402.
[2] Goldenberg C, Tanguy A, Barrat J L. Particle displacements in the elastic deformation of amorphous materials: Local fluctuations vs. non-affine field[J]. EPL (Europhysics Letters), 2007, 80(1): 16003.
[3] Goldhirsch I, Goldenberg C. On the microscopic foundations of elasticity[J]. The European Physical Journal E: Soft Matter and Biological Physics, 2002, 9(3): 245-251.
[4] Spatial correlation of elastic heterogeneity tunes the deformation behavior of metallic glasses
[5] Guo, N., Zhao, J., 2014. Local fluctuations and spatial correlations in granular flows under constant-volume quasistatic shear. Phys. Rev. E - Stat. Nonlinear, Soft Matter Phys. 89, 1–16.
[6] Chikkadi, V., & Schall, P. (2012). Nonaffine measures of particle displacements in sheared colloidal glasses. Physical Review E, 85(3), 031402. Cubuk, E. D., Ivancic, R. J. S., Schoenholz, S. S., Strickland, D. J., Basu, A., Davidson, Z. S., … Liu, A. J. (2017). Structure-property relationships from universal signatures of plasticity in disordered solids. Science, 358(6366), 1033–1037.
[7] Malins, A., Eggers, J., Tanaka, H., & Royall, C. P. (2013). Lifetimes and lengthscales of structural motifs in a model glassformer. Faraday Discussions, 167, 405–423.
[8] Watanabe, K., Kawasaki, T., & Tanaka, H. (2011). Structural origin of enhanced slow dynamics near a wall in glass-forming systems. Nature Materials, 10(7), 512–520.
[9] Hu, Y. C., Tanaka, H., & Wang, W. H. (2017). Impact of spatial dimension on structural ordering in metallic glass. Physical Review E, 96(2), 1–6. Kawasaki, T., Araki, T., & Tanaka, H. (2007). Correlation between dynamic heterogeneity and medium-range order in two-dimensional glass-forming liquids. Physical Review Letters, 99(21), 2–5.
[10] Malins, A., Eggers, J., Royall, C. P., Williams, S. R., & Tanaka, H. (2013). Identification of long-lived clusters and their link to slow dynamics in a model glass former. Journal of Chemical Physics, 138(12).
'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sys import argv,exit
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy import optimize
import numba as nb
# import pysal
# from numpy import *
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
# str转bool方法
#
def str_to_bool(str):
    return True if str.lower() == 'true' else False

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Angles between two n-dimensional vectors in Python
#
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
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))*180.0/np.pi

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
# Exponential function
# Critical-like behaviour of glass-forming liquids. Nature Materials, 9(4), 324–331. https://doi.org/10.1038/nmat2634
#
def log_OZ_3D_func(x, a, b):
    # 3D OZ correlation relation
    return np.log10(a) - np.log10(x)/2.0 - x/b*np.log10(np.e)

def log_OZ_2D_func(x, a, b):
    # 2D OZ correlation relation
    return np.log10(a) - np.log10(x)/4.0 - x/b*np.log10(np.e)

def OZ_3D_func(x, a, b):
    # 3D OZ correlation relation
    return a*x**(-1/2.0)*np.exp(-x/b)

def OZ_2D_func(x, a, b):
    # 2D OZ correlation relation
    return a*x**(-1/4.0)*np.exp(-x/b)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Power function
#
def power_func(x, a, b):
    # power law decay
    return a*x**(-b)

def log_power_func(x, a, b):
    # power law decay
    return np.log10(a) - b*np.log10(x)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Exponential function
#
def exp_func(x, a, b):
    # power law decay
    return a*np.exp(-x/b)

def log_exp_func(x, a, b):
    # power law decay
    return np.log10(a) - x/b*np.log10(np.e)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Power law distribution with exponential cutoff
# Truncated power law
#
def truncated_powerlaw(x, amp, alpha, beta):
    return amp*x**(-alpha)*np.exp(-x/beta)

def log_truncated_powerlaw(x, amp, alpha, beta):
    # power law decay
    return np.log10(amp) - alpha*np.log10(x) - (x/beta)*np.log10(np.e)

def fit_truncated_powerlaw(xdata, ydata, yerror):
    popt, pcov = optimize.curve_fit(log_truncated_powerlaw, xdata, np.log10(ydata), bounds=([0, 0, 0], [1000, 1000, 1000]))
    amp, alpha, beta = popt
    return popt

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
def is_outlier(points, threshold=2.0):
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
def plot_fittingline(ax, param, xdata, xerr, ydata, yerr, xlabel, ylabel, color='blue', label='Fit'):

    # plt.figure(figsize=(6,6))
    if (np.where(xerr!=0)[0].shape[0] != 0) or (np.where(yerr!=0)[0].shape[0] != 0):
        ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt='o', ecolor=color, color=color, elinewidth=2, capsize=4)
    else:
        ax.scatter(xdata, ydata, marker='o', color=color)

    if len(param) == 2:
        amp = param[0]
        index = param[1]
        ax.plot(xdata, power_func(xdata, amp, index), color=color, linewidth=2, label=label+"τ= %4.3f" %index)  #画拟合直线
        # ax.text(5e-5, 5e-3, "τ= %4.3f" %index, fontsize=12)
    else:
        amp = param[0]
        alpha = param[1]
        beta = param[2]
        ax.plot(xdata, truncated_powerlaw(xdata, amp, alpha, beta), color=color, linewidth=2, label=label+"τ= %4.3f, s$\mathregular{_*}$= %4.3f" % (alpha, beta))  #画拟合直线
        # ax.text(1e-4, 1e-1, "τ= %4.3f, s$\mathregular{_*}$= %4.3f" % (alpha, beta), fontsize=12)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(fontsize=12)
    # plt.savefig(file_path, dpi=500, bbox_inches = 'tight')
    # plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://studywolf.wordpress.com/2017/11/21/matplotlib-legends-for-mean-and-confidence-interval-plots/
# Mean and confidence interval plot
#
def plot_mean_and_CI(x, y, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(x, ub, lb, color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(x, y, color_mean)

class LegendObject(object):
    def __init__(self, facecolor='red', edgecolor='white', dashed=False):
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dashed = dashed

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle(
            # create a rectangle that is filled with color
            [x0, y0], width, height, facecolor=self.facecolor,
            # and whose edges are the faded color
            edgecolor=self.edgecolor, lw=3)
        handlebox.add_artist(patch)

        # if we're creating the legend for a dashed line,
        # manually add the dash in to our rectangle
        if self.dashed:
            patch1 = mpatches.Rectangle(
                [x0 + 2*width/5, y0], width/5, height, facecolor=self.edgecolor,
                transform=handlebox.get_transform())
            handlebox.add_artist(patch1)

        return patch

#@nb.jit(nopython=False)
def numba_correlation(Par_coord, Par_inside_region, d50, r_step, r_list,
                      Par_potential, Par_nonaffine_um, Par_temperature,
                      Par_D2min, Par_shear_strain):

    r_num = len(r_list)
    cr_count = np.zeros(r_num)
    cr_potential = np.zeros(r_num)
    cr_nonaffine_um = np.zeros(r_num)
    cr_temperature = np.zeros(r_num)
    cr_D2min = np.zeros(r_num)
    cr_shear_strain = np.zeros(r_num)

    normalized_cr_potential = np.zeros(r_num)
    normalized_cr_nonaffine_um = np.zeros(r_num)
    normalized_cr_temperature = np.zeros(r_num)
    normalized_cr_D2min = np.zeros(r_num)
    normalized_cr_shear_strain = np.zeros(r_num)

    square_mean_potential = np.mean(Par_potential*Par_potential)
    mean_square_potential = (np.mean(Par_potential))**2.0
    square_mean_nonaffine_um = np.mean(Par_nonaffine_um*Par_nonaffine_um)
    mean_square_nonaffine_um = (np.mean(Par_nonaffine_um))**2.0
    square_mean_temperature = np.mean(Par_temperature*Par_temperature)
    mean_square_temperature = (np.mean(Par_temperature))**2.0
    square_mean_D2min = np.mean(Par_D2min*Par_D2min)
    mean_square_D2min = (np.mean(Par_D2min))**2.0
    square_mean_shear_strain = np.mean(Par_shear_strain*Par_shear_strain)
    mean_square_shear_strain = (np.mean(Par_shear_strain))**2.0

    Par_num = Par_coord.shape[0]
    for i in range(Par_num):
        if not Par_inside_region[i]: continue
        for j in range(i, Par_num):
            if not Par_inside_region[j]: continue
            if i == j: continue
            # distij = vertex_distance(Par_coord[i], Par_coord[j])/d50
            distij = np.sqrt((Par_coord[i][0] - Par_coord[j][0])**2.0 + (Par_coord[i][1] - Par_coord[j][1])**2.0 + (Par_coord[i][2] - Par_coord[j][2])**2.0)
            distij /= d50
            k = int(np.ceil(distij/r_step)) - 1
            if (k >= r_num): continue
            if (k < 0): k = 0
            cr_count[k] += 1.0

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #  scalar product correlation function
            cr_potential[k] += Par_potential[i]*Par_potential[j]
            cr_nonaffine_um[k] += Par_nonaffine_um[i]*Par_nonaffine_um[j]
            cr_temperature[k] += Par_temperature[i]*Par_temperature[j]
            cr_D2min[k] += Par_D2min[i]*Par_D2min[j]
            cr_shear_strain[k] += Par_shear_strain[i]*Par_shear_strain[j]

    for k in range(r_num):
        if cr_count[k]:
            cr_potential[k] /= cr_count[k]
            cr_nonaffine_um[k] /= cr_count[k]
            cr_temperature[k] /= cr_count[k]
            cr_D2min[k] /= cr_count[k]
            cr_shear_strain[k] /= cr_count[k]
        else:
            cr_potential[k] = 0.0
            cr_nonaffine_um[k] = 0.0
            cr_temperature[k] = 0.0
            cr_D2min[k] = 0.0
            cr_shear_strain[k] = 0.0

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2.2 normalized spatial correlation function
        #
        normalized_cr_potential[k] = (cr_potential[k] - mean_square_potential)/(square_mean_potential - mean_square_potential)
        normalized_cr_nonaffine_um[k] = (cr_nonaffine_um[k] - mean_square_nonaffine_um)/(square_mean_nonaffine_um - mean_square_nonaffine_um)
        normalized_cr_temperature[k] = (cr_temperature[k] - mean_square_temperature)/(square_mean_temperature - mean_square_temperature)
        normalized_cr_D2min[k] = (cr_D2min[k] - mean_square_D2min)/(square_mean_D2min - mean_square_D2min)
        normalized_cr_shear_strain[k] = (cr_shear_strain[k] - mean_square_shear_strain)/(square_mean_shear_strain - mean_square_shear_strain)

    range_correlation = {}
    range_correlation['potential'] = normalized_cr_potential
    range_correlation['nonaffine_um'] = normalized_cr_nonaffine_um
    range_correlation['temperature'] = normalized_cr_temperature
    range_correlation['D2min'] = normalized_cr_D2min
    range_correlation['shear_strain'] = normalized_cr_shear_strain

    return range_correlation

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Spatial correlation
#
def CalSpatialCorrelation(case, test_id, d50, shear_strain, shear_rate, time_step, steady_strain, scenario, strain_window):

    file_path = os.path.pardir + '/' + case
    # dump files
    dump_path = file_path + '/test-' + str(test_id) + '/particle potential'
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
    steady_frame = int(start_frame + (end_frame - start_frame)*steady_strain/shear_strain)
    frame_interval = (end_frame - start_frame)/scenario
    frame_list = np.arange(start_frame, end_frame, frame_interval)
    frame_list = np.append(frame_list, end_frame)
    frame_list = frame_list.astype(int)

    steady_range = frame_list > steady_frame
    frame_list = frame_list[steady_range]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. Array initialization
    #
    sns.set()
    frame_num = len(frame_list)
    frame_strain = [[] for i in range(frame_num)]
    frame_cr_potential = [[] for i in range(frame_num)]
    frame_cr_nonaffine_um = [[] for i in range(frame_num)]
    frame_cr_temperature = [[] for i in range(frame_num)]
    frame_cr_D2min = [[] for i in range(frame_num)]
    frame_cr_shear_strain = [[] for i in range(frame_num)]

    r_step = 0.2
    r_list = np.arange(1.0, 20, r_step)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. Load data set
    #
    # data file path and results file path
    output_path = file_path + '/test-' + str(test_id) + '/strain window-' + str(round(strain_window, 3)) + '/spatial correlation'
    dynamics_path = file_path + '/test-' + str(test_id) + '/strain window-' + str(round(strain_window, 3)) + '/particle dynamics'
    potential_path = file_path + '/test-' + str(test_id) + '/particle potential'
    mkdir(output_path)
    mkdir(dynamics_path)
    mkdir(potential_path)

    for idx, frame in enumerate(frame_list):
        strain = frame_time[frame]*shear_rate
        frame_strain[idx] = str(round(strain, 3))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2.1 Particle nonaffine measures (by Falk & Langer, it measures the deviation from affine motion and symmetry)
        #
        particle_info = open(dynamics_path + '/Particle dynamics-' + str(frame) + '.dump', 'r')
        alllines = particle_info.readlines()
        lines = alllines[9:]
        particle_info.close()
        for i in range(len(lines)):
            if (lines[i] == '\n'): del lines[i]
        Par_id                = np.array([int(line.strip().split(' ')[0]) for line in lines])     # 字段以逗号分隔，这里取得是第1列
        Par_type              = np.array([int(line.strip().split(' ')[1]) for line in lines])     # 字段以逗号分隔，这里取得是第2列
        Par_radius            = np.array([float(line.strip().split(' ')[2]) for line in lines])   # 字段以逗号分隔，这里取得是第3列
        Par_xcor              = np.array([float(line.strip().split(' ')[3]) for line in lines])   # 字段以逗号分隔，这里取得是第4列
        Par_ycor              = np.array([float(line.strip().split(' ')[4]) for line in lines])   # 字段以逗号分隔，这里取得是第5列
        Par_zcor              = np.array([float(line.strip().split(' ')[5]) for line in lines])   # 字段以逗号分隔，这里取得是第6列
        Par_um                = np.array([float(line.strip().split(' ')[9]) for line in lines])   # 字段以逗号分隔，这里取得是第10列
        Par_nonaffine_um      = np.array([float(line.strip().split(' ')[13]) for line in lines])  # 字段以逗号分隔，这里取得是第14列
        Par_temperature       = np.array([float(line.strip().split(' ')[14]) for line in lines])  # 字段以逗号分隔，这里取得是第15列
        Par_D2min             = np.array([float(line.strip().split(' ')[15]) for line in lines])  # 字段以逗号分隔，这里取得是第16列
        Par_volumetric_strain = np.array([float(line.strip().split(' ')[16]) for line in lines])  # 字段以逗号分隔，这里取得是第17列
        Par_shear_strain      = np.array([float(line.strip().split(' ')[17]) for line in lines])  # 字段以逗号分隔，这里取得是第18列

        Par_id = Par_id[Par_type == 1]
        Par_xcor = Par_xcor[Par_type == 1]
        Par_ycor = Par_ycor[Par_type == 1]
        Par_zcor = Par_zcor[Par_type == 1]
        Par_nonaffine_um = Par_nonaffine_um[Par_type == 1]
        Par_temperature = Par_temperature[Par_type == 1]
        Par_D2min = Par_D2min[Par_type == 1]
        Par_shear_strain = Par_shear_strain[Par_type == 1]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2.2 Particle potential energy
        #
        particle_info = open(potential_path + '/Particle potential-' + str(frame) + '.dump', 'r')
        alllines = particle_info.readlines()
        lines = alllines[9:]
        particle_info.close()
        for i in range(len(lines)):
            if (lines[i] == '\n'): del lines[i]
        Par_id        = np.array([int(line.strip().split(' ')[0]) for line in lines])  # 字段以逗号分隔，这里取得是第1列
        Par_type      = np.array([int(line.strip().split(' ')[1]) for line in lines])  # 字段以逗号分隔，这里取得是第2列
        Par_potential = np.array([float(line.strip().split(' ')[6]) for line in lines])  # 字段以逗号分隔，这里取得是第7列
        Par_id = Par_id[Par_type == 1]
        Par_potential = Par_potential[Par_type == 1]

        Par_num = len(Par_id)
        Par_coord = np.stack((Par_xcor, Par_ycor, Par_zcor), axis=1)
        # Par_inside_region = check_inside_region(Par_coord, boundary_gap=[5*d50, 2*d50, 2*d50])
        Par_inside_region = np.ones(Par_num).astype(bool)

        print(60*'*')
        print('Frame:            %d' %frame)
        print('Shear strain:     %.5f' %strain)
        print('Particle number:  %d' %Par_num)
        print(60*'*')

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2. The existence of q-Gaussian distribution for the nonaffine displacements implies that the variables are long-range correlated, and it is also consistent
        #    with the obeservations of power-law decayed distributions. The power-law distribution that we observe gives evidence of critical behavior of the driven glass.
        #    To elucidate this critical behavior, we measure how far the fluctuations of non-affine displacements extend in space.
        #    We determine the  spatial correlation function
        #
        range_correlation = numba_correlation(Par_coord, Par_inside_region, d50, r_step, r_list,
                                              Par_potential, Par_nonaffine_um, Par_temperature,
                                              Par_D2min, Par_shear_strain)
        frame_cr_potential[idx] = range_correlation['potential']
        frame_cr_nonaffine_um[idx] = range_correlation['nonaffine_um']
        frame_cr_temperature[idx] = range_correlation['temperature']
        frame_cr_D2min[idx] = range_correlation['D2min']
        frame_cr_shear_strain[idx] = range_correlation['shear_strain']

    frame_cr_potential = np.transpose(np.array(frame_cr_potential))
    frame_cr_nonaffine_um = np.transpose(np.array(frame_cr_nonaffine_um))
    frame_cr_temperature = np.transpose(np.array(frame_cr_temperature))
    frame_cr_D2min = np.transpose(np.array(frame_cr_D2min))
    frame_cr_shear_strain = np.transpose(np.array(frame_cr_shear_strain))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3. Save results to dump files
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fw = open(output_path + '/Spatial correlation.dump','wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(r_list, fw)
    pickle.dump(frame_strain, fw)
    pickle.dump(frame_cr_potential, fw)
    pickle.dump(frame_cr_nonaffine_um, fw)
    pickle.dump(frame_cr_temperature, fw)
    pickle.dump(frame_cr_D2min, fw)
    pickle.dump(frame_cr_shear_strain, fw)
    fw.close()

def PostProcess(case, test_id, d50):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. Save results to dump files
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    file_path = os.path.pardir + '/' + case
    output_path = file_path + '/test-' + str(test_id) + '/strain window-' + str(round(strain_window, 3)) + '/spatial correlation'
    fw = open(output_path + '/Spatial correlation.dump', 'rb')
    r_list = pickle.load(fw)
    frame_strain = pickle.load(fw)
    frame_cr_potential = pickle.load(fw)
    frame_cr_nonaffine_um = pickle.load(fw)
    frame_cr_temperature = pickle.load(fw)
    frame_cr_D2min = pickle.load(fw)
    frame_cr_shear_strain = pickle.load(fw)
    frame_num = len(frame_strain)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. Analysis
    #
    df_cr_potential = pd.DataFrame(frame_cr_potential, index=r_list, columns=frame_strain)
    df_potential_stats = pd.DataFrame({'Mean': np.mean(frame_cr_potential, axis=1), 'Std':np.std(frame_cr_potential, axis=1)}, index=r_list)
    df_potential_stats['LB'] = df_potential_stats['Mean'].values - 1.96*df_potential_stats['Std'].values/np.sqrt(frame_num)
    df_potential_stats['UB'] = df_potential_stats['Mean'].values + 1.96*df_potential_stats['Std'].values/np.sqrt(frame_num)
    df_cr_potential = pd.concat([df_cr_potential, df_potential_stats], axis=1)

    df_cr_D2min = pd.DataFrame(frame_cr_D2min, index=r_list, columns=frame_strain)
    df_D2min_stats = pd.DataFrame({'Mean': np.mean(frame_cr_D2min, axis=1), 'Std':np.std(frame_cr_D2min, axis=1)}, index=r_list)
    df_D2min_stats['LB'] = df_D2min_stats['Mean'].values - 1.96*df_D2min_stats['Std'].values/np.sqrt(frame_num)
    df_D2min_stats['UB'] = df_D2min_stats['Mean'].values + 1.96*df_D2min_stats['Std'].values/np.sqrt(frame_num)
    df_cr_D2min = pd.concat([df_cr_D2min, df_D2min_stats], axis=1)

    df_cr_shear_strain = pd.DataFrame(frame_cr_shear_strain, index=r_list, columns=frame_strain)
    df_shear_strain_stats = pd.DataFrame({'Mean': np.mean(frame_cr_shear_strain, axis=1), 'Std':np.std(frame_cr_shear_strain, axis=1)}, index=r_list)
    df_shear_strain_stats['LB'] = df_shear_strain_stats['Mean'].values - 1.96*df_shear_strain_stats['Std'].values/np.sqrt(frame_num)
    df_shear_strain_stats['UB'] = df_shear_strain_stats['Mean'].values + 1.96*df_shear_strain_stats['Std'].values/np.sqrt(frame_num)
    df_cr_shear_strain = pd.concat([df_cr_shear_strain, df_shear_strain_stats], axis=1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3. exponential curve fitting
    #    3D Ornstein– Zernike correlation function
    #    references:
    #    [1] Kawasaki, T., & Tanaka, H. (2011). Structural signature of slow dynamics and dynamic heterogeneity in two-dimensional colloidal liquids: Glassy structural order. Journal of Physics Condensed Matter, 23(19).
    #    [2] Kawasaki, T., Araki, T., & Tanaka, H. (2007). Correlation between dynamic heterogeneity and medium-range order in two-dimensional glass-forming liquids. Physical Review Letters, 99(21), 2–5.
    #    [3] Tanaka, H., Kawasaki, T., Shintani, H., & Watanabe, K. (2010). Critical-like behaviour of glass-forming liquids. Nature Materials, 9(4), 324–331.
    xdata = r_list
    lb, ub, threshold = 2, 20, 1e-6
    frame_D2min_fit_params = [[] for i in range(frame_num)]
    frame_shear_strain_fit_params = [[] for i in range(frame_num)]
    for idx, strain in enumerate(frame_strain):
        ydata = df_cr_D2min.iloc[:, idx]
        fit_range = (ydata >= threshold) & (xdata <= ub) & (xdata >= lb)
        popt1, pcov1 = optimize.curve_fit(exp_func, xdata[fit_range], ydata[fit_range])
        frame_D2min_fit_params[idx] = popt1

        ydata = df_cr_shear_strain.iloc[:, idx]
        fit_range = (ydata >= threshold) & (xdata <= ub) & (xdata >= lb)
        popt1, pcov1 = optimize.curve_fit(exp_func, xdata[fit_range], ydata[fit_range])
        frame_shear_strain_fit_params[idx] = popt1

    df_D2min_fit_params = pd.DataFrame(np.array(frame_D2min_fit_params), columns=['Amp', 'Xi'])
    df_shear_strain_fit_params = pd.DataFrame(np.array(frame_shear_strain_fit_params), columns=['Amp', 'Xi'])

    xdata, ydata = r_list, df_cr_D2min['Mean']
    fit_range = (ydata >= threshold) & (xdata <= ub) & (xdata >= lb)
    power_coeff = fit_truncated_powerlaw(xdata[fit_range], ydata[fit_range], np.zeros(np.sum(fit_range == True)))
    D2min_popt1, pcov1 = optimize.curve_fit(exp_func, xdata[fit_range], ydata[fit_range])
    D2min_fit_params = power_coeff

    xdata, ydata = r_list, df_cr_shear_strain['Mean']
    fit_range = (ydata >= threshold) & (xdata <= ub) & (xdata >= lb)
    power_coeff = fit_truncated_powerlaw(xdata[fit_range], ydata[fit_range], np.zeros(np.sum(fit_range == True)))
    shear_strain_popt1, pcov1 = optimize.curve_fit(exp_func, xdata[fit_range], ydata[fit_range])
    shear_strain_fit_params = power_coeff

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3. Visualization setting
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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. Plot
    #
    nrows, ncols, size = 1, 2, 4
    fig = plt.figure(figsize=(ncols*size, nrows*size))
    ax1 = plt.subplot(nrows, ncols, 1)
    ax2 = plt.subplot(nrows, ncols, 2)

    ax1.scatter(r_list, df_cr_D2min['Mean'], s=20, color=colors[0], marker=markers[0])
    ax1.plot(r_list, truncated_powerlaw(r_list, D2min_fit_params[0], D2min_fit_params[1], D2min_fit_params[2]), linestyle='-', color=colors[1], linewidth=2,
             label="α=%4.3f, β=%4.3f" %(D2min_fit_params[1], D2min_fit_params[2]))
    ax1.plot(r_list, exp_func(r_list, D2min_popt1[0], D2min_popt1[1]), linestyle='-', color=colors[2], linewidth=2,
             label="ξ=%4.3f" % (D2min_popt1[1]))
    # ax1.errorbar(strain_interval, df_cr_D2min['Mean'].values, xerr=None, yerr=df_cr_D2min['Std'].values,
    #              fmt=markers[0], ecolor=colors[0], color=colors[0], elinewidth=2, capsize=4)

    ax2.scatter(r_list, df_cr_shear_strain['Mean'], s=20, color=colors[0], marker=markers[0])
    ax2.plot(r_list, truncated_powerlaw(r_list, shear_strain_fit_params[0], shear_strain_fit_params[1], shear_strain_fit_params[2]), color=colors[1], linestyle='-', linewidth=2,
             label="α=%4.3f, β=%4.3f" %(shear_strain_fit_params[1], shear_strain_fit_params[2]))
    ax2.plot(r_list, exp_func(r_list, shear_strain_popt1[0], shear_strain_popt1[1]), linestyle='-', color=colors[2], linewidth=2,
             label="ξ=%4.3f" % (shear_strain_popt1[1]))
    # ax2.errorbar(strain_interval, y=df_cr_shear_strain['Mean'].values, yerr=df_cr_shear_strain['Std'].values,
    #              fmt=markers[1], ecolor=colors[1], color=colors[1], elinewidth=2, capsize=4)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('$r(d_{50})$', fontsize=12)
    ax1.set_ylabel('$C_{D_{min}^2}(r)$', fontsize=12)
    ax1.set_xlim(xmin=1)
    ax1.set_ylim(ymax=1)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_title('Particle D2min', fontsize=12)
    ax1.legend(fontsize=12)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('$r(d_{50})$', fontsize=12)
    ax2.set_ylabel('$C_{\epsilon_{\gamma}}(r)$', fontsize=12)
    ax2.set_xlim(xmin=1)
    ax2.set_ylim(ymax=1)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_title('Particle shear strain', fontsize=12)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    file_path = output_path + '/Spatial correlation.png'
    plt.savefig(file_path, dpi=600, bbox_inches='tight')
    plt.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 5. Output
    #
    writer = pd.ExcelWriter(output_path + '/Spatial correlation.xlsx')
    df_cr_D2min.to_excel(writer, sheet_name='Particle D2min')
    df_cr_shear_strain.to_excel(writer, sheet_name='Particle shear strain')
    df_D2min_fit_params.to_excel(writer, sheet_name='Exponential fitting of D2min')
    df_shear_strain_fit_params.to_excel(writer, sheet_name='Exponential fitting of e')
    writer.save()
    writer.close()


#==================================================================
# S T A R T
#
if __name__ == '__main__':
    
    file_path = None
    file_name = None
    case = 'shear-rate-2-press-1e6'
    test_id = 1
    time_step = 2e-8
    shear_rate = 2.0
    shear_strain = 5.0
    steady_strain = 1.0
    strain_window = 0.001
    scenario = 100
    d50 = 0.001
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
        elif (argList[i][:4] == "-ste"):
            i += 1
            steady_strain = float(argList[i])
        elif (argList[i][:4] == "-sce"):
            i += 1
            scenario = int(argList[i])
        elif (argList[i][:4] == "-vis"):
            i += 1
            visualization = str_to_bool(argList[i])
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
    print("Steady state upon: %.5f" % steady_strain)
    print("Strain window:     %.5f" % strain_window)
    print("Scenario:          %d" % scenario)
    print(60 * '~')
    if visualization:
        PostProcess(case, test_id, d50)
    else:
        CalSpatialCorrelation(case, test_id, d50, shear_strain, shear_rate, time_step, steady_strain, scenario,
                               strain_window)

