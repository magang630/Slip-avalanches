# -*- coding: utf-8 -*-
'''
CalTemporalCorrelation.py
Script to calculate the time correlation of plasticity.
Usage:
python CalTemporalCorrelation.py -test 1
python CalTemporalCorrelation.py -test 2
References:
[1] Denisov, D. V., Lörincz, K.A., Uhl, J.T., Dahmen, K.A., Schall, P., 2016. Universality of slip avalanches in flowing granular matter. Nat. Commun. 7. doi:10.1038/ncomms10641
'''
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sys import argv, exit
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import pearsonr, spearmanr
# from scipy import stats
# from scipy.signal import savgol_filter
# from scipy.ndimage import gaussian_filter1d
# from scipy.ndimage import median_filter
# from scipy.signal import medfilt
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
# A spatial CG function, which is a normalized non-negative function with a single maximum at R=0 and a characteristic depth w (i.e. the CG scale)
# e.g. in 2D, a Gaussian coarse-graining function.
#
def CG_function(pt, pt_i, depth):
    R = vertex_distance(pt, pt_i)
    depth = np.float(depth)
    # return np.exp(-(R/depth)**2.0)/(np.pi*depth**2.0)
    return np.exp(-0.5*(R/depth)**2.0)/(np.sqrt(2.0*np.pi)*depth)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# array normalization
#
def array_normalization(array):
    # return (array - np.min(array))/(np.mean(array) - np.min(array))
    return array/np.mean(array)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# split the arr into N chunks
#
def chunks(arr, m):
    n = int(np.ceil(len(arr)/float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]

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
# q-Exponential function
#
def func(x, q, lamda):
    return (2 - q)*lamda*(1 - lamda*(1 - q)*x)**(1/(1 - q))

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
    return Cn3*np.mean(array**4.0)/np.power(np.mean(array**2.0), 2.0) - 1.0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# standard error
#
def standard_error(sample):
    std = np.std(sample, ddof=0)
    standard_error = std/np.sqrt(len(sample))
    return standard_error

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# array normalization
#
def array_normalization(array):
    return (array - np.min(array))/(np.mean(array) - np.min(array))

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
        points = points[:, None]

    # 数组的中位数
    median = np.median(points, axis=0)

    # 计算方差
    diff = np.sum((points - median)**2, axis=-1)
    # 标准差
    diff = np.sqrt(diff)
    # 中位数绝对偏差
    med_abs_deviation = np.median(diff)

    # compute modified Z-score
    # http://www.itl.nist.gov/div898/handbook/eda/section4/eda43.htm#Iglewicz
    modified_z_score = 0.6745*diff/med_abs_deviation

    # return a mask for each outlier
    return modified_z_score > threshold

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If your data is well-behaved, you can fit a power-law function by first converting to a linear equation by using the logarithm.
# Then use the optimize function to fit a straight line. Notice that we are weighting by positional uncertainties during the fit.
# Also, the best-fit parameters uncertainties are estimated from the variance-covariance matrix.
# You should read up on when it may not be appropriate to use this form of error estimation.
# Refereces: https://scipy-cookbook.readthedocs.io/items/FittingData.html#Fitting-a-power-law-to-data-with-errors
#
def powerlaw(x, amp, index):
    # Define function for calculating a power law
    powerlaw = lambda x, amp, index: amp*(x**index)

def fit_powerlaw(xdata, ydata, yerr, weight=False):
    ##########
    # Fitting the data -- Least Squares Method
    ##########

    # Power-law fitting is best done by first converting
    # to a linear equation and then fitting to a straight line.
    # Note that the `logyerr` term here is ignoring a constant prefactor.
    #
    #  y = a*x^b
    #  log10(y) = log10(a) + b*log10(x)
    #

    # Define function for calculating a power law
    powerlaw = lambda x, amp, index: amp*(x**index)

    log10x = np.log10(xdata)
    log10y = np.log10(ydata)
    if weight:
        log10yerr = yerr/ydata
    else:
        log10yerr = np.ones(len(log10x))
    # log10yerr[np.where(log10yerr == 0)] = 1

    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1]*x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x))/err

    pinit = [10.0, -10.0]
    out = optimize.leastsq(errfunc, pinit, args=(log10x, log10y, log10yerr), full_output=1)

    pfinal = out[0]
    covar = out[1]
    # print(pfinal)
    # print(covar)

    index = pfinal[1]
    amp = 10.0**pfinal[0]

    indexErr = np.sqrt(covar[1][1])
    ampErr = np.sqrt(covar[0][0])*amp

    ##########
    # Plotting data
    ##########

    # plt.clf()
    # plt.subplot(2, 1, 1)
    # plt.plot(xdata, powerlaw(xdata, amp, index))  # Fit
    # plt.errorbar(xdata, ydata, yerr=yerr, fmt='k.')  # Data
    # plt.text(5, 6.5, 'Ampli = %5.2f +/- %5.2f' % (amp, ampErr))
    # plt.text(5, 5.5, 'Index = %5.2f +/- %5.2f' % (index, indexErr))
    # plt.title('Best Fit Power Law')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.xlim(1, 11)
    #
    # plt.subplot(2, 1, 2)
    # plt.loglog(xdata, powerlaw(xdata, amp, index))
    # plt.errorbar(xdata, ydata, yerr=yerr, fmt='k.')  # Data
    # plt.xlabel('X (log scale)')
    # plt.ylabel('Y (log scale)')
    # plt.xlim(1.0, 11)

    return pfinal

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
    popt, pcov = optimize.curve_fit(log_truncated_powerlaw, xdata, np.log10(ydata), bounds=([0, 0, 0], [100, 100, 100]))
    amp, alpha, beta = popt
    return popt

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Time correlation
# [1] Denisov, D. V., Lörincz, K.A., Uhl, J.T., Dahmen, K.A., Schall, P., 2016. Universality of slip avalanches in flowing granular matter. Nat. Commun. 7.
#
#@nb.jit(nopython=False)
def numba_correlation(Par_metrics_t0, Par_metrics_t1, type=None):

    Par_num, metrics_num = Par_metrics_t0.shape
    metrics_corr = np.zeros(metrics_num)
    for i in range(metrics_num):
        if type == 'pearson':
            corr, p_value = pearsonr(Par_metrics_t0[:, i], Par_metrics_t1[:, i])
        elif type == 'spearman':
            corr, p_value = spearmanr(Par_metrics_t0[:, i], Par_metrics_t1[:, i])
        else:
            mean_metric_t0 = np.mean(Par_metrics_t0[:, i])
            mean_metric_t1 = np.mean(Par_metrics_t1[:, i])
            Chi_square = np.mean((Par_metrics_t0[:, i] - mean_metric_t0)**2.0)
            corr = np.mean((Par_metrics_t1[:, i] - mean_metric_t1)*(Par_metrics_t0[:, i] - mean_metric_t0))/Chi_square
        metrics_corr[i] = corr

    return metrics_corr

def TemporalCorrelation(case, test_id, shear_strain, shear_rate, steady_strain, time_step, scenario, strain_window):

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
    truncated_powerlaw = lambda x, amp, alpha, beta: amp*x**(-alpha)*np.exp(-x/beta)
    powerlaw = lambda x, amp, index: amp*(x**index)

    strain_interval = np.arange(1.0*strain_window, 40*strain_window, strain_window)
    frame_window = int(strain_window/shear_rate/time_step)
    frame_interval = np.arange(1*frame_window, 40*frame_window, frame_window)

    frame_num = len(frame_list)
    frame_strain = [[] for i in range(frame_num)]
    frame_time_cr_nonaffine_um = [[] for i in range(frame_num)]
    frame_time_cr_temperature = [[] for i in range(frame_num)]
    frame_time_cr_D2min = [[] for i in range(frame_num)]
    frame_time_cr_shear_strain = [[] for i in range(frame_num)]
    frame_time_cr_potential = [[] for i in range(frame_num)]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. Load data set
    #
    # data file path and results file path
    output_path = file_path + '/test-' + str(test_id) + '/strain window-' + str(round(strain_window, 3)) + '/temporal correlation'
    dynamics_path = file_path + '/test-' + str(test_id) + '/strain window-' + str(round(strain_window, 3)) + '/particle dynamics'
    potential_path = file_path + '/test-' + str(test_id) + '/particle potential'
    mkdir(output_path)
    mkdir(dynamics_path)
    mkdir(potential_path)

    for idx, frame in enumerate(frame_list):
        if frame+np.max(frame_interval) > end_frame: continue
        strain = (frame - start_frame)*time_step*shear_rate
        frame_strain[idx] = str(round(strain, 3))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2.1 Particle nonaffine measures at γ
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
        Par_nonaffine_um_t0 = Par_nonaffine_um[Par_type == 1]
        Par_temperature_t0 = Par_temperature[Par_type == 1]
        Par_D2min_t0 = Par_D2min[Par_type == 1]
        Par_shear_strain_t0 = Par_shear_strain[Par_type == 1]

        D2min_threshold = np.percentile(Par_D2min_t0, 0)
        Par_selected = Par_D2min_t0 > D2min_threshold
        Par_nonaffine_um_t0 = Par_nonaffine_um_t0[Par_selected]
        Par_temperature_t0 = Par_temperature_t0[Par_selected]
        Par_D2min_t0 = Par_D2min_t0[Par_selected]
        Par_shear_strain_t0 = Par_shear_strain_t0[Par_selected]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2.2 Particle potential energy at γ
        #
        # particle_info = open(potential_path + '/Particle potential-' + str(frame) + '.dump', 'r')
        # alllines = particle_info.readlines()
        # lines = alllines[9:]
        # particle_info.close()
        # for i in range(len(lines)):
        #     if (lines[i] == '\n'): del lines[i]
        # Par_id        = np.array([int(line.strip().split(' ')[0]) for line in lines])   # 字段以逗号分隔，这里取得是第1列
        # Par_type      = np.array([int(line.strip().split(' ')[1]) for line in lines])   # 字段以逗号分隔，这里取得是第2列
        # Par_potential = np.array([float(line.strip().split(' ')[6]) for line in lines]) # 字段以逗号分隔，这里取得是第7列
        # frame_par_potential_t0 = Par_potential[Par_type == 1]

        Par_metrics_t0 = np.stack((Par_nonaffine_um_t0, Par_temperature_t0, Par_D2min_t0, Par_shear_strain_t0), axis=1)
        time_corr = [[] for i in range(len(frame_interval))]
        for jdx, frame_shift in enumerate(frame_interval):

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 2.1 Particle nonaffine measures at γ+δγ
            #
            particle_info = open(dynamics_path + '/Particle dynamics-' + str(frame + frame_shift) + '.dump', 'r')
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
            Par_nonaffine_um_t1 = Par_nonaffine_um[Par_type == 1]
            Par_temperature_t1 = Par_temperature[Par_type == 1]
            Par_D2min_t1 = Par_D2min[Par_type == 1]
            Par_shear_strain_t1 = Par_shear_strain[Par_type == 1]

            Par_nonaffine_um_t1 = Par_nonaffine_um_t1[Par_selected]
            Par_temperature_t1 = Par_temperature_t1[Par_selected]
            Par_D2min_t1 = Par_D2min_t1[Par_selected]
            Par_shear_strain_t1 = Par_shear_strain_t1[Par_selected]

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 2.2 Particle potential energy at γ+δγ
            #
            # particle_info = open(potential_path + '/Particle potential-' + str(frame + frame_shift) + '.dump', 'r')
            # alllines = particle_info.readlines()
            # lines = alllines[9:]
            # particle_info.close()
            # for i in range(len(lines)):
            #     if (lines[i] == '\n'): del lines[i]
            # Par_id        = np.array([int(line.strip().split(' ')[0]) for line in lines])   # 字段以逗号分隔，这里取得是第1列
            # Par_type      = np.array([int(line.strip().split(' ')[1]) for line in lines])   # 字段以逗号分隔，这里取得是第2列
            # Par_potential = np.array([float(line.strip().split(' ')[6]) for line in lines]) # 字段以逗号分隔，这里取得是第7列
            # frame_par_potential_t1 = Par_potential[Par_type == 1]

            Par_metrics_t1 = np.stack((Par_nonaffine_um_t1, Par_temperature_t1, Par_D2min_t1, Par_shear_strain_t1), axis=1)
            metrics_corr = numba_correlation(Par_metrics_t0, Par_metrics_t1, type=None) # pearson spearman
            time_corr[jdx] = metrics_corr

        time_corr = np.array(time_corr)
        frame_time_cr_nonaffine_um[idx] = time_corr[:, 0]
        frame_time_cr_temperature[idx] = time_corr[:, 1]
        frame_time_cr_D2min[idx] = time_corr[:, 2]
        frame_time_cr_shear_strain[idx] = time_corr[:, 3]

    frame_strain = list(filter(lambda x: len(x) > 0, frame_strain))
    frame_time_cr_nonaffine_um = list(filter(lambda x: len(x) > 0, frame_time_cr_nonaffine_um))
    frame_time_cr_temperature = list(filter(lambda x: len(x) > 0, frame_time_cr_temperature))
    frame_time_cr_D2min = list(filter(lambda x: len(x) > 0, frame_time_cr_D2min))
    frame_time_cr_shear_strain = list(filter(lambda x: len(x) > 0, frame_time_cr_shear_strain))

    frame_num = len(frame_strain)
    frame_time_cr_nonaffine_um = np.transpose(np.array(frame_time_cr_nonaffine_um))
    frame_time_cr_temperature = np.transpose(np.array(frame_time_cr_temperature))
    frame_time_cr_D2min = np.transpose(np.array(frame_time_cr_D2min))
    frame_time_cr_shear_strain = np.transpose(np.array(frame_time_cr_shear_strain))

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

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. Analysis
    #
    df_time_cr_D2min = pd.DataFrame(frame_time_cr_D2min, index=strain_interval, columns=frame_strain)
    df_D2min_stats = pd.DataFrame({'Mean': np.mean(df_time_cr_D2min, axis=1), 'Std':np.std(df_time_cr_D2min, axis=1)}, index=strain_interval)
    df_D2min_stats['LB'] = df_D2min_stats['Mean'].values - 1.96*df_D2min_stats['Std'].values/np.sqrt(frame_num)
    df_D2min_stats['UB'] = df_D2min_stats['Mean'].values + 1.96*df_D2min_stats['Std'].values/np.sqrt(frame_num)
    df_time_cr_D2min = pd.concat([df_time_cr_D2min, df_D2min_stats], axis=1)

    df_time_cr_shear_strain = pd.DataFrame(frame_time_cr_shear_strain, index=strain_interval, columns=frame_strain)
    df_shear_strain_stats = pd.DataFrame({'Mean': np.mean(df_time_cr_shear_strain, axis=1), 'Std':np.std(df_time_cr_shear_strain, axis=1)}, index=strain_interval)
    df_shear_strain_stats['LB'] = df_shear_strain_stats['Mean'].values - 1.96*df_shear_strain_stats['Std'].values/np.sqrt(frame_num)
    df_shear_strain_stats['UB'] = df_shear_strain_stats['Mean'].values + 1.96*df_shear_strain_stats['Std'].values/np.sqrt(frame_num)
    df_time_cr_shear_strain = pd.concat([df_time_cr_shear_strain, df_shear_strain_stats], axis=1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Time correlation fitting by Power law
    # Denisov, D. V., Lörincz, K.A., Uhl, J.T., Dahmen, K.A., Schall, P., 2016. Universality of slip avalanches in flowing granular matter. Nat. Commun. 7.
    #
    xdata = strain_interval
    lb, ub, threshold = 1*strain_window, 40*strain_window, 1e-5
    frame_D2min_fit_params = [[] for i in range(frame_num)]
    frame_shear_strain_fit_params = [[] for i in range(frame_num)]
    # frame_potential_fit_params = [[] for i in range(frame_num)]
    for idx, strain in enumerate(frame_strain):
        ydata = df_time_cr_D2min.iloc[:, idx]
        fit_range = (ydata >= threshold) & (xdata <= ub) & (xdata >= lb)
        # popt1, pcov1 = optimize.curve_fit(log_OZ_3D_func, xdata[fit_range], np.log10(ydata[fit_range]), bounds=(0, [1000., 1000.]))
        # power_coeff = fit_truncated_powerlaw(xdata[fit_range], ydata[fit_range], np.zeros(np.sum(fit_range == True)))
        power_coeff = fit_powerlaw(xdata[fit_range], ydata[fit_range], np.zeros(np.sum(fit_range == True)), weight=False)
        frame_D2min_fit_params[idx] = power_coeff

        ydata = df_time_cr_shear_strain.iloc[:, idx]
        fit_range = (ydata >= threshold) & (xdata <= ub) & (xdata >= lb)
        # popt1, pcov1 = optimize.curve_fit(log_OZ_3D_func, xdata[fit_range], np.log10(ydata[fit_range]), bounds=(0, [1000., 1000.]))
        power_coeff = fit_powerlaw(xdata[fit_range], ydata[fit_range], np.zeros(np.sum(fit_range == True)), weight=False)
        frame_shear_strain_fit_params[idx] = power_coeff

    ydata = df_time_cr_D2min.Mean
    fit_range = (ydata >= threshold) & (xdata <= ub) & (xdata >= lb)
    power_coeff = fit_powerlaw(xdata[fit_range], ydata[fit_range], np.zeros(np.sum(fit_range == True)), weight=False)
    D2min_fit_params = power_coeff

    ydata = df_time_cr_shear_strain.Mean
    fit_range = (ydata >= threshold) & (xdata <= ub) & (xdata >= lb)
    power_coeff = fit_powerlaw(xdata[fit_range], ydata[fit_range], np.zeros(np.sum(fit_range == True)), weight=False)
    shear_strain_fit_params = power_coeff

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 5. Plot
    #
    nrows, ncols, size = 1, 2, 4
    fig = plt.figure(figsize=(ncols*size, nrows*size))
    ax1 = plt.subplot(nrows, ncols, 1)
    ax2 = plt.subplot(nrows, ncols, 2)

    df_D2min_fit_params = pd.DataFrame(np.array(frame_D2min_fit_params), columns=['Amp', 'Index'])
    amp_mean = df_D2min_fit_params['Amp'].mean()
    index_mean, index_std = df_D2min_fit_params['Index'].mean(), df_D2min_fit_params['Index'].std()
    ax1.scatter(strain_interval, df_time_cr_D2min['Mean'], s=20, color=colors[0], marker=markers[0])
    ax1.plot(strain_interval, powerlaw(strain_interval, 10**D2min_fit_params[0], D2min_fit_params[1]), linestyle='-', color=colors[0], linewidth=2,
             label="τ=%4.3f±%4.3f" %(-index_mean, index_std))
    # ax1.errorbar(strain_interval, df_time_cr_D2min['Mean'].values, xerr=None, yerr=df_time_cr_D2min['Std'].values,
    #              fmt=markers[0], ecolor=colors[0], color=colors[0], elinewidth=2, capsize=4)

    df_shear_strain_fit_params = pd.DataFrame(np.array(frame_shear_strain_fit_params), columns=['Amp', 'Index'])
    amp_mean = df_shear_strain_fit_params['Amp'].mean()
    index_mean, index_std = df_shear_strain_fit_params['Index'].mean(), df_shear_strain_fit_params['Index'].std()
    ax2.scatter(strain_interval, df_time_cr_shear_strain['Mean'], s=20, color=colors[1], marker=markers[1])
    ax2.plot(strain_interval, powerlaw(strain_interval, 10**shear_strain_fit_params[0], shear_strain_fit_params[1]), linestyle='-', color=colors[1], linewidth=2,
             label="τ=%4.3f±%4.3f" %(-index_mean, index_std))
    # ax2.errorbar(strain_interval, y=df_time_cr_shear_strain['Mean'].values, yerr=df_time_cr_shear_strain['Std'].values,
    #              fmt=markers[1], ecolor=colors[1], color=colors[1], elinewidth=2, capsize=4)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Strain, $Δγ$', fontsize=12)
    ax1.set_ylabel('Strain correlation, $C_{D_{min}^2}(Δγ)$', fontsize=12)
    ax1.set_xlim(xmin=0.5*strain_window)
    ax1.set_ylim(ymax=1)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_title('Particle D2min', fontsize=12)
    ax1.legend(fontsize=12)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Strain, $Δγ$', fontsize=12)
    ax2.set_ylabel('Strain correlation, $C_{\epsilon_{\gamma}}(Δγ)$', fontsize=12)
    ax2.set_xlim(xmin=0.5*strain_window)
    ax2.set_ylim(ymax=1)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_title('Particle shear strain', fontsize=12)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    file_path = output_path + '/Temporal correlation.png'
    plt.savefig(file_path, dpi=600, bbox_inches='tight')
    plt.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 6. Output
    #
    writer = pd.ExcelWriter(output_path + '/Temporal correlation.xlsx')
    df_time_cr_D2min.to_excel(writer, sheet_name='Particle D2min')
    df_time_cr_shear_strain.to_excel(writer, sheet_name='Particle shear strain')
    df_D2min_fit_params.to_excel(writer, sheet_name='Powerlaw fitting of corr(Demin)')
    df_shear_strain_fit_params.to_excel(writer, sheet_name='Powerlaw fitting of corr(e)')
    writer.save()
    writer.close()

# ==================================================================
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
    strain_window = 0.01
    scenario = 50
    d50 = 0.0010
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
        elif (argList[i][:4] == "-rat"):
            i += 1
            shear_rate = float(argList[i])
        elif (argList[i][:2] == "-w"):
            i += 1
            strain_window = float(argList[i])
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
    TemporalCorrelation(case, test_id, shear_strain, shear_rate, steady_strain, time_step, scenario, strain_window)
