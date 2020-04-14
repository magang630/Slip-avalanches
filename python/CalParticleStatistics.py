# -*- coding: utf-8 -*-
'''
CalStressTensor.py
Script to calculate the assembly averaged stress tensor based on contact force and branch vector.
Usage:
python CalParticleStatistics.py -test 1
python CalParticleStatistics.py -test 2
References:
[1] Dorostkar, O., Guyer, R. A., Johnson, P. A., Marone, C. & Carmeliet, J. On the role of fluids in stick-slip dynamics of saturated granular fault gouge using a coupled computational fluid dynamics-discrete element approach. J. Geophys. Res. Solid Earth 122, 3689¨C3700 (2017).
[2] Cao, P. et al. Nanomechanics of slip avalanches in amorphous plasticity. J. Mech. Phys. Solids 114, 158¨C171 (2018).
[3] Bi, D. & Chakraborty, B. Rheology of granular materials: Dynamics in a stress landscape. Philos. Trans. R. Soc. A Math. Phys. Eng. Sci. 367, 5073¨C5090 (2009).
[4] Gao, K., Guyer, R., Rougier, E., Ren, C. X. & Johnson, P. A. From Stress Chains to Acoustic Emission. Phys. Rev. Lett. 123, 48003 (2019).
'''
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sys import argv, exit
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# import powerlaw
from scipy import optimize
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
    #  log(y) = log(a) + b*log(x)
    #

    # Define function for calculating a power law
    powerlaw = lambda x, amp, index: amp*(x**index)

    logx = np.log10(xdata)
    logy = np.log10(ydata)
    if weight:
        logyerr = yerr/ydata
    else:
        logyerr = np.ones(len(logx))
    # logyerr[np.where(logyerr == 0)] = 1

    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1]*x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x))/err

    pinit = [10.0, -10.0]
    out = optimize.leastsq(errfunc, pinit, args=(logx, logy, logyerr), full_output=1)

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
    # plt.show()

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
    popt, pcov = optimize.curve_fit(log_truncated_powerlaw, xdata, np.log10(ydata), bounds=([0, 0, 0], [1000, 1000, 1000]))
    amp, alpha, beta = popt
    return popt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
def plot_fittingline(ax, param, xdata, xerr, ydata, yerr, xlabel, ylabel, color, label):
    # Define function for calculating a power law
    powerlaw = lambda x, amp, index: amp*(x**index)
    truncated_powerlaw = lambda x, amp, alpha, beta: amp*x**(-alpha)*np.exp(-x/beta)

    # plt.figure(figsize=(6,6))
    if (np.where(xerr!=0)[0].shape[0] != 0) or (np.where(yerr!=0)[0].shape[0] != 0):
        ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt='o', ecolor=color, color=color, elinewidth=2, capsize=4)
    else:
        ax.scatter(xdata, ydata, marker='o', color=color)
        # ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt='o', ecolor=color, color=color, elinewidth=2, capsize=4)

    if len(param) == 2:
        amp = 10**param[0]
        index = param[1]
        ax.plot(xdata, powerlaw(xdata, amp, index), color=color, linewidth=2, label=label+"τ= %4.3f" %index)  #画拟合直线
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
    ax.tick_params(axis='both', labelsize=12)
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# histogram with log form
#
def histgram_logform(data, step=0.05, func='power'):
    # lb, ub = np.floor(np.log10(np.min(Par_potential))), np.ceil(np.log10(np.max(Par_potential)))
    lb, ub = np.floor(np.log10(np.percentile(data, 1))), np.ceil(np.log10(np.max(data)))
    bins_try = np.arange(lb, ub, step)
    bins_try = 10**bins_try
    hist, bin_edges = np.histogram(data, bins=bins_try, density=False)
    # delete the bin without data points
    bins = np.delete(bins_try, np.where(hist == 0)[0])
    hist, bin_edges = np.histogram(data, bins=bins, density=False)

    digitized = np.digitize(data, bins)
    data_binning = np.array([[data[digitized == i].mean(), data[digitized == i].std()] for i in range(1, len(bins))])
    bin_value, bin_range = data_binning[:, 0], data_binning[:, 1]
    bin_value = np.array([0.5*(bin_edges[i] + bin_edges[i+1]) for i in range(len(bin_edges)-1)])
    frequency = hist/(np.sum(hist)*np.diff(bin_edges))
    frequency = frequency/np.sum(frequency)

    data_selected = bin_value/np.mean(data) > 5e-1
    xdata, ydata, yerr  = bin_value[data_selected], frequency[data_selected], np.zeros(len(bin_value[data_selected]))
    if func == 'power':
        fit_params = fit_powerlaw(xdata, ydata, yerr)
    else:
        fit_params = fit_truncated_powerlaw(xdata, ydata, yerr)

    return bin_value, bin_range, frequency, fit_params

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Bootstrap resampling with Python
# https://nbviewer.jupyter.org/gist/aflaxman/6871948
#
def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if isinstance(X, pd.Series):
        X = X.copy()
        X.index = range(len(X.index))
    if n == None:
        n = len(X)

    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = np.array(X[resample_i])  # TODO: write a test demonstrating why array() is important
    return X_resample


def ParticleStatistics(case, test_id, shear_strain, shear_rate, steady_strain, time_step, scenario, strain_window):

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

    frame_num = len(frame_list)
    frame_strain = [[] for i in range(frame_num)]
    # frame_par_potential = [[] for i in range(frame_num)]
    frame_par_D2min = [[] for i in range(frame_num)]
    frame_par_shear_strain = [[] for i in range(frame_num)]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. Load data set
    #
    # data file path and results file path
    output_path = file_path + '/test-' + str(test_id) + '/strain window-' + str(round(strain_window, 3)) + '/particle statistics'
    dynamics_path = file_path + '/test-' + str(test_id) + '/strain window-' + str(round(strain_window, 3)) + '/particle dynamics'
    potential_path = file_path + '/test-' + str(test_id) + '/particle potential'
    mkdir(output_path)
    mkdir(dynamics_path)
    mkdir(potential_path)

    for idx, frame in enumerate(frame_list):
        strain = (frame - start_frame)*time_step*shear_rate
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
        frame_par_D2min[idx] = Par_D2min[Par_type == 1]
        frame_par_shear_strain[idx] = Par_shear_strain[Par_type == 1]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2.2 Particle potential energy
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
        # Par_id = Par_id[Par_type == 1]
        # frame_par_potential[idx] = Par_potential[Par_type == 1]

    # df_allframes = pd.DataFrame(np.stack((frame_par_potential, frame_par_D2min), axis=1), columns=['U', 'D2min'], index=frame_index)
    # frame_par_potential = np.transpose(np.array(frame_par_potential))
    frame_par_D2min = np.transpose(np.array(frame_par_D2min))
    frame_par_shear_strain = np.transpose(np.array(frame_par_shear_strain))
    # df_par_potential =pd.DataFrame(frame_par_potential, index=Par_id, columns=frame_strain)
    df_par_D2min = pd.DataFrame(frame_par_D2min, index=Par_id, columns=frame_strain)
    df_par_shear_strain = pd.DataFrame(frame_par_shear_strain, index=Par_id, columns=frame_strain)
    for i in range(frame_num):
        # frame_par_potential[:, i] = frame_par_potential[:, i]/np.mean(frame_par_potential[:, i])
        frame_par_D2min[:, i] = frame_par_D2min[:, i]/np.mean(frame_par_D2min[:, i])
        frame_par_shear_strain[:, i] = frame_par_shear_strain[:, i]/np.mean(frame_par_shear_strain[:, i])

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

    nrows, ncols, size = 1, 3, 4
    fig = plt.figure(figsize=(ncols*size, nrows*size))
    ax1 = plt.subplot(nrows, ncols, 1)
    ax2 = plt.subplot(nrows, ncols, 2)
    ax3 = plt.subplot(nrows, ncols, 3)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. Par D2min frequency distribution
    #
    D2min_value, D2min_range, D2min_frequency, D2min_fit_params = histgram_logform(frame_par_D2min.reshape(1, frame_par_D2min.size), step=0.05, func='power')
    df_D2min_frequency = pd.DataFrame(np.stack((D2min_value, D2min_frequency), axis=1), columns=['D2min', 'PDF'])
    ax1.scatter(D2min_value, D2min_frequency, s=20, color=colors[0], marker=markers[0])
    # ax1.plot(D2min_value, powerlaw(D2min_value, 10**D2min_fit_params[0], D2min_fit_params[1]), linestyle=':', color=colors[0], linewidth=2,
    #          label="τ=%4.3f" %(D2min_fit_params[1]))

    frame_fit_params = [[] for i in range(frame_num)]
    for k, cl in enumerate(df_par_D2min.columns):
        D2min_value, D2min_range, D2min_frequency, D2min_fit_params = histgram_logform(df_par_D2min[cl].values/df_par_D2min[cl].mean(), step=0.05, func='power')
        frame_fit_params[k] = D2min_fit_params

    df_D2min_fit_params = pd.DataFrame(np.array(frame_fit_params), columns=['Amp', 'Index'])
    amp_mean = df_D2min_fit_params['Amp'].mean()
    index_mean, index_std = df_D2min_fit_params['Index'].mean(), df_D2min_fit_params['Index'].std()
    ax1.plot(D2min_value, powerlaw(D2min_value, 10**amp_mean, index_mean), linestyle='-', color=colors[0], linewidth=2,
             label="τ=%4.3f±%4.3f" %(-index_mean, index_std))

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('$D_{min}^2$', fontsize=12)
    ax1.set_ylabel('$P(D_{min}^2)$', fontsize=12)
    ax1.set_ylim(ymax=1)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_title('Particle D2min', fontsize=12)
    ax1.legend(fontsize=12)
    # plt.tight_layout()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4.2 Par shear strain frequency distribution
    #
    shear_strain_value, shear_strain_range, shear_strain_frequency, shear_strain_fit_params = histgram_logform(frame_par_shear_strain.reshape(1, frame_par_shear_strain.size), step=0.05, func='power')
    df_shear_strain_frequency = pd.DataFrame(np.stack((shear_strain_value, shear_strain_frequency), axis=1), columns=['Shear strain', 'PDF'])
    ax2.scatter(shear_strain_value, shear_strain_frequency, s=20, color=colors[1], marker=markers[1])
    # ax2.plot(shear_strain_value, powerlaw(shear_strain_value, 10**shear_strain_fit_params[0], shear_strain_fit_params[1]), linestyle=':', color=colors[1], linewidth=2,
    #          label="τ=%4.3f" %(shear_strain_fit_params[1]))

    frame_fit_params = [[] for i in range(frame_num)]
    for k, cl in enumerate(df_par_shear_strain.columns):
        shear_strain_value, shear_strain_range, shear_strain_frequency, shear_strain_fit_params = histgram_logform(df_par_shear_strain[cl].values/df_par_shear_strain[cl].mean(), step=0.05, func='power')
        frame_fit_params[k] = shear_strain_fit_params

    df_shear_strain_fit_params = pd.DataFrame(np.array(frame_fit_params), columns=['Amp', 'Index'])
    amp_mean = df_shear_strain_fit_params['Amp'].mean()
    index_mean, index_std = df_shear_strain_fit_params['Index'].mean(), df_shear_strain_fit_params['Index'].std()
    ax2.plot(shear_strain_value, powerlaw(shear_strain_value, 10**amp_mean, index_mean), linestyle='-', color=colors[1], linewidth=2,
             label="τ=%4.3f±%4.3f" % (-index_mean, index_std))

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('${\epsilon_{\gamma}}$', fontsize=12)
    ax2.set_ylabel('$P({\epsilon_{\gamma}})$', fontsize=12)
    ax2.set_ylim(ymax=1)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_title('Particle shear strain', fontsize=12)
    ax2.legend(fontsize=12)
    # plt.tight_layout()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 5. Particle potential energy frequency distribution
    #
    # potential_value, potential_range, potential_frequency, potential_fit_params = histgram_logform(frame_par_potential.reshape(1, frame_par_potential.size), step=0.05, func='power')
    # df_potential_frequency = pd.DataFrame(np.stack((potential_value, potential_frequency), axis=1), columns=['U', 'PDF'])
    # ax2.scatter(potential_value, potential_frequency, s=20, color=colors[1], marker=markers[1])
    # # ax2.plot(potential_value, powerlaw(potential_value, 10**potential_fit_params[0], potential_fit_params[1]), linestyle=':', color=colors[1], linewidth=2,
    # #          label="τ=%4.3f" %(potential_fit_params[1]))
    #
    # frame_fit_params = [[] for i in range(frame_num)]
    # for k, cl in enumerate(df_par_potential.columns):
    #     potential_value, potential_range, potential_frequency, potential_fit_params = histgram_logform(df_par_potential[cl].values/df_par_potential[cl].mean(), step=0.05, func='power')
    #     frame_fit_params[k] = potential_fit_params
    #
    # df_potential_fit_params = pd.DataFrame(np.array(frame_fit_params), columns=['Amp', 'Index'])
    # amp_mean = df_potential_fit_params['Amp'].mean()
    # index_mean, index_std = df_potential_fit_params['Index'].mean(), df_potential_fit_params['Index'].std()
    # ax2.plot(potential_value, powerlaw(potential_value, 10**amp_mean, index_mean), linestyle='-', color=colors[1], linewidth=2,
    #          label="τ=%4.3f±%4.3f" % (-index_mean, index_std))
    #
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    # ax2.set_xlabel('$U$', fontsize=12)
    # ax2.set_ylabel('$P(U)$', fontsize=12)
    # ax2.set_ylim(ymax=1)
    # ax2.tick_params(axis='both', labelsize=12)
    # ax2.set_title('Particle potential energy', fontsize=12)
    # ax2.legend(fontsize=12)
    # # plt.tight_layout()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 6. Potetial energy and D2min fluctuation
    #    we use the standard deviation of the logarithm of fractional changes, termed the volatility in the Black-Scholes model for price evolution.
    #    It provides a dimensionless, baseline-independent measure of the rate of change of a discretized time series.
    #    References:
    #    [1] https://tinytrader.io/2019/01/02/how-to-calculate-historical-price-volatility-with-python/
    #    [2] Murphy, K.A., Dahmen, K.A., Jaeger, H.M., 2019. Transforming Mesoscale Granular Plasticity Through Particle Shape. Phys. Rev. X 9, 11014.
    mean_par_D2min = np.mean(frame_par_D2min, axis=1)
    std_par_D2min = np.std(frame_par_D2min, axis=1)
    var_par_D2min = np.var(frame_par_D2min, axis=1)

    mean_par_shear_strain = np.mean(frame_par_shear_strain, axis=1)
    std_par_shear_strain = np.std(frame_par_shear_strain, axis=1)
    var_par_shear_strain = np.var(frame_par_shear_strain, axis=1)

    volatility_par_D2min = np.zeros(frame_par_D2min.shape[0])
    volatility_par_shear_strain = np.zeros(frame_par_shear_strain.shape[0])
    for i in range(frame_par_shear_strain.shape[0]):
        D2min_series = pd.DataFrame(frame_par_D2min[i,:], columns=['D2min'])
        # calculate the logarithm of fractional changes
        D2min_series['Return'] = (np.log(D2min_series['D2min']/D2min_series['D2min'].shift(-1)))
        D2min_volatility = np.std(D2min_series.Return)
        volatility_par_D2min[i] = D2min_volatility

        shear_strain_series = pd.DataFrame(frame_par_shear_strain[i,:], columns=['Shear strain'])
        # calculate the logarithm of fractional changes
        shear_strain_series['Return'] = (np.log(shear_strain_series['Shear strain']/shear_strain_series['Shear strain'].shift(-1)))
        shear_strain_volatility = np.std(shear_strain_series.Return)
        volatility_par_shear_strain[i] = shear_strain_volatility

    df_par_stats = pd.DataFrame({'Mean_D2min': mean_par_D2min, 'Std_D2min': std_par_D2min, 'Variance_D2min': var_par_D2min, 'Vol_D2min': volatility_par_D2min,
                                 'Mean_shear_strain': mean_par_shear_strain, 'Std_shear_strain': std_par_shear_strain, 'Variance_shear_strain': var_par_shear_strain, 'Vol_shear_strain': volatility_par_shear_strain})

    sns.distplot(df_par_stats['Vol_D2min'], kde=True, color=colors[0], bins=100, label='$D_{min}^2$', ax=ax3)
    sns.distplot(df_par_stats['Vol_shear_strain'], kde=True, color=colors[1], bins=100, label='${\epsilon_{\gamma}}$', ax=ax3)
    ax3.set_xlabel('Volatility, $V$', fontsize=12)
    ax3.set_ylabel('Frequency density', fontsize=12)
    ax3.tick_params(axis='both', labelsize=12)
    ax3.legend(prop={'size': 12})

    plt.tight_layout()
    # plt.suptitle('Energy barrier landscape', fontsize=12)
    file_path = output_path + '/Statistics of particle D2min and shear strain.png'
    plt.savefig(file_path, dpi=600, bbox_inches='tight')
    plt.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 7. Output
    #
    writer = pd.ExcelWriter(output_path + '/Particle D2min and shear strain statistics.xlsx')
    df_D2min_frequency.to_excel(writer, sheet_name='PDF of D2min')
    df_shear_strain_frequency.to_excel(writer, sheet_name='PDF of shear strain')
    df_D2min_fit_params.to_excel(writer, sheet_name='Powerlaw fitting of P(D2min)')
    df_shear_strain_fit_params.to_excel(writer, sheet_name='Powerlaw fitting of P(e)')
    df_par_stats.to_excel(writer, sheet_name='Fluctuation of D2min and e')
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
    strain_window = 0.001
    scenario = 100
    d50 = 0.001
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
    ParticleStatistics(case, test_id, shear_strain, shear_rate, steady_strain, time_step, scenario, strain_window)
