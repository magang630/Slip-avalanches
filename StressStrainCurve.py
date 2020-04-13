# -*- coding: utf-8 -*-
'''
StressStrainCurve.py
Script to calculate the assembly averaged stress tensor based on contact force and branch vector.
Usage:
python StressStrainCurve.py -test 1 -filter median -param 21
python StressStrainCurve.py -test 2 -filter median -param 21
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
from scipy import optimize
from scipy import stats
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import median_filter
from scipy.signal import medfilt
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
# array normalization
#
def array_normalization(array):
    return (array - np.min(array))/(np.mean(array) - np.min(array))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# split the arr into N chunks
#
def chunks(arr, m):
    n = int(np.ceil(len(arr)/float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]

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
# standard error
#
def standard_error(sample):
    std=np.std(sample, ddof=0)
    standard_error=std/np.sqrt(len(sample))
    return standard_error

#################################################
# Author : 余欢
# Date : Dec 28, 2015    4:09:29 PM
# company : 南京师范大学--大数据实验室
# description : 清理异常值
# https://blog.csdn.net/redaihanyu/article/details/50421773
#################################################
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

    popt, pcov = optimize.curve_fit(log_truncated_powerlaw, xdata, np.log10(ydata), bounds=([0, 0, 0], [100, 100, 100]))
    amp, alpha, beta = popt
    return popt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
def plot_fittingline(ax, sample, power_law, xdata, xerr, ydata, yerr, xlabel, ylabel):

    # Define function for calculating a power law
    powerlaw = lambda x, amp, index: amp*(x**index)
    truncated_powerlaw = lambda x, amp, alpha, beta: amp*x**(-alpha)*np.exp(-x/beta)

    # plt.figure(figsize=(6,6))
    if ylabel == 'stress drop':
        ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt='o', ecolor='b', color='b', elinewidth=2, capsize=4)
    else:
        ax.scatter(xdata, ydata, marker='o', color='b')
        # ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt='o', ecolor='b', color='b', elinewidth=2, capsize=4)

    if len(power_law) == 2:
        amp = 10**power_law[0]
        index = power_law[1]
        ax.plot(xdata, powerlaw(xdata, amp, index), color="r", linewidth=2)  #画拟合直线
        ax.text(5e-5, 5e-3, "τ= %4.3f" %index, fontsize=12)
    else:
        amp = power_law[0]
        alpha = power_law[1]
        beta = power_law[2]
        ax.plot(xdata, truncated_powerlaw(xdata, amp, alpha, beta), color="r", linewidth=2)  #画拟合直线
        ax.text(1e-4, 1e-1, "τ= %4.3f, s$\mathregular{_*}$= %4.3f" % (alpha, beta), fontsize=12)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    # ax.legend(fontsize=12)
    # plt.savefig(file_path, dpi=500, bbox_inches = 'tight')
    # plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Bootstrap resampling with Python
# https://nbviewer.jupyter.org/gist/aflaxman/6871948
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

def StressStrainCurve(case, test_id, shear_rate, delta_strain, strain_interval, filter_method, filter_param):

    # data file path and results file path
    file_path = os.path.pardir + '/' + case
    data_path = file_path + '/test-' + str(test_id)
    output_path = file_path + '/test-' + str(test_id) + '/stress drop'
    mkdir(output_path)
    sns.set()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. Array initialization
    #
    domain_depth = 0.02
    domain_length = 0.04
    stress_drop = []
    strain_duration = []
    data_index = []

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. Load data set
    #
    force_info = open(data_path + '/output.dat', 'r')
    alllines = force_info.readlines()
    lines = alllines[4001:]
    force_info.close()
    for i in range(len(lines)):
        if (lines[i] == '\n'): del lines[i]
    time           = np.array([float(line.strip().split(' ')[0]) for line in lines])  # 字段以空格分隔，这里取得是第1列
    system_height  = np.array([float(line.strip().split(' ')[1]) for line in lines])  # 字段以空格分隔，这里取得是第2列
    system_volume  = np.array([float(line.strip().split(' ')[2]) for line in lines])  # 字段以空格分隔，这里取得是第3列
    solid_fraction = np.array([float(line.strip().split(' ')[3]) for line in lines])  # 字段以空格分隔，这里取得是第4列
    normal_force   = np.array([float(line.strip().split(' ')[4]) for line in lines])  # 字段以空格分隔，这里取得是第5列
    shear_force_top= np.array([float(line.strip().split(' ')[5]) for line in lines])  # 字段以空格分隔，这里取得是第6列
    shear_force_bot= np.array([float(line.strip().split(' ')[6]) for line in lines])  # 字段以空格分隔，这里取得是第7列
    coord_num      = np.array([float(line.strip().split(' ')[7]) for line in lines])  # 字段以空格分隔，这里取得是第8列

    time0 = time[0]
    height0 = system_height[0]
    shear_strain = (time - time0)*shear_rate
    volumetric_strain = (height0 - system_height)/height0
    normal_pressure = normal_force/(domain_depth*domain_length)*1e-6
    shear_stress = shear_force_bot/(domain_depth*domain_length)*1e-6
    shear_stress /= np.mean(normal_pressure)

    # force_info = open(data_path + '/Particle assembly stress.dat', 'r')
    # alllines = force_info.readlines()
    # lines = alllines
    # force_info.close()
    # for i in range(len(lines)):
    #     if (lines[i] == '\n'): del lines[i]
    # time1 = np.array([float(line.strip().split(',')[1]) for line in lines])  # 字段以空格分隔，这里取得是第2列
    # sxx   = np.array([float(line.strip().split(',')[2]) for line in lines])  # 字段以空格分隔，这里取得是第3列
    # syy   = np.array([float(line.strip().split(',')[3]) for line in lines])  # 字段以空格分隔，这里取得是第4列
    # szz   = np.array([float(line.strip().split(',')[4]) for line in lines])  # 字段以空格分隔，这里取得是第5列
    # sxy   = np.array([float(line.strip().split(',')[5]) for line in lines])  # 字段以空格分隔，这里取得是第6列
    # sxz   = np.array([float(line.strip().split(',')[6]) for line in lines])  # 字段以空格分隔，这里取得是第7列
    # syz   = np.array([float(line.strip().split(',')[7]) for line in lines])  # 字段以空格分隔，这里取得是第8列
    # sxy = abs(sxy)*1e-6

    # plt.figure(figsize=(8,6))
    # plt.plot(time1, sxy, linewidth=2, label='sxy')
    # plt.xlabel('time', fontsize=15)
    # plt.ylabel('sxy', fontsize=15)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.legend(fontsize=15)
    # plt.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.5)
    # plt.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2.1 Sampling from the original data set
    #
    data_interval = int(round(strain_interval/delta_strain))
    solid_fraction = solid_fraction[::data_interval]
    shear_stress = shear_stress[::data_interval]
    shear_strain = shear_strain[::data_interval]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2.2 Data smooth
    #     To remove fluctuations from the original time series of the shear stress obtained from the simulations,
    #     we smoothed the time series using different filters, such as Savitzky–Golay filter, gaussian filter, and median filter.
    # References:
    # [1] Niiyama, T., Wakeda, M., Shimokawa, T., Ogata, S., 2019. Structural relaxation affecting shear-transformation avalanches in metallic glasses. Phys. Rev. E 100, 1–10.
    # [2] Cao, P., Short, M.P., Yip, S., 2019. Potential energy landscape activations governing plastic flows in glass rheology. Proc. Natl. Acad. Sci. U. S. A. 116, 18790–18797.

    if 'gaussian' in filter_method:
        # https: // docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html
        shear_stress_fld = gaussian_filter1d(shear_stress, sigma=filter_param) # standard deviation of shear stress
    elif 'median' in filter_method:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html
        shear_stress_fld = medfilt(shear_stress, kernel_size=int(filter_param))
        # shear_stress_fld = median_filter(shear_stress, size=filter_param, mode='nearest')
    elif 'savgol' in filter_method:
        # Apply a Savitzky-Golay filter to an array
        # https://codeday.me/bug/20170710/39369.html
        # http://scipy.github.io/devdocs/generated/scipy.signal.savgol_filter.html
        # https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
        # A Savitzky–Golay filter is a digital filter that can be applied to a set of digital data points for the purpose of smoothing the data.
        shear_stress_fld = savgol_filter(shear_stress, window_length=int(filter_param), polyorder=3)
    else:
        print('original data')

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(shear_strain, shear_stress, color='k', linewidth=1, label='original stress data')
    ax1.plot(shear_strain, shear_stress_fld, color='r', linewidth=2, label='filtered stress data')
    ax1.set_xlabel('shear strain, $γ$', fontsize=15, labelpad=5)
    ax1.set_ylabel('shear stress, $τ$ (MPa)', fontsize=15, labelpad=5)
    ax1.tick_params(axis='both', labelsize=15)
    # ax1.set_xlim(0.9, 1)
    # ax1.set_ylim(0.24, 0.27)

    ax2 = ax1.twinx()
    ax2.plot(shear_strain, solid_fraction, color='c', linewidth=2, label='solid fraction')
    ax2.set_ylabel('solid fraction', fontsize=15, labelpad=5)
    ax2.tick_params(axis="y", labelsize=15)

    fig.legend(loc='center', fontsize=15)
    plt.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.5)
    plt.grid()
    plt.savefig(output_path + '/Macroscopic responses.png', dpi=600, bbox_inches='tight')
    plt.show()


# ==================================================================
# S T A R T
#
if __name__ == '__main__':

    file_path = None
    file_name = None
    case = 'shear-rate-2-press-1e6'
    test_id = 1
    shear_rate = 10
    delta_strain = 1e-5
    strain_interval = 1e-5
    filter_method = 'median'  # savgol, gaussian, median
    filter_param = 11
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
        elif (argList[i][:4] == "-del"):
            i += 1
            delta_strain = float(argList[i])
        elif (argList[i][:2] == "-i"):
            i += 1
            strain_interval = float(argList[i])
        elif (argList[i][:2] == "-f"):
            i += 1
            filter_method = str(argList[i])
        elif (argList[i][:2] == "-p"):
            i += 1
            filter_param = float(argList[i])
        elif (argList[i][:2] == "-h"):
            print(__doc__)
            exit(0)
        i += 1
    
    print("Running case:      %s" % case)
    print("Test id:           %d" % test_id)
    print("Shear rate:        %.5f" % shear_rate)
    print("Delta strain:      %.5f" % delta_strain)
    print("Strain interval:   %.5f" % strain_interval)
    print("filter_method:     %s" % filter_method)
    print("filter_param:      %.5f" % filter_param)
    StressStrainCurve(case, test_id, shear_rate, delta_strain, strain_interval, filter_method, filter_param)
