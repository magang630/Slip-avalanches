# -*- coding: utf-8 -*-
'''
CalStressTensor.py
Script to calculate the assembly averaged stress tensor based on contact force and branch vector.
Usage:
python CalStressDrop.py -test 1 -filter median -param 21
python CalStressDrop.py -case shear-rate-10-press-1e5 -test 1
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

def histgram_logform(data, minimum, step=0.05):
    # lb, ub = np.floor(np.log10(np.min(Par_potential))), np.ceil(np.log10(np.max(Par_potential)))
    lb, ub = np.floor(np.log10(minimum)), np.ceil(np.log10(np.max(data)))
    bins_try = np.arange(lb, ub, step)
    bins_try = 10**bins_try
    hist, bin_edges = np.histogram(data, bins=bins_try, density=False)
    # delete the bin without data points
    bins = np.delete(bins_try, np.where(hist == 0)[0])
    hist, bin_edges = np.histogram(data, bins=bins, density=False)

    digitized = np.digitize(data, bins)
    data_binning = np.array([[data[digitized == i].mean(), data[digitized == i].std()] for i in range(1, len(bins))])
    bin_value, bin_range = data_binning[:, 0], data_binning[:, 1]
    # bin_value = np.array([0.5*(bin_edges[i] + bin_edges[i+1]) for i in range(len(bin_edges)-1)])
    frequency = hist/(np.sum(hist)*np.diff(bin_edges))
    frequency = frequency/np.sum(frequency)

    return bin_value, bin_range, frequency, digitized

def CalStressDrops(case, test_id, shear_rate, delta_strain, steady_strain, strain_interval,
                   filter_method, filter_param, threshold, k_sampling, M0):

    # data file path and results file path
    data_path = os.path.pardir + '/' + case + '/test-' + str(test_id)
    output_path = os.path.pardir + '/' + case + '/test-' + str(test_id) + '/stress drop'
    mkdir(output_path)

    truncated_powerlaw = lambda x, amp, alpha, beta: amp*x**(-alpha)*np.exp(-x/beta)
    powerlaw = lambda x, amp, index: amp*(x**index)

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
    # 2. Load data set
    #
    domain_depth = 0.02
    domain_length = 0.04
    time_resolution = strain_interval/shear_rate

    force_info = open(data_path + '/output.dat', 'r')
    alllines = force_info.readlines()
    lines = alllines[801:]
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
    shear_time = time - time0
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
    volumetric_strain = volumetric_strain[::data_interval]
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
    elif 'ewma' in filter_method:
        # Exponentially-weighted moving average
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ewm.html?highlight=ewma
        # http://connor-johnson.com/2014/02/01/smoothing-with-exponentially-weighted-moving-averages/
        df_shear_stress_fwd = pd.Series(shear_stress)
        df_shear_stress_bwd = pd.Series(shear_stress[::-1])
        fwd = df_shear_stress_fwd.ewm(span=filter_param).mean() # take EWMA in fwd direction
        bwd = df_shear_stress_bwd.ewm(span=filter_param).mean() # take EWMA in bwd direction
        shear_stress_fld = np.vstack((fwd, bwd[::-1]))          # lump fwd and bwd together
        shear_stress_fld = np.mean(shear_stress_fld, axis=0)    # average
    else:
        print('original data')

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(111)
    ax1.plot(shear_strain, shear_stress, color=colors[0], alpha=0.1, linewidth=1, label='Original stress data')
    ax1.plot(shear_strain, shear_stress_fld, color=colors[0], linewidth=1, label='Filtered stress data')
    ax1.set_xlabel('shear strain, $\gamma$', fontsize=12, labelpad=5)
    ax1.set_ylabel('shear stress, ' + r'$\tau$(MPa)', fontsize=12, labelpad=5)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlim(0., 5)
    # ax1.set_ylim(0.24, 0.27)

    ax2 = ax1.twinx()
    ax2.plot(shear_strain, volumetric_strain, color=colors[1], linewidth=1, label='Volumetric strain')
    ax2.set_ylabel('volumetric strain, $\epsilon_v$', fontsize=12, labelpad=5)
    ax2.tick_params(axis="y", labelsize=12)

    fig.legend(loc='center', fontsize=12)
    plt.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.5)
    plt.grid()
    plt.savefig(output_path + '/Macroscopic responses.png', dpi=600, bbox_inches='tight')
    plt.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3. Identify the stress drop、strain duration
    #
    steady_range = shear_strain > steady_strain
    shear_time = shear_time[steady_range]
    shear_strain = shear_strain[steady_range]
    shear_stress = shear_stress_fld[steady_range]
    avalanches = []

    # https://stackoverflow.com/questions/16841729/how-do-i-compute-the-derivative-of-an-array-in-python
    # dydx = np.diff(shear_stress)/np.diff(shear_strain)
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.gradient.html
    dydx = np.gradient(shear_stress, shear_strain)
    d2ydx2 = np.gradient(dydx, shear_strain)

    drop_pos = dydx < 0
    drop_slice = []
    slice = []
    for i in range(len(shear_stress)):
        if drop_pos[i]:
            slice.append(i)
        else:
            if len(slice) > 1:
                # combine two drop slice if they are two close
                if len(drop_slice) == 0:
                    drop_slice.append(slice)
                else:
                    if (slice[0] - drop_slice[-1][-1]) <= 0:
                        drop_slice[-1].extend(slice)
                    else:
                        drop_slice.append(slice)
            slice = []

    for i, slice in enumerate(drop_slice):
        drop_start = np.min(slice)
        drop_end = np.max(slice)
        stress_drop = shear_stress[drop_start] - shear_stress[drop_end]
        strain_duration = shear_strain[drop_end] - shear_strain[drop_start]
        time_duration = strain_duration/shear_rate
        t0 = shear_time[drop_start]
        t1 = shear_time[drop_end]
        if i == 0:
            waiting_time, waiting_strain = 0, 0
        else:
            previous_drop_end = np.max(drop_slice[i-1])
            waiting_strain = shear_strain[drop_start] -shear_strain[previous_drop_end]
            waiting_time = shear_time[drop_start] -shear_time[previous_drop_end]
        avalanches.append([stress_drop, strain_duration, time_duration, waiting_strain, waiting_time, t0, t1])
    avalanches = np.array(avalanches)
    df_avalanches = pd.DataFrame(avalanches, columns=['Stress_drop', 'Strain_duration', 'Time_duration', 'Waiting_strain', 'Waiting_time', 'T0', 'T1'])

    # Remove the very small stress drops
    df_avalanches = df_avalanches[df_avalanches['Stress_drop'] > threshold]
    df_avalanches = df_avalanches[df_avalanches['Strain_duration'] < 0.01]
    dataset_size = len(df_avalanches)
    for i in range(dataset_size):
        if i == 0:
            df_avalanches.iloc[i]['Waiting_time'], df_avalanches.iloc[i]['Waiting_strain'] = 0, 0
        else:
            df_avalanches.iloc[i]['Waiting_time'] = df_avalanches.iloc[i]['T0'] - df_avalanches.iloc[i-1]['T1']
            df_avalanches.iloc[i]['Waiting_strain'] = df_avalanches.iloc[i]['Waiting_time']*shear_rate
    df_avalanches.reset_index()

    writer = pd.ExcelWriter(output_path + '/Stress avalanches.xlsx')
    df_avalanches.to_excel(writer, sheet_name='Stress avalanches')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. Statistical analysis
    #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4.1 Frequency density of stress drop
    #
    stress_drop = df_avalanches['Stress_drop'].values
    bin_value, bin_range, frequency, digitized = histgram_logform(stress_drop, threshold, 0.05)
    data_selected = (frequency > 1e-6)
    fit_params1 = fit_truncated_powerlaw(bin_value[data_selected], frequency[data_selected], np.zeros(np.sum(data_selected == True)))
    stress_drop_pdf = pd.DataFrame(np.stack((bin_value, frequency), axis=1), columns=['Stress_drop', 'PDF'])
    stress_drop_pdf.to_excel(writer, sheet_name='PDF of stress drops')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4.2 Frequency density of time duration
    #
    time_duration = df_avalanches['Time_duration'].values
    bin_value, bin_range, frequency, digitized = histgram_logform(time_duration, time_resolution, 0.05)
    data_selected = (frequency > 1e-6)
    fit_params2 = fit_truncated_powerlaw(bin_value[data_selected], frequency[data_selected], np.zeros(np.sum(data_selected == True)))
    time_duration_pdf = pd.DataFrame(np.stack((bin_value, frequency), axis=1), columns=['Time_duration', 'PDF'])
    time_duration_pdf.to_excel(writer, sheet_name='PDF of time duration')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4.3 Time duration vs stress drop
    #
    time_duration_binning = np.array([[time_duration[digitized == i].mean(), time_duration[digitized == i].std()] for i in range(1, len(bin_value)+1)])
    stress_drop_binning = np.array([[stress_drop[digitized == i].mean(), stress_drop[digitized == i].std()] for i in range(1, len(bin_value)+1)])
    # https://stackoverflow.com/questions/2831516/isnotnan-functionality-in-numpy-can-this-be-more-pythonic
    time_duration, duration_range = time_duration_binning[:, 0], time_duration_binning[:, 1]
    stress_drop, drop_range = stress_drop_binning[:, 0], stress_drop_binning[:, 1]
    fit_params3 = fit_powerlaw(time_duration_binning[:, 0][data_selected], stress_drop_binning[:, 0][data_selected], stress_drop_binning[:, 1][data_selected], weight=False)
    df_duration_drop = pd.DataFrame(np.stack((time_duration, duration_range, stress_drop, drop_range), axis=1), columns=['Time_duration', 'Duration_range', 'Stress_drop', 'Drop_range'])
    df_duration_drop.to_excel(writer, sheet_name='Duration vs Drop')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4.4 Frequency density of interevent time (waiting time of avalanche events)
    #
    waiting_time = df_avalanches['Waiting_time'].values
    bin_value, bin_range, frequency, digitized = histgram_logform(waiting_time, time_resolution, 0.05)
    data_selected = (frequency > 1e-6)
    fit_params4 = fit_truncated_powerlaw(bin_value[data_selected], frequency[data_selected], np.zeros(np.sum(data_selected == True)))
    waiting_time_pdf = pd.DataFrame(np.stack((bin_value, frequency), axis=1), columns=['Waiting_time', 'PDF'])
    waiting_time_pdf.to_excel(writer, sheet_name='PDF of inter-event time')

    writer.save()
    writer.close()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4.4 Bootstrap resampling
    #
    bs_fit_params1 = [[] for i in range(k_sampling)]
    bs_fit_params2 = [[] for i in range(k_sampling)]
    bs_fit_params3 = [[] for i in range(k_sampling)]
    bs_fit_params4 = [[] for i in range(k_sampling)]
    for k in range(k_sampling):
        data_resample = bootstrap_resample(np.arange(dataset_size), n=int(dataset_size*0.33))

        df_resample_avalanches = df_avalanches.iloc[data_resample]
        stress_drop = df_resample_avalanches['Stress_drop']
        time_duration = df_resample_avalanches['Time_duration']
        waiting_time = df_resample_avalanches['Waiting_time']

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4.4.1 Frequency density of stress drop
        #
        bin_value, bin_range, frequency, digitized = histgram_logform(stress_drop, threshold, 0.05)
        data_selected = (frequency > 1e-6)
        power_coeff = fit_truncated_powerlaw(bin_value[data_selected], frequency[data_selected], np.zeros(np.sum(data_selected == True)))
        bs_fit_params1[k] = power_coeff

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4.4.2 Frequency density of time duration
        #
        bin_value, bin_range, frequency, digitized = histgram_logform(time_duration, time_resolution, 0.05)
        data_selected = (frequency > 1e-6)
        power_coeff = fit_truncated_powerlaw(bin_value[data_selected], frequency[data_selected], np.zeros(np.sum(data_selected == True)))
        bs_fit_params2[k] = power_coeff

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4.4.3 Stress drop vs strain duration
        #
        time_duration_binning = np.array([[time_duration[digitized == i].mean(), time_duration[digitized == i].std()] for i in range(1, len(bin_value)+1)])
        stress_drop_binning = np.array([[stress_drop[digitized == i].mean(), stress_drop[digitized == i].std()] for i in range(1, len(bin_value)+1)])
        # https://stackoverflow.com/questions/2831516/isnotnan-functionality-in-numpy-can-this-be-more-pythonic
        power_coeff = fit_powerlaw(time_duration_binning[:,0][data_selected], stress_drop_binning[:,0][data_selected], stress_drop_binning[:,1][data_selected], weight=False)
        bs_fit_params3[k] = power_coeff

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4.4.4 Frequency density of interevent time (waiting time of avalanche events)
        #
        bin_value, bin_range, frequency, digitized = histgram_logform(waiting_time, threshold, 0.05)
        data_selected = (frequency > 1e-6)
        power_coeff = fit_truncated_powerlaw(bin_value[data_selected], frequency[data_selected], np.zeros(np.sum(data_selected == True)))
        bs_fit_params4[k] = power_coeff

    df_fit_params1 = pd.DataFrame(np.array(bs_fit_params1), index=np.arange(1, 1+k_sampling, 1), columns=['Amp', 'Alpha', 'Beta'])
    df_fit_params2 = pd.DataFrame(np.array(bs_fit_params2), index=np.arange(1, 1+k_sampling, 1), columns=['Amp', 'Alpha', 'Beta'])
    df_fit_params3 = pd.DataFrame(np.array(bs_fit_params3), index=np.arange(1, 1+k_sampling, 1), columns=['Amp', 'Tau'])
    df_fit_params4 = pd.DataFrame(np.array(bs_fit_params4), index=np.arange(1, 1+k_sampling, 1), columns=['Amp', 'Alpha', 'Beta'])

    writer = pd.ExcelWriter(output_path + '/Parameter space of data fitting.xlsx')
    df_fit_params1.to_excel(writer, sheet_name='Stress drop')
    df_fit_params2.to_excel(writer, sheet_name='Time duration')
    df_fit_params3.to_excel(writer, sheet_name='Duration vs Drop')
    df_fit_params4.to_excel(writer, sheet_name='Inter-event time')
    writer.save()
    writer.close()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 5 Plot
    #
    nrows, ncols, size = 2, 2, 4
    fig = plt.figure(figsize=(ncols*size, nrows*size))
    ax1 = plt.subplot(nrows, ncols, 1)
    ax2 = plt.subplot(nrows, ncols, 2)
    ax3 = plt.subplot(nrows, ncols, 3)
    ax4 = plt.subplot(nrows, ncols, 4)

    amp_mean, alpha_mean, beta_mean = df_fit_params1['Amp'].median(), df_fit_params1['Alpha'].median(), df_fit_params1['Beta'].median()
    amp_std, alpha_std, beta_std = df_fit_params1['Amp'].std(), df_fit_params1['Alpha'].std(), df_fit_params1['Beta'].std()
    ax1.scatter(stress_drop_pdf['Stress_drop'], stress_drop_pdf['PDF'], s=15, marker=markers[0], color=colors[0])
    ax1.plot(stress_drop_pdf['Stress_drop'], truncated_powerlaw(stress_drop_pdf['Stress_drop'], fit_params1[0], fit_params1[1], fit_params1[2]),
             color=colors[0], linewidth=2, label='Fit line')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('stress drop, $s$', fontsize=12)
    ax1.set_ylabel('$P(s)$', fontsize=12)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_title("τ=%4.3f±%4.3f, $s\mathregular{_*}$=%4.3f±%4.3f" % (alpha_mean, alpha_std, beta_mean, beta_std), fontsize=12)
    # ax1.legend(fontsize=12)

    amp_mean, alpha_mean, beta_mean = df_fit_params2['Amp'].median(), df_fit_params2['Alpha'].median(), df_fit_params2['Beta'].median()
    amp_std, alpha_std, beta_std = df_fit_params2['Amp'].std(), df_fit_params2['Alpha'].std(), df_fit_params2['Beta'].std()
    ax2.scatter(time_duration_pdf['Time_duration'], time_duration_pdf['PDF'], s=15, marker=markers[0], color=colors[1])
    ax2.plot(time_duration_pdf['Time_duration'], truncated_powerlaw(time_duration_pdf['Time_duration'], fit_params2[0], fit_params2[1], fit_params2[2]),
             color=colors[1], linewidth=2, label='Fit line')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('time duration, $T_n$', fontsize=12)
    ax2.set_ylabel('$P(T_n)$', fontsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_title("τ=%4.3f±%4.3f, $T\mathregular{_*}$=%4.3f±%4.3f" % (alpha_mean, alpha_std, beta_mean, beta_std), fontsize=12)
    # ax2.legend(fontsize=12)

    amp_mean, tau_mean = df_fit_params3['Amp'].median(), df_fit_params3['Tau'].median()
    amp_std, tau_std = df_fit_params3['Amp'].std(), df_fit_params3['Tau'].std()
    ax3.scatter(df_duration_drop['Time_duration'], df_duration_drop['Stress_drop'], s=15, marker=markers[0], color=colors[2])
    ax3.plot(df_duration_drop['Time_duration'], powerlaw(df_duration_drop['Time_duration'], 10**fit_params3[0], fit_params3[1]), color=colors[2], linewidth=2, label='Fit line')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('time duration, $T_n$', fontsize=12)
    ax3.set_ylabel('stress drop, $s$', fontsize=12)
    ax3.tick_params(axis='both', labelsize=12)
    ax3.set_title("τ=%4.3f±%4.3f" % (tau_mean, tau_std), fontsize=12)
    # ax3.legend(fontsize=12)

    amp_mean, alpha_mean, beta_mean = df_fit_params4['Amp'].median(), df_fit_params4['Alpha'].median(), df_fit_params4['Beta'].median()
    amp_std, alpha_std, beta_std = df_fit_params4['Amp'].std(), df_fit_params4['Alpha'].std(), df_fit_params4['Beta'].std()
    ax4.scatter(waiting_time_pdf['Waiting_time'], waiting_time_pdf['PDF'], s=15, marker=markers[0], color=colors[3])
    ax4.plot(waiting_time_pdf['Waiting_time'], truncated_powerlaw(waiting_time_pdf['Waiting_time'], fit_params4[0], fit_params4[1], fit_params4[2]),
             color=colors[3], linewidth=2, label='Fit line')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('inter-event time, ' + r'$\tau_\ell$', fontsize=12)
    ax4.set_ylabel(r'$P({\tau_\ell})$', fontsize=12)
    ax4.tick_params(axis='both', labelsize=12)
    ax4.set_title("τ=%4.3f±%4.3f, $T\mathregular{_*}$=%4.3f±%4.3f" % (alpha_mean, alpha_std, beta_mean, beta_std), fontsize=12)
    # ax4.legend(fontsize=12)

    # fig.legend(loc='center', fontsize=15)
    plt.tight_layout()
    file_path = output_path + '/Statistics of stress drop.png'
    plt.savefig(file_path, dpi=600, bbox_inches='tight')
    plt.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 6 Statistics of granular avalanches and earthquakes
    #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 6.1 Mainshocks: By defining the large stress drops as mainshocks, and other shocks are either foreschocks or aftershocks
    #     References:
    #     [1] Hatano, T., Narteau, C., Shebalin, P., 2015. Common dependence on stress for the statistics of granular avalanches and earthquakes. Sci. Rep. 5, 1–9. doi:10.1038/srep12280
    #     [2] Johnson, P.A., Jia, X., 2005. Nonlinear dynamics, granular media and dynamic earthquake triggering. Nature 437, 871–874. doi:10.1038/nature04015
    #     [3] Lherminier, S., Planet, R., Vehel, V.L.D., Simon, G., Vanel, L., Måløy, K.J., Ramos, O., 2019. Continuously Sheared Granular Matter Reproduces in Detail Seismicity Laws. Phys. Rev. Lett. 122, 218501.
    #
    df_mainshocks = df_avalanches[df_avalanches['Stress_drop'] >= M0]
    df_weakshocks = df_avalanches[df_avalanches['Stress_drop'] < M0]
    # Now, the waiting time is the time interval between mainshocks
    for i in range(len(df_mainshocks)):
        if i == 0:
            df_mainshocks.iloc[i]['Waiting_time'] = df_mainshocks.iloc[i]['Time_duration']
            df_mainshocks.iloc[i]['Waiting_strain'] = df_mainshocks.iloc[i]['Strain_duration']
        else:
            df_mainshocks.iloc[i]['Waiting_time'] = df_mainshocks.iloc[i]['T0'] - df_mainshocks.iloc[i-1]['T1']
            df_mainshocks.iloc[i]['Waiting_strain'] = df_mainshocks.iloc[i]['Waiting_time']*shear_rate
    Ttol = shear_time.max() - shear_time.min()
    seismicity_rate = len(df_mainshocks)/Ttol
    characteristic_time = 1/seismicity_rate
    mainshocks_sita = df_mainshocks['Waiting_time']/characteristic_time

    # Foreshocks and aftershocks
    digitized = np.digitize(df_weakshocks.index, df_mainshocks.index)
    weakshocks_type = [[] for i in range(len(df_weakshocks))]
    weakshocks_time = np.zeros(len(df_weakshocks))
    for i, j in enumerate(digitized):
        if digitized[i] <= 0:
            weakshocks_type[i] = 'Foreshocks'
            weakshocks_time[i] = df_mainshocks.iloc[j]['T0'] - df_weakshocks.iloc[i]['T1']
        elif (digitized[i] > 0) & (digitized[i] < len(df_mainshocks)):
            T_fore = df_mainshocks.iloc[j]['T0'] - df_weakshocks.iloc[i]['T1']
            T_after = df_weakshocks.iloc[i]['T0'] - df_mainshocks.iloc[j-1]['T1']
            if T_after <= T_fore:
                weakshocks_type[i] = 'Aftershocks'
                weakshocks_time[i] = T_after
            else:
                weakshocks_type[i] = 'Foreshocks'
                weakshocks_time[i] = T_fore
        else:
            weakshocks_type[i] = 'Aftershocks'
            weakshocks_time[i] = df_weakshocks.iloc[i]['T0'] - df_mainshocks.iloc[j-1]['T1']
    df_weakshocks_type = pd.DataFrame(weakshocks_type, columns=['Type'], index=df_weakshocks.index)
    df_weakshocks_time = pd.DataFrame(weakshocks_time, columns=['Tl'], index=df_weakshocks.index)
    df_weakshocks = pd.concat([df_weakshocks, df_weakshocks_type, df_weakshocks_time], axis=1)

    writer = pd.ExcelWriter(output_path + '/Earthquake.xlsx')
    df_mainshocks.to_excel(writer, sheet_name='Mainshocks')
    df_weakshocks.to_excel(writer, sheet_name='Foreshocks and aftershocks')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 6.2 Frequency density of inter-event times, in terms of θ = τ/τ∗.
    #
    bin_value, bin_range, frequency, digitized = histgram_logform(mainshocks_sita, np.min(mainshocks_sita), step=0.05)
    data_selected = (frequency > 1e-6)
    fit_params1 = fit_truncated_powerlaw(bin_value[data_selected], frequency[data_selected], np.zeros(np.sum(data_selected == True)))
    sita_pdf = pd.DataFrame(np.stack((bin_value, frequency), axis=1), columns=['Sita', 'PDF'])
    sita_pdf.to_excel(writer, sheet_name='PDF of sita')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 6.3 Foreshock and aftershock analysis
    #
    foreshock_tl = df_weakshocks[df_weakshocks['Type'] == 'Foreshocks']['Tl']
    aftershock_tl = df_weakshocks[df_weakshocks['Type'] == 'Aftershocks']['Tl']
    bin_value, bin_range, frequency, digitized = histgram_logform(foreshock_tl, np.min(foreshock_tl), step=0.05)
    foreshock_tl_pdf = pd.DataFrame(np.stack((bin_value, frequency), axis=1), columns=['Tl_fore', 'PDF'])

    bin_value, bin_range, frequency, digitized = histgram_logform(aftershock_tl, np.min(aftershock_tl), step=0.05)
    aftershock_tl_pdf = pd.DataFrame(np.stack((bin_value, frequency), axis=1), columns=['Tl_after', 'PDF'])

    foreshock_tl_pdf.to_excel(writer, sheet_name='Foreshock rate')
    aftershock_tl_pdf.to_excel(writer, sheet_name='Aftershock rate')
    writer.save()
    writer.close()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 6.4 Plot
    #
    nrows, ncols, size = 1, 2, 4
    fig = plt.figure(figsize=(ncols*size, nrows*size))
    ax1 = plt.subplot(nrows, ncols, 1)
    ax2 = plt.subplot(nrows, ncols, 2)

    ax1.plot(sita_pdf['Sita'], sita_pdf['PDF'], marker=markers[0], color=colors[0])
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'${\Theta}$', fontsize=12)
    ax1.set_ylabel(r'$P({\Theta})$', fontsize=12)
    ax1.tick_params(axis='both', labelsize=12)
    # ax1.legend(fontsize=12)

    ax2.plot(foreshock_tl_pdf['Tl_fore'], foreshock_tl_pdf['PDF'], marker=markers[0], color=colors[0], label='Foreshocks')
    ax2.plot(aftershock_tl_pdf['Tl_after'], aftershock_tl_pdf['PDF'], marker=markers[1], color=colors[1], label='Aftershocks')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('$t$', fontsize=12)
    ax2.set_ylabel('$n(t)$', fontsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    file_path = output_path + '/Earthquake.png'
    plt.savefig(file_path, dpi=600, bbox_inches='tight')
    plt.show()

# ==================================================================
# S T A R T
#
if __name__ == '__main__':

    file_path = None
    file_name = None
    case = 'shear-rate-2-press-1e6'
    test_id = 1
    shear_rate = 2
    delta_strain = 1e-5
    steady_strain = 1.0
    strain_interval = 1e-5
    filter_method = 'ewma'  # savgol, gaussian, median, ewma
    filter_param = 11
    threshold = 1e-5
    M0 = 1e-2
    k_sampling = 1000
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
        elif (argList[i][:4] == "-ste"):
            i += 1
            steady_strain = float(argList[i])
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
    print("Steady state upon: %.5f" % steady_strain)
    print("filter_method:     %s" % filter_method)
    print("filter_param:      %.5f" % filter_param)
    print("Drop threshold:    %.5f" % threshold)
    print(60 * '~')
    CalStressDrops(case, test_id, shear_rate, delta_strain, steady_strain, strain_interval, filter_method, filter_param, threshold, k_sampling, M0)
