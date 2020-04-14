# -*- coding: cp936 -*-
'''
Usage: python ClusterAnalysis.py -select 0.05
'''

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sys import argv, exit
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import optimize
import pickle
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

def fitpowerlaw(xdata, ydata, yerr, weight):
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
def hist_fit(df_clusters, d50, step=0.05):

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. Variable definition
    #
    fit_params = []

    Par_volume = 1.0/6*np.pi*d50**3.0
    # cluster_size = df_clusters['Cluster_par_volume']/Par_volume
    cluster_size = df_clusters['Cluster_size']
    cluster_gyration = df_clusters['Gyration_radius']/d50
    cluster_anisotropy = df_clusters['Shape_anisotropy']
    cluster_sphericity = (36*np.pi*df_clusters['Vor_volume']**2.0)**(1/3.0)/df_clusters['Vor_area']
    cluster_convexity = df_clusters['Vor_volume']/df_clusters['Vor_convex_volume']
    cluster_Q4 = df_clusters['Boo_Q4']
    cluster_Q6 = df_clusters['Boo_Q6']
    cluster_Q8 = df_clusters['Boo_Q8']
    cluster_Q10 = df_clusters['Boo_Q10']
    cluster_Q12 = df_clusters['Boo_Q12']
    # cluster_domokos = (1.0/df_clusters['Axis_L'] + 1.0/df_clusters['Axis_I'] + 1.0/df_clusters['Axis_S'])*np.sqrt(df_clusters['Axis_L']**2.0+df_clusters['Axis_I']**2.0+df_clusters['Axis_S']**2.0)/np.sqrt(3.0)
    # cluster_wentworth = (df_clusters['Axis_L'] + df_clusters['Axis_I'])/(2.0*df_clusters['Axis_S'])
    # cluster_krumbein = ((df_clusters['Axis_I']*df_clusters['Axis_S'])/(df_clusters['Axis_L']**2.0))**(1/3.0)
    # cluster_corey = np.sqrt(df_clusters['Axis_S']**2.0/(df_clusters['Axis_L']*df_clusters['Axis_I']))
    # cluster_max_projection = (df_clusters['Axis_S']**2.0/(df_clusters['Axis_L']*df_clusters['Axis_I']))**(1/3.0)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. Frequency distribution of cluster size
    # Binning data in python with scipy/numpy
    # https://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
    #
    # first_edge, last_edge = 0, 1000
    # num_bins = int((last_edge - first_edge)/2)
    # bins_try = np.linspace(start=first_edge, stop=last_edge, num=num_bins + 1, endpoint=True)

    # bins = np.linspace(0, 3.5, 100)
    # bins = 10**bins
    # bins = np.logspace(0, 3.5, 71)
    bins_try = np.arange(0.0, np.round(np.log10(np.max(cluster_size)), 2), step)
    bins_try = 10**bins_try
    hist, bin_edges = np.histogram(cluster_size, bins=bins_try, density=False)
    # delete the bin without data points
    bins = np.delete(bins_try, np.where(hist == 0)[0])
    hist, bin_edges = np.histogram(cluster_size, bins=bins, density=False)

    digitized = np.digitize(cluster_size, bins)
    cluster_size_binning = np.array([[cluster_size[digitized == i].mean(), cluster_size[digitized == i].std()] for i in range(1, len(bins))])
    frequency = hist/(np.sum(hist)*np.diff(bin_edges))
    frequency = frequency/np.sum(frequency)

    ub = np.percentile(cluster_size, 100)
    data_selected = (cluster_size_binning[:,0] <= ub)
    power_coeff = fitpowerlaw(cluster_size_binning[:,0][data_selected], frequency[data_selected], np.zeros(np.sum(data_selected == True)), weight=False)
    Tau = -power_coeff[1]
    fit_params.extend([power_coeff[0], Tau])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3. Cluster gyration radius versus cluster size
    # Binning data in python with scipy/numpy
    # https://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
    #
    cluster_gyration_binning= np.array([[cluster_gyration[digitized == i].mean(), cluster_gyration[digitized == i].std()] for i in range(1, len(bins))])
    # https://stackoverflow.com/questions/2831516/isnotnan-functionality-in-numpy-can-this-be-more-pythonic
    isnotnan = ~np.isnan(cluster_size_binning[:,0])
    power_coeff = fitpowerlaw(cluster_size_binning[:,0][data_selected], cluster_gyration_binning[:,0][data_selected], cluster_gyration_binning[:,1][data_selected], weight=False)
    fractal_dimension = 1.0/power_coeff[1]
    fit_params.extend([power_coeff[0], fractal_dimension])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. Cluster shape descriptor versus gyration radius
    # Binning data in python with scipy/numpy
    # https://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
    #
    cluster_anisotropy_binning = np.array([[cluster_anisotropy[digitized == i].mean(), cluster_anisotropy[digitized == i].std()] for i in range(1, len(bins))])
    cluster_sphericity_binning = np.array([[cluster_sphericity[digitized == i].mean(), cluster_sphericity[digitized == i].std()] for i in range(1, len(bins))])
    cluster_convexity_binning  = np.array([[cluster_convexity[digitized == i].mean(), cluster_convexity[digitized == i].std()] for i in range(1, len(bins))])
    cluster_Q4_binning         = np.array([[cluster_Q4[digitized == i].mean(), cluster_Q4[digitized == i].std()] for i in range(1, len(bins))])
    cluster_Q6_binning         = np.array([[cluster_Q6[digitized == i].mean(), cluster_Q6[digitized == i].std()] for i in range(1, len(bins))])
    cluster_Q8_binning         = np.array([[cluster_Q8[digitized == i].mean(), cluster_Q8[digitized == i].std()] for i in range(1, len(bins))])
    cluster_Q10_binning        = np.array([[cluster_Q10[digitized == i].mean(), cluster_Q10[digitized == i].std()] for i in range(1, len(bins))])
    cluster_Q12_binning        = np.array([[cluster_Q12[digitized == i].mean(), cluster_Q12[digitized == i].std()] for i in range(1, len(bins))])

    cluster_statistics = np.stack((cluster_size_binning[:, 0], frequency, cluster_gyration_binning[:, 0],
                                   cluster_anisotropy_binning[:, 0], cluster_sphericity_binning[:, 0], cluster_convexity_binning[:, 0],
                                   cluster_Q4_binning[:, 0], cluster_Q6_binning[:, 0], cluster_Q8_binning[:, 0],
                                   cluster_Q10_binning[:, 0], cluster_Q12_binning[:, 0]), axis=1)
    df_cluster_statistics = pd.DataFrame(cluster_statistics, columns=['Cluster_size', 'Frequency', 'Gyration_radius', 'Shape_anisotropy', 'Sphericity', 'Convexity',
                                                                      'Q4', 'Q6', 'Q8', 'Q10', 'Q12'])

    return fit_params, df_cluster_statistics

def DataAnalysis(case, test_id, d50, strain_window, selected_ratio, linkage, k_sampling):

    file_path = os.path.pardir + '/' + case
    output_path = file_path + '/test-' + str(test_id) + '/strain window-' + str(round(strain_window, 3)) +'/cluster' + '-' + str(selected_ratio)
    mkdir(output_path)

    truncated_powerlaw = lambda x, amp, alpha, beta: amp*x**(-alpha)*np.exp(-x/beta)
    powerlaw = lambda x, amp, index: amp*(x**index)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. Load cluster information from dump file
    #
    fw = open(output_path + '/Clusters.dump', 'rb')
    frame_stz_clusters = pickle.load(fw)
    frame_random_clusters = pickle.load(fw)
    fw.close()

    frame_stz_clusters['Nx_trimmed'] = frame_stz_clusters.pop('nx_trimmed')
    frame_stz_clusters['DBSCAN'] = frame_stz_clusters.pop('db')
    frame_random_clusters['Nx_trimmed'] = frame_random_clusters.pop('nx_trimmed')
    frame_random_clusters['DBSCAN'] = frame_random_clusters.pop('db')
    frame_stz_clusters.pop('nx')
    frame_random_clusters.pop('nx')
    frame_num = len(frame_stz_clusters['DBSCAN'])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. Store the cluster info in Pandas DataFrames
    #
    columns = ['Type', 'Cluster_size', 'Cluster_par_volume', 'Gyration_radius', 'Shape_anisotropy',
               'Boo_Q2', 'Boo_Q4', 'Boo_Q6', 'Boo_Q8', 'Boo_Q10', 'Boo_Q12',
               'Axis_L', 'Axis_I', 'Axis_S', 'Vor_area', 'Vor_volume', 'Vor_convex_volume']
    columns_info = ['Type']
    columns_data = ['Cluster_size', 'Cluster_par_volume', 'Gyration_radius', 'Shape_anisotropy',
                    'Boo_Q2', 'Boo_Q4', 'Boo_Q6', 'Boo_Q8', 'Boo_Q10', 'Boo_Q12',
                    'Axis_L', 'Axis_I', 'Axis_S', 'Vor_area', 'Vor_volume', 'Vor_convex_volume']
    df_allframe_stz_clusters = pd.DataFrame(columns=columns)
    df_allframe_random_clusters = pd.DataFrame(columns=columns)
    for frame_index, frame_clusters in enumerate(frame_stz_clusters[linkage]):
        df_frame_clusters = pd.DataFrame(frame_clusters, columns=columns_data, index=[frame_index for i in range(len(frame_clusters))])
        df_frame_clusters['Type'] = 'STZ'
        df_allframe_stz_clusters = df_allframe_stz_clusters.append(df_frame_clusters, ignore_index=False)

    for frame_index, frame_clusters in enumerate(frame_random_clusters[linkage]):
        df_frame_clusters = pd.DataFrame(frame_clusters, columns=columns_data, index=[frame_index for i in range(len(frame_clusters))])
        df_frame_clusters['Type'] = 'Random'
        df_allframe_random_clusters = df_allframe_random_clusters.append(df_frame_clusters, ignore_index=False)

    df_allframe_clusters = pd.concat([df_allframe_stz_clusters, df_allframe_random_clusters], axis=0)
    del df_allframe_stz_clusters, df_allframe_random_clusters, df_frame_clusters
    del frame_stz_clusters, frame_random_clusters
    gc.collect()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3. Visualization setting
    #
    # sns.set_style('ticks')
    plt.style.use('seaborn-paper')
    my_palette = "bright"  # deep, muted, pastel, bright, dark, colorblind, Set3, husl, Paired
    sns.set_palette(my_palette)
    # colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    # colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
    colors = sns.color_palette(my_palette, 10)
    markers = ['o', '^', 's']
    nrows, ncols, size = 2, 2, 4
    fig = plt.figure(figsize=(size*ncols, size*nrows))
    ax1 = fig.add_subplot(nrows, ncols, 1)
    ax2 = fig.add_subplot(nrows, ncols, 2)
    ax3 = fig.add_subplot(nrows, ncols, 3)
    ax4 = fig.add_subplot(nrows, ncols, 4)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. Frequency distribution
    #
    category = ['STZ', 'Random']
    writer = pd.ExcelWriter(output_path + '/Cluster analysis-' + linkage + '.xlsx')
    for idx, cl_type in enumerate(category):
        df_clusters = df_allframe_clusters.loc[(df_allframe_clusters['Type'] == cl_type)]
        fit_params, df_cluster_statistics = hist_fit(df_clusters, d50, step=0.05)
        df_cluster_statistics.to_excel(writer, sheet_name='Statistics of ' + cl_type + ' clusters')

        ax1.scatter(df_cluster_statistics['Cluster_size'], df_cluster_statistics['Frequency'], s=20, color=colors[idx], marker=markers[idx])
        ax2.scatter(df_cluster_statistics['Cluster_size'], df_cluster_statistics['Gyration_radius'], s=20, color=colors[idx], marker=markers[idx])
        ax3.scatter(df_cluster_statistics['Gyration_radius'], df_cluster_statistics['Shape_anisotropy'], s=20, color=colors[idx], marker=markers[idx], label=cl_type)
        ax4.scatter(df_cluster_statistics['Gyration_radius'], df_cluster_statistics['Q10'], s=20, color=colors[idx], marker=markers[idx], label=cl_type)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 5. Bootstrap resampling
    #
    fit_params_bootstrap = []
    for k in range(k_sampling):
        frame_resample = bootstrap_resample(np.arange(frame_num), n=int(frame_num*0.33))
        df_resampled_clusters = df_allframe_clusters.loc[frame_resample]
        for idx, cl_type in enumerate(category):
            df_clusters = df_resampled_clusters[(df_resampled_clusters['Type'] == cl_type)]
            fit_params, df_cluster_statistics = hist_fit(df_clusters, d50, step=0.05)
            fit_params.extend([cl_type])
            fit_params_bootstrap.append(fit_params)

    df_fit_params = pd.DataFrame(np.array(fit_params_bootstrap), columns=['Amp1', 'Tau', 'Amp2', 'Fractal', 'Type'])
    df_fit_params[['Amp1', 'Tau', 'Amp2', 'Fractal']] = df_fit_params[['Amp1', 'Tau', 'Amp2', 'Fractal']].apply(pd.to_numeric)
    for idx, cl_type in enumerate(category):
        fit_params = df_fit_params[(df_fit_params['Type'] == cl_type)]
        amp1_mean = fit_params['Amp1'].mean()
        tau_mean = fit_params['Tau'].mean()
        tau_std = fit_params['Tau'].std()

        amp2_mean = fit_params['Amp2'].mean()
        Fractal_mean = fit_params['Fractal'].mean()
        Fractal_std = fit_params['Fractal'].std()

        df_clusters = df_allframe_clusters.loc[(df_allframe_clusters['Type'] == cl_type)]
        xdata = np.arange(0.0, np.round(np.log10(np.max(df_clusters['Cluster_size'].max())), 2), 0.05)
        xdata = 10**xdata
        ax1.plot(xdata, powerlaw(xdata, 10**amp1_mean, -tau_mean), color=colors[idx], linewidth=2, label=cl_type + ": τ=%4.3f±%4.3f" % (tau_mean, tau_std))
        ax2.plot(xdata, powerlaw(xdata, 10**amp2_mean, 1/Fractal_mean), color=colors[idx], linewidth=2, label=cl_type + ": D=%4.3f±%4.3f" % (Fractal_mean, Fractal_std))

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Cluster size, $s$', fontsize=12)
    ax1.set_ylabel('$P(s)$', fontsize=12)
    ax1.set_ylim([1e-5, 1])
    ax1.tick_params(axis='both', labelsize=12)
    ax1.legend(fontsize=12)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Cluster size, $s$', fontsize=12)
    ax2.set_ylabel('Gyration radius, $R\mathregular{_g}$', fontsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.legend(fontsize=12)

    ax3.set_xscale('linear')
    ax3.set_yscale('linear')
    ax3.set_xlabel('Gyration radius, $R\mathregular{_g}$', fontsize=12)
    ax3.set_ylabel('Shape anisotropy, ' + r'$\alpha$', fontsize=12)
    ax3.tick_params(axis='both', labelsize=12)
    ax3.legend(fontsize=12)

    ax4.set_xscale('linear')
    ax4.set_yscale('linear')
    ax4.set_xlabel('Gyration radius, $R\mathregular{_g}$', fontsize=12)
    ax4.set_ylabel('Q10', fontsize=12)
    ax4.tick_params(axis='both', labelsize=12)
    ax4.legend(fontsize=12)

    plt.tight_layout()
    file_path = output_path + '/Cluster analysis-' + linkage + '.png'
    plt.savefig(file_path, dpi=600, bbox_inches='tight')
    # plt.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 6. Output
    #
    df_fit_params[(df_fit_params['Type'] == 'STZ')].to_excel(writer, sheet_name='Powerlaw fitting of STZ')
    df_fit_params[(df_fit_params['Type'] == 'Random')].to_excel(writer, sheet_name='Powerlaw fitting of Random')
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
    d50 = 0.001
    strain_window = 0.001
    selected_ratio = 0.05
    k_sampling = 1000
    linkage = 'DBSCAN'    # 'Nx_trimmed', 'DBSCAN'
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
        elif (argList[i][:4] == "-dia"):
            i += 1
            d50 = float(argList[i])
        elif (argList[i][:4] == "-sel"):
            i += 1
            selected_ratio = float(argList[i])
        elif (argList[i][:2] == "-w"):
            i += 1
            strain_window = float(argList[i])
        elif (argList[i][:2] == "-h"):
            print(__doc__)
            exit(0)
        i += 1
    DataAnalysis(case, test_id, d50, strain_window, selected_ratio, linkage, k_sampling)
