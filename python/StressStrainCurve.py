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


def StressStrainCurve(case, test_id, shear_rate, delta_strain, strain_interval, filter_method, filter_param):

    # data file path and results file path
    file_path = os.path.pardir + '/' + case
    data_path = file_path + '/test-' + str(test_id)
    output_path = file_path + '/test-' + str(test_id) + '/stress drop'
    mkdir(output_path)
    sns.set()

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

    force_info = open(data_path + '/output.dat', 'r')
    alllines = force_info.readlines()
    lines = alllines[81:]
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

    # if 'gaussian' in filter_method:
    #     # https: // docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html
    #     shear_stress_fld = gaussian_filter1d(shear_stress, sigma=filter_param) # standard deviation of shear stress
    # elif 'median' in filter_method:
    #     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html
    #     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html
    #     shear_stress_fld = medfilt(shear_stress, kernel_size=int(filter_param))
    #     # shear_stress_fld = median_filter(shear_stress, size=filter_param, mode='nearest')
    # elif 'savgol' in filter_method:
    #     # Apply a Savitzky-Golay filter to an array
    #     # https://codeday.me/bug/20170710/39369.html
    #     # http://scipy.github.io/devdocs/generated/scipy.signal.savgol_filter.html
    #     # https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
    #     # A Savitzky–Golay filter is a digital filter that can be applied to a set of digital data points for the purpose of smoothing the data.
    #     shear_stress_fld = savgol_filter(shear_stress, window_length=int(filter_param), polyorder=3)
    # elif 'ewma' in filter_method:
    #     # Exponentially-weighted moving average
    #     # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ewm.html?highlight=ewma
    #     # http://connor-johnson.com/2014/02/01/smoothing-with-exponentially-weighted-moving-averages/
    #     df_shear_stress_fwd = pd.Series(shear_stress)
    #     df_shear_stress_bwd = pd.Series(shear_stress[::-1])
    #     fwd = df_shear_stress_fwd.ewm(span=filter_param).mean() # take EWMA in fwd direction
    #     bwd = df_shear_stress_bwd.ewm(span=filter_param).mean() # take EWMA in bwd direction
    #     shear_stress_fld = np.vstack((fwd, bwd[::-1]))          # lump fwd and bwd together
    #     shear_stress_fld = np.mean(shear_stress_fld, axis=0)    # average
    # else:
    #     print('original data')

    shear_stress_median = medfilt(shear_stress, kernel_size=int(filter_param))
    shear_stress_savgol = savgol_filter(shear_stress, window_length=int(filter_param), polyorder=3)
    df_shear_stress_fwd = pd.Series(shear_stress)
    df_shear_stress_bwd = pd.Series(shear_stress[::-1])
    fwd = df_shear_stress_fwd.ewm(span=filter_param).mean()  # take EWMA in fwd direction
    bwd = df_shear_stress_bwd.ewm(span=filter_param).mean()  # take EWMA in bwd direction
    shear_stress_ewma = np.vstack((fwd, bwd[::-1]))  # lump fwd and bwd together
    shear_stress_ewma = np.mean(shear_stress_ewma, axis=0)  # average

    df_shear_stress = pd.DataFrame(np.stack((shear_stress, shear_stress_median, shear_stress_savgol, shear_stress_ewma), axis=1),
                                  columns=['Data', 'Median', 'Savgol', 'EWMA'])

    # calculate the logarithm of fractional changes
    df_shear_stress['Data_return'] = (np.log(df_shear_stress['Data']/df_shear_stress['Data'].shift(-1)))
    df_shear_stress['Median_return'] = (np.log(df_shear_stress['Median']/df_shear_stress['Median'].shift(-1)))
    df_shear_stress['Savgol_return'] = (np.log(df_shear_stress['Savgol']/df_shear_stress['Savgol'].shift(-1)))
    df_shear_stress['EWMA_return'] = (np.log(df_shear_stress['EWMA']/df_shear_stress['EWMA'].shift(-1)))

    data_volatility = np.std(df_shear_stress.Data_return)
    median_volatility = np.std(df_shear_stress.Median_return)
    savgol_volatility = np.std(df_shear_stress.Savgol_return)
    ewma_volatility = np.std(df_shear_stress.EWMA_return)

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(111)
    ax1.plot(shear_strain, shear_stress, color=colors[0], alpha=0.5, linewidth=1, label='data: ' + str(round(data_volatility, 6)))
    ax1.plot(shear_strain, shear_stress_median+0.01, color=colors[1], linewidth=1, label='median: ' + str(round(median_volatility, 6)))
    ax1.plot(shear_strain, shear_stress_savgol+0.02, color=colors[2], linewidth=1, label='savgol: ' + str(round(savgol_volatility, 6)))
    ax1.plot(shear_strain, shear_stress_ewma+0.03, color=colors[3], linewidth=1, label='ewma: ' + str(round(ewma_volatility, 6)))

    ax1.set_xlabel('shear strain, $\gamma$', fontsize=12, labelpad=5)
    ax1.set_ylabel('shear stress, ' + r'$\tau$(MPa)', fontsize=12, labelpad=5)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlim(1.0, 1.1)
    ax1.set_ylim(0.12, 0.22)

    ax2 = ax1.twinx()
    ax2.plot(shear_strain, volumetric_strain, color=colors[4], linewidth=1, label='Volumetric strain')
    ax2.set_ylabel('volumetric strain, $\epsilon_v$', fontsize=12, labelpad=5)
    ax2.tick_params(axis="y", labelsize=12)

    fig.legend(fontsize=12)
    plt.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.5)
    plt.grid()
    plt.show()
    plt.savefig(output_path + '/Macroscopic responses.png', dpi=600, bbox_inches='tight')



# ==================================================================
# S T A R T
#
if __name__ == '__main__':

    file_path = None
    file_name = None
    case = 'shear-rate-0.5-press-1e7'
    test_id = 1
    shear_rate = 0.5
    delta_strain = 1e-5
    strain_interval = 1e-5
    filter_method = 'ewma'  # savgol, gaussian, median, ewma
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
