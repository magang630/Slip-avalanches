# -*- coding: cp936 -*-
'''
CalCluster.py
Script to identify force chains.
Usage: python CalCluster.py -test 1 -select 0.05 -strain 5.0 -delta_strain 0.01 -rate 0.5 -steady 0.5 -interval 5 -group_num 5 -dump False

'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import time
import numpy as np
import random
import networkx as nx
import os
import copy
import pickle
import pyvoro
# import pysal
# import boo
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sys import argv,exit
from collections import defaultdict
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.manifold import TSNE
# from sklearn.mixture import BayesianGaussianMixture
# from sklearn.cluster import KMeans
# from sklearn.cluster import MiniBatchKMeans
# from sklearn.cluster import MeanShift
# from sklearn.cluster import SpectralClustering
# from sklearn.cluster import birch
# from sklearn.cluster import estimate_bandwidth
# from sklearn.cluster import DBSCAN
# from sklearn.mixture import GaussianMixture
# from sklearn import metrics
# from sklearn.neighbors import kneighbors_graph
# from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.sparse import csr_matrix
from scipy.spatial import ConvexHull
from scipy import special
# from scipy.spatial import Voronoi, voronoi_plot_2d
# from kneed import DataGenerator, KneeLocator
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
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
# str转bool方法
#
def str_to_bool(str):
    return True if str.lower() == 'true' else False

# ----------------------------------------------------------------------
# some dictionary functions
#
def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]

def dict_initiation(key):
    key = list(key)
    value = [0 for i in range(len(key))]
    return dict(zip(key, value))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# array normalization
#
def array_normalization(array):
    return (array - np.min(array))/(np.mean(array) - np.min(array))

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

#----------------------------------------------------------------------
# unit normal vector of plane defined by points a, b, and c
#
def unit_normal(a, b, c):
    x = det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
    y = det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
    z = det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)

#----------------------------------------------------------------------
# dot product of vectors a and b
#
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

#----------------------------------------------------------------------
# cross product of vectors a and b
#
def cross(a, b):
    x = a[1]*b[2] - a[2]*b[1]
    y = a[2]*b[0] - a[0]*b[2]
    z = a[0]*b[1] - a[1]*b[0]
    return (x, y, z)

#----------------------------------------------------------------------
# calculate polar and azimuth angles from (x,y,z) vector
#
def calc_beta_rad(pvec):
    '''
    polar angle [0, pi]
    '''
    return np.arccos(pvec[2])  # arccos:[0, pi]

def calc_gamma_rad(pvec):
    '''
    azimuth angle [0, 2pi]
    '''
    gamma = np.arctan2(pvec[1], pvec[0])
    if gamma < 0.0:
        gamma += 2*np.pi
    return gamma

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Distance of two particles
#
def dist(pt_1, pt_2):
    pt_1 = np.array((pt_1[0], pt_1[1], pt_1[2]))
    pt_2 = np.array((pt_2[0], pt_2[1], pt_2[2]))
    return np.linalg.norm(pt_1-pt_2)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Distance of two vertexes
#
def vertex_distance(a, b):
    return np.sqrt((a[0] - b[0])**2.0 + (a[1] - b[1])**2.0 + (a[2] - b[2])**2.0)

#----------------------------------------------------------------------
# area of polygon poly
#
def PolygonArea(poly):
    if len(poly) < 3: # not a plane - no area
        return 0

    total = [0, 0, 0]
    for i in range(len(poly)):
        vi1 = poly[i]
        if i is len(poly)-1:
            vi2 = poly[0]
        else:
            vi2 = poly[i+1]
        prod = cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)

#----------------------------------------------------------------------
# find area of polygon from xyz coordinates
# https://stackoverflow.com/questions/12642256/python-find-area-of-polygon-from-xyz-coordinates/12643315
# determinant of matrix a
#
def det(a):
    return a[0][0]*a[1][1]*a[2][2] + a[0][1]*a[1][2]*a[2][0] + a[0][2]*a[1][0]*a[2][1] - a[0][2]*a[1][1]*a[2][0] - a[0][1]*a[1][0]*a[2][2] - a[0][0]*a[1][2]*a[2][1]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Volume of convex hull with QHull from SciPy
#
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

def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d)))/6

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# find if a point lies in the convex hull of a point cloud
# https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
# a point is in the hull if and only if for every equation (describing the facets)
# the dot product between the point and the normal vector (eq[:-1]) plus the offset (eq[-1]) is less than or equal to zero.
#
def point_in_hull(point, hull, tolerance=1e-12):
    return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Python Code for principal component analysis
# Reference：
# [1] Implementing a Principal Component Analysis (PCA)– in Python, step by step: http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
# [2] Fonseca J, O’Sullivan C, Coop M R, et al. Non-invasive characterization of particle morphology of natural sands[J]. Soils and Foundations, 2012, 52(4): 712-722.
# [3] http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
# [4] http://stackoverflow.com/questions/24733185/volume-of-convex-hull-with-qhull-from-scipy
#
def cluster_shape(cluster_par_xcor, cluster_par_ycor, cluster_par_zcor):
    """
    python script for cluster shape analysis
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. Taking the whole dataset
    #
    cluster_par_xcor = np.array(cluster_par_xcor)
    cluster_par_ycor = np.array(cluster_par_ycor)
    cluster_par_zcor = np.array(cluster_par_zcor)
    cluster_particle = len(cluster_par_xcor)
    dataSet = np.hstack((cluster_par_xcor[:, np.newaxis], cluster_par_ycor[:, np.newaxis], cluster_par_zcor[:, np.newaxis])).T
    assert dataSet.shape == (3, cluster_particle), "The matrix has not the dimensions 3x40"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. Computing the d-dimensional mean vector
    #
    mean_x = np.mean(dataSet[0, :])
    mean_y = np.mean(dataSet[1, :])
    mean_z = np.mean(dataSet[2, :])
    mean_vector = np.array([[mean_x], [mean_y], [mean_z]])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3A. Computing the Scatter Matrix
    #
    scatter_matrix = np.zeros((3, 3))
    for i in range(dataSet.shape[1]):
        scatter_matrix += (dataSet[:, i].reshape(3, 1) - mean_vector).dot((dataSet[:, i].reshape(3, 1) - mean_vector).T)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3B. Computing the Covariance Matrix (alternatively to the scatter matrix)
    #
    cov_mat = np.cov([dataSet[0, :], dataSet[1, :], dataSet[2, :]])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4A. Computing eigenvectors and corresponding eigenvalues
    #
    # eigenvectors and eigenvalues for the from the scatter matrix
    eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
    # eigenvectors and eigenvalues for the from the covariance matrix
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

    for i in range(len(eig_val_sc)):
        eigvec_sc = eig_vec_sc[:, i].reshape(1, 3).T
        eigvec_cov = eig_vec_cov[:, i].reshape(1, 3).T
        assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4B. Checking the eigenvector-eigenvalue calculation
    #
    for i in range(len(eig_val_sc)):
        eigv = eig_vec_sc[:, i].reshape(1, 3).T
        np.testing.assert_array_almost_equal(scatter_matrix.dot(eigv), eig_val_sc[i]*eigv,
                                             decimal=6, err_msg='', verbose=True)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 5A. Sorting the eigenvectors by decreasing eigenvalues
    #
    for ev in eig_vec_sc:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
        # instead of 'assert' because of rounding errors

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:, i]) for i in range(len(eig_val_sc))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    # eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 5B. Choosing k eigenvectors with the largest eigenvalues(没有排序)
    #
    matrix_w = np.hstack((eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1), eig_pairs[2][1].reshape(3, 1)))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 6. Transforming the samples onto the new subspace
    #
    transformed = matrix_w.T.dot(dataSet)
    assert transformed.shape == (3, cluster_particle), "The matrix is not 2x40 dimensional."

    xmax = np.max(transformed[0, :])
    xmin = np.min(transformed[0, :])
    ymax = np.max(transformed[1, :])
    ymin = np.min(transformed[1, :])
    zmax = np.max(transformed[2, :])
    zmin = np.min(transformed[2, :])

    length = []
    length.append(xmax - xmin)
    length.append(ymax - ymin)
    length.append(zmax - zmin)
    length.sort()
    length.reverse()
    length = np.array(length)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 7. Constructing a convex hull of the fragment
    #
    pts = np.hstack((cluster_par_xcor[:, np.newaxis], cluster_par_ycor[:, np.newaxis], cluster_par_zcor[:, np.newaxis]))
    volume_convex_hull = convex_hull_volume_bis(pts)

    return volume_convex_hull, length

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

#----------------------------------------------------------------------
# Bond orientation order parameter
# We use a modified definition of BOO in which each bond is weighted by its length
#
def boo_ql(origin, pt_coord, pt_weight, l, modified):
    # l: order of symmetry
    # origin: center of the cluster
    # pt_coord:
    m_list = np.arange(-l, l+1, 1)
    pt_num = len(pt_coord)
    ql = 0
    qlm = np.zeros(len(m_list)).astype(complex)

    bond_weight = np.zeros(pt_num)
    polar_angle = np.zeros(pt_num)
    azimuth_angle = np.zeros(pt_num)
    for j in range(pt_num):
        bond_weight[j] = vertex_distance(pt_coord[j], origin)*pt_weight[j]
        bond = unit_vector((pt_coord[j] - origin))
        polar_angle[j] = calc_beta_rad(bond)
        azimuth_angle[j] = calc_gamma_rad(bond)
    sum_weight = np.sum(bond_weight)

    if modified:
        for i, m in enumerate(m_list):
            for j in range(pt_num):
                qlm[i] += special.sph_harm(m, l, azimuth_angle[j], polar_angle[j])*bond_weight[j]/sum_weight
    else:
        for i, m in enumerate(m_list):
            for j in range(pt_num):
                qlm[i] += special.sph_harm(m, l, azimuth_angle[j], polar_angle[j])
            qlm[i] /= pt_num
    ql = np.sum(np.abs(qlm)**2.0)
    ql = np.sqrt(4*np.pi/(2*l+1)*ql)

    return ql

#----------------------------------------------------------------------
# A Gini coefficient calculator in Python.
# This is a function that calculates the Gini coefficient of a numpy array. Gini coefficients are often used to quantify income inequality
# https://github.com/oliviaguest/gini
# https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
#
def gini(x, w=None):
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x*sorted_w, dtype=float)
        return (np.sum(cumxw[1:]*cumw[:-1] - cumxw[:-1]*cumw[1:]) /
                (cumxw[-1]*cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2*np.sum(cumx)/cumx[-1])/n

#----------------------------------------------------------------------
# 2D Visualize the clustering
#
#def plot_clustering_2d(X_red, labels, title, output_path, frame):
#    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
#    X_red = (X_red - x_min)/(x_max - x_min)
#
#    plt.figure(figsize=(6, 4))
#    for i in range(X_red.shape[0]):
#        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
#                 color=plt.cm.nipy_spectral(labels[i]/10.),
#                 fontdict={'weight': 'bold', 'size': 9})
#
#    plt.xticks([])
#    plt.yticks([])
#    plt.title(title, size=17)
#    plt.axis('off')
#    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#    plt.savefig(output_path + '/Clusters-metrics-' + str(frame) + '-' + title + '.png', format='png', dpi=1000, bbox_inches='tight')
#    plt.close()

# ----------------------------------------------------------------------
# 3D Visualize the clustering
#
#def plot_clustering_3d(Par_coord, labels, title, output_path, frame):
#    fig = plt.figure()
#    ax = p3.Axes3D(fig)
#    ax.view_init(7, -80)
#    for l in np.unique(labels):
#        ax.scatter(Par_coord[labels == l, 0], Par_coord[labels == l, 1], Par_coord[labels == l, 2],
#                   color=plt.cm.jet(float(l)/np.max(labels + 1)),
#                   s=20, edgecolor='k')
#    plt.title(title, size=17)
#    plt.savefig(output_path + '/' + title + '-' + str(frame) +  '.png', format='png', dpi=1000, bbox_inches='tight')
#    plt.close()

#----------------------------------------------------------------------
# 2D Visualize the clustering
#
#def plot_clustering_2d(X, labels, output_path, frame):

#    plt.figure(figsize=(6, 6))
#    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', alpha=0.3)
#    # x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
#    # y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
#    # x_min, x_max = 0, np.percentile(X[:, 0], 99)
#    # y_min, y_max = 0, np.percentile(X[:, 1], 99)
#    x_min, x_max = 10**-5, 10
#    y_min, y_max = 10**-5, 10
#    plt.title('Particle classification', size=17)
#    plt.axis('on')
#    plt.xlim(x_min, x_max)
#    plt.ylim(y_min, y_max)
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.xlabel('D2min')
#    plt.ylabel('granular temperature')
#    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#    plt.savefig(output_path + '/Particle classification-' + str(frame) + '.png', format='png', dpi=1000, bbox_inches='tight')
#    plt.close()

# ----------------------------------------------------------------------
# Visualizing Histograms with Matplotlib
#
def plot_hist(data, output_path, frame):
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=data, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of particle D2min')
    # plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq/10)*10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig(output_path + '/Histogram of particle D2min-' + str(frame) +  '.png', format='png', dpi=1000, bbox_inches='tight')
    plt.close()

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

# ----------------------------------------------------------------------
# DBSCAN
#    
def DBSCAN_v2(neighbor_list, density):
    neighbor_list_dbscan = copy.deepcopy(neighbor_list)
    mark = dict()
    for i in list(neighbor_list_dbscan.keys()):
        if neighbor_list_dbscan[i] == []:
            neighbor_list_dbscan[i].append(i)
            mark[i] = 1
        elif len(neighbor_list_dbscan[i]) < int(density):
            neighbor_list_dbscan[i].append(i)
            mark[i] = 2
        else:
            neighbor_list_dbscan[i].append(i)
            mark[i] = 3
    Core_cluster = dict()
    CoreNeighbor = dict()
    CorePoints = get_key(mark, 3)
    for i in range(len(CorePoints)):
        CoreNeighbor[i] = neighbor_list_dbscan[CorePoints[i]]
    for i in range(len(CoreNeighbor)):
        for j in range(len(CoreNeighbor) - 1, -1, -1):
            if len(set(CoreNeighbor[j]) & set(CoreNeighbor[i])) > 0 and i != j:
                CoreNeighbor[i] = list(set(CoreNeighbor[i]) | set(CoreNeighbor[j]))
                CoreNeighbor[j] = list()
    j = 0
    for i in range(len(CoreNeighbor)):
        if len(CoreNeighbor[i]) > 0:
            Core_cluster[j] = CoreNeighbor[i]
            j = j + 1
    Core_cluster = list(Core_cluster.values())

    Border_cluster = dict()
    BorderNeighbor = dict()
    BorderPoints = get_key(mark, 2)
    j = 0
    for i in range(len(BorderPoints)):
        if len(set(neighbor_list_dbscan[BorderPoints[i]]) & set(CorePoints)) <= 0:
            BorderNeighbor[j] = neighbor_list_dbscan[BorderPoints[i]]
            j = j + 1
    for i in range(len(BorderNeighbor)):
        for j in range(len(BorderNeighbor) - 1, -1, -1):
            if len(set(BorderNeighbor[j]) & set(BorderNeighbor[i])) > 0 and i != j:
                BorderNeighbor[i] = list(set(BorderNeighbor[i]) | set(BorderNeighbor[j]))
                BorderNeighbor[j] = list()
    j = 0
    for i in range(len(BorderNeighbor)):
        if len(BorderNeighbor[i]) > 0:
            Border_cluster[j] = BorderNeighbor[i]
            j = j + 1
    Border_cluster = list(Border_cluster.values())

    Noise_cluster = []
    NoisePoints = get_key(mark, 1)
    for i in NoisePoints:
        Noise_cluster.append([i])
    cluster_all = Core_cluster + Border_cluster + Noise_cluster

    return cluster_all

def DBSCAN_v1(neighbor_list, density):
    neighbor_list_dbscan = copy.deepcopy(neighbor_list)
    mark = dict()
    for i in list(neighbor_list_dbscan.keys()):
        if neighbor_list_dbscan[i] == []:
            mark[i] = 1
        elif len(neighbor_list_dbscan[i]) < int(density):
            mark[i] = 2
        else:
            mark[i] = 3
    mark_done = dict()
    for i in list(neighbor_list_dbscan.keys()):
        mark_done[i] = 0
    cluster_all = []
    for i in list(neighbor_list_dbscan.keys()):
        if mark_done[i] == 1:continue
        if mark[i] < 3:continue
        clu = neighbor_list_dbscan[i]
        cluster = [i]
        mark_done[i] = 1
        for j in clu:
            if mark_done[j] == 0:
                cluster.append(j)
                mark_done[j] = 1
            for k in neighbor_list_dbscan[j]:
                if k in clu or k in cluster or mark_done[k] == 1:continue
                if mark[k] == 3:
                    clu.append(k)
                    cluster.append(k)
                    mark_done[k] = 1
                elif mark[k] == 2:
                    cluster.append(k)
                    mark_done[k] = 1
                else:
                    continue
        cluster = list(set(cluster))
        cluster_all.append(cluster)
    for i in list(neighbor_list_dbscan.keys()):
        if mark_done[i] == 1:continue
        if mark[i] == 2:
            clu = neighbor_list_dbscan[i]
            cluster = [i]
            mark_done[i] = 1
            for j in clu:
                if mark_done[j] == 0:
                    if mark[j] == 2:
                        cluster.append(j)
                        mark_done[j] = 1
                for k in neighbor_list_dbscan[j]:
                    if mark_done[k] == 1:continue
                    if mark[k] == 2:
                        clu.append(k)
                        cluster.append(k)
                        mark_done[j] = 1
            cluster = list(set(cluster))
            cluster_all.append(cluster)
        elif mark[i] == 1:
            cluster_all.append([i])
            mark_done[i] = 1

    return cluster_all

def cal_connectivity_and_distance_matrix(Par_id, Par_coord, Par_radius, Par_selected, vor, vor_nearest_neighbors,
                                         epsilon, frame):
    Par_num = len(Par_id)
    Par_num_selected = np.sum(Par_selected == True)
    Par_index = np.arange(Par_num)
    Par_index_all = Par_index[Par_selected]  # 选中的颗粒，在所有颗粒中的编号
    Par_index_selected = dict(zip(Par_index_all, range(Par_num_selected)))  # 所有颗粒中，选中的部分颗粒，其在这部分中的编号
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2.1 Define the structure of the dataset
    #     Create a sparse matrix capturing local connectivity. Larger number of neighbors will give more homogeneous clusters to the cost of computation time.
    #     A very large number of neighbors gives more evenly distributed cluster sizes, but may not impose the local manifold structure of the data
    #
    # Assign graph attributes when creating a new graph
    t0 = time.time()
    G = nx.Graph(series=frame)
    # Add one edge at a time and add edge attributes using add_edge()
    neighbor_list = defaultdict(list)
    for i in range(Par_num):
        if not Par_selected[i]: continue
        G.add_node(i)

        # Hard connectivity
        # neighbor_par_i = vor_nearest_neighbors[i]
        # for j in neighbor_par_i:
        #     # From these remaining particles, filter out all particles that are not in contact with
        #     # any other particles, and all particles that are only in contact with one other particles.
        #     if Par_selected[j]: G.add_edge(i, j)

        # Soft connectivity
        vertices_coord = np.array(vor[i]['vertices'])
        for k, face in enumerate(vor[i]['faces']):
            if face['adjacent_cell'] < 0: continue
            j = face['adjacent_cell']
            if not Par_selected[j]: continue
            neighbor_par_j = vor_nearest_neighbors[j]
            if i not in neighbor_par_j: continue
            vertices_list = face['vertices']
            polygon = vertices_coord[vertices_list]
            face_area = PolygonArea(polygon)
            clearance = vertex_distance(Par_coord[i], Par_coord[j]) - Par_radius[i] - Par_radius[j]
            diameter = 4.0*Par_radius[i]*Par_radius[j]/(Par_radius[i] + Par_radius[j])
            # print("Weight  ", face_area/clearance/diameter, diameter/clearance)
            # if face_area/clearance >= epsilon*diameter:
            if clearance <= epsilon*diameter:
                G.add_edge(i, j)
                neighbor_list[i].append(j)

    # Distance matrix for hierarchical clustering with distancethresholdd:
    # distance_matrix = np.ones([Par_num_selected, Par_num_selected])*10000
    # np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.zeros([Par_num_selected, Par_num_selected])

    # Modify graph to prevent further change by adding or removing nodes or edges
    if not nx.is_frozen(G): nx.freeze(G)
    G_trimmed = G.copy()

    # for sub_graph in nx.connected_component_subgraphs(G):
    for sub_graph in [G.subgraph(c) for c in nx.connected_components(G)]:
        sub_graph_trimmed = sub_graph.copy()
        sub_graph_num = nx.number_connected_components(sub_graph_trimmed)
        while True:
            if sub_graph_trimmed.number_of_edges() <= 20: break
            edge_betweenness = nx.edge_betweenness_centrality(sub_graph_trimmed, normalized=True, weight=None)
            edge_betweenness = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)  # https://www.cnblogs.com/yoyoketang/p/9147052.html
            edge_betweenness = dict(edge_betweenness)
            betweenness = list(edge_betweenness.values())
            key_list = list(edge_betweenness.keys())
            key = key_list[0]
            edge_gini = gini(betweenness[0:]) - gini(betweenness[1:])
            edge_deviation = (edge_betweenness[key] - np.mean(betweenness[1:]))/np.std(betweenness[1:])
            if edge_deviation >= 3:
                sub_graph_trimmed.remove_edge(key[0], key[1])
                if nx.number_connected_components(sub_graph_trimmed) > sub_graph_num:
                    G_trimmed.remove_edge(key[0], key[1])
                    sub_graph_num = nx.number_connected_components(sub_graph_trimmed)
                else:
                    sub_graph_trimmed.add_edge(key[0], key[1])
                    break
                # print("Connected: ", nx.is_connected(sub_graph_trimmed))  # Return True if the graph is connected, false otherwise
                # print(len(edge_betweenness), edge_gini, edge_deviation, sub_graph_num)
            else:
                break

        # node_list = list(sub_graph.nodes())
        # for i in range(sub_graph.number_of_nodes()):
        #     for j in range(i+1, sub_graph.number_of_nodes()):
        #         node_i = node_list[i]
        #         node_j = node_list[j]
        #         try:
        #             distance_matrix[Par_index_selected[node_i], Par_index_selected[node_j]] = nx.shortest_path_length(sub_graph, source=node_i, target=node_j)
        #         except:
        #             distance_matrix[Par_index_selected[node_i], Par_index_selected[node_j]] = 10000
        #         distance_matrix[Par_index_selected[node_j], Par_index_selected[node_i]] = distance_matrix[Par_index_selected[node_i], Par_index_selected[node_j]]

    # Connectivity for hierarchical clustering with structured dataset
    row, col, data = [], [], []
    for u, v in G_trimmed.edges():
        row.append(Par_index_selected[u])
        col.append(Par_index_selected[v])
        data.append(1.0)
    # Compressed Sparse Row matrix:
    connectivity = csr_matrix((data, (row, col)), shape=(Par_num_selected, Par_num_selected))

    # Print short summary of information for the graph G and G_trimmed
    print(60 * '~')
    print(nx.info(G))
    print("Number of connected components: ", nx.number_connected_components(G))  # Returns the number of connected components
    print(nx.info(G_trimmed))
    print("Number of connected components: ", nx.number_connected_components(G_trimmed))  # Returns the number of connected components

    elapsed_time = time.time() - t0
    print("Elapsed time: %.2fs" % elapsed_time)
    print(60 * '~')

    # connectivity:     hierarchical clustering
    # distance_matrix:  hierarchical clustering
    # neighbor_list:    DBSCAN
    # G:                pure connection
    # G_trimmed:        connection with trimmed graph
    return G, G_trimmed, connectivity, distance_matrix, neighbor_list

# ----------------------------------------------------------------------
# Clustering by different algorithms
#
def particle_clustering_nx(d50, Par_id, Par_coord, Par_radius, Par_volume, Par_D2min, Par_shear_strain, Par_selected,
                           vor, Graph, trimmed, output_path, frame, dump_results, title):

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. Data definition
    #
    Par_num = len(Par_id)
    Par_num_selected = np.sum(Par_selected == True)
    Par_weight = Par_volume
    Par_type = np.ones(Par_num)
    Par_index = np.arange(Par_num)
    Par_index_all = Par_index[Par_selected]                                 # 选中的颗粒，在所有颗粒中的编号
    Par_index_selected = dict(zip(Par_index_all, range(Par_num_selected)))  # 所有颗粒中，选中的部分颗粒，其在这部分中的编号
    Par_index_all = np.array(Par_index_all)

    # Par_coord_selected = Par_coord[Par_selected]
    # Par_type_selected = Par_type[Par_selected]
    # Par_id_selected = Par_id[Par_selected]
    # Par_radius_selected = Par_radius[Par_selected]
    # Par_D2min_selected = Par_D2min[Par_selected]
    # Par_shear_strain_selected = Par_shear_strain[Par_selected]
    # Par_weight_selected = Par_weight[Par_selected]
    # Par_volume_selected = Par_volume[Par_selected]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2.2 Cluster identification and geometrical analysis
    #     RDCI: A novel method of cluster analysis and applications thereof in sample molecular simulations
    #     https: // pubs.acs.org.ccindex.cn/doi/full/10.1021/jp2065612?src = recsys
    #     http://www.sbg.bio.ic.ac.uk/maxcluster/
    nx_clusters = []
    nx_clusters_summary = []
    nx_clusters_nonaffinity = []
    nx_par_num_cluster = 0
    Par_nx_cluster_id = np.zeros(Par_num).astype(int)
    cluster_id = 0
    for i, sub_graph in enumerate(nx.connected_component_subgraphs(Graph)):
        cluster = list(sub_graph.nodes())
        nx_clusters.append(cluster)
        cluster_par_num = sub_graph.number_of_nodes()
        if cluster_par_num < 2: continue
        nx_par_num_cluster += cluster_par_num
        cluster_size = np.sqrt(cluster_par_num)
        cluster_id += 1

        cluster_xcen = np.sum(Par_coord[cluster, 0]*Par_weight[cluster])/np.sum(Par_weight[cluster])
        cluster_ycen = np.sum(Par_coord[cluster, 1]*Par_weight[cluster])/np.sum(Par_weight[cluster])
        cluster_zcen = np.sum(Par_coord[cluster, 2]*Par_weight[cluster])/np.sum(Par_weight[cluster])
        cluster_center = np.array([cluster_xcen, cluster_ycen, cluster_zcen])
        cluster_par_volume = 0
        cluster_gyration_radius = 0.0
        cluster_gyration_tensor = np.zeros([3, 3])
        cluster_inertia_tensor = np.zeros([3, 3])
        cluster_voronoi_volume = 0
        cluster_voronoi_area = 0
        cluster_voronoi_vertices = []
        for j in cluster:
            Par_nx_cluster_id[j] = cluster_id
            cluster_par_volume += Par_volume[j]
            cluster_gyration_radius += Par_weight[j]*vertex_distance(Par_coord[j], cluster_center)**2.0
            for k in cluster:
                pos_relative = Par_coord[k] - Par_coord[j]
                cluster_gyration_tensor += np.outer(pos_relative, pos_relative)*Par_weight[j]*Par_weight[k]
            pos_relative = Par_coord[j] - cluster_center
            # cluster_gyration_tensor += np.outer(pos_relative, pos_relative)
            cluster_inertia_tensor += Par_weight[j]*(np.sum(pos_relative**2.0)*np.eye(3,3) - np.outer(pos_relative, pos_relative))

            cluster_voronoi_volume += vor[j]['volume']
            vertices_coord = np.array(vor[j]['vertices'])
            cluster_voronoi_vertices.extend(vertices_coord)
            for k, face in enumerate(vor[j]['faces']):
                if (face['adjacent_cell'] in cluster): continue
                vertices_list = face['vertices']
                polygon = vertices_coord[vertices_list]
                cluster_voronoi_area += PolygonArea(polygon)

        cluster_gyration_radius = np.sqrt(cluster_gyration_radius/np.sum(Par_weight[cluster]))
        # cluster_gyration_tensor = cluster_gyration_tensor/cluster_par_num
        cluster_gyration_tensor = cluster_gyration_tensor/(2*np.sum(Par_weight[cluster])**2.0)
        cluster_inertia_tensor = cluster_inertia_tensor/np.sum(Par_weight[cluster])
        if cluster_par_num == 1:
            cluster_shape_anisotropy = 0.0
        else:
            eigenvalues, eigenvectors = np.linalg.eig(cluster_gyration_tensor)
            cluster_eigenvalues = np.abs(eigenvalues)
            cluster_eigenvalues.sort()
            # sphericity
            cluster_b = cluster_eigenvalues[2] - 0.5*(cluster_eigenvalues[0] + cluster_eigenvalues[1])
            # cylindricity
            cluster_c = cluster_eigenvalues[1] - cluster_eigenvalues[0]
            cluster_shape_anisotropy = (cluster_b**2.0 + 3*cluster_c**2.0/4.0)/(cluster_gyration_radius**4.0)

        cluster_voronoi_vertices = np.array(cluster_voronoi_vertices)
        cluster_voronoi_convex_volume, cluster_voronoi_axies = cluster_shape(cluster_voronoi_vertices[:,0], cluster_voronoi_vertices[:,1], cluster_voronoi_vertices[:,2])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # A Python package to compute bond orientational order parameters as defined by Steinhardt Physical Review B (1983) doi:10.1103/PhysRevB.28.784.
        #
        # Steinhardt’s bond orientational order parameter is a popular method (>20k citations of the original paper) to identify local symmetries in an assembly of particles in 3D.
        # It can be used in particle-based simulations (typically molecular dynamics, brownian dynamics, monte-carlo, etc.)
        # or in particle-tracking experiments (colloids, granular materials) where the coordinates of all particles are known.
        # reference:
        # [1] https://pyboo.readthedocs.io/en/latest/intro.html
        # [2] Xia, C. et al. The structural origin of the hard-sphere glass transition in granular packing. Nat. Commun. 6, 1–9 (2015).
        # [3] Teich, E. G., Van Anders, G., Klotsa, D., Dshemuchadse, J. & Glotzer, S. C. Clusters of polyhedra in spherical confinement. Proc. Natl. Acad. Sci. U. S. A. (2016). doi:10.1073/pnas.1524875113
        # [4] Leocmach, M. & Tanaka, H. Roles of icosahedral and crystal-like order in the hard spheres glass transition. Nat. Commun. (2012). doi:10.1038/ncomms1974

        cluster_par_coord = np.stack((Par_coord[cluster, 0], Par_coord[cluster, 1], Par_coord[cluster, 2]), axis=1)
        cluster_par_weight = Par_weight[cluster] # cluster_par_weight = np.ones(len(cluster))
        cluster_boo_Q2 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=2, modified=True)
        cluster_boo_Q4 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=4, modified=True)
        cluster_boo_Q6 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=6, modified=True)
        cluster_boo_Q8 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=8, modified=True)
        cluster_boo_Q10 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=10, modified=True)
        cluster_boo_Q12 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=12, modified=True)

        # define coordinates and bond network
        #
        # cluster_par_coord = np.vstack((cluster_center, cluster_par_coord))
        # cluster_bond = np.array([[0, k+1] for k in range(cluster_par_num)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # tensorial bond orientational order
        #
        # cluster_boo_q4m = boo.bonds2qlm(cluster_par_coord, cluster_bond, l=4)
        # cluster_boo_q6m = boo.bonds2qlm(cluster_par_coord, cluster_bond, l=6)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Rotational invarients
        # The tensorial q?mq?m coefficients are dependent of the orientation of the reference axis.
        # That is why we have to compute quantities that are rotationally invarients.
        # Rotational invarients give more precise information about the type of structures.
        #
        # cluster_boo_Q4 = boo.ql(cluster_boo_q4m[0])   # the second order invarient indicates the strength of the ??-fold symetry.
        # cluster_boo_Q6 = boo.ql(cluster_boo_q6m[0])
        # cluster_boo_W4 = boo.wl(cluster_boo_q4m[0])   # the third order invarient allows to discriminate different types of ?-fold symetric structures
        # cluster_boo_W6 = boo.wl(cluster_boo_q6m[0])

        cluster_nonaffinity = np.mean(Par_D2min[cluster])
        cluster_local_strain = np.mean(Par_shear_strain[cluster])
        nx_clusters_nonaffinity.append(cluster_nonaffinity)
        nx_clusters_summary.append([cluster_par_num, cluster_par_volume, cluster_gyration_radius, cluster_shape_anisotropy,
                                    cluster_boo_Q2, cluster_boo_Q4, cluster_boo_Q6, cluster_boo_Q8, cluster_boo_Q10, cluster_boo_Q12,
                                    cluster_voronoi_axies[0], cluster_voronoi_axies[1], cluster_voronoi_axies[2], cluster_voronoi_area, cluster_voronoi_volume, cluster_voronoi_convex_volume])

    nx_clusters_num = len(nx_clusters_summary)
    nx_clusters_sort = np.zeros(nx_clusters_num)
    for i, index in enumerate(np.argsort(nx_clusters_nonaffinity)):
        nx_clusters_sort[index] = nx_clusters_num - i

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3. Output the geometry of cluster
    #
    if dump_results:
        if trimmed:
            myOutputFile = open(output_path + '/' + title + '-clusters by trimmed contiguity-' + str(frame) + '.dump', 'w')
        else:
            myOutputFile = open(output_path + '/' + title + '-clusters by contiguity-' + str(frame) + '.dump', 'w')

        myOutputFile.write("ITEM: TIMESTEP\n")
        myOutputFile.write("%-d\n" % frame)
        myOutputFile.write("ITEM: NUMBER OF ATOMS\n")
        myOutputFile.write("%-d\n" % nx_par_num_cluster)
        myOutputFile.write("ITEM: BOX BOUNDS pp ff pp\n")
        myOutputFile.write("0 0.04\n")
        myOutputFile.write("-0.005 0.025\n")
        myOutputFile.write("0 0.02\n")
        myOutputFile.write("ITEM: ATOMS id type radius x y z cluster_id cluster_rank cluster_par_num cluster_par_volume cluster_gyration cluster_anisotropy "
                           "Q2 Q4 Q6 Q8 Q10 Q12 major_axis intermediate_axis minor_axis voronoi_area voronoi_volume voronoi_convex_volume\n")
        for i in range(Par_num):
            cluster_id = Par_nx_cluster_id[i]
            if cluster_id:
                myOutputFile.write("%d %d %.8f %.8f %.8f %.8f %d %d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n"
                       % (Par_id[i], Par_type[i], Par_radius[i], Par_coord[i, 0], Par_coord[i, 1], Par_coord[i, 2],
                          cluster_id, nx_clusters_sort[cluster_id-1],
                          nx_clusters_summary[cluster_id-1][0], nx_clusters_summary[cluster_id-1][1], nx_clusters_summary[cluster_id-1][2],
                          nx_clusters_summary[cluster_id-1][3], nx_clusters_summary[cluster_id-1][4], nx_clusters_summary[cluster_id-1][5],
                          nx_clusters_summary[cluster_id-1][6], nx_clusters_summary[cluster_id-1][7], nx_clusters_summary[cluster_id-1][8],
                          nx_clusters_summary[cluster_id-1][9], nx_clusters_summary[cluster_id-1][10], nx_clusters_summary[cluster_id-1][11],
                          nx_clusters_summary[cluster_id-1][12], nx_clusters_summary[cluster_id-1][13], nx_clusters_summary[cluster_id-1][14],
                          nx_clusters_summary[cluster_id-1][15]))
            # else:
            #     myOutputFile.write("%d %d %.8f %.8f %.8f %.8f %d %d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n"
            #                        % (Par_id[i], Par_type[i], Par_radius[i], Par_coord[i, 0], Par_coord[i, 1], Par_coord[i, 2],
            #                           cluster_id, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        myOutputFile.close()

    # skl_clusters_summary = nx_clusters_summary
    return nx_clusters_summary

# ----------------------------------------------------------------------
# Clustering by different algorithms
#
def particle_clustering_skl(d50, Par_id, Par_coord, Par_radius, Par_volume, Par_D2min, Par_shear_strain, Par_selected,
                            vor, connectivity, linkage, G_trimmed, output_path, frame, dump_results, title):

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. Data definition
    #
    Par_num = len(Par_id)
    Par_num_selected = np.sum(Par_selected == True)
    Par_weight = Par_volume
    Par_type = np.ones(Par_num)
    Par_index = np.arange(Par_num)
    Par_index_all = Par_index[Par_selected]                                 # 选中的颗粒，在所有颗粒中的编号
    Par_index_selected = dict(zip(Par_index_all, range(Par_num_selected)))  # 所有颗粒中，选中的部分颗粒，其在这部分中的编号
    Par_index_all = np.array(Par_index_all)

    Par_coord_selected = Par_coord[Par_selected]
    # Par_type_selected = Par_type[Par_selected]
    # Par_id_selected = Par_id[Par_selected]
    # Par_radius_selected = Par_radius[Par_selected]
    # Par_D2min_selected = Par_D2min[Par_selected]
    # Par_shear_strain_selected = Par_shear_strain[Par_selected]
    # Par_weight_selected = Par_weight[Par_selected]
    # Par_volume_selected = Par_volume[Par_selected]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. Clustering of unlabeled data
    #    references:
    #    [1] https://scikit-learn.org/stable/modules/clustering.html#
    #    [2] https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
    #    [3] http://sklearn.lzjqsdd.com/auto_examples/cluster/plot_cluster_comparison.html

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  2.2 n*m array of observations on m attributes across n samples.
    #
    # Par_attr1 = np.log10(Par_D2min_selected)
    # Par_attr2 = np.log10(Par_shear_strain_selected)
    # Par_attr1 = array_normalization(Par_D2min_selected)
    # Par_attr2 = array_normalization(Par_shear_strain_selected)
    # Par_attributes = np.stack((Par_attr1, Par_attr2), axis=1)
    # scaler = RobustScaler(quantile_range=(10, 90))
    # Par_attributes = scaler.fit_transform(np.stack((Par_D2min_selected, Par_shear_strain_selected), axis=1))
    Par_attributes = Par_coord_selected

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2.3 Clustering with connectivity constraints
    #     The hierarchical clustering is restricted to the nearest neighbors graph: it’s a hierarchical clustering with structure prior.
    #
    # linkage criteria: 'average', 'complete', 'ward', 'single'
    # Comparing different hierarchical linkage methods on toy datasets
    # The main observations to make are:
    # single linkage is fast, and can perform well on non-globular data, but it performs poorly in the presence of noise.
    # average and complete linkage perform well on cleanly separated globular clusters, but have mixed results otherwise.
    # Ward is the most effective method for noisy data.
    print("Compute hierarchical clustering...")
    t0 = time.time()
    skl_clustering = AgglomerativeClustering(n_clusters=nx.number_connected_components(G_trimmed), connectivity=connectivity, linkage=linkage).fit(Par_attributes)
    elapsed_time = time.time() - t0
    skl_clusters_num = np.max(skl_clustering.labels_) + 1

    # Hierarchical clustering summary
    print("linkage criteria:                            %s" % linkage)
    print("Number of regions:                           %d" % skl_clusters_num)
    print("Number of leaves in the hierarchical tree:   %d" % skl_clustering.n_leaves_)
    print("Number of connected components in the graph: %d" % skl_clustering.n_connected_components_)
    print("Elapsed time:                                %.2fs" % elapsed_time)
    print(60 * '~')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2.4 statistics over the hierarchical clusters
    #     references:
    #     [1] https://en.wikipedia.org/wiki/Radius_of_gyration
    skl_clusters_nonaffinity = np.zeros(skl_clusters_num)
    skl_clusters_summary = [[] for i in range(skl_clusters_num)]
    skl_clusters = [[] for i in range(skl_clusters_num)]
    skl_labels = skl_clustering.labels_
    skl_par_num_cluster = 0
    Par_skl_cluster_id = np.zeros(Par_num).astype(int)
    for i in range(Par_num_selected):
        # cluster_id = skl_labels[i]
        skl_clusters[skl_labels[i]].append(Par_index_all[i])

    cluster_id = 0
    for i, cluster in enumerate(skl_clusters):
        cluster_par_num = len(cluster)
        if cluster_par_num < 2: continue
        skl_par_num_cluster += cluster_par_num
        cluster_size = np.sqrt(cluster_par_num)
        cluster_id += 1

        cluster_xcen = np.sum(Par_coord[cluster, 0]*Par_weight[cluster])/np.sum(Par_weight[cluster])
        cluster_ycen = np.sum(Par_coord[cluster, 1]*Par_weight[cluster])/np.sum(Par_weight[cluster])
        cluster_zcen = np.sum(Par_coord[cluster, 2]*Par_weight[cluster])/np.sum(Par_weight[cluster])
        cluster_center = np.array([cluster_xcen, cluster_ycen, cluster_zcen])
        cluster_par_volume = 0
        cluster_gyration_radius = 0.0
        cluster_gyration_tensor = np.zeros([3, 3])
        cluster_inertia_tensor = np.zeros([3, 3])
        cluster_voronoi_volume = 0
        cluster_voronoi_area = 0
        cluster_voronoi_vertices = []
        for j in cluster:
            Par_skl_cluster_id[j] = cluster_id
            cluster_par_volume += Par_volume[j]
            cluster_gyration_radius += Par_weight[j]*vertex_distance(Par_coord[j], cluster_center)**2.0
            for k in cluster:
                pos_relative = Par_coord[j] - Par_coord[k]
                cluster_gyration_tensor += np.outer(pos_relative, pos_relative)*Par_weight[j]*Par_weight[j]
            pos_relative = Par_coord[j] - cluster_center
            # cluster_gyration_tensor += np.outer(pos_relative, pos_relative)
            cluster_inertia_tensor += Par_weight[j]*(np.sum(pos_relative**2.0)*np.eye(3,3) - np.outer(pos_relative, pos_relative))

            cluster_voronoi_volume += vor[j]['volume']
            vertices_coord = np.array(vor[j]['vertices'])
            cluster_voronoi_vertices.extend(vertices_coord)
            for k, face in enumerate(vor[j]['faces']):
                if (face['adjacent_cell'] in cluster): continue
                vertices_list = face['vertices']
                polygon = vertices_coord[vertices_list]
                cluster_voronoi_area += PolygonArea(polygon)

        cluster_gyration_radius = np.sqrt(cluster_gyration_radius/np.sum(Par_weight[cluster]))
        # cluster_gyration_tensor = cluster_gyration_tensor/cluster_par_num
        cluster_gyration_tensor = cluster_gyration_tensor/(2*np.sum(Par_weight[cluster])**2.0)
        cluster_inertia_tensor = cluster_inertia_tensor/np.sum(Par_weight[cluster])
        if cluster_par_num == 1:
            cluster_shape_anisotropy = 0.0
        else:
            eigenvalues, eigenvectors = np.linalg.eig(cluster_gyration_tensor)
            cluster_eigenvalues = np.abs(eigenvalues)
            cluster_eigenvalues.sort()
            # sphericity
            cluster_b = cluster_eigenvalues[2] - 0.5*(cluster_eigenvalues[0] + cluster_eigenvalues[1])
            # cylindricity
            cluster_c = cluster_eigenvalues[1] - cluster_eigenvalues[0]
            cluster_shape_anisotropy = (cluster_b**2.0 + 3*cluster_c**2.0/4.0)/(cluster_gyration_radius**4.0)

        cluster_voronoi_vertices = np.array(cluster_voronoi_vertices)
        cluster_voronoi_convex_volume, cluster_voronoi_axies = cluster_shape(cluster_voronoi_vertices[:,0], cluster_voronoi_vertices[:,1], cluster_voronoi_vertices[:,2])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # A Python package to compute bond orientational order parameters as defined by Steinhardt Physical Review B (1983) doi:10.1103/PhysRevB.28.784.
        #
        # Steinhardt’s bond orientational order parameter is a popular method (>20k citations of the original paper) to identify local symmetries in an assembly of particles in 3D.
        # It can be used in particle-based simulations (typically molecular dynamics, brownian dynamics, monte-carlo, etc.)
        # or in particle-tracking experiments (colloids, granular materials) where the coordinates of all particles are known.
        # reference:
        # [1] https://pyboo.readthedocs.io/en/latest/intro.html
        # [2] Xia, C. et al. The structural origin of the hard-sphere glass transition in granular packing. Nat. Commun. 6, 1–9 (2015).
        # [3] Teich, E. G., Van Anders, G., Klotsa, D., Dshemuchadse, J. & Glotzer, S. C. Clusters of polyhedra in spherical confinement. Proc. Natl. Acad. Sci. U. S. A. (2016). doi:10.1073/pnas.1524875113
        # [4] Leocmach, M. & Tanaka, H. Roles of icosahedral and crystal-like order in the hard spheres glass transition. Nat. Commun. (2012). doi:10.1038/ncomms1974

        cluster_par_coord = np.stack((Par_coord[cluster, 0], Par_coord[cluster, 1], Par_coord[cluster, 2]), axis=1)
        cluster_par_weight = Par_weight[cluster] # cluster_par_weight = np.ones(len(cluster))
        cluster_boo_Q2 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=2, modified=True)
        cluster_boo_Q4 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=4, modified=True)
        cluster_boo_Q6 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=6, modified=True)
        cluster_boo_Q8 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=8, modified=True)
        cluster_boo_Q10 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=10, modified=True)
        cluster_boo_Q12 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=12, modified=True)

        # define coordinates and bond network
        #
        # cluster_par_coord = np.vstack((cluster_center, cluster_par_coord))
        # cluster_bond = np.array([[0, k+1] for k in range(cluster_par_num)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # tensorial bond orientational order
        #
        # cluster_boo_q4m = boo.bonds2qlm(cluster_par_coord, cluster_bond, l=4)
        # cluster_boo_q6m = boo.bonds2qlm(cluster_par_coord, cluster_bond, l=6)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Rotational invarients
        # The tensorial q?mq?m coefficients are dependent of the orientation of the reference axis.
        # That is why we have to compute quantities that are rotationally invarients.
        # Rotational invarients give more precise information about the type of structures.
        #
        # cluster_boo_Q4 = boo.ql(cluster_boo_q4m[0])   # the second order invarient indicates the strength of the ??-fold symetry.
        # cluster_boo_Q6 = boo.ql(cluster_boo_q6m[0])
        # cluster_boo_W4 = boo.wl(cluster_boo_q4m[0])   # the third order invarient allows to discriminate different types of ?-fold symetric structures
        # cluster_boo_W6 = boo.wl(cluster_boo_q6m[0])

        cluster_nonaffinity = np.mean(Par_D2min[cluster])
        cluster_local_strain = np.mean(Par_shear_strain[cluster])
        skl_clusters_nonaffinity[i] = cluster_nonaffinity
        skl_clusters_summary[i] = [cluster_par_num, cluster_par_volume, cluster_gyration_radius, cluster_shape_anisotropy,
                                   cluster_boo_Q2, cluster_boo_Q4, cluster_boo_Q6, cluster_boo_Q8, cluster_boo_Q10, cluster_boo_Q12,
                                   cluster_voronoi_axies[0], cluster_voronoi_axies[1], cluster_voronoi_axies[2], cluster_voronoi_area, cluster_voronoi_volume, cluster_voronoi_convex_volume]

    skl_clusters_sort = np.zeros(skl_clusters_num)
    for i, index in enumerate(np.argsort(skl_clusters_nonaffinity)):
        skl_clusters_sort[index] = skl_clusters_num - i

    # ----------------------------------------------------------------------
    # 2.5 Visualize the clustering
    #
    # 2D Plot result
    # plot_clustering_2d(Par_attributes, skl_labels, "structured hierarchical clustering", output_path, frame)

    # ----------------------------------------------------------------------
    # 3D Plot result
    # fig_title = title + " clusters by structured hierarchical clustering"
    # plot_clustering_3d(Par_coord_selected, skl_labels, fig_title , output_path, frame)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3. Output the geometry of cluster
    #
    if dump_results:
        myOutputFile = open(output_path + '/' + title + '-clusters by hierarchical clustering-' + str(frame) + '.dump', 'w')
        myOutputFile.write("ITEM: TIMESTEP\n")
        myOutputFile.write("%-d\n" % frame)
        myOutputFile.write("ITEM: NUMBER OF ATOMS\n")
        myOutputFile.write("%-d\n" % skl_par_num_cluster)
        myOutputFile.write("ITEM: BOX BOUNDS pp ff pp\n")
        myOutputFile.write("0 0.04\n")
        myOutputFile.write("-0.005 0.025\n")
        myOutputFile.write("0 0.02\n")
        myOutputFile.write("ITEM: ATOMS id type radius x y z cluster_id cluster_rank cluster_par_num cluster_par_volume cluster_gyration cluster_anisotropy "
                           "Q2 Q4 Q6 Q8 Q10 Q12 major_axis intermediate_axis minor_axis voronoi_area voronoi_volume voronoi_convex_volume\n")
        for i in range(Par_num):
            cluster_id = Par_skl_cluster_id[i]
            if cluster_id:
                myOutputFile.write("%d %d %.8f %.8f %.8f %.8f %d %d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n"
                                   % (Par_id[i], Par_type[i], Par_radius[i], Par_coord[i, 0], Par_coord[i, 1], Par_coord[i, 2],
                                      cluster_id, skl_clusters_sort[cluster_id-1],
                                      skl_clusters_summary[cluster_id-1][0], skl_clusters_summary[cluster_id-1][1], skl_clusters_summary[cluster_id-1][2],
                                      skl_clusters_summary[cluster_id-1][3], skl_clusters_summary[cluster_id-1][4], skl_clusters_summary[cluster_id-1][5],
                                      skl_clusters_summary[cluster_id-1][6], skl_clusters_summary[cluster_id-1][7], skl_clusters_summary[cluster_id-1][8],
                                      skl_clusters_summary[cluster_id-1][9], skl_clusters_summary[cluster_id-1][10], skl_clusters_summary[cluster_id-1][11],
                                      skl_clusters_summary[cluster_id-1][12], skl_clusters_summary[cluster_id-1][13], skl_clusters_summary[cluster_id-1][14],
                                      skl_clusters_summary[cluster_id-1][15]))
            # else:
            #     myOutputFile.write("%d %d %.8f %.8f %.8f %.8f %d %d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n"
            #                        % (Par_id[i], Par_type[i], Par_radius[i], Par_coord[i, 0], Par_coord[i, 1], Par_coord[i, 2],
            #                           cluster_id, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        myOutputFile.close()

    # skl_clusters_summary = nx_clusters_summary
    return skl_clusters_summary

# ----------------------------------------------------------------------
# Clustering by different algorithms
#
def particle_clustering_dbscan(d50, Par_id, Par_coord, Par_radius, Par_volume, Par_D2min, Par_shear_strain, Par_selected,
                               vor, neighbor_list, output_path, frame, dump_results, title):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. Data definition
    #
    Par_num = len(Par_id)
    Par_num_selected = np.sum(Par_selected == True)
    Par_weight = Par_volume
    Par_type = np.ones(Par_num)
    Par_index = np.arange(Par_num)
    Par_index_all = Par_index[Par_selected]  # 选中的颗粒，在所有颗粒中的编号
    Par_index_selected = dict(zip(Par_index_all, range(Par_num_selected)))  # 所有颗粒中，选中的部分颗粒，其在这部分中的编号
    Par_index_all = np.array(Par_index_all)

    # Par_coord_selected = Par_coord[Par_selected]
    # Par_type_selected = Par_type[Par_selected]
    # Par_id_selected = Par_id[Par_selected]
    # Par_radius_selected = Par_radius[Par_selected]
    # Par_D2min_selected = Par_D2min[Par_selected]
    # Par_shear_strain_selected = Par_shear_strain[Par_selected]
    # Par_weight_selected = Par_weight[Par_selected]
    # Par_volume_selected = Par_volume[Par_selected]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. DBSCAN analysis
    db_connected_component = DBSCAN_v1(neighbor_list, 3)
    db_clusters = []
    db_clusters_summary = []
    db_clusters_nonaffinity = []
    db_par_num_cluster = 0
    Par_db_cluster_id = np.zeros(Par_num).astype(int)
    
    cluster_id = 0
    for i, sub_graph in enumerate(db_connected_component):
        cluster = sub_graph
        cluster_par_num = len(sub_graph)
        if cluster_par_num < 2: continue
        db_clusters.append(cluster)
        db_par_num_cluster += cluster_par_num
        cluster_size = np.sqrt(cluster_par_num)
        cluster_id += 1

        cluster_xcen = np.sum(Par_coord[cluster, 0]*Par_weight[cluster])/np.sum(Par_weight[cluster])
        cluster_ycen = np.sum(Par_coord[cluster, 1]*Par_weight[cluster])/np.sum(Par_weight[cluster])
        cluster_zcen = np.sum(Par_coord[cluster, 2]*Par_weight[cluster])/np.sum(Par_weight[cluster])
        cluster_center = np.array([cluster_xcen, cluster_ycen, cluster_zcen])
        cluster_par_volume = 0
        cluster_gyration_radius = 0.0
        cluster_gyration_tensor = np.zeros([3, 3])
        cluster_inertia_tensor = np.zeros([3, 3])
        cluster_voronoi_volume = 0
        cluster_voronoi_area = 0
        cluster_voronoi_vertices = []
        for j in cluster:
            Par_db_cluster_id[j] = cluster_id
            cluster_par_volume += Par_volume[j]
            cluster_gyration_radius += Par_weight[j]*vertex_distance(Par_coord[j], cluster_center)**2.0
            for k in cluster:
                pos_relative = Par_coord[j] - Par_coord[k]
                cluster_gyration_tensor += np.outer(pos_relative, pos_relative)*Par_weight[j]*Par_weight[k]
            pos_relative = Par_coord[j] - cluster_center
            # cluster_gyration_tensor += np.outer(pos_relative, pos_relative)
            cluster_inertia_tensor += Par_weight[j]*(np.sum(pos_relative**2.0)*np.eye(3,3) - np.outer(pos_relative, pos_relative))

            cluster_voronoi_volume += vor[j]['volume']
            vertices_coord = np.array(vor[j]['vertices'])
            cluster_voronoi_vertices.extend(vertices_coord)
            for k, face in enumerate(vor[j]['faces']):
                if (face['adjacent_cell'] in cluster): continue
                vertices_list = face['vertices']
                polygon = vertices_coord[vertices_list]
                cluster_voronoi_area += PolygonArea(polygon)

        cluster_gyration_radius = np.sqrt(cluster_gyration_radius/np.sum(Par_weight[cluster]))
        # cluster_gyration_tensor = cluster_gyration_tensor/cluster_par_num
        cluster_gyration_tensor = cluster_gyration_tensor/(2*np.sum(Par_weight[cluster])**2.0)
        cluster_inertia_tensor = cluster_inertia_tensor/np.sum(Par_weight[cluster])
        if cluster_par_num == 1:
            cluster_shape_anisotropy = 0.0
        else:
            eigenvalues, eigenvectors = np.linalg.eig(cluster_gyration_tensor)
            cluster_eigenvalues = np.abs(eigenvalues)
            cluster_eigenvalues.sort()
            # sphericity
            cluster_b = cluster_eigenvalues[2] - 0.5*(cluster_eigenvalues[0] + cluster_eigenvalues[1])
            # cylindricity
            cluster_c = cluster_eigenvalues[1] - cluster_eigenvalues[0]
            cluster_shape_anisotropy = (cluster_b**2.0 + 3*cluster_c**2.0/4.0)/(cluster_gyration_radius**4.0)

        cluster_voronoi_vertices = np.array(cluster_voronoi_vertices)
        cluster_voronoi_convex_volume, cluster_voronoi_axies = cluster_shape(cluster_voronoi_vertices[:,0], cluster_voronoi_vertices[:,1], cluster_voronoi_vertices[:,2])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # A Python package to compute bond orientational order parameters as defined by Steinhardt Physical Review B (1983) doi:10.1103/PhysRevB.28.784.
        #
        # Steinhardt’s bond orientational order parameter is a popular method (>20k citations of the original paper) to identify local symmetries in an assembly of particles in 3D.
        # It can be used in particle-based simulations (typically molecular dynamics, brownian dynamics, monte-carlo, etc.)
        # or in particle-tracking experiments (colloids, granular materials) where the coordinates of all particles are known.
        # reference:
        # [1] https://pyboo.readthedocs.io/en/latest/intro.html
        # [2] Xia, C. et al. The structural origin of the hard-sphere glass transition in granular packing. Nat. Commun. 6, 1–9 (2015).
        # [3] Teich, E. G., Van Anders, G., Klotsa, D., Dshemuchadse, J. & Glotzer, S. C. Clusters of polyhedra in spherical confinement. Proc. Natl. Acad. Sci. U. S. A. (2016). doi:10.1073/pnas.1524875113
        # [4] Leocmach, M. & Tanaka, H. Roles of icosahedral and crystal-like order in the hard spheres glass transition. Nat. Commun. (2012). doi:10.1038/ncomms1974

        cluster_par_coord = np.stack((Par_coord[cluster, 0], Par_coord[cluster, 1], Par_coord[cluster, 2]), axis=1)
        cluster_par_weight = Par_weight[cluster] # cluster_par_weight = np.ones(len(cluster))
        cluster_boo_Q2 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=2, modified=True)
        cluster_boo_Q4 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=4, modified=True)
        cluster_boo_Q6 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=6, modified=True)
        cluster_boo_Q8 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=8, modified=True)
        cluster_boo_Q10 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=10, modified=True)
        cluster_boo_Q12 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=12, modified=True)

        # define coordinates and bond network
        #
        # cluster_par_coord = np.vstack((cluster_center, cluster_par_coord))
        # cluster_bond = np.array([[0, k+1] for k in range(cluster_par_num)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # tensorial bond orientational order
        #
        # cluster_boo_q4m = boo.bonds2qlm(cluster_par_coord, cluster_bond, l=4)
        # cluster_boo_q6m = boo.bonds2qlm(cluster_par_coord, cluster_bond, l=6)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Rotational invarients
        # The tensorial q?mq?m coefficients are dependent of the orientation of the reference axis.
        # That is why we have to compute quantities that are rotationally invarients.
        # Rotational invarients give more precise information about the type of structures.
        #
        # cluster_boo_Q4 = boo.ql(cluster_boo_q4m[0])   # the second order invarient indicates the strength of the ??-fold symetry.
        # cluster_boo_Q6 = boo.ql(cluster_boo_q6m[0])
        # cluster_boo_W4 = boo.wl(cluster_boo_q4m[0])   # the third order invarient allows to discriminate different types of ?-fold symetric structures
        # cluster_boo_W6 = boo.wl(cluster_boo_q6m[0])

        cluster_nonaffinity = np.mean(Par_D2min[cluster])
        cluster_local_strain = np.mean(Par_shear_strain[cluster])
        db_clusters_nonaffinity.append(cluster_nonaffinity)
        db_clusters_summary.append([cluster_par_num, cluster_par_volume, cluster_gyration_radius, cluster_shape_anisotropy,
                                    cluster_boo_Q2, cluster_boo_Q4, cluster_boo_Q6, cluster_boo_Q8, cluster_boo_Q10, cluster_boo_Q12,
                                    cluster_voronoi_axies[0], cluster_voronoi_axies[1], cluster_voronoi_axies[2], cluster_voronoi_area, cluster_voronoi_volume, cluster_voronoi_convex_volume])

    db_clusters_num = len(db_clusters_summary)
    db_clusters_sort = np.zeros(db_clusters_num)
    for i, index in enumerate(np.argsort(db_clusters_nonaffinity)):
        db_clusters_sort[index] = db_clusters_num - i

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3. Output the geometry of cluster
    #
    if dump_results:
        myOutputFile = open(output_path + '/' + title + '-clusters by DBSCAN-' + str(frame) + '.dump', 'w')
        myOutputFile.write("ITEM: TIMESTEP\n")
        myOutputFile.write("%-d\n" % frame)
        myOutputFile.write("ITEM: NUMBER OF ATOMS\n")
        myOutputFile.write("%-d\n" % db_par_num_cluster)
        myOutputFile.write("ITEM: BOX BOUNDS pp ff pp\n")
        myOutputFile.write("0 0.04\n")
        myOutputFile.write("-0.005 0.025\n")
        myOutputFile.write("0 0.02\n")
        myOutputFile.write("ITEM: ATOMS id type radius x y z cluster_id cluster_rank cluster_par_num cluster_par_volume cluster_gyration cluster_anisotropy "
                           "Q2 Q4 Q6 Q8 Q10 Q12 major_axis intermediate_axis minor_axis voronoi_area voronoi_volume voronoi_convex_volume\n")
        for i in range(Par_num):
            cluster_id = Par_db_cluster_id[i]
            if cluster_id:
                myOutputFile.write("%d %d %.8f %.8f %.8f %.8f %d %d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n"
                       % (Par_id[i], Par_type[i], Par_radius[i], Par_coord[i, 0], Par_coord[i, 1], Par_coord[i, 2],
                          cluster_id, db_clusters_sort[cluster_id-1],
                          db_clusters_summary[cluster_id-1][0], db_clusters_summary[cluster_id-1][1], db_clusters_summary[cluster_id-1][2],
                          db_clusters_summary[cluster_id-1][3], db_clusters_summary[cluster_id-1][4], db_clusters_summary[cluster_id-1][5],
                          db_clusters_summary[cluster_id-1][6], db_clusters_summary[cluster_id-1][7], db_clusters_summary[cluster_id-1][8],
                          db_clusters_summary[cluster_id-1][9], db_clusters_summary[cluster_id-1][10], db_clusters_summary[cluster_id-1][11],
                          db_clusters_summary[cluster_id-1][12], db_clusters_summary[cluster_id-1][13], db_clusters_summary[cluster_id-1][14],
                          db_clusters_summary[cluster_id-1][15]))
            # else:
            #     myOutputFile.write("%d %d %.8f %.8f %.8f %.8f %d %d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n"
            #                        % (Par_id[i], Par_type[i], Par_radius[i], Par_coord[i, 0], Par_coord[i, 1], Par_coord[i, 2],
            #                           cluster_id, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        myOutputFile.close()

    return db_clusters_summary

# ----------------------------------------------------------------------
# Clustering by different algorithms
#
def particle_clustering(d50, Par_id, Par_coord, Par_radius, Par_volume, Par_D2min, Par_shear_strain, Par_selected,
                        vor_nearest_neighbors, vor, linkage, epsilon, output_path, frame, dump_results, title):

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. Data definition
    #
    Par_num = len(Par_id)
    Par_num_selected = np.sum(Par_selected == True)
    Par_weight = Par_volume
    Par_type = np.ones(Par_num)
    Par_index = np.arange(Par_num)
    Par_index_all = Par_index[Par_selected]                                 # 选中的颗粒，在所有颗粒中的编号
    Par_index_selected = dict(zip(Par_index_all, range(Par_num_selected)))  # 所有颗粒中，选中的部分颗粒，其在这部分中的编号
    Par_index_all = np.array(Par_index_all)

    Par_coord_selected = Par_coord[Par_selected]
    # Par_type_selected = Par_type[Par_selected]
    # Par_id_selected = Par_id[Par_selected]
    # Par_radius_selected = Par_radius[Par_selected]
    # Par_D2min_selected = Par_D2min[Par_selected]
    # Par_shear_strain_selected = Par_shear_strain[Par_selected]
    # Par_weight_selected = Par_weight[Par_selected]
    # Par_volume_selected = Par_volume[Par_selected]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2.1 Creating a graph (constructing graphs node-by-node or edge-by-edge)
    #
    # Assign graph attributes when creating a new graph
    t0 = time.time()
    G = nx.Graph(series=frame)
    row, col, data = [], [], []
    # Add one edge at a time and add edge attributes using add_edge()
    neighbor_list = dict().fromkeys(Par_index_all, [])
    for i in range(Par_num):
        if not Par_selected[i]: continue
        G.add_node(i)

        # Hard connectivity
        # neighbor_list = vor_nearest_neighbors[i]
        # for j in neighbor_list:
        #     # From these remaining particles, filter out all particles that are not in contact with
        #     # any other particles, and all particles that are only in contact with one other particles.
        #     if Par_selected[j]: G.add_edge(i, j)

        # Soft connectivity
        vertices_coord = np.array(vor[i]['vertices'])
        for k, face in enumerate(vor[i]['faces']):
            if face['adjacent_cell'] < 0: continue
            j = face['adjacent_cell']
            if not Par_selected[j]: continue
            neighbor_par_j = vor_nearest_neighbors[j]
            if i not in neighbor_par_j: continue
            vertices_list = face['vertices']
            polygon = vertices_coord[vertices_list]
            face_area = PolygonArea(polygon)
            clearance = vertex_distance(Par_coord[i], Par_coord[j]) - Par_radius[i] - Par_radius[j]
            diameter = 4.0*Par_radius[i]*Par_radius[j]/(Par_radius[i] + Par_radius[j])
            # print("Weight  ", face_area/clearance/diameter, diameter/clearance)
            # if face_area/clearance >= epsilon*diameter:
            if clearance <= epsilon*diameter:
                G.add_edge(i, j)
                row.append(Par_index_selected[i])
                col.append(Par_index_selected[j])
                data.append(1.0)
                neighbor_list[i].append(j)

    # Compressed Sparse Row matrix:
    connectivity = csr_matrix((data, (row, col)), shape=(Par_num_selected, Par_num_selected))

    # Modify graph to prevent further change by adding or removing nodes or edges
    if not nx.is_frozen(G): nx.freeze(G)
    # Print short summary of information for the graph G
    print(nx.info(G))
    print("Connected: ", nx.is_connected(G))  # Return True if the graph is connected, false otherwise
    elapsed_time = time.time() - t0
    print("Elapsed time: %.2fs" % elapsed_time)
    print(60 * '~')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2.2 Cluster identification and geometrical analysis
    #     RDCI: A novel method of cluster analysis and applications thereof in sample molecular simulations
    #     https: // pubs.acs.org.ccindex.cn/doi/full/10.1021/jp2065612?src = recsys
    #     http://www.sbg.bio.ic.ac.uk/maxcluster/
    nx_clusters = []
    nx_clusters_summary = []
    nx_clusters_nonaffinity = []
    nx_par_num_cluster = 0
    Par_nx_cluster_id = np.zeros(Par_num).astype(int)
    for i, sub_graph in enumerate(nx.connected_component_subgraphs(G)):
        cluster = list(sub_graph.nodes())
        cluster_par_num = sub_graph.number_of_nodes()
        if cluster_par_num < 2: continue
        nx_clusters.append(cluster)
        nx_par_num_cluster += cluster_par_num
        cluster_size = np.sqrt(cluster_par_num)
        cluster_id = i + 1

        cluster_xcen = np.sum(Par_coord[cluster, 0]*Par_weight[cluster])/np.sum(Par_weight[cluster])
        cluster_ycen = np.sum(Par_coord[cluster, 1]*Par_weight[cluster])/np.sum(Par_weight[cluster])
        cluster_zcen = np.sum(Par_coord[cluster, 2]*Par_weight[cluster])/np.sum(Par_weight[cluster])
        cluster_center = np.array([cluster_xcen, cluster_ycen, cluster_zcen])
        cluster_par_volume = 0
        cluster_gyration_radius = 0.0
        cluster_gyration_tensor = np.zeros([3, 3])
        cluster_inertia_tensor = np.zeros([3, 3])
        cluster_voronoi_volume = 0
        cluster_voronoi_area = 0
        cluster_voronoi_vertices = []
        for j in cluster:
            Par_nx_cluster_id[j] = cluster_id
            cluster_par_volume += Par_volume[j]
            cluster_gyration_radius += Par_weight[j]*vertex_distance(Par_coord[j], cluster_center)**2.0
            for k in cluster:
                pos_relative = Par_coord[j] - Par_coord[k]
                cluster_gyration_tensor += np.outer(pos_relative, pos_relative)*Par_weight[j]*Par_weight[k]
            pos_relative = Par_coord[j] - cluster_center
            # cluster_gyration_tensor += np.outer(pos_relative, pos_relative)
            cluster_inertia_tensor += Par_weight[j]*(np.sum(pos_relative**2.0)*np.eye(3,3) - np.outer(pos_relative, pos_relative))

            cluster_voronoi_volume += vor[j]['volume']
            vertices_coord = np.array(vor[j]['vertices'])
            cluster_voronoi_vertices.extend(vertices_coord)
            for k, face in enumerate(vor[j]['faces']):
                if (face['adjacent_cell'] in cluster): continue
                vertices_list = face['vertices']
                polygon = vertices_coord[vertices_list]
                cluster_voronoi_area += PolygonArea(polygon)

        cluster_gyration_radius = np.sqrt(cluster_gyration_radius/np.sum(Par_weight[cluster]))
        # cluster_gyration_tensor = cluster_gyration_tensor/cluster_par_num
        cluster_gyration_tensor = cluster_gyration_tensor/(2*np.sum(Par_weight[cluster])**2.0)
        cluster_inertia_tensor = cluster_inertia_tensor/np.sum(Par_weight[cluster])
        if cluster_par_num == 1:
            cluster_shape_anisotropy = 0.0
        else:
            eigenvalues, eigenvectors = np.linalg.eig(cluster_gyration_tensor)
            cluster_eigenvalues = np.abs(eigenvalues)
            cluster_eigenvalues.sort()
            # sphericity
            cluster_b = cluster_eigenvalues[2] - 0.5*(cluster_eigenvalues[0] + cluster_eigenvalues[1])
            # cylindricity
            cluster_c = cluster_eigenvalues[1] - cluster_eigenvalues[0]
            cluster_shape_anisotropy = (cluster_b**2.0 + 3*cluster_c**2.0/4.0)/(cluster_gyration_radius**4.0)

        cluster_voronoi_vertices = np.array(cluster_voronoi_vertices)
        cluster_voronoi_convex_volume, cluster_voronoi_axies = cluster_shape(cluster_voronoi_vertices[:,0], cluster_voronoi_vertices[:,1], cluster_voronoi_vertices[:,2])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # A Python package to compute bond orientational order parameters as defined by Steinhardt Physical Review B (1983) doi:10.1103/PhysRevB.28.784.
        #
        # Steinhardt’s bond orientational order parameter is a popular method (>20k citations of the original paper) to identify local symmetries in an assembly of particles in 3D.
        # It can be used in particle-based simulations (typically molecular dynamics, brownian dynamics, monte-carlo, etc.)
        # or in particle-tracking experiments (colloids, granular materials) where the coordinates of all particles are known.
        # reference:
        # [1] https://pyboo.readthedocs.io/en/latest/intro.html
        # [2] Xia, C. et al. The structural origin of the hard-sphere glass transition in granular packing. Nat. Commun. 6, 1–9 (2015).
        # [3] Teich, E. G., Van Anders, G., Klotsa, D., Dshemuchadse, J. & Glotzer, S. C. Clusters of polyhedra in spherical confinement. Proc. Natl. Acad. Sci. U. S. A. (2016). doi:10.1073/pnas.1524875113
        # [4] Leocmach, M. & Tanaka, H. Roles of icosahedral and crystal-like order in the hard spheres glass transition. Nat. Commun. (2012). doi:10.1038/ncomms1974

        cluster_par_coord = np.stack((Par_coord[cluster, 0], Par_coord[cluster, 1], Par_coord[cluster, 2]), axis=1)
        cluster_par_weight = Par_weight[cluster] # cluster_par_weight = np.ones(len(cluster))
        cluster_boo_Q2 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=2, modified=True)
        cluster_boo_Q4 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=4, modified=True)
        cluster_boo_Q6 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=6, modified=True)
        cluster_boo_Q8 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=8, modified=True)
        cluster_boo_Q10 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=10, modified=True)
        cluster_boo_Q12 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=12, modified=True)

        # define coordinates and bond network
        #
        # cluster_par_coord = np.vstack((cluster_center, cluster_par_coord))
        # cluster_bond = np.array([[0, k+1] for k in range(cluster_par_num)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # tensorial bond orientational order
        #
        # cluster_boo_q4m = boo.bonds2qlm(cluster_par_coord, cluster_bond, l=4)
        # cluster_boo_q6m = boo.bonds2qlm(cluster_par_coord, cluster_bond, l=6)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Rotational invarients
        # The tensorial q?mq?m coefficients are dependent of the orientation of the reference axis.
        # That is why we have to compute quantities that are rotationally invarients.
        # Rotational invarients give more precise information about the type of structures.
        #
        # cluster_boo_Q4 = boo.ql(cluster_boo_q4m[0])   # the second order invarient indicates the strength of the ??-fold symetry.
        # cluster_boo_Q6 = boo.ql(cluster_boo_q6m[0])
        # cluster_boo_W4 = boo.wl(cluster_boo_q4m[0])   # the third order invarient allows to discriminate different types of ?-fold symetric structures
        # cluster_boo_W6 = boo.wl(cluster_boo_q6m[0])

        cluster_nonaffinity = np.mean(Par_D2min[cluster])
        cluster_local_strain = np.mean(Par_shear_strain[cluster])
        nx_clusters_nonaffinity.append(cluster_nonaffinity)
        nx_clusters_summary.append([cluster_par_num, cluster_par_volume, cluster_gyration_radius, cluster_shape_anisotropy,
                                    cluster_boo_Q2, cluster_boo_Q4, cluster_boo_Q6, cluster_boo_Q8, cluster_boo_Q10, cluster_boo_Q12,
                                    cluster_voronoi_axies[0], cluster_voronoi_axies[1], cluster_voronoi_axies[2], cluster_voronoi_area, cluster_voronoi_volume, cluster_voronoi_convex_volume])

    nx_clusters_num = len(nx_clusters_summary)
    nx_clusters_sort = np.zeros(nx_clusters_num)
    for i, index in enumerate(np.argsort(nx_clusters_nonaffinity)):
        nx_clusters_sort[index] = nx_clusters_num - i

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DBSCAN analysis
    db_connected_component = DBSCAN(neighbor_list,3)
    db_clusters = []
    db_clusters_summary = []
    db_clusters_nonaffinity = []
    db_par_num_cluster = 0
    Par_db_cluster_id = np.zeros(Par_num).astype(int)
    
    cluster_id = 0
    for i, sub_graph in enumerate(db_connected_component):
        cluster = sub_graph
        cluster_par_num = len(sub_graph)
        if cluster_par_num < 2: continue
        db_clusters.append(cluster)
        db_par_num_cluster += cluster_par_num
        cluster_size = np.sqrt(cluster_par_num)
        cluster_id += 1

        cluster_xcen = np.sum(Par_coord[cluster, 0]*Par_weight[cluster])/np.sum(Par_weight[cluster])
        cluster_ycen = np.sum(Par_coord[cluster, 1]*Par_weight[cluster])/np.sum(Par_weight[cluster])
        cluster_zcen = np.sum(Par_coord[cluster, 2]*Par_weight[cluster])/np.sum(Par_weight[cluster])
        cluster_center = np.array([cluster_xcen, cluster_ycen, cluster_zcen])
        cluster_par_volume = 0
        cluster_gyration_radius = 0.0
        cluster_gyration_tensor = np.zeros([3, 3])
        cluster_inertia_tensor = np.zeros([3, 3])
        cluster_voronoi_volume = 0
        cluster_voronoi_area = 0
        cluster_voronoi_vertices = []
        for j in cluster:
            Par_db_cluster_id[j] = cluster_id
            cluster_par_volume += Par_volume[j]
            cluster_gyration_radius += Par_weight[j]*vertex_distance(Par_coord[j], cluster_center)**2.0
            for k in cluster:
                pos_relative = Par_coord[j] - Par_coord[k]
                cluster_gyration_tensor += np.outer(pos_relative, pos_relative)*Par_weight[j]*Par_weight[k]
            pos_relative = Par_coord[j] - cluster_center
            # cluster_gyration_tensor += np.outer(pos_relative, pos_relative)
            cluster_inertia_tensor += Par_weight[j]*(np.sum(pos_relative**2.0)*np.eye(3,3) - np.outer(pos_relative, pos_relative))

            cluster_voronoi_volume += vor[j]['volume']
            vertices_coord = np.array(vor[j]['vertices'])
            cluster_voronoi_vertices.extend(vertices_coord)
            for k, face in enumerate(vor[j]['faces']):
                if (face['adjacent_cell'] in cluster): continue
                vertices_list = face['vertices']
                polygon = vertices_coord[vertices_list]
                cluster_voronoi_area += PolygonArea(polygon)

        cluster_gyration_radius = np.sqrt(cluster_gyration_radius/np.sum(Par_weight[cluster]))
        # cluster_gyration_tensor = cluster_gyration_tensor/cluster_par_num
        cluster_gyration_tensor = cluster_gyration_tensor/(2*np.sum(Par_weight[cluster])**2.0)
        cluster_inertia_tensor = cluster_inertia_tensor/np.sum(Par_weight[cluster])
        if cluster_par_num == 1:
            cluster_shape_anisotropy = 0.0
        else:
            eigenvalues, eigenvectors = np.linalg.eig(cluster_gyration_tensor)
            cluster_eigenvalues = np.abs(eigenvalues)
            cluster_eigenvalues.sort()
            # sphericity
            cluster_b = cluster_eigenvalues[2] - 0.5*(cluster_eigenvalues[0] + cluster_eigenvalues[1])
            # cylindricity
            cluster_c = cluster_eigenvalues[1] - cluster_eigenvalues[0]
            cluster_shape_anisotropy = (cluster_b**2.0 + 3*cluster_c**2.0/4.0)/(cluster_gyration_radius**4.0)

        cluster_voronoi_vertices = np.array(cluster_voronoi_vertices)
        cluster_voronoi_convex_volume, cluster_voronoi_axies = cluster_shape(cluster_voronoi_vertices[:,0], cluster_voronoi_vertices[:,1], cluster_voronoi_vertices[:,2])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # A Python package to compute bond orientational order parameters as defined by Steinhardt Physical Review B (1983) doi:10.1103/PhysRevB.28.784.
        #
        # Steinhardt’s bond orientational order parameter is a popular method (>20k citations of the original paper) to identify local symmetries in an assembly of particles in 3D.
        # It can be used in particle-based simulations (typically molecular dynamics, brownian dynamics, monte-carlo, etc.)
        # or in particle-tracking experiments (colloids, granular materials) where the coordinates of all particles are known.
        # reference:
        # [1] https://pyboo.readthedocs.io/en/latest/intro.html
        # [2] Xia, C. et al. The structural origin of the hard-sphere glass transition in granular packing. Nat. Commun. 6, 1–9 (2015).
        # [3] Teich, E. G., Van Anders, G., Klotsa, D., Dshemuchadse, J. & Glotzer, S. C. Clusters of polyhedra in spherical confinement. Proc. Natl. Acad. Sci. U. S. A. (2016). doi:10.1073/pnas.1524875113
        # [4] Leocmach, M. & Tanaka, H. Roles of icosahedral and crystal-like order in the hard spheres glass transition. Nat. Commun. (2012). doi:10.1038/ncomms1974

        cluster_par_coord = np.stack((Par_coord[cluster, 0], Par_coord[cluster, 1], Par_coord[cluster, 2]), axis=1)
        cluster_par_weight = Par_weight[cluster] # cluster_par_weight = np.ones(len(cluster))
        cluster_boo_Q2 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=2, modified=True)
        cluster_boo_Q4 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=4, modified=True)
        cluster_boo_Q6 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=6, modified=True)
        cluster_boo_Q8 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=8, modified=True)
        cluster_boo_Q10 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=10, modified=True)
        cluster_boo_Q12 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=12, modified=True)

        # define coordinates and bond network
        #
        # cluster_par_coord = np.vstack((cluster_center, cluster_par_coord))
        # cluster_bond = np.array([[0, k+1] for k in range(cluster_par_num)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # tensorial bond orientational order
        #
        # cluster_boo_q4m = boo.bonds2qlm(cluster_par_coord, cluster_bond, l=4)
        # cluster_boo_q6m = boo.bonds2qlm(cluster_par_coord, cluster_bond, l=6)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Rotational invarients
        # The tensorial q?mq?m coefficients are dependent of the orientation of the reference axis.
        # That is why we have to compute quantities that are rotationally invarients.
        # Rotational invarients give more precise information about the type of structures.
        #
        # cluster_boo_Q4 = boo.ql(cluster_boo_q4m[0])   # the second order invarient indicates the strength of the ??-fold symetry.
        # cluster_boo_Q6 = boo.ql(cluster_boo_q6m[0])
        # cluster_boo_W4 = boo.wl(cluster_boo_q4m[0])   # the third order invarient allows to discriminate different types of ?-fold symetric structures
        # cluster_boo_W6 = boo.wl(cluster_boo_q6m[0])

        cluster_nonaffinity = np.mean(Par_D2min[cluster])
        cluster_local_strain = np.mean(Par_shear_strain[cluster])
        db_clusters_nonaffinity.append(cluster_nonaffinity)
        db_clusters_summary.append([cluster_par_num, cluster_par_volume, cluster_gyration_radius, cluster_shape_anisotropy,
                                   cluster_boo_Q2, cluster_boo_Q4, cluster_boo_Q6, cluster_boo_Q8, cluster_boo_Q10, cluster_boo_Q12,
                                   cluster_voronoi_volume, cluster_voronoi_area, cluster_voronoi_convex_volume, cluster_voronoi_axies[0], cluster_voronoi_axies[1], cluster_voronoi_axies[2]])

    db_clusters_num = len(db_clusters_summary)
    db_clusters_sort = np.zeros(db_clusters_num)
    for i, index in enumerate(np.argsort(nx_clusters_nonaffinity)):
        db_clusters_sort[index] = db_clusters_num - i

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3. Clustering of unlabeled data
    #    references:
    #    [1] https://scikit-learn.org/stable/modules/clustering.html#
    #    [2] https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
    #    [3] http://sklearn.lzjqsdd.com/auto_examples/cluster/plot_cluster_comparison.html

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3.1 Define the structure of the dataset
    #     Create a sparse matrix capturing local connectivity. Larger number of neighbors will give more homogeneous clusters to the cost of computation time.
    #     A very large number of neighbors gives more evenly distributed cluster sizes, but may not impose the local manifold structure of the data
    #
    # Compressed Sparse Row matrix:
    # row, col, data = [], [], []
    # for i, neighbor_list in enumerate(vor_nearest_neighbors):
    #     if not Par_selected[i]: continue
    #     for j in neighbor_list:
    #         if not Par_selected[j]: continue
    #         row.append(Par_index_selected[i])
    #         col.append(Par_index_selected[j])
    #         data.append(1.0)
    # connectivity = csr_matrix((data, (row, col)), shape=(Par_num_selected, Par_num_selected))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  3.2 n*m array of observations on m attributes across n samples.
    #
    # Par_attr1 = np.log10(Par_D2min_selected)
    # Par_attr2 = np.log10(Par_shear_strain_selected)
    # Par_attr1 = array_normalization(Par_D2min_selected)
    # Par_attr2 = array_normalization(Par_shear_strain_selected)
    # Par_attributes = np.stack((Par_attr1, Par_attr2), axis=1)
    # scaler = RobustScaler(quantile_range=(10, 90))
    # Par_attributes = scaler.fit_transform(np.stack((Par_D2min_selected, Par_shear_strain_selected), axis=1))
    Par_attributes = Par_coord_selected

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3.3 Clustering with connectivity constraints
    #     The hierarchical clustering is restricted to the nearest neighbors graph: it’s a hierarchical clustering with structure prior.
    #
    # linkage criteria: 'average', 'complete', 'ward', 'single'
    # Comparing different hierarchical linkage methods on toy datasets
    # The main observations to make are:
    # single linkage is fast, and can perform well on non-globular data, but it performs poorly in the presence of noise.
    # average and complete linkage perform well on cleanly separated globular clusters, but have mixed results otherwise.
    # Ward is the most effective method for noisy data.
    print("Compute hierarchical clustering...")
    skl_clusters_num = nx_clusters_num
    t0 = time.time()
    skl_clustering = AgglomerativeClustering(linkage=linkage, connectivity=connectivity, n_clusters=skl_clusters_num).fit(Par_attributes)
    elapsed_time = time.time() - t0

    # Hierarchical clustering summary
    print("linkage criteria:                            %s" % linkage)
    print("Number of regions:                           %d" % skl_clusters_num)
    print("Number of leaves in the hierarchical tree:   %d" % skl_clustering.n_leaves_)
    print("Number of connected components in the graph: %d" % skl_clustering.n_connected_components_)
    print("Elapsed time:                                %.2fs" % elapsed_time)
    print(60 * '~')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3.4 statistics over the hierarchical clusters
    #     references:
    #     [1] https://en.wikipedia.org/wiki/Radius_of_gyration
    skl_clusters_nonaffinity = np.zeros(skl_clusters_num)
    skl_clusters_summary = [[] for i in range(skl_clusters_num)]
    skl_clusters = [[] for i in range(skl_clusters_num)]
    skl_labels = skl_clustering.labels_
    skl_par_num_cluster = 0
    Par_skl_cluster_id = np.zeros(Par_num).astype(int)
    for i in range(Par_num_selected):
        # cluster_id = skl_labels[i]
        skl_clusters[skl_labels[i]].append(Par_index_all[i])

    for i, cluster in enumerate(skl_clusters):
        cluster_par_num = len(cluster)
        skl_par_num_cluster += cluster_par_num
        cluster_size = np.sqrt(cluster_par_num)
        cluster_id = i + 1

        cluster_xcen = np.sum(Par_coord[cluster, 0]*Par_weight[cluster])/np.sum(Par_weight[cluster])
        cluster_ycen = np.sum(Par_coord[cluster, 1]*Par_weight[cluster])/np.sum(Par_weight[cluster])
        cluster_zcen = np.sum(Par_coord[cluster, 2]*Par_weight[cluster])/np.sum(Par_weight[cluster])
        cluster_center = np.array([cluster_xcen, cluster_ycen, cluster_zcen])
        cluster_par_volume = 0
        cluster_gyration_radius = 0.0
        cluster_gyration_tensor = np.zeros([3, 3])
        cluster_inertia_tensor = np.zeros([3, 3])
        cluster_voronoi_volume = 0
        cluster_voronoi_area = 0
        cluster_voronoi_vertices = []
        for j in cluster:
            Par_skl_cluster_id[j] = cluster_id
            cluster_par_volume += Par_volume[j]
            cluster_gyration_radius += Par_weight[j]*vertex_distance(Par_coord[j], cluster_center)**2.0
            for k in cluster:
                pos_relative = Par_coord[j] - Par_coord[k]
                cluster_gyration_tensor += np.outer(pos_relative, pos_relative)*Par_weight[j]*Par_weight[k]
            pos_relative = Par_coord[j] - cluster_center
            # cluster_gyration_tensor += np.outer(pos_relative, pos_relative)
            cluster_inertia_tensor += Par_weight[j]*(np.sum(pos_relative**2.0)*np.eye(3,3) - np.outer(pos_relative, pos_relative))

            cluster_voronoi_volume += vor[j]['volume']
            vertices_coord = np.array(vor[j]['vertices'])
            cluster_voronoi_vertices.extend(vertices_coord)
            for k, face in enumerate(vor[j]['faces']):
                if (face['adjacent_cell'] in cluster): continue
                vertices_list = face['vertices']
                polygon = vertices_coord[vertices_list]
                cluster_voronoi_area += PolygonArea(polygon)

        cluster_gyration_radius = np.sqrt(cluster_gyration_radius/np.sum(Par_weight[cluster]))
        # cluster_gyration_tensor = cluster_gyration_tensor/cluster_par_num
        cluster_gyration_tensor = cluster_gyration_tensor/(2*np.sum(Par_weight[cluster])**2.0)
        cluster_inertia_tensor = cluster_inertia_tensor/np.sum(Par_weight[cluster])
        if cluster_par_num == 1:
            cluster_shape_anisotropy = 0.0
        else:
            eigenvalues, eigenvectors = np.linalg.eig(cluster_gyration_tensor)
            cluster_eigenvalues = np.abs(eigenvalues)
            cluster_eigenvalues.sort()
            # sphericity
            cluster_b = cluster_eigenvalues[2] - 0.5*(cluster_eigenvalues[0] + cluster_eigenvalues[1])
            # cylindricity
            cluster_c = cluster_eigenvalues[1] - cluster_eigenvalues[0]
            cluster_shape_anisotropy = (cluster_b**2.0 + 3*cluster_c**2.0/4.0)/(cluster_gyration_radius**4.0)

        cluster_voronoi_vertices = np.array(cluster_voronoi_vertices)
        cluster_voronoi_convex_volume, cluster_voronoi_axies = cluster_shape(cluster_voronoi_vertices[:,0], cluster_voronoi_vertices[:,1], cluster_voronoi_vertices[:,2])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # A Python package to compute bond orientational order parameters as defined by Steinhardt Physical Review B (1983) doi:10.1103/PhysRevB.28.784.
        #
        # Steinhardt’s bond orientational order parameter is a popular method (>20k citations of the original paper) to identify local symmetries in an assembly of particles in 3D.
        # It can be used in particle-based simulations (typically molecular dynamics, brownian dynamics, monte-carlo, etc.)
        # or in particle-tracking experiments (colloids, granular materials) where the coordinates of all particles are known.
        # reference:
        # [1] https://pyboo.readthedocs.io/en/latest/intro.html
        # [2] Xia, C. et al. The structural origin of the hard-sphere glass transition in granular packing. Nat. Commun. 6, 1–9 (2015).
        # [3] Teich, E. G., Van Anders, G., Klotsa, D., Dshemuchadse, J. & Glotzer, S. C. Clusters of polyhedra in spherical confinement. Proc. Natl. Acad. Sci. U. S. A. (2016). doi:10.1073/pnas.1524875113
        # [4] Leocmach, M. & Tanaka, H. Roles of icosahedral and crystal-like order in the hard spheres glass transition. Nat. Commun. (2012). doi:10.1038/ncomms1974

        cluster_par_coord = np.stack((Par_coord[cluster, 0], Par_coord[cluster, 1], Par_coord[cluster, 2]), axis=1)
        cluster_par_weight = Par_weight[cluster] # cluster_par_weight = np.ones(len(cluster))
        cluster_boo_Q2 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=2, modified=True)
        cluster_boo_Q4 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=4, modified=True)
        cluster_boo_Q6 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=6, modified=True)
        cluster_boo_Q8 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=8, modified=True)
        cluster_boo_Q10 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=10, modified=True)
        cluster_boo_Q12 = boo_ql(cluster_center, cluster_par_coord, cluster_par_weight, l=12, modified=True)

        # define coordinates and bond network
        #
        # cluster_par_coord = np.vstack((cluster_center, cluster_par_coord))
        # cluster_bond = np.array([[0, k+1] for k in range(cluster_par_num)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # tensorial bond orientational order
        #
        # cluster_boo_q4m = boo.bonds2qlm(cluster_par_coord, cluster_bond, l=4)
        # cluster_boo_q6m = boo.bonds2qlm(cluster_par_coord, cluster_bond, l=6)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Rotational invarients
        # The tensorial q?mq?m coefficients are dependent of the orientation of the reference axis.
        # That is why we have to compute quantities that are rotationally invarients.
        # Rotational invarients give more precise information about the type of structures.
        #
        # cluster_boo_Q4 = boo.ql(cluster_boo_q4m[0])   # the second order invarient indicates the strength of the ??-fold symetry.
        # cluster_boo_Q6 = boo.ql(cluster_boo_q6m[0])
        # cluster_boo_W4 = boo.wl(cluster_boo_q4m[0])   # the third order invarient allows to discriminate different types of ?-fold symetric structures
        # cluster_boo_W6 = boo.wl(cluster_boo_q6m[0])

        cluster_nonaffinity = np.mean(Par_D2min[cluster])
        cluster_local_strain = np.mean(Par_shear_strain[cluster])
        skl_clusters_nonaffinity[i] = cluster_nonaffinity
        skl_clusters_summary[i] = [cluster_par_num, cluster_par_volume, cluster_gyration_radius, cluster_shape_anisotropy,
                                   cluster_boo_Q2, cluster_boo_Q4, cluster_boo_Q6, cluster_boo_Q8, cluster_boo_Q10, cluster_boo_Q12,
                                   cluster_voronoi_axies[0], cluster_voronoi_axies[1], cluster_voronoi_axies[2], cluster_voronoi_area, cluster_voronoi_volume, cluster_voronoi_convex_volume]

    skl_clusters_sort = np.zeros(skl_clusters_num)
    for i, index in enumerate(np.argsort(skl_clusters_nonaffinity)):
        skl_clusters_sort[index] = skl_clusters_num - i

    # ----------------------------------------------------------------------
    # 3.5 Visualize the clustering
    #
    # 2D Plot result
    # plot_clustering_2d(Par_attributes, skl_labels, "structured hierarchical clustering", output_path, frame)

    # ------------------------------------------------------------------
    # fig_title = title + " clusters by structured hierarchical clustering"
    # plot_clustering_3d(Par_coord_selected, skl_labels, fig_title , output_path, frame)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. Output the geometry of cluster
    #
    if dump_results:
        myOutputFile = open(output_path + '/' + title + '-clusters by contiguity constrained-' + str(frame) + '.dump', 'w')
        myOutputFile.write("ITEM: TIMESTEP\n")
        myOutputFile.write("%-d\n" % frame)
        myOutputFile.write("ITEM: NUMBER OF ATOMS\n")
        myOutputFile.write("%-d\n" % Par_num)
        myOutputFile.write("ITEM: BOX BOUNDS pp ff pp\n")
        myOutputFile.write("0 0.04\n")
        myOutputFile.write("-0.005 0.025\n")
        myOutputFile.write("0 0.02\n")
        myOutputFile.write("ITEM: ATOMS id type radius x y z cluster_id cluster_rank cluster_par_num cluster_par_volume cluster_gyration cluster_anisotropy "
                           "Q2 Q4 Q6 Q8 Q10 Q12 major_axis intermediate_axis minor_axis voronoi_area voronoi_volume voronoi_convex_volume\n")
        for i in range(Par_num):
            cluster_id = Par_nx_cluster_id[i]
            if cluster_id:
                myOutputFile.write("%d %d %.8f %.8f %.8f %.8f %d %d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n"
                                   % (Par_id[i], Par_type[i], Par_radius[i], Par_coord[i, 0], Par_coord[i, 1], Par_coord[i, 2],
                                      cluster_id, nx_clusters_sort[cluster_id-1],
                                      nx_clusters_summary[cluster_id-1][0], nx_clusters_summary[cluster_id-1][1], nx_clusters_summary[cluster_id-1][2],
                                      nx_clusters_summary[cluster_id-1][3], nx_clusters_summary[cluster_id-1][4], nx_clusters_summary[cluster_id-1][5],
                                      nx_clusters_summary[cluster_id-1][6], nx_clusters_summary[cluster_id-1][7], nx_clusters_summary[cluster_id-1][8],
                                      nx_clusters_summary[cluster_id-1][9], nx_clusters_summary[cluster_id-1][10], nx_clusters_summary[cluster_id-1][11],
                                      nx_clusters_summary[cluster_id-1][12], nx_clusters_summary[cluster_id-1][13], nx_clusters_summary[cluster_id-1][14],
                                      nx_clusters_summary[cluster_id-1][15]))
            else:
                myOutputFile.write("%d %d %.8f %.8f %.8f %.8f %d %d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n"
                                   % (Par_id[i], Par_type[i], Par_radius[i], Par_coord[i, 0], Par_coord[i, 1], Par_coord[i, 2],
                                      cluster_id, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        myOutputFile.close()

        myOutputFile = open(output_path + '/' + title + '-clusters by DBSCAN-' + str(frame) + '.dump', 'w')
        myOutputFile.write("ITEM: TIMESTEP\n")
        myOutputFile.write("%-d\n" % frame)
        myOutputFile.write("ITEM: NUMBER OF ATOMS\n")
        myOutputFile.write("%-d\n" % Par_num)
        myOutputFile.write("ITEM: BOX BOUNDS pp ff pp\n")
        myOutputFile.write("0 0.04\n")
        myOutputFile.write("-0.005 0.025\n")
        myOutputFile.write("0 0.02\n")
        myOutputFile.write("ITEM: ATOMS id type radius x y z cluster_id cluster_rank cluster_par_num cluster_par_volume cluster_gyration cluster_anisotropy "
                           "Q2 Q4 Q6 Q8 Q10 Q12 major_axis intermediate_axis minor_axis voronoi_area voronoi_volume voronoi_convex_volume\n")
        for i in range(Par_num):
            cluster_id = Par_db_cluster_id[i]
            if cluster_id:
                myOutputFile.write("%d %d %.8f %.8f %.8f %.8f %d %d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n"
                                   % (Par_id[i], Par_type[i], Par_radius[i], Par_coord[i, 0], Par_coord[i, 1], Par_coord[i, 2],
                                      cluster_id, db_clusters_sort[cluster_id-1],
                                      db_clusters_summary[cluster_id-1][0], db_clusters_summary[cluster_id-1][1], db_clusters_summary[cluster_id-1][2],
                                      db_clusters_summary[cluster_id-1][3], db_clusters_summary[cluster_id-1][4], db_clusters_summary[cluster_id-1][5],
                                      db_clusters_summary[cluster_id-1][6], db_clusters_summary[cluster_id-1][7], db_clusters_summary[cluster_id-1][8],
                                      db_clusters_summary[cluster_id-1][9], db_clusters_summary[cluster_id-1][10], db_clusters_summary[cluster_id-1][11],
                                      db_clusters_summary[cluster_id-1][12], db_clusters_summary[cluster_id-1][13], db_clusters_summary[cluster_id-1][14],
                                      db_clusters_summary[cluster_id-1][15]))
            else:
                myOutputFile.write("%d %d %.8f %.8f %.8f %.8f %d %d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n"
                                   % (Par_id[i], Par_type[i], Par_radius[i], Par_coord[i, 0], Par_coord[i, 1], Par_coord[i, 2],
                                      cluster_id, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        myOutputFile.close()

        myOutputFile = open(output_path + '/' + title + '-clusters by hierarchical clustering-' + str(frame) + '.dump', 'w')
        myOutputFile.write("ITEM: TIMESTEP\n")
        myOutputFile.write("%-d\n" % frame)
        myOutputFile.write("ITEM: NUMBER OF ATOMS\n")
        myOutputFile.write("%-d\n" % Par_num)
        myOutputFile.write("ITEM: BOX BOUNDS pp ff pp\n")
        myOutputFile.write("0 0.04\n")
        myOutputFile.write("-0.005 0.025\n")
        myOutputFile.write("0 0.02\n")
        myOutputFile.write("ITEM: ATOMS id type radius x y z cluster_id cluster_rank cluster_par_num cluster_par_volume cluster_gyration cluster_anisotropy cluster_nonaffinity cluster_strain "
                           "voronoi_volume voronoi_area voronoi_convex_volume major_axis intermediate_axis minor_axis Q2 Q4 Q6 Q8 Q10 Q12\n")
        for i in range(Par_num):
            cluster_id = Par_skl_cluster_id[i]
            if cluster_id:
                myOutputFile.write("%d %d %.8f %.8f %.8f %.8f %d %d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n"
                                   % (Par_id[i], Par_type[i], Par_radius[i], Par_coord[i, 0], Par_coord[i, 1], Par_coord[i, 2],
                                      cluster_id, skl_clusters_sort[cluster_id-1],
                                      skl_clusters_summary[cluster_id-1][0], skl_clusters_summary[cluster_id-1][1], skl_clusters_summary[cluster_id-1][2],
                                      skl_clusters_summary[cluster_id-1][3], skl_clusters_summary[cluster_id-1][4], skl_clusters_summary[cluster_id-1][5],
                                      skl_clusters_summary[cluster_id-1][6], skl_clusters_summary[cluster_id-1][7], skl_clusters_summary[cluster_id-1][8],
                                      skl_clusters_summary[cluster_id-1][9], skl_clusters_summary[cluster_id-1][10], skl_clusters_summary[cluster_id-1][11],
                                      skl_clusters_summary[cluster_id-1][12], skl_clusters_summary[cluster_id-1][13], skl_clusters_summary[cluster_id-1][14],
                                      skl_clusters_summary[cluster_id-1][15]))
            else:
                myOutputFile.write("%d %d %.8f %.8f %.8f %.8f %d %d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n"
                                   % (Par_id[i], Par_type[i], Par_radius[i], Par_coord[i, 0], Par_coord[i, 1], Par_coord[i, 2],
                                      cluster_id, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        myOutputFile.close()

    # skl_clusters_summary = nx_clusters_summary
    return nx_clusters_summary, db_clusters_summary, skl_clusters_summary


def DefineClusters(case, test_id, d50, selected_ratio, epsilon, linkage, shear_strain, shear_rate, time_step,
                   steady_strain, dump_results, scenario, strain_window):

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

    start_frame = min(dump_frame)
    end_frame = max(dump_frame)
    shear_strain = (end_frame - start_frame)*time_step*shear_rate
    steady_frame = int(start_frame + (end_frame - start_frame)*steady_strain/shear_strain)
    frame_interval = (end_frame - start_frame)/scenario
    frame_list = np.arange(start_frame, end_frame, frame_interval)
    frame_list = np.append(frame_list, end_frame)
    frame_list = frame_list.astype(int)

    time_window = strain_window/shear_rate
    frame_window = int(time_window/time_step)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # array initialization
    #
    frame_percolation = []
    frame_stz_clusters = {'nx': [], 'nx_trimmed': [], 'db': []}
    frame_nonaffine_clusters = {'nx': [], 'nx_trimmed': [], 'db': []}
    frame_random_clusters = {'nx': [], 'nx_trimmed': [], 'db': []}

    # frame_stz_nx_clusters = []
    # frame_stz_nx_trimmed_clusters = []
    # frame_stz_db_clusters = []
    # frame_stz_skl_clusters  = []

    # frame_nonaffine_nx_clusters = []
    # frame_nonaffine_nx_trimmed_clusters = []
    # frame_nonaffine_db_clusters = []
    # frame_nonaffine_skl_clusters = []

    # frame_random_nx_clusters = []
    # frame_random_nx_trimmed_clusters = []
    # frame_random_db_clusters = []
    # frame_random_skl_clusters = []

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 循环开始，提取每一步数据
    #
    output_path = file_path + '/test-' + str(test_id) + '/strain window-' + str(round(strain_window, 3)) +'/cluster' + '-' + str(selected_ratio)
    data_path = file_path + '/test-' + str(test_id) + '/strain window-' + str(round(strain_window, 3)) + '/particle dynamics'
    mkdir(output_path)
    mkdir(data_path)

    frame_num = 0
    for idx, frame in enumerate(frame_list):
        if idx == 0 or frame <= steady_frame: continue
        frame_num += 1

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1.1 particle D2min and particle temperature (by Falk & Langer, it measures the deviation from affine motion and symmetry)
        #
        particle_info = open(data_path + '/Particle dynamics-' + str(frame) + '.dump', 'r')
        alllines = particle_info.readlines()
        lines = alllines[9:]
        particle_info.close()
        for i in range(len(lines)):
            if (lines[i] == '\n'): del lines[i]
        Par_id = np.array([int(line.strip().split(' ')[0]) for line in lines])             # 字段以逗号分隔，这里取得是第1列
        Par_type = np.array([int(line.strip().split(' ')[1]) for line in lines])           # 字段以逗号分隔，这里取得是第2列
        Par_radius = np.array([float(line.strip().split(' ')[2]) for line in lines])       # 字段以逗号分隔，这里取得是第3列
        Par_xcor = np.array([float(line.strip().split(' ')[3]) for line in lines])         # 字段以逗号分隔，这里取得是第4列
        Par_ycor = np.array([float(line.strip().split(' ')[4]) for line in lines])         # 字段以逗号分隔，这里取得是第5列
        Par_zcor = np.array([float(line.strip().split(' ')[5]) for line in lines])         # 字段以逗号分隔，这里取得是第6列
        Par_um = np.array([float(line.strip().split(' ')[9]) for line in lines])           # 字段以逗号分隔，这里取得是第10列
        Par_nonaffine_um = np.array([float(line.strip().split(' ')[13]) for line in lines]) # 字段以逗号分隔，这里取得是第14列
        Par_temperature = np.array([float(line.strip().split(' ')[14]) for line in lines])  # 字段以逗号分隔，这里取得是第15列
        Par_D2min = np.array([float(line.strip().split(' ')[15]) for line in lines])        # 字段以逗号分隔，这里取得是第16列
        Par_volumetric_strain = np.array([float(line.strip().split(' ')[16]) for line in lines])  # 字段以逗号分隔，这里取得是第17列
        Par_shear_strain = np.array([float(line.strip().split(' ')[17]) for line in lines])  # 字段以逗号分隔，这里取得是第18列

        Par_id = Par_id[Par_type == 1]
        Par_radius = Par_radius[Par_type == 1]
        Par_xcor = Par_xcor[Par_type == 1]
        Par_ycor = Par_ycor[Par_type == 1]
        Par_zcor = Par_zcor[Par_type == 1]
        Par_D2min = Par_D2min[Par_type == 1]
        Par_shear_strain = Par_shear_strain[Par_type == 1]
        Par_volumetric_strain = Par_volumetric_strain[Par_type == 1]
        Par_nonaffine_um = Par_nonaffine_um[Par_type == 1]
        Par_temperature = Par_temperature[Par_type == 1]
        Par_diameter = 2.0*Par_radius
        Par_volume = 4./3.0*np.pi*Par_radius**3.0
        Par_num = np.size(Par_id)

        Par_coord = np.stack((Par_xcor, Par_ycor, Par_zcor), axis=1)
        # Par_inside_region = check_inside_region(Par_coord, boundary_gap=[0*d50, 0*d50, 0*d50])
        Par_inside_region = np.ones(Par_num).astype(bool)
        Par_id_region = Par_id[Par_inside_region]
        # Par_id = Par_id[Par_inside_region]
        # Par_radius = Par_radius[Par_inside_region]
        # Par_xcor = Par_xcor[Par_inside_region]
        # Par_ycor = Par_ycor[Par_inside_region]
        # Par_zcor = Par_zcor[Par_inside_region]
        # Par_D2min = Par_D2min[Par_inside_region]
        # Par_volumetric_strain = Par_volumetric_strain[Par_inside_region]
        # Par_shear_strain = Par_shear_strain[Par_inside_region]
        # Par_temperature = Par_temperature[Par_inside_region]

        print(60*'*')
        print('Start frame:   %d' %(frame_list[idx] - frame_window))
        print('End frame:     %d' %frame_list[idx])
        print('Frame interval:%d' %frame_interval)
        print('Time window:   %.6f' %time_window)
        print('Strain window: %.6f' %strain_window)
        print(60*'*')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1.3 Knee-point detection of histogram of particle nonaffinity
        #     references:
        #     [1] https://realpython.com/python-histograms/
        #     [2] https://github.com/arvkevi/kneed

        # max_D2min, min_D2min = Par_D2min.max(), Par_D2min.min()
        # max_D2min = np.percentile(Par_D2min, 98)
        # Par_D2min_nooutlier = Par_D2min[Par_D2min < max_D2min]
        #
        # num_bins = 1000
        # first_edge, last_edge = min_D2min, max_D2min
        # bins = np.linspace(start=first_edge, stop=last_edge, num=num_bins + 1, endpoint = True)
        # hist, bin_edges = np.histogram(Par_D2min_nooutlier, bins=bins, density=True)
        # plot_hist(Par_D2min_nooutlier, output_path, frame)
        #
        # bin_midpoint = []
        # for i in range(num_bins):
        #     bin_midpoint.append(0.5*(bin_edges[i] + bin_edges[i+1]))
        # kneedle = KneeLocator(bin_midpoint[10:], hist[10:], curve='concave', direction='decreasing')
        # D2min_threshold = kneedle.knee

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2. Spatial Weights based on the current configuration
        #    We construct the spatial weights matrix, which is used to define the nearest neighbors of each particle.
        #
        # Par_coord = np.stack((Par_xcor, Par_ycor, Par_zcor), axis=1)
        # vor_nearest_neighbors = pysal.weights.DistanceBand(Par_coord, threshold=1.4*d50, binary=True)
        # vor_nearest_neighbors = [[] for i in range(Par_num)]
        # vor = Voronoi(Par_coord, furthest_site=False, incremental=False)
        # for ridge in vor.ridge_points:
        #     vor_nearest_neighbors[ridge[0]].append(ridge[1])
        #     vor_nearest_neighbors[ridge[1]].append(ridge[0])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2.1 Call the voro++ library
        #
        Par_coord = np.stack((Par_xcor, Par_ycor, Par_zcor), axis=1)
        domain_xmin, domain_xmax = np.min(Par_coord[:, 0] - Par_radius), np.max(Par_coord[:, 0] + Par_radius)
        domain_ymin, domain_ymax = np.min(Par_coord[:, 1] - Par_radius), np.max(Par_coord[:, 1] + Par_radius)
        domain_zmin, domain_zmax = np.min(Par_coord[:, 2] - Par_radius), np.max(Par_coord[:, 2] + Par_radius)
        # domain_xmin, domain_xmax = 0.0, 0.04
        # domain_ymin, domain_ymax = np.min(Par_coord[:, 1] - Par_radius), np.max(Par_coord[:, 1] + Par_radius)
        # domain_zmin, domain_zmax = 0.0, 0.02
        container = [[domain_xmin, domain_xmax], [domain_ymin, domain_ymax], [domain_zmin, domain_zmax]]
        vor = pyvoro.compute_voronoi(Par_coord, limits=container, dispersion=4*d50, radii=Par_radius, periodic=[False, False, False])

        # t0 = time.time()
        # vor_nearest_neighbors = [[] for i in range(Par_num)]
        # for par_i, cell_i in enumerate(vor):
        #     faces_i = cell_i['faces']
        #     neighbor_par_i = []
        #     for k in range(len(faces_i)):
        #         if faces_i[k]['adjacent_cell'] >= 0:
        #             par_j = faces_i[k]['adjacent_cell']
        #             cell_j = vor[par_j]
        #             faces_j = cell_j['faces']
        #             neighbor_par_j = []
        #             for l in range(len(faces_j)):
        #                 if faces_j[l]['adjacent_cell'] >= 0: neighbor_par_j.append(faces_j[l]['adjacent_cell'])
        #             if par_i in neighbor_par_j: neighbor_par_i.append(par_j)
        #     vor_nearest_neighbors[par_i] = neighbor_par_i
        # elapsed_time = time.time() - t0
        # print("Elapsed time: %.2fs" % elapsed_time)

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

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3. cluster: particle number, cluster size, gyration radius, cluster shape
        #    references:
        #    [1] Kou, B., Cao, Y., Li, J., Xia, C., Li, Z., Dong, H., … Wang, Y. (2018). Translational and rotational dynamical heterogeneities in granular systems. Physical Review Letters, 121(1), 1–9.
        #    [2] Xia, C., Li, J., Cao, Y., Kou, B., Xiao, X., Fezzaa, K., … Wang, Y. (2015). The structural origin of the hard-sphere glass transition in granular packing. Nature Communications, 6, 1–9.
        #    [3] Panaitescu, A., Reddy, K. A., & Kudrolli, A. (2012). Nucleation and crystal growth in sheared granular sphere packings. Physical Review Letters, 108(10), 1–5.
        #    [4] Cubuk, E. D., Ivancic, R. J. S., Schoenholz, S. S., Strickland, D. J., Basu, A., Davidson, Z. S., … Liu, A. J. (2017). Structure-property relationships from universal signatures of plasticity in disordered solids. Science, 358(6366), 1033–1037.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3.1 Unsupervised learning for particle classification.
        #
        # Data preparation
        # https://sklearn.org/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
        # http://www.dataivy.cn/blog/3-9-标准化，让运营数据落入相同的范围/

        start, end = 0.0, 1.0
        iteration = 0
        early_stopping = False
        while True:
            nonaffine_rank = (start + end)/2.0
            D2min_threshold = np.percentile(Par_D2min, 100*(1 - nonaffine_rank))
            Temp_threshold = np.percentile(Par_temperature, 100*(1 - nonaffine_rank))
            Par_selected = (Par_D2min > D2min_threshold)*(Par_temperature > Temp_threshold)
            Par_D2min_high = Par_D2min[Par_selected]
            Par_shear_strain_high = Par_shear_strain[Par_selected]
            Par_nonaffine_um_high = Par_nonaffine_um[Par_selected]
            Par_temperature_high = Par_temperature[Par_selected]
            Par_id_high = Par_id[Par_selected]

            Par_attributes = np.stack((Par_D2min_high, Par_temperature_high), axis=1)
            # Par_attributes_scaled = preprocessing.scale(Par_attributes)  # Center to the mean and component wise scale to unit variance.
            # StandardScaler, MinMaxScaler is very sensitive to the presence of outliers.
            # Par_attributes_scaled = preprocessing.MinMaxScaler().fit_transform(Par_attributes)
            # Par_attributes_scaled = preprocessing.StandardScaler().fit_transform(Par_attributes)
            Par_attributes_scaled = preprocessing.RobustScaler(quantile_range=(0, 99)).fit_transform(Par_attributes)
            # Par_attributes_scaled = preprocessing.QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal').fit_transform(Par_attributes)
            # Par_attributes_normalized = preprocessing.normalize(Par_attributes_scaled, norm='l1')

            # Hierarchical clustering, K-means, DBSCAN
            # clusters = DBSCAN(eps=0.01, min_samples=min_samples).fit(Par_attributes_scaled)
            # clusters = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete').fit(Par_attributes_scaled)
            # clusters = KMeans(n_clusters=2, random_state=0).fit(Par_attributes_scaled)
            # clusters = SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity="nearest_neighbors").fit(Par_attributes_scaled)
            # clusters = MiniBatchKMeans(n_clusters=2).fit(Par_attributes_scaled)
            # labels = clusters.labels_

            # Gaussian Mixture Model
            # https://blog.csdn.net/csdn_inside/article/details/85267341
            # https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html#sphx-glr-auto-examples-mixture-plot-gmm-py GaussianMixture BayesianGaussianMixture
            # https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html
            # https://www.zhihu.com/question/33467075
            # covariance_type : {‘full’, ‘tied’, ‘diag’, ‘spherical’}, defaults to ‘full’s
            # full 指每个分量有各自不同的标准协方差矩阵，完全协方差矩阵（元素都不为零）
            # tied 指所有分量有相同的标准协方差矩阵（HMM 会用到,不好用，不用考虑）
            # diag 指每个分量有各自不同对角协方差矩阵（非对角为零，对角不为零）
            # spherical 指每个分量有各自不同的简单协方差矩阵，球面协方差矩阵（非对角为零，对角完全相同，球面特性）
            gmm = GaussianMixture(n_components=2, covariance_type='diag').fit(Par_attributes_scaled)
            labels = gmm.predict(Par_attributes_scaled)
            cluster_id_selected = np.argmax(gmm.means_[:,0])
            Par_selected = (labels == cluster_id_selected)
            Par_num_selected = np.sum(Par_selected == True)
            Par_selected_ratio = np.float(Par_num_selected)/Par_num
            print(nonaffine_rank, Par_selected_ratio)
            if Par_selected_ratio > (selected_ratio + 0.0010):
                end = nonaffine_rank
            elif Par_selected_ratio < (selected_ratio - 0.0010) and Par_selected_ratio >= 0.01:
                start = nonaffine_rank
            elif Par_selected_ratio < 0.01:
                early_stopping = True
                break
            else:
                Par_id_STZ = Par_id_high[Par_selected]
                break

            if iteration > 40:
                early_stopping = True
                break
            iteration += 1

        # plt.scatter(Par_attributes[:, 0], Par_attributes[:, 1], c=labels, cmap='rainbow')
        # Par_attributes = np.stack((Par_D2min, Par_temperature), axis=1)
        # Par_attributes_scaled = preprocessing.RobustScaler(quantile_range=(0, 99)).fit_transform(Par_attributes)
        # plot_clustering_2d(Par_attributes_scaled, labels, output_path, frame)
        # python CalCluster.py -rank 0.05 -linkage average -test 1 -start 6400000 -strain 1.0 -delta_strain 0.01 -rate 0.1 -steady 0.25

        # https://blog.csdn.net/weixin_40604987/article/details/79292493
        # key = np.unique(labels)
        # result = {}
        # for k in key:
        #     mask = (labels == k)
        #     arr_new = labels[mask]
        #     v = arr_new.size
        #     result[k] = v
        # print(Par_num, result, cluster_id_selected)

        if early_stopping:
            start, end = 0.0, 0.4
            while True:
                # try:
                #     D2min_threshold = np.percentile(Par_D2min, 100*(1 - rank))
                #     Temp_threshold = np.percentile(Par_temperature, 100*(1 - rank))
                # except UnboundLocalError:
                #     rank = 0.1
                #     D2min_threshold = np.percentile(Par_D2min, 100*(1 - rank))
                #     Temp_threshold = np.percentile(Par_temperature, 100*(1 - rank))

                rank = (start + end)/2.0
                D2min_threshold = np.percentile(Par_D2min, 100*(1 - rank))
                Temp_threshold = np.percentile(Par_temperature, 100*(1 - rank))
                Par_selected = (Par_D2min > D2min_threshold)*(Par_temperature > Temp_threshold)
                Par_id_STZ = Par_id[Par_selected]
                Par_num_selected = np.sum(Par_selected == True)
                Par_selected_ratio = np.float(Par_num_selected)/Par_num
                print(rank, Par_selected_ratio)
                if Par_selected_ratio < (selected_ratio - 0.0010):
                    start = rank
                elif Par_selected_ratio > (selected_ratio + 0.0010):
                    end = rank
                else:
                    break

        Par_selected = np.zeros(Par_num, dtype=bool)
        for i, id in enumerate(Par_id):
            if ((id in Par_id_STZ) and (id in Par_id_region)): Par_selected[i] = True

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3.2 Particles in plastic state defined by a D2min threshold
        #
        # In Science paper (Cubuk et al. 2017), different thresholds are used.  A particle at time t is said to be rearranging if D2_min > D2_min_0. D2_min in units of squared mean particle diameter.
        # Sorted_Par_D2min = sorted(Par_D2min, reverse=True)
        # D2min_threshold = Sorted_Par_D2min[int(Par_num*nonaffine_rank)]
        # D2min_threshold = np.percentile(Par_D2min, 100*(1-nonaffine_rank))
        # Par_selected = Par_D2min > D2min_threshold

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # STZ clusters
        #
        Par_num_selected = np.sum(Par_selected == True)
        G, G_trimmed, connectivity, distance_matrix, neighbor_list = cal_connectivity_and_distance_matrix(Par_id, Par_coord,
                                                                                                          Par_radius, Par_selected,
                                                                                                          vor, vor_nearest_neighbors, epsilon, frame)

        # trimmed = False
        # stz_nx_clusters = particle_clustering_nx(d50, Par_id, Par_coord, Par_radius, Par_volume, Par_D2min, Par_shear_strain, Par_selected,
        #                                          vor, G, trimmed, output_path, frame, dump_results, title='STZ')

        trimmed = True
        stz_nx_trimmed_clusters = particle_clustering_nx(d50, Par_id, Par_coord, Par_radius, Par_volume, Par_D2min, Par_shear_strain, Par_selected,
                                                         vor, G_trimmed, trimmed, output_path, frame, dump_results, title='STZ')

        stz_db_clusters = particle_clustering_dbscan(d50, Par_id, Par_coord, Par_radius, Par_volume, Par_D2min, Par_shear_strain, Par_selected,
                                                     vor, neighbor_list, output_path, frame, dump_results, title='STZ')

        # stz_skl_clusters = particle_clustering_skl(d50, Par_id, Par_coord, Par_radius, Par_volume, Par_D2min, Par_shear_strain, Par_selected,
        #                                            vor, connectivity, linkage, G_trimmed, output_path, frame, dump_results, title='STZ')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # NONAFFINE clusters
        #
        # D2min_threshold = np.percentile(Par_D2min, 100*(1-Par_selected_ratio))
        # Par_selected = Par_D2min > D2min_threshold
        # G, G_trimmed, connectivity, distance_matrix, neighbor_list = cal_connectivity_and_distance_matrix(Par_id, Par_coord,
        #                                                                                                   Par_radius, Par_selected,
        #                                                                                                   vor, vor_nearest_neighbors, epsilon, frame)
        #
        # trimmed = False
        # nonaffine_nx_clusters = particle_clustering_nx(d50, Par_id, Par_coord, Par_radius, Par_volume, Par_D2min, Par_shear_strain, Par_selected,
        #                                                vor, G, trimmed, output_path, frame, dump_results, title='NONAFFINE')
        #
        # trimmed = True
        # nonaffine_nx_trimmed_clusters = particle_clustering_nx(d50, Par_id, Par_coord, Par_radius, Par_volume, Par_D2min, Par_shear_strain, Par_selected,
        #                                                        vor, G_trimmed, trimmed, output_path, frame, dump_results, title='NONAFFINE')
        #
        # nonaffine_db_clusters = particle_clustering_dbscan(d50, Par_id, Par_coord, Par_radius, Par_volume, Par_D2min, Par_shear_strain, Par_selected,
        #                                                   vor, neighbor_list, output_path, frame, dump_results, title='NONAFFINE')

        # nonaffine_skl_clusters = particle_clustering_skl(d50, Par_id, Par_coord, Par_radius, Par_volume, Par_D2min, Par_shear_strain, Par_selected,
        #                                                  vor, connectivity, linkage, G_trimmed, output_path, frame, dump_results, title='NONAFFINE')


        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # RANDOM clusters
        #
        Par_index = np.arange(Par_num)
        random.shuffle(Par_index)
        Par_index_selected = random.sample(list(Par_index), Par_num_selected)
        Par_selected = np.zeros(Par_num, dtype=bool)
        for i in range(Par_num):
            if i in Par_index_selected: Par_selected[i] = True

        G, G_trimmed, connectivity, distance_matrix, neighbor_list = cal_connectivity_and_distance_matrix(Par_id, Par_coord,
                                                                                                          Par_radius, Par_selected,
                                                                                                          vor, vor_nearest_neighbors, epsilon, frame)

        # trimmed = False
        # random_nx_clusters = particle_clustering_nx(d50, Par_id, Par_coord, Par_radius, Par_volume, Par_D2min, Par_shear_strain, Par_selected,
        #                                             vor, G, trimmed, output_path, frame, dump_results, title='RANDOM')

        trimmed = True
        random_nx_trimmed_clusters = particle_clustering_nx(d50, Par_id, Par_coord, Par_radius, Par_volume, Par_D2min, Par_shear_strain, Par_selected,
                                                            vor, G_trimmed, trimmed, output_path, frame, dump_results, title='RANDOM')

        random_db_clusters = particle_clustering_dbscan(d50, Par_id, Par_coord, Par_radius, Par_volume, Par_D2min, Par_shear_strain, Par_selected,
                                                        vor, neighbor_list, output_path, frame, dump_results, title='RANDOM')

        # random_skl_clusters = particle_clustering_skl(d50, Par_id, Par_coord, Par_radius, Par_volume, Par_D2min, Par_shear_strain, Par_selected,
        #                                               vor, connectivity, linkage, G_trimmed, output_path, frame, dump_results, title='RANDOM')

        print(60 * '~')
        print("Type:         contiguity   dbscan")
        print("STZ clusters:       %4d    %4d" % (len(stz_nx_trimmed_clusters), len(stz_db_clusters)))
        print("RANDOM clusters:    %4d    %4d" % (len(random_nx_trimmed_clusters), len(random_db_clusters)))
        print(60 * '~')

        frame_stz_clusters['nx_trimmed'].append(stz_nx_trimmed_clusters)
        frame_stz_clusters['db'].append(stz_db_clusters)

        frame_random_clusters['nx_trimmed'].append(random_nx_trimmed_clusters)
        frame_random_clusters['db'].append(random_db_clusters)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. Save results to dump files
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fw = open(output_path + '/Clusters.dump','wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(frame_stz_clusters, fw)
    pickle.dump(frame_random_clusters, fw)
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
    selected_ratio = 0.10
    linkage = 'average'
    shear_strain = 5.0
    shear_rate = 2.0
    steady_strain = 0.2
    scenario = 200
    strain_window = 0.001
    dump_results = True
    epsilon = 100.0
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
        elif (argList[i][:4] == "-dia"):
            i += 1
            d50 = float(argList[i])
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
        elif (argList[i][:2] == "-l"):
            i += 1
            linkage = argList[i]
        elif (argList[i][:4] == "-sel"):
            i += 1
            selected_ratio = float(argList[i])
        elif (argList[i][:4] == "-dum"):
            i += 1
            dump_results = str_to_bool(argList[i])
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
    print("Particle selected: %.5f" % selected_ratio)
    print("Linkage method:    %s" % linkage)
    print("Scenario:          %d" % scenario)
    print("Dump results:      %s" % dump_results)
    print(60 * '~')
    DefineClusters(case, test_id, d50, selected_ratio, epsilon, linkage, shear_strain, shear_rate, time_step,
                   steady_strain, dump_results, scenario, strain_window)