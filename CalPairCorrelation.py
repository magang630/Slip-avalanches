# -*- coding: cp936 -*-
"""
CalAnisotropy.py
Script to calculate the evolution of anisotropy.
Usage: python CalPairCorrelation.py -test 1 -strain 4.0 -delta_strain 0.01 -rate 0.5 -interval 200
Requirements:
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
# import time
import pickle
import pyvoro
import numpy as np
import pandas as pd
from sys import argv,exit
from scipy.special import legendre
from scipy.special import jv
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
# from openpyxl import load_workbook
# from math import *
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
# Distance of two vertexes
#
def vertex_distance(a, b):
    return np.sqrt((a[0] - b[0])**2.0 + (a[1] - b[1])**2.0 + (a[2] - b[2])**2.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Returns the unit vector of the vector
#
def unit_vector(vector):
    return vector/np.linalg.norm(vector)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plane wave expansion
# In physics, the plane wave expansion expresses a plane wave as a sum of spherical waves
# references:
# [1] https://en.wikipedia.org/wiki/Plane_wave_expansion
# [2] https://en.wikipedia.org/wiki/Legendre_polynomials
# [3] https://en.wikipedia.org/wiki/Bessel_function#Spherical_Bessel_functions
def plane_wave_expansion(wave_vector, branch_vector, lota):
    unit_wave_vector = unit_vector(wave_vector)
    unit_branch_vector = unit_vector(branch_vector)

    x1 = np.linalg.norm(wave_vector)*np.linalg.norm(branch_vector)
    x2 = np.dot(unit_wave_vector, unit_branch_vector)
    summation = 0
    for n in range(lota):
        # The spherical Bessel function jn is related to the ordinary Bessel function Jn
        # jv: Bessel function of the first kind of real order v
        spherical_bessal = np.sqrt(np.pi/(2*x1))*jv((n+0.5),x1)

        # Generate the nth-order Legendre polynomial
        legendre_polynomial = legendre(n)
        summation += (2*n+1)*np.power(np.complex(0,1),n)*spherical_bessal*legendre_polynomial(x2)

    return summation

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def calPairCorrelation(case, test_id, d50, shear_rate, shear_strain, time_step, steady_strain, scenario):

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
    # 1. array and parameter initialization
    #
    frame_strain = []
    frame_pcf = []
    frame_sq_gr = []
    frame_sq_damping = []
    frame_order = []

    delta_r = 0.05*d50
    r_num = 100
    rlist = [i*delta_r for i in range(r_num)]

    q_num = 100
    delta_q = 0.2/d50
    wave_modulus = [(i+1)*delta_q for i in range(q_num)]

    domain_length, domain_depth = 0.04, 0.02
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. Loop over each frame
    #
    for idx, frame in enumerate(frame_list):
        strain = (frame - start_frame)*time_step*shear_rate
        frame_strain.append(strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Taking the whole dataset
        #
        particle_info = open(dump_path + '/dump-' + str(frame) + '.sample', 'r')
        alllines = particle_info.readlines()
        lines = alllines[9:]
        particle_info.close()
        for i in range(len(lines)):
            if (lines[i] == '\n'): del lines[i]
        Par_id0   = np.array([int(line.strip().split(' ')[0]) for line in lines])    # 字段以逗号分隔，这里取得是第1列
        Par_type0 = np.array([int(line.strip().split(' ')[1]) for line in lines])  # 字段以逗号分隔，这里取得是第2列
        Par_radius0 = np.array([float(line.strip().split(' ')[2]) for line in lines])  # 字段以逗号分隔，这里取得是第3列
        Par_xcor0 = np.array([float(line.strip().split(' ')[3]) for line in lines])  # 字段以逗号分隔，这里取得是第4列
        Par_ycor0 = np.array([float(line.strip().split(' ')[4]) for line in lines])  # 字段以逗号分隔，这里取得是第5列
        Par_zcor0 = np.array([float(line.strip().split(' ')[5]) for line in lines])  # 字段以逗号分隔，这里取得是第6列

        sorted_indices = np.argsort(Par_id0)
        Par_id = Par_id0[sorted_indices]
        Par_type = Par_type0[sorted_indices]
        Par_radius = Par_radius0[sorted_indices]
        Par_xcor = Par_xcor0[sorted_indices]
        Par_ycor = Par_ycor0[sorted_indices]
        Par_zcor = Par_zcor0[sorted_indices]

        Par_id = Par_id[Par_type == 1]
        Par_radius = Par_radius[Par_type == 1]
        Par_xcor = Par_xcor[Par_type == 1]
        Par_ycor = Par_ycor[Par_type == 1]
        Par_zcor = Par_zcor[Par_type == 1]
        Par_diameter = 2.0*Par_radius
        Par_num = np.size(Par_id)
        Par_volume = 4./3.0*np.pi*Par_radius**3.0

        assembly_polydispersity = np.sqrt(np.mean(Par_diameter*Par_diameter) - np.power(np.mean(Par_diameter),2))/np.mean(np.mean(Par_diameter))
        # print assembly_polydispersity

        # neighbor particles in the current configuration
        Par_coord = np.stack((Par_xcor, Par_ycor, Par_zcor), axis=1)
        domain_xmin, domain_xmax = 0.0, domain_length
        domain_zmin, domain_zmax = 0.0, domain_depth
        # domain_xmin, domain_xmax = np.min(Par_coord[:, 0] - Par_radius), np.max(Par_coord[:, 0] + Par_radius)
        domain_ymin, domain_ymax = np.min(Par_coord[:, 1] - Par_radius), np.max(Par_coord[:, 1] + Par_radius)
        # domain_zmin, domain_zmax = np.min(Par_coord[:, 2] - Par_radius), np.max(Par_coord[:, 2] + Par_radius)
        container = [[domain_xmin, domain_xmax], [domain_ymin, domain_ymax], [domain_zmin, domain_zmax]]
        vor = pyvoro.compute_voronoi(Par_coord, limits=container, dispersion=4*d50, radii=Par_radius, periodic=[True, False, True])

        Par_inside_region = check_inside_region(Par_coord, boundary_gap=[5*d50, 5*d50, 5*d50])
        Par_num_region = np.sum(Par_inside_region == True)
        Par_volume_region = Par_volume[Par_inside_region]
        # Par_coord_region = Par_coord[Par_inside_region]

        print(60*'*')
        print('Frame:                  %d' % frame)
        print('Shear strain:           %.2f' % strain)
        print('Particle number in ROI: %d' % Par_num_region)
        print(60*'*')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3. Pair correlation function g(r) and static stucture factor s(q) of the region of interest
        #    Pair correlation function g(r) is a simple measure of structural correlations in granular materials.
        #    Using local g(r), entropy of a particle i can be defined as S2,i. Structural entropy measures the loss of entropy due to positional correlations.
        #    References:
        #    [1] Tong, H., & Xu, N. (2014). Order parameter for structural heterogeneity in disordered solids. Physical Review E, 90(1), 010401.
        #    [2] Yang, X., Liu, R., Yang, M., Wang, W. H., & Chen, K. (2016). Structures of Local Rearrangements in Soft Colloidal Glasses. Physical Review Letters, 116(23), 1–6.
        #    [3] https://homepage.univie.ac.at/franz.vesely/simsp/dx/node22.html
        #
        # domain_volume = convex_hull_volume_bis(Par_coord_region)
        # assembly_num_dens = Par_num_region/domain_volume

        cr_count = np.zeros(r_num)
        assembly_pcf = np.zeros(r_num)
        assembly_sq_gr = np.zeros(q_num)
        assembly_sq_damping = np.zeros(q_num)
        assembly_s2 = 0
        domain_volume = 0

        # assembly_sq = np.zeros(q_num)
        # assembly_sq_radial = np.zeros([q_num, wave_number]).astype(complex)
        Par_reduced_vor_vol = np.zeros(Par_num_region)
        pointer = 0
        for i in range(Par_num):
            if not Par_inside_region[i]: continue
            domain_volume += vor[i]['volume']
            Par_reduced_vor_vol[pointer] = vor[i]['volume']/Par_volume[i]
            pointer += 1
            for j in range(i, Par_num):
                if i == j: continue
                distij = vertex_distance(Par_coord[i], Par_coord[j])
                k = np.ceil(distij/delta_r).astype(int) - 1
                if (k >= r_num): continue
                if (k < 0): k = 0
                cr_count[k] += 1.0
        assembly_num_dens = Par_num_region/domain_volume

        for k in range(r_num):
            shell_volume = 4.0/3.0*np.pi*((rlist[k] + delta_r)**3.0 - rlist[k]**3.0)
            if cr_count[k]:
                assembly_pcf[k] = (2.0*cr_count[k]/Par_num_region)/(shell_volume*assembly_num_dens)
                if np.abs(assembly_pcf[k] - 1.0) <= 0.01: assembly_pcf[k] = 1.0
            else:
                assembly_pcf[k] = 1e-8
            assembly_s2 += shell_volume*(assembly_pcf[k]*np.log(assembly_pcf[k]) - (assembly_pcf[k] - 1.0))

        assembly_s2 *= -0.5*assembly_num_dens
        assembly_sf = np.sum(Par_volume_region)/domain_volume
        assembly_var_free_vol = np.mean(Par_reduced_vor_vol**2.0) - np.mean(Par_reduced_vor_vol)**2.0

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # The structure factor can be calculated from the particle configuration. The calculations, however, involve the summation over all pairs of particles in the system,
        # leading to the quadratic dependence of the computational cost on the number of particles and making the calculations
        # expensive for large systems. An alternative approach to calculations of s(q) is to substitute the double summation over particle positions by integration over
        # pair distribution function. In calculation of g(r), the maximum value of r, rmax, is limited  by the size of system. The truncation of the numerical integration
        # at rmax induces spurious ripples with a period of △=2π/rmax. A damping function W(r) can be used to replace the sharp step function at rmax
        # by a smoothly decreasing contribution from the density function at large interparticle distances and approaching zero at rmax
        #    References:
        #    [1] http://www.nyu.edu/classes/tuckerman/stat.mech/lectures/lecture_9/node3.html
        #    [2] https://en.wikipedia.org/wiki/Radial_distribution_function
        #
        for l, q in enumerate(wave_modulus):
            for k, gr in enumerate(assembly_pcf):
                # damping function W(r)
                temp = np.pi*(rlist[k]+0.5*delta_r)/(r_num*delta_r)
                damping = np.sin(temp)/temp
                assembly_sq_gr[l] += np.sin(q*(rlist[k]+0.5*delta_r))*(gr - 1.0)*(rlist[k]+0.5*delta_r)*delta_r/q
                assembly_sq_damping[l] += np.sin(q*(rlist[k]+0.5*delta_r))*(gr - 1.0)*(rlist[k]+0.5*delta_r)*delta_r*damping/q
        assembly_sq_gr = 1 + 4*np.pi*assembly_num_dens*assembly_sq_gr
        assembly_sq_damping = 1 + 4*np.pi*assembly_num_dens*assembly_sq_damping

        frame_pcf.append(assembly_pcf)
        frame_sq_gr.append(assembly_sq_gr)
        frame_sq_damping.append(assembly_sq_damping)
        frame_order.append({'solid fraction': assembly_sf, 'number density': assembly_num_dens, 'polydispersity': assembly_polydispersity, 'structural entropy': assembly_s2, 'variance of reduced Voronoi': assembly_var_free_vol})

    frame_pcf = np.array(frame_pcf)
    frame_sq_gr = np.array(frame_sq_gr)
    frame_sq_damping = np.array(frame_sq_damping)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. Output
    #
    output_path = file_path + '/test-' + str(test_id) + '/structural characterization'
    mkdir(output_path)

    columns = frame_strain
    index = np.array(rlist)/d50
    df_pcf = pd.DataFrame(frame_pcf.transpose(), index=index, columns=columns)
    df_pcf.to_excel(output_path + '/Pair correlation function.xlsx')

    writer = pd.ExcelWriter(output_path + '/Static structure factor.xlsx')
    index = np.array(wave_modulus)*d50
    df_sq_gr = pd.DataFrame(frame_sq_gr.transpose(), index=index, columns=columns)
    df_sq_damping = pd.DataFrame(frame_sq_damping.transpose(), index=index, columns=columns)
    df_sq_gr.to_excel(writer, sheet_name='No damping')
    df_sq_damping.to_excel(writer, sheet_name='Damping')
    writer.save()
    writer.close()

    # s_s2 = pd.Series(frame_s2, index=frame_strain)
    # s_s2.to_excel(output_path + '/Structure entropy.xlsx')
    df_order = pd.DataFrame(frame_order, index=frame_strain)
    df_order.to_excel(output_path + '/Structure order.xlsx')

    # myOutputFile = open(output_path + '/Pair correlation function.dat','w')
    # for j in range(r_num):
    #     myOutputFile.write("%12.8f " %(rlist[j]/d50))
    #     for i in range(len(frame_pcf)):
    #         myOutputFile.write("%12.8f " %frame_pcf[i][j])
    #     myOutputFile.write("\n")
    # myOutputFile.close()
    #
    # myOutputFile = open(output_path + '/Static structure factor.dat', 'w')
    # for j in range(q_num):
    #     myOutputFile.write("%12.8f " %(wave_modulus[j]*d50))
    #     for i in range(len(frame_sq_gr)):
    #         myOutputFile.write("%12.8f  " %(frame_sq_gr[i][j]))
    #     myOutputFile.write("\n")
    # myOutputFile.write("~~~~~~~~~~~~~~~~~\n")
    #
    # for j in range(q_num):
    #     myOutputFile.write("%12.8f " %(wave_modulus[j]*d50))
    #     for i in range(len(frame_sq_damping)):
    #         myOutputFile.write("%12.8f  " %(frame_sq_damping[i][j]))
    #     myOutputFile.write("\n")
    # myOutputFile.close()
    #
    # myOutputFile = open(output_path + '/Structure entropy.dat','w')
    # for key, value in frame_s2.items():
    #     myOutputFile.write("%12.8f, %12.8f" %(key, value))
    # myOutputFile.close()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 5. Save results to dump files
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fw = open(output_path + '/Pair correlation function and structure factor.dump','wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(frame_pcf, fw)
    pickle.dump(frame_sq_gr, fw)
    pickle.dump(frame_sq_damping, fw)
    pickle.dump(frame_order, fw)

    fw.close()

#==================================================================
# S T A R T
#
if __name__ == '__main__':
    
    file_path = None
    file_name = None
    case = 'shear-rate-2-press-1e6'
    test_id = 1
    time_step = 2e-8
    shear_rate = 0.1
    shear_strain = 1.0
    delta_strain = 0.01
    steady_strain = 0.0
    d50 = 0.001
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
        elif (argList[i][:4] == "-rat"):
            i += 1
            shear_rate = float(argList[i])
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
    print("Scenario:    %d" % scenario)
    print(60 * '~')
    calPairCorrelation(case, test_id, d50, shear_rate, shear_strain, time_step, steady_strain, scenario)
