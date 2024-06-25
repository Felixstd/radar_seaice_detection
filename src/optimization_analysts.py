import sys

import edge_detector_ice_raw as edi
import numpy as np
import matplotlib.pyplot as plt
import plot_radar as plr

from concurrent.futures import ThreadPoolExecutor, as_completed

original_stdout = sys.stdout

"""
This script is used to run optmization with the analysts images. 
"""

def read_parameters_optmization(num, param, shape, reshape = True) : 
    
    total_3 = np.load(SaveDir_Optimized+'OptimizedParams_3_'+num+'.npy', allow_pickle=True).item()[param]
    total_5 = np.load(SaveDir_Optimized+'OptimizedParams_5_'+num+'.npy', allow_pickle=True).item()[param]
    total_7 = np.load(SaveDir_Optimized+'OptimizedParams_7_'+num+'.npy', allow_pickle=True).item()[param]
    total_9 = np.load(SaveDir_Optimized+'OptimizedParams_9_'+num+'.npy', allow_pickle=True).item()[param]
    total_11 = np.load(SaveDir_Optimized+'OptimizedParams_11_'+num+'.npy', allow_pickle=True).item()[param]
    total_13 = np.load(SaveDir_Optimized+'OptimizedParams_13_'+num+'.npy', allow_pickle=True).item()[param]
    
    if reshape:
        total_3 = np.reshape(total_3, shape)
        total_5 = np.reshape(total_5, shape)
        total_7 = np.reshape(total_7, shape)
        total_9 = np.reshape(total_9, shape)
        total_11 = np.reshape(total_11, shape)
        total_13 = np.reshape(total_13, shape)
    
    # total_3 = np.zeros_like(total_5)
    
    return total_3, total_5, total_7, total_9, total_11, total_13

def parallel_optimizing(kernel):
        return edi.optimizing_RadAnalysts(concentration_analysers, kernel, num_analysts, step_opt,
                                    './Optimization_img_algo', masksDir)


ParamsDir = './'
# ImgDir_analysts = './Optimization_img_analysers/'
ImgDir_analysts = './Opt_img_Analysts/' #This one has the fog
VidDir_algorithm = '/aos/home/fstdenis/ICE_RADAR/Data_Unc/Test_imgs_algo/'
Savedir = '/aos/home/fstdenis/ICE_RADAR/Data_Unc/Test_imgs_algo/saved_params/'
masksDir = "/storage/fstdenis/Barrow_RADAR/ICE_RADAR_MISC/IdentTypeIce/edge_detection/masks/"

#---------- Code to obtain the concentration from the analyst images ----------#
"""
    SIC_analysts_2.npy contains the data of the analysts without the added fog frames present in Optimization_img_analysers
    SIC_analysts_Fog.npy contains the data of the analysts with the added fog frames present in Opt_img_Analysts
"""
# polygon_list_time_1, polygons_area_analysers, concentration_analysers = \
#     identification_edges_group(ImgDir_analysts, False)

# dict_conc = {'Analysts' : concentration_analysers}
# np.save('./SIC_analysts_Fog.npy', dict_conc)

# ------- Optimizing Steps ---------# 

#to analyse the run, set one to 1 and the others to 0
Added_Fog = 1
NoAdded_Fog = 0
Border = 0
optimizing = 0

num_analysts = 10
step_opt = 1
# 
# kernels = [(3,3)]
kernels = [(3,3), (5,5), (7, 7), (9, 9), (11, 11), (13, 13)]

x = np.arange(0, 126, step_opt)
y = np.arange(0, 126, step_opt)

X, Y = np.meshgrid(x, y)
shape = X.shape

if Border: 
    SaveDir_Optimized = '/storage/fstdenis/Barrow_RADAR/saved_run/RADAR/saved_OptimizedParams/'
    figDir = './Results_Borders/'
    concentration_comp = np.load('./SIC_analysts_2.npy', allow_pickle=True).item()
    analysts = 14*[0] + 14*[1] + 14*[2] + 14*[3] + 14*[4] + 14*[5] + 14*[6] + 14*[7] + 14*[8] + 14*[9]
    num = '9'
    
    
if Added_Fog:
    SaveDir_Optimized = "/storage/fstdenis/Barrow_RADAR/Optimized_Parameters_WithoutBorders/OptParams_withAddedFog/"
    figDir = './Results_NoBorders_WithAddedFog/'
    concentration_comp = np.load('./SIC_analysts_Fog.npy', allow_pickle=True).item()
    analysts = 16*[0] + 16*[1] + 16*[2] + 16*[3] + 16*[4] + 16*[5] + 16*[6] + 16*[7] + 16*[8] + 16*[9]
    num = '11'
    
    
if NoAdded_Fog: 
    SaveDir_Optimized = "/storage/fstdenis/Barrow_RADAR/Optimized_Parameters_WithoutBorders/OptParams_withoutAddedFog/"
    figDir = './Results_NoBorders_WithoutAddedFog/'
    concentration_comp = np.load('./SIC_analysts_2.npy', allow_pickle=True).item()
    analysts = 14*[0] + 14*[1] + 14*[2] + 14*[3] + 14*[4] + 14*[5] + 14*[6] + 14*[7] + 14*[8] + 14*[9]
    num = '10'

concentration_analysers = np.asarray(concentration_comp['Analysts'])[1:]


std_analyser = np.zeros(14)
for i in range(0, 14):
    std_analyser[i] = np.std(concentration_analysers[i::14])

max_std_analysers = np.max(std_analyser)
min_std_analysers = np.min(std_analyser)
std_average_analysers = np.mean(std_analyser)

if optimizing:

# Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as executor:
        # Submit tasks to the executor
        futures = {executor.submit(edi.parallel_optimizing, kernel): kernel for kernel in kernels}

        # Wait for all tasks to complete
        for future in as_completed(futures):
            kernel = futures[future]
            try:
                result = future.result()
                low_total, high_total, slope_total, slope_elim, intercept_total, intercept_elim, rmse_total, rmse_elim, mbe_total, mbe_elim, \
                slope_inter0_total, slope_inter0_elim, concentration_ice, iterations = result
                opt_dict = {'Low' : low_total, 'High' : high_total, 'Slope' : slope_total, \
                'Slope_elim' : slope_elim, 'Intercept' : intercept_total, 'Intercept_elim' : intercept_elim, \
                'RMSE' : rmse_total, 'RMSE_elim' : rmse_elim, 'MBE' : mbe_total, 'MBE_elim' : mbe_elim, \
                    'Slope_0' : slope_inter0_total, 'Slope_0_elim' : slope_inter0_elim, 'SIC' : concentration_ice, 'IT' : iterations} 
                np.save(SaveDir_Optimized+'OptimizedParams_'+str(kernel[0])+'_11.npy', opt_dict)
                print(f"Kernel: {kernel}, Result: {result}")
            except Exception as e:
                print(f"Kernel: {kernel} generated an exception: {e}")    

slope_total_3, slope_total_5, slope_total_7, slope_total_9, slope_total_11, slope_total_13 = read_parameters_optmization(num, 'Slope', shape)
mbe_total_3, mbe_total_5, mbe_total_7, mbe_total_9, mbe_total_11, mbe_total_13 = read_parameters_optmization(num, 'MBE', shape)
RMSE_total_3, RMSE_total_5, RMSE_total_7, RMSE_total_9, RMSE_total_11, RMSE_total_13 = read_parameters_optmization(num, 'RMSE', shape)
intercept_total_3, intercept_total_5, intercept_total_7, intercept_total_9, intercept_total_11,intercept_total_13 = read_parameters_optmization(num, 'Intercept', shape)
SIC_total_3, SIC_total_5, SIC_total_7, SIC_total_9, SIC_total_11, SIC_total_13 = read_parameters_optmization(num, 'SIC', shape, False)
iterations = np.load(SaveDir_Optimized+'OptimizedParams_13_'+num+'.npy', allow_pickle=True).item()['IT']
iterations = np.reshape(iterations, shape)

RMSE_kernels = [RMSE_total_3, RMSE_total_5, RMSE_total_7, RMSE_total_9, RMSE_total_11, RMSE_total_13]
MBE_kernels  = [mbe_total_3, mbe_total_5, mbe_total_7, mbe_total_9, mbe_total_11, mbe_total_13]
Slope_kernels = [slope_total_3, slope_total_5, slope_total_7, slope_total_9, slope_total_11, slope_total_13]
Intercept_kernels = [intercept_total_3, intercept_total_5, intercept_total_7, intercept_total_9, intercept_total_11,intercept_total_13]


idx_min_3, idx_min_5, idx_min_7, idx_min_9, idx_min_11, idx_min_13 = edi.find_min_kernels(RMSE_kernels, shape)
idx_min = [idx_min_3, idx_min_5, idx_min_7, idx_min_9, idx_min_11, idx_min_13]

min_mbe_3, min_mbe_5, min_mbe_7, min_mbe_9, min_mbe_11, min_mbe_13 = edi.min_value_kernel(MBE_kernels, idx_min)
min_RMSE_3, min_RMSE_5, min_RMSE_7, min_RMSE_9, min_RMSE_11, min_RMSE_13 = edi.min_value_kernel(RMSE_kernels, idx_min)
intercept_min_3, intercept_min_5, intercept_min_7, intercept_min_9, intercept_min_11, intercept_min_13 = edi.min_value_kernel(Intercept_kernels, idx_min)
slope_min_3, slope_min_5, slope_min_7, slope_min_9, slope_min_11, slope_min_13 = edi.min_value_kernel(Slope_kernels, idx_min)

#finding the minima

idx_min_SIC3 = iterations[idx_min_3]
idx_min_SIC5 = iterations[idx_min_5]
idx_min_SIC7 = iterations[idx_min_7]
idx_min_SIC9 = iterations[idx_min_9]
idx_min_SIC11 = iterations[idx_min_11]
idx_min_SIC13 = iterations[idx_min_13]

SIC_min_3 = SIC_total_3[idx_min_SIC3]
SIC_min_5 = SIC_total_5[idx_min_SIC5]
SIC_min_7 = SIC_total_7[idx_min_SIC7]
SIC_min_9 = SIC_total_9[idx_min_SIC9]
SIC_min_11 = SIC_total_11[idx_min_SIC11]
SIC_min_13 = SIC_total_13[idx_min_SIC13]

min_Tlow, min_Thigh = idx_min_7[0], idx_min_7[1]
RMSE_Tlow = RMSE_total_7[min_Tlow, :]
MBE_Tlow = mbe_total_7[min_Tlow, :]

figures_analysts = True
if figures_analysts:
    plt.figure()
    plt.plot(np.arange(len(RMSE_Tlow)), RMSE_Tlow, color = 'r', label = 'RMSE at $T_{low}$ = '+str(min_Tlow))
    plt.plot(np.arange(len(MBE_Tlow)), MBE_Tlow, color = 'b', label = 'MBE at $T_{low}$ = '+str(min_Tlow))
    plt.plot(min_Thigh, RMSE_Tlow[min_Thigh], marker = 'x', markersize = 5, color = 'black')
    plt.plot(min_Thigh, MBE_Tlow[min_Thigh], marker = 'x', markersize = 5, color = 'black')
    plt.legend()
    # plt.axis('equal')
    plt.grid('--')
    plt.xlabel(r'$T_{High}$')
    
    plt.ylabel('Error')
    plt.savefig(figDir+'tlow_RMSE.png', dpi = 500, bbox_inches = 'tight')
    
    fig, ax1 = plt.subplots()

    color = 'red'
    ax1.set_xlabel(r'$T_{High}$')
    ax1.set_ylabel('RMSE', color=color)
    # plt.grid('--')
    ax1.plot(np.arange(len(RMSE_Tlow)), RMSE_Tlow, color = 'r', label = 'MBE at $T_{low}$ = '+str(min_Tlow))
    ax1.plot(min_Thigh, RMSE_Tlow[min_Thigh], marker = 'x', markersize = 5, color = 'black')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'blue'
    ax2.set_ylabel('MBE', color=color)  # we already handled the x-label with ax1
    ax2.plot(np.arange(len(MBE_Tlow)), MBE_Tlow, color = 'b', label = 'RMSE at $T_{low}$ = '+str(min_Tlow))
    ax2.plot(min_Thigh, MBE_Tlow[min_Thigh], marker = 'x', markersize = 5, color = 'black')
    ax2.tick_params(axis='y', labelcolor=color)
    # plt.grid('--')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(figDir+'errors_Opt.png', dpi = 500, bbox_inches = 'tight')
    
    
    rmse_min = np.array([min_RMSE_3, min_RMSE_5, min_RMSE_7, min_RMSE_9, min_RMSE_11, min_RMSE_13])
    slopes_min = np.array([slope_min_3, slope_min_5, slope_min_7, slope_min_9, slope_min_11, slope_min_13])
    # rmse_min = np.array([ min_RMSE_5, min_RMSE_7, min_RMSE_9, min_RMSE_11, min_RMSE_13])
    # slopes_min = np.array([ slope_min_5, slope_min_7, slope_min_9, slope_min_11, slope_min_13])
    com_slopes_min = np.abs(1-(slopes_min))
    kernel = ['(3,3)', '(5, 5)', '(7,7)', '(9,9)', '(11,11)', '(13, 13)']
    # kernel = [ '(5, 5)', '(7,7)', '(9,9)', '(11,11)', '(13, 13)']
    
    width = 0.25
    x = np.arange(len(kernel))
    x2 = x + width/2
    
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.bar(x-width/2, com_slopes_min, width = width, linewidth = 1, color = 'b', fill = True, hatch = '/', edgecolor = 'b')#label=bar_labels, color=bar_colors)
    ax.set_ylabel(r'$|1 - Slopes|$', color = 'b')
    ax.set_xlabel('Kernels')
    ax2 = ax.twinx()
    ax2.bar(x2, rmse_min, linewidth = 1, color = 'r',fill = True, hatch = '/', edgecolor = 'r', width = width)
    ax2.set_ylabel('RMSE', color = 'r')
    plt.xticks(np.arange(len(com_slopes_min)), kernel)
    plt.savefig(figDir+'Param_minimum.png', dpi = 500, bbox_inches = 'tight')
            
        
    # plr.plot_optimizing(X, Y, mbe_total_3, mbe_total_5, mbe_total_7, mbe_total_13, 'MBE', 'MBE_optimizing.png')
    # # plr.plot_optimizing(X, Y, sumError_3, sumError_5, sumError_7, sumError_13, 'MBE + RMSE', 'Sum_optimizing.png')
    # plr.plot_optimizing(X, Y, RMSE_total_3, RMSE_total_5, RMSE_total_7, RMSE_total_13, 'RMSE', 'RMSE_optimizing.png')
    # plr.plot_optimizing(X, Y, slope_total_3, slope_total_5, slope_total_7, slope_total_13, 'Slope', 'Slope_optimizing.png')
    # plr.plot_optimizing(X, Y, intercept_total_3, intercept_total_5, intercept_total_7, intercept_total_13, 'Intercept', 'Intercept_optimizing.png')
    
    plr.plot_analysts(concentration_analysers, SIC_min_3, slope_min_3, intercept_min_3, analysts, figDir+'SIC_uncertainty_3.png')
    plr.plot_analysts(concentration_analysers, SIC_min_5, slope_min_5, intercept_min_5,analysts, figDir+'SIC_uncertainty_5.png')
    plr.plot_analysts(concentration_analysers, SIC_min_7, slope_min_7, intercept_min_7,analysts, figDir+'SIC_uncertainty_7.png')
    plr.plot_analysts(concentration_analysers, SIC_min_9, slope_min_9, intercept_min_9,analysts, figDir+'SIC_uncertainty_9.png')
    plr.plot_analysts(concentration_analysers, SIC_min_11, slope_min_11, intercept_min_11,analysts, figDir+'SIC_uncertainty_11.png')
    plr.plot_analysts(concentration_analysers, SIC_min_13, slope_min_13, intercept_min_13,analysts, figDir+'SIC_uncertainty_13.png')
    

    with open(figDir+'results_Opt.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print('Analysers Average STD (min, max):', std_average_analysers, '(', min_std_analysers, max_std_analysers, ')')
        
        
        print('Min RMSE (MBE, RMSE) for (3,3) kernel : ', (min_mbe_3, min_RMSE_3))
        print('Min RMSE (MBE, RMSE) for (5,5) kernel : ', (min_mbe_5, min_RMSE_5))
        print('Min RMSE (MBE, RMSE) for (7,7) kernel : ', (min_mbe_7, min_RMSE_7))
        print('Min RMSE (MBE, RMSE) for (9,9) kernel : ', (min_mbe_9, min_RMSE_9))
        print('Min RMSE (MBE, RMSE) for (11,11) kernel : ', (min_mbe_11, min_RMSE_11))
        print('Min RMSE (MBE, RMSE) for (13,13) kernel : ', (min_mbe_13, min_RMSE_13)) 
        
        print('Parameters for the minimal RMSE (MBE, RMSE) (3,3): ', idx_min_3)
        print('Parameters for the minimal RMSE (MBE, RMSE) (5,5): ', idx_min_5)
        print('Parameters for the minimal RMSE (MBE, RMSE) (7,7): ', idx_min_7)
        print('Parameters for the minimal RMSE (MBE, RMSE) (9,9): ', idx_min_9)
        print('Parameters for the minimal RMSE (MBE, RMSE) (11,11): ', idx_min_11)
        print('Parameters for the minimal RMSE (MBE, RMSE) (13,13): ', idx_min_13)
        
        print('Slope for (3,3) kernel : ', slope_min_3)
        print('Slope for (5,5) kernel : ', slope_min_5)
        print('Slope for (7,7) kernel : ', slope_min_7)
        print('Slope for (9,9) kernel : ', slope_min_9)
        print('Slope for (11,11) kernel : ', slope_min_11)
        print('Slope for (13,13) kernel : ', slope_min_13)
        
        print('intercept for (3,3) kernel ', intercept_min_3)
        print('intercept for (5,5) kernel : ', intercept_min_5)
        print('intercept for (7,7) kernel : ', intercept_min_7)
        print('intercept for (9,9) kernel : ', intercept_min_9)
        print('intercept for (11,11) kernel : ', intercept_min_11)
        print('intercept for (13,13) kernel : ', intercept_min_13)
        

    sys.stdout = original_stdout
    