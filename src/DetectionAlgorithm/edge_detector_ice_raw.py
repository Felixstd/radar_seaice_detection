# %%
"""
************************************************************************
                        Edge Detection Algorithm

    Edge detection algorithm for sea ice radar images. This was specifically made 
    for the coastal radar present in Utqiagvik, AK, but it can be adapted for 
    other types of radar images. 
    
    This file contains all of the main functions of the detection algorithm.
    
    Also, it contains the functions for the optimization with the analysts. 

    Revision History
    ----------------
    Ver             Date (dd-mm-yy)        Authors

    V1.0             04-07-24               F. St-Denis

    Address : Dept. of Atmospheric and Oceanic Sciences, McGill University
    -------   Montreal, Quebec, Canada
    Email   :  felix.st-denis@mail.mcgill.ca

************************************************************************
"""

import warnings, time, os, sys, numba

import cv2 as cv 
import numpy as np
import xarray as xr
import netCDF4 as nc
import plot_radar as plr
import extract_video as vid
import matplotlib.pyplot as plt 

from numba import njit, jit
from shapely import geometry
from scipy.interpolate import griddata
from shapely.validation import make_valid
from sklearn.linear_model import LinearRegression

original_stdout = sys.stdout

warnings.filterwarnings("ignore")

def savefile_missing(SavingDirectory, DayDir, xg, yg):
    ds1 = xr.Dataset(
    data_vars={
        "interpolation": (("x", "y"), np.zeros_like(xg)*np.nan)
        # "individual_maps":(("t", "i", "j"), result)
    },
    coords = {"latitude": (["x","y"], xg),
                            "longitude": (["x","y"], yg)
    }
    )
    ds1.to_netcdf(SavingDirectory+"interpolated_"+DayDir+"_modis1km.nc")

@njit()
def rmse(SIC_calculated, SIC_observations) : 
    
    '''
    Function used to calculate the Root-Mean Square error between two datasets. 
    It's used to compare the calculated SIC with the best-line fit when comparing the observed and algorithm
    SIC. 
    
    RMSE = ((sum((X_i - Y_i)**2)/N)^(1/2)
    
    Input :
        SIC_calculated (array): calculated SIC
        SIC_observations (array): observed SIC, in our case it's the analysts' one. 
        
    Output : 
        error (array): the computed RMSE
    '''
    
    error = np.sqrt(np.nanmean((SIC_calculated - SIC_observations)**2))
    
    return error

def mbe(SIC_calculated, SIC_observations) : 
    
    """
    Function used to calculate the Mean Bias error between two datasets. 
    It's used to compare two datasets. It tells us our far one is from the other
    
    RMSE = ((sum((X_i - Y_i))/N)
    
    Input :
        SIC_calculated (array): calculated SIC
        SIC_observations (array): observed SIC, in our case it's the analysts' one. 
        
    Output : 
        error (array): the MBE
    """
    
    error = np.nanmean(SIC_calculated - SIC_observations)
    return error

def canny_edge_opencv(img_list, low, high, land_mask, fill_area = False) : 
    
    """
    Function that applies the canny edge algorithm to a list of radar images. 
    
    To the the value for the land to be, we use the value of two pixels taken in the ocean. 
    If the either one of the two is lower on the gray scale, the land will be blacker. 
    
    Inputs : 
        img_list (list of arrays): list of images
        low (float): lower threshold for the detection algorithm (int)
        high (float): highest threshold for the detection algorithm (int)
        land_mask (array): Binary mask differientiating the land from the ocean (1 : land, 0 : ocean)
        fill_area (Bool): Set to true when computing the complete field of view of the radar, otherwise set to false. 
        
    Returns:
        edges_list (list of arrays): List of canny edge outputs corresponding to the img_list
        gray_list (list of arrays): same as img_list but the images are now on the gray scale
    """

    
    #setting the initial parameters and list to be appended to
    it = 0
    edges_list = []
    gray_list = []
    
    #main loop
    for img in img_list : 
        
        print('Canny edge algorithm iteration : ', it)
        
        #Finding the land
        idx_land = np.where(land_mask == 1)
        
        #pixel test to determine the land fill value
        first_pixel_test  = np.mean(img[50, 250])
        second_pixel_test = np.mean(img[250, 50])
        
        if fill_area :
            fill_value = 255
        
        else : 

            if first_pixel_test < 35 or second_pixel_test < 35 : 
                fill_value = 15
            
            else : 
                fill_value = 56
            
        #setting the color of the land
        img[idx_land] = fill_value   
        
        #Finding the pixels to get rid of the borders
        ret,bin = cv.threshold(img,5,255,cv.THRESH_BINARY)

        #specifying the kernel
        kernel = np.ones((7,7),np.uint8)
        
        #creating the mask
        erosion = cv.erode(bin,kernel,iterations = 1)
        

        #Computing the edges with the Canny algorithm
        result_canny = cv.Canny(img, low, high)
        
        #remoing the contours
        result = cv.bitwise_and(result_canny,result_canny,mask = erosion)

        #appending to the lists
        edges_list.append(result)
        gray_list.append(img)
        
        it += 1
        
    return edges_list, gray_list

    
def find_contours(img_list, kernel = (3,3), recog = False) : 
    
    """
    Function used to find the contours associated with the edges detected by the canny edge detection algorithm. 
    Uses openCv to compute those contours. The kernel represents number of neighbouring pixels. At the end, 
    a polygon is created for each contour found. 
    
    Inputs: 
        img_list (list of arrays): list containing the edges of every images (output of the canny edge algorithm)
        kernel (tuple) kernel used for the find contours algorithm. 
        recog (bool) set to True when finding the complete radar range. 
        
    Returns:
        polygon_img_list (list of arrays): lists of lists containing the Polygons representing the contours in each images in img_list 
        polygon_area_list (list of arrays): lists of lists containing the Polygon's areas. 
    """
    
    #setting the lists
    polygon_img_list = []
    polygon_area_list = []
    
    it = 0
    #main loop through the images
    for img in img_list : 
        
        print('Finding contours iteration : ', it)
        
        # setting the kernel for the morphological operations
        if recog : 
            kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5)) 
        else : 
            kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel) 

        dilate = cv.dilate(img, kernel_ellipse, iterations=1) #dilating the images to get better resolution

        #finding the contours
        (cnts, _) = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
        
        #lists for every image
        polygon_img = []
        polygon_area = []
        
        #loop through the image's contours
        for cont in cnts : 
            area = cv.contourArea(cont) #computing the contour area
            if (area > 100) :  #if the contour's too small, don't use it
                
                contour = np.squeeze(cont) #to make the polygons
                polygon = geometry.Polygon(contour) #make a polygon instance for each contours
                
                polygon_img.append(polygon)
                polygon_area.append(polygon.area)
                
        polygon_img_list.append(polygon_img)
        polygon_area_list.append(polygon_area)
        it += 1
    
    return polygon_img_list, polygon_area_list

def make_valid_polygon(polygon_list) : 
    
    """
    Function used to make valid polygons that are not valid for shapely so this function makes them valid. 
    We need to do that because to make some tests between them, they all need to be valid. 

    Inputs : 
        polygon_list (list of arrays): list of polygons
        
    Returns:
        polygon_list_valid (list of arrays): lists of valid polygons
    """
    
    polygon_list_valid = []
    #loop through the polygons
    for it in range(len(polygon_list)) : 
        
        polygon = polygon_list[it]
        valid_poly = polygon.is_valid #test validity
        
        if valid_poly : 
            #do nothing
            polygon_v = polygon
        else : 
            #make them valid
            polygon_v = make_valid(polygon)
        
        polygon_list_valid.append(polygon_v)
        
    return polygon_list_valid

def check_inside_contours(polygon_list, polygon_area_list) :
    
    """
    Function use to test if a polygon is inside the other. If yes, the polygon inside is then remove so it doesn't
    count when computing the final area. We count everything inside bigger ones as sea ice. 

    Inputs : 
        polygon_list (list of arrays): list of polygons found by the find_contours and canny_edge()
        polygon_area_list (list of arrays): lists of areas associated to each polygons
        
    Returns:
        polygon_list_checked (list of arrays): updated list of polygons found by the find_contours and canny_edge()
        polygon_area_list_checked (list of arrays): updated list of areas
    """
    
    polygon_list_checked = []
    polygon_area_list_checked = []
    
    #main loop
    for it in range(len(polygon_list)) : 
        
        print('Checking contours : ', it)
        #make them arrays
        polygons = np.array(polygon_list[it])
        polygons_area = np.array(polygon_area_list[it])
        
        #make the polygons valid
        polygons_v = make_valid_polygon(polygons)
        
        idx = 0
        idx_remove = [] #lists of the indices of the polygons that need to be removed
        
        for poly in polygons_v : 
            contains = poly.contains(polygons_v) #test if one contains the other 
            contains[idx] = False #setting to false so the initial shpe doesnt contain itself
            
            #finding the polygons to be removed
            idx_contains = list(np.where(contains == True)[0])      
            if len(idx_contains) != 0 :      
                idx_remove.extend(idx_contains)
            idx += 1
            
        if len(idx_remove) != 0 :
            #delete them
            polygons_rmv = np.delete(polygons, np.unique(idx_remove))
            polygons_area_rmv = np.delete(polygons_area, np.unique(idx_remove))
        
        else : 
            #do nothing
            polygons_rmv = polygons
            polygons_area_rmv  = polygons_area
        
        polygon_list_checked.append(polygons_rmv)
        polygon_area_list_checked.append(polygons_area_rmv)
    
        
    return polygon_list_checked, polygon_area_list_checked

def find_polygon_area(polygon_area_vid, total_radar_range) : 
    
    """
    Function used to find the sea ice concentration from the polygon's areas. 
    
    
    Inputs:
        polygon_area_vid (list of arrays): lists of the polygons areas
        total_radar_range (list of floats): area of the total area of the radar range (float)

    Returns:
        tot_area_ice_vid (list of floats): lists for every images of the total ice area
        tot_area_ow_vid (list of floats): lists for every images of the total open water area
        concentration_ice (list of floats): lists for every images of SIC
        concentration_ow (list of floats): lists for every images of Open water Concentration
    """
    
    #initializing the lists
    tot_area_ice_vid = []
    tot_area_ow_vid = []
    
    concentration_ice = []
    concentration_ow = []

    #loop through the frames
    for it in range(len(polygon_area_vid)) : 
        
        polygon_area_it = polygon_area_vid[it]

        #compute the areas
        tot_area_ice = np.sum(polygon_area_it)
        tot_area_ow = total_radar_range - tot_area_ice

        tot_area_ice_vid.append(tot_area_ice)
        tot_area_ow_vid.append(tot_area_ow)
        
        #compute the concentrations
        concentration_ice.append(tot_area_ice/total_radar_range)
        concentration_ow.append(tot_area_ow/total_radar_range)
        
    return tot_area_ice_vid, tot_area_ow_vid, concentration_ice, concentration_ow

def identification_ice_ow(low, high, kernel, DataDirectory, MasksDirectory, SavingDirectory, plotting = False, saving = False) : 
    
    """
    This function is used to compute the SIC for every frame available. It contains every step of the main detection
    algorithm.
    
    We start by loading everything needed. Then, for every frame, we :
        1. Apply the Canny edge algofithm
        2. Find the sea ice contours 
        3. Remove the unwanted contours
        4. Calculate the areas 
        
    We can also plot the contours for every frame but this slow down the code a lot.
    
    It's important to note that in order to run this code, you should use the optimized parameters found by the 
    optmisation step.
    
    Inputs:
        low: lower threshold for the canny edge detection (int)
        high: higher threshold for the canny edge detection (int)
        kernel: kernel used for the find contours
        DataDirectory: Directory where the frames are
        MasksDirectory: Directory where the masks are
        SavingDirectory: Directory where to save the data
        plotting: usually set to False, but it the contours need to be plotted, set to True
        
        
    """
    
    #Loading the different parameters

    print('Reading Mask')
    land_mask = np.load('../data/land_mask_radar.npy', allow_pickle=True).item()['mask']
    
    #----- Computing the area of the reference frame for the sea ice concentration -----#
    print('Calulating the total area of the radar range')

    img_tot_area = '../data/UAFIceRadar_20220219_014400_crop_geo.tif'

    img = vid.read_img(img_tot_area)
    
    edges_first_img, gray_list = canny_edge_opencv([img], 0, 1, land_mask, False)

    contours_first, poly_area_first = find_contours(edges_first_img, (7,7))
    checked_contours_first, checked_area_list = check_inside_contours(contours_first, poly_area_first)
    total_area_radar = np.amax(checked_area_list)

    #loop through the years
    for YearDir in os.listdir(DataDirectory) : 
        YearDir = str(YearDir)
        #loop through the months
        for MonthDir in os.listdir(DataDirectory+YearDir) : 
            MonthDir = str(MonthDir)
            #loop through the days
            for DayDir in os.listdir(DataDirectory+YearDir+'/'+MonthDir) : 
                DayDir = str(DayDir)
                #Create the saving directory
                saveDir = SavingDirectory+DayDir+'/'

                try:
                    os.mkdir(saveDir)
                    print("Folder %s created!" % saveDir)
                except FileExistsError:
                    print("Folder exists")
                
                # #reading the images
                img_list_day = vid.extract_img_folder(DataDirectory+YearDir+'/'+MonthDir+'/'+DayDir)
                # img_list_day = vid.extract_img_folder('/storage/fstdenis/Barrow_RADAR/RAW_Data/2023/02/20230209/')
                
                
                i = 0
                idx_remove = []
                for img in img_list_day:
                    shape = img.shape
                    
                    if shape != (900, 900):
                        idx_remove.append(i)
                    i+=1
                if len(idx_remove) > 0:
                    
                    img_list_day = np.delete(img_list_day, idx_remove)
                    
                if len(img_list_day) == 0 :
                    polygon_list_time = []
                    gray_list = []
                    concentration_ice = [np.nan]
                    saving_dic = {'Polygons' : polygon_list_time, 'Img' : gray_list, 'Conc_ice' : concentration_ice}
                    np.save(saveDir+'saved_params'+DayDir+'.npy', saving_dic)

                else : 
                        
                    print('\n', 'Starting Canny edge algorithm')
                    start_time =  time.time()
                    edge_list_cv, gray_list = canny_edge_opencv(img_list_day, low, high, land_mask)
                    print('Time for canny with OpenCV : ', time.time() - start_time)

                    print('\n', 'Finding the contours')
                    start_time = time.time()
                    polygon_list_time_1, polygon_area_vid = find_contours(edge_list_cv, kernel)
                    print('Time for finding contours : ', time.time() - start_time)
                    
                    print('Removing contours : ')
                    start_time =  time.time()
                    polygon_list_time, poly_area_time\
                        = check_inside_contours(polygon_list_time_1, polygon_area_vid)
                    print('Time for canny with OpenCV : ', time.time() - start_time)
                    
                    print('Computing the areas')
                    tot_area_vid_ice, tot_area_vid_ow, concentration_ice, concentration_ow \
                        = find_polygon_area(poly_area_time, total_area_radar)
                                            
                    if plotting : 
                        print('Now Plotting the run')
                        plr.plot_edge_finding(gray_list, polygon_list_time, 0, './')
                    #saving the paramters
                    
                    if saving:
                        saving_dic = {'Polygons' : polygon_list_time, 'Img' : gray_list, 'Conc_ice' : concentration_ice}
                        np.save(saveDir+'saved_params'+DayDir+'.npy', saving_dic) 
        
    return img_list_day, gray_list, edge_list_cv, polygon_list_time, polygon_list_time_1, concentration_ice

def identification_edges_group(ImgDir, plotting) :
    
    """
    Function used to compute the SIC from the contours identified by the analysts with an electronic tablet. 
    
    It identifies the red contours made by the analysts in every frame that they drew on. It's based on the fact
    that it takes the red part on every frame. 

    Inputs: 
        VidDir_Cont: Location of the analysts video
        plotting: set to true for plotting
        
    Returns:
        _type_: _description_
    """
    
    #setting the color range
    lower_red = (38, 134, 128)
    upper_red = (179, 255, 255)
    
    
    concentration_analysers = []
    
    for num_analyst in sorted(os.listdir(ImgDir)):
        print(ImgDir+num_analyst)
        #reading the video with the analyst frames.
        img_list_contours = vid.extract_imgs(ImgDir+num_analyst)
        print(img_list_contours)
        #initializing the lists
        it = 0
        polygons_area_analysers = []
        #loop to extract the drawed contours
        for img in img_list_contours : 
            print('Img : ', it)
            print(img.shape)
            test_image1 = cv.cvtColor(img, cv.COLOR_BGR2HSV) #converting to hsv
            test_image2 = cv.inRange(test_image1, lower_red, upper_red) #taking only the red parts

            #identifying the contours
            polygon_list_time_1, polygon_area_vid = find_contours([test_image2], (7,7), True)
            
            #removing the insides contours
            polygon_list_time_checked, polygon_area_vid_check = check_inside_contours(polygon_list_time_1, polygon_area_vid)

            if plotting : 
                for poly in polygon_list_time_checked[0] : 
                    plt.plot(*poly.exterior.xy, color = 'g')
                plt.savefig(f'./test_img_{it}.png', dpi = 500)
                plt.close()
                
            
            polygons_area_analysers.append(polygon_area_vid_check[0])

            it += 1
        if num_analyst == '0':
            #loop to calculate the SIC
            tot_area = polygons_area_analysers[0][0]
        
        for areas in polygons_area_analysers : 
            tot_ice_area = np.sum(areas) #summing the polygons area
            concentration_analysers.append(tot_ice_area/tot_area) #calculating the concentration
        
    return polygon_list_time_1, polygons_area_analysers, concentration_analysers

def optimizing_RadAnalysts(Concentration_analysts, kernel, num_analysts, step_opt, File, MasksDirectory) : 
    
    
    """
    Function used for the optimisation step.
    
    It identifies the contours on the same images as the analysts in order to compare the two and to find the 
    best thresholds for the canny edge algorithm and the best kernel for the find_contours.
    
    It computes a lot of different parameters for a linear regression with a forced intercept at 0 and not. 
    Parameters : 
        RMSE
        MBE
        Slope
        Intercept
        
    """
    
    #Adding nans to outliers
    Conc_analysts = np.copy(Concentration_analysts)
    
    #Setting the Lin Reg classes
    linReg = LinearRegression(fit_intercept=True)
    linReg_noslope = LinearRegression(fit_intercept=False) #intercept forced to 0
    
    #----Reading parameters that will not change during the optimization -----#

    #reading the images, the same
    img_list_1 = vid.extract_imgs(File)
    
    print('Reading Mask')
    land_mask = np.load(MasksDirectory+'land_mask_3.npy', allow_pickle=True).item()['mask']
    print(img_list_1)
    
    print('Computing the Total area')
    edges_first_img, gray_list = canny_edge_opencv([img_list_1[0]], 0, 1, land_mask, False)

    contours_first, poly_area_first = find_contours(edges_first_img, kernel)
    checked_contours_first, checked_area_list = check_inside_contours(contours_first, poly_area_first)
    total_area_radar = np.amax(checked_area_list)
    # for poly in checked_contours_first[0] : 
    #     plt.plot(*poly.exterior.xy, color = 'g')
    # plt.savefig(f'./test.png', dpi = 500)
    # plt.close()
    
    #---- Setting the parameters range ----#
    low_optimize = np.arange(0, 126, step_opt)
    high_optimize = np.arange(0, 126, step_opt)
    
    low_total = []
    high_total = []
    slope_linReg_total = []
    slope_linReg_ElimFar = []
    intercept_linReg_total = []
    intercept_linreg_elim = []
    rmse_linreg_total = []
    rmse_linreg_elim = [] 
    mbe_linreg_total = []
    mbe_linreg_elim = [] 
    slope_intercept0_total = []
    slope_intercept0_elim = []
    concentration_ice_total = []
    iterations = []
    

    it = 0
    #loop through the different thresholds combinaitions. 
    for low in low_optimize : 
        for high in high_optimize : 
            print('Kernel, threshold:', kernel, low, high)
            low = 38
            high = 82
            kernel = (13, 13)

            #computing the concentrations
            concentration_ice = detection_only(img_list_1, low, high, kernel, \
                total_area_radar, land_mask)
                        
            concentration_ice = np.tile(concentration_ice, num_analysts)
            
            mask = ~np.isnan(Conc_analysts) & ~np.isnan(concentration_ice)

            #fitting with and without intercepts
            LinReg_RADAnalysts = linReg.fit(Concentration_analysts.reshape(-1,1), concentration_ice.reshape(-1,1))
            LinReg_Without = linReg.fit(Conc_analysts[mask].reshape(-1,1), concentration_ice[mask].reshape(-1,1))
            
            LinReg_RADAnalysts_0slope = linReg_noslope.fit(Concentration_analysts.reshape(-1,1), concentration_ice.reshape(-1,1))
            LinReg_Without_0slope = linReg_noslope.fit(Conc_analysts[mask].reshape(-1,1), concentration_ice[mask].reshape(-1,1))
            
            #computing the errors
            rmse_linreg = rmse(concentration_ice, Concentration_analysts*1 + 0)
            rmse_linreg_nan = rmse(concentration_ice[mask], Conc_analysts[mask]*1 + 0)
            
            mbe_ice = mbe(concentration_ice, Concentration_analysts)
            mbe_nan = mbe(concentration_ice[mask], Conc_analysts[mask])
            
            #appending to the different lists
            slope_linReg = LinReg_RADAnalysts.coef_[0][0]
            slope_linReg_Without = LinReg_Without.coef_[0][0]
            intercept_linReg = LinReg_RADAnalysts.intercept_[0]
            intercept_Without = LinReg_Without.intercept_[0]
            
            low_total.append(low)
            high_total.append(high)
            slope_linReg_total.append(slope_linReg)
            slope_linReg_ElimFar.append(slope_linReg_Without)
            intercept_linReg_total.append(intercept_linReg)
            intercept_linreg_elim.append(intercept_Without)
            rmse_linreg_total.append(rmse_linreg)
            rmse_linreg_elim.append(rmse_linreg_nan)
            mbe_linreg_total.append(mbe_ice)
            mbe_linreg_elim.append(mbe_nan)     
            slope_intercept0_total.append(LinReg_RADAnalysts_0slope)   
            slope_intercept0_elim.append(LinReg_Without_0slope)
            concentration_ice_total.append(concentration_ice)
            iterations.append(it)
            it+=1
    
    
    return low_total, high_total, slope_linReg_total, slope_linReg_ElimFar, \
        intercept_linReg_total, intercept_linreg_elim, rmse_linreg_total, rmse_linreg_elim, mbe_linreg_total, \
            mbe_linreg_elim, slope_intercept0_total, slope_intercept0_elim, concentration_ice_total, iterations

def detection_only(img_list, low, high, kernel, tot_area_radar, land_mask) : 
    
    # print('\n', 'Starting Canny edge algorithm')
    start_time =  time.time()
    edge_list_cv, gray_list = canny_edge_opencv(img_list, low, high, land_mask)
    # print('Time for canny with OpenCV : ', time.time() - start_time)

    # print('\n', 'Finding the contours')
    start_time = time.time()
    polygon_list_time_1, polygon_area_vid = find_contours(edge_list_cv, kernel)
    # print('Time for finding contours : ', time.time() - start_time)
    
    
    # print('Removing contours : ')
    start_time =  time.time()
    polygon_list_time, poly_area_time\
        = check_inside_contours(polygon_list_time_1, polygon_area_vid)
    # print('Time for canny with OpenCV : ', time.time() - start_time)
    
    # print('Calculating the Concentration')
    tot_area_vid_ice, tot_area_vid_ow, concentration_ice, concentration_ow \
        = find_polygon_area(poly_area_time, tot_area_radar)
        
    # plr.plot_edge_finding(gray_list, polygon_list_time, 0, './')

    return concentration_ice


def find_min_kernels(Parameters_list, shape) : 
    
    idx_min_3 = np.unravel_index(np.argmin(abs(Parameters_list[0]), axis=None), shape)
    idx_min_5 = np.unravel_index(np.argmin(abs(Parameters_list[1]), axis=None), shape)
    idx_min_7 = np.unravel_index(np.argmin(abs(Parameters_list[2]), axis=None), shape)
    idx_min_9 = np.unravel_index(np.argmin(abs(Parameters_list[3]), axis=None), shape)
    idx_min_11 = np.unravel_index(np.argmin(abs(Parameters_list[4]), axis=None), shape)
    idx_min_13 = np.unravel_index(np.argmin(abs(Parameters_list[5]), axis=None), shape)
    
    
    return idx_min_3, idx_min_5, idx_min_7, idx_min_9, idx_min_11, idx_min_13

def min_value_kernel(Parameters_list, idx_min_list) : 
    
    min_3 = Parameters_list[0][idx_min_list[0]]
    min_5 = Parameters_list[1][idx_min_list[1]]
    min_7 = Parameters_list[2][idx_min_list[2]]
    min_9 = Parameters_list[3][idx_min_list[3]]
    min_11 = Parameters_list[4][idx_min_list[4]]
    min_13 = Parameters_list[5][idx_min_list[5]]
    
    
    return min_3, min_5, min_7, min_9, min_11, min_13

@jit(nopython=True)
def is_inside_sm(polygon, point):
    length = len(polygon)-1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii<length:
        dy  = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy<0 or dy2<0:
                F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

                if point[0] > F: # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F: # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2==0 and (point[0]==polygon[jj][0] or (dy==0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0])<=0)):
                return 2

        ii = jj
        jj += 1

    #print 'intersections =', intersections
    return intersections & 1  

@njit(parallel=True)
def is_inside_sm_parallel(points, polygon):
    """
    This function is used to go trough a list of points and test if wether the points 
    are inside or not of the specified polygon. 
    
    This function uses the 

    Args:
        points (array, float): array containing the points
        polygon (array, float): array containing the edges of the polygon

    Returns:
        is_inside: (Array, bool): array saying if 1 (inside) or 0 (outside)
    """
    num_points= len(points)
    is_inside = np.zeros(num_points) 
    
    
    for i in numba.prange(num_points):
        is_inside[i] = is_inside_sm(polygon,points[i])
        
        
    return is_inside

def identify_ice_images(img, polygon_list_time, points_ocean, idx_ocean, shape_img = (900,900)):
                        
    polygon_list_img = polygon_list_time[img]
    #! here changed to nan
    ice_img = np.zeros(shape_img)
    
    for polygon in polygon_list_img:
        coordinates_array = np.asarray(polygon.exterior.coords)
        ice = is_inside_sm_parallel(points_ocean, coordinates_array)
        # ice = np.array([polygon.contains(points_ocean_geometry)])
        
        ice_img[idx_ocean] += ice
        
    ice_img[ice_img > 1] = 1
    ice_img = np.fliplr(ice_img)

    # print('Identification Done: ', img)
    return ice_img

@njit()
def coarse_grain(ice_img, window_size, latitude_radar, longitude_radar, shape_img = (900, 900)):
    
    number_window_i = shape_img[0]//window_size
    number_window_j = shape_img[1]//window_size
    window_step_i = np.arange(0, shape_img[0], window_size)
    window_step_j = np.arange(0, shape_img[1], window_size)  
    # print(window_step_j, window_step_i)       
    ice_modis_grid = np.zeros((number_window_i, number_window_j))
    lat_modis_grid = np.zeros_like(ice_modis_grid)
    lon_modis_grid = np.zeros_like(ice_modis_grid)

        #taking the mean on the new grid
    for ice_i, i in enumerate(window_step_j):
        for ice_j, j in enumerate(window_step_i):

            ice_modis_grid[ice_i, ice_j] = np.nanmean(ice_img[i:i+window_size, j:j+window_size])
            lat_modis_grid[ice_i, ice_j] = latitude_radar[i+window_size//2, j+window_size//2]
            lon_modis_grid[ice_i, ice_j] = longitude_radar[i+window_size//2, j+window_size//2]
            

    return ice_modis_grid, lat_modis_grid, lon_modis_grid

def identification_interpolation_modis(low, high, kernel, file_points, file_coordinates, Res,DataDirectory, MasksDirectory, SavingDirectory, ParamsDirectory, saving = False, interpolation = True, daily_correlation = False) : 
    
    """
    In order that the MODIS SIC and the radar SIC are oriented in the same direction, we need to fliplr the radar SIC. 
    This function is used to regrid the radar SIC on the same grid as the merged MODIS-AMSR2. It is build on the detection algorithm which
    goes as follow:

    
    We start by loading everything needed. Then, for every frame, we :
        1. Apply the Canny edge algofithm
        2. Find the sea ice contours 
        3. Remove the unwanted contours
        4. Calculate the areas 
        5. Regrid on the 1kmx1km grid
        
    We can also plot the contours for every frame but this slow down the code a lot.
    
    It's important to note that in order to run this code, you should use the optimized parameters found by the 
    optmisation step.
    
    Inputs:
        low: lower threshold for the canny edge detection (int)
        high: higher threshold for the canny edge detection (int)
        kernel: kernel used for the find contours
        DataDirectory: Directory where the frames are
        MasksDirectory: Directory where the masks are
        SavingDirectory: Directory where to save the data
        plotting: usually set to False, but it the contours need to be plotted, set to True

    """
    
    #Loading the different parameters

    print('HERE: Reading Mask')
    print(MasksDirectory)
    land_mask = np.load(MasksDirectory+'land_mask_3.npy', allow_pickle=True).item()['mask']
    idx_ocean = np.where(land_mask == 0) #finding the points where the ocean ice
    points_ocean = np.vstack(idx_ocean).T
    points_ocean_geometry = np.array([geometry.Point(point[0], point[1]) for point in points_ocean])
    
    #reading the indexes associated to the right grid
    
    #idx of the full grid around utq
    idx_grid_MOD = np.load(ParamsDirectory+file_points, allow_pickle=True).item()['idx_grid']
    #idx of the points of interest
    idx_point_MOD = np.load(ParamsDirectory+file_points, allow_pickle=True).item()['idx_points']
    idx_i_MOD = np.load(ParamsDirectory+file_points, allow_pickle=True).item()['idx_i']
    idx_j_MOD = np.load(ParamsDirectory+file_points, allow_pickle=True).item()['idx_j']
    
    #latitudes of the radar
    latitude_radar = np.fliplr(np.load('Latitude_Longitude_Modis/latlon_radar.npy', allow_pickle=True).item()['latitude'])
    #longitudes of the radar
    longitude_radar = np.fliplr(np.load('Latitude_Longitude_Modis/latlon_radar.npy', allow_pickle=True).item()['longtitude'])
    
    coordinates = nc.Dataset(file_coordinates, mode='r')
    try:
        
        latitudes_grid  = coordinates['lat'][:][idx_grid_MOD[:, 0], idx_grid_MOD[:, 1]]
        longitudes_grid = coordinates['lon'][:][idx_grid_MOD[:, 0], idx_grid_MOD[:, 1]]
    except:
        latitudes_grid  = coordinates['latitude'][:][idx_grid_MOD[:, 0], idx_grid_MOD[:, 1]]
        longitudes_grid = coordinates['longitude'][:][idx_grid_MOD[:, 0], idx_grid_MOD[:, 1]]
        
    xg, yg = np.reshape(latitudes_grid, (len(idx_j_MOD), len(idx_i_MOD))), np.reshape(longitudes_grid, (len(idx_j_MOD), len(idx_i_MOD)))
    
    #loop through the years
    for YearDir in os.listdir(DataDirectory) : 
        YearDir = str(YearDir)
        try:
            os.mkdir(SavingDirectory+YearDir)
            print("Folder %s created!" % SavingDirectory+YearDir)
        except FileExistsError:
            print("Folder exists")
        
        #loop through the months
        for MonthDir in os.listdir(DataDirectory+YearDir) : 
            MonthDir = str(MonthDir)
            
            #skip those months because no modis
            if MonthDir == '06' or MonthDir == '07' or MonthDir == '08' or MonthDir == '09':
                continue
            # if (YearDir == '2023' and MonthDir == '02'):
            #loop through the days
            for DayDir in os.listdir(DataDirectory+YearDir+'/'+MonthDir) : 
                DayDir = str(DayDir)
                print('Analysing: ', DayDir)
                #Create the saving directory
                if (DayDir == '20220414') or (DayDir == '20220413') or (DayDir == '20220415'):
                # if DayDir == '20230526':
                    saveDir = SavingDirectory+YearDir+'/'+MonthDir+'/'
                    try:
                        os.mkdir(saveDir)
                        print("Folder %s created!" % saveDir)
                    except FileExistsError:
                        print("Folder exists")
                
                    # #reading the images
                    img_list_day = vid.extract_img_folder(DataDirectory+YearDir+'/'+MonthDir+'/'+DayDir)
                    num_img_day = len(img_list_day)
                    i = 0
                    idx_remove = []
                    
                    for img in img_list_day:
                        shape = img.shape
                        
                        if shape != (900, 900):
                            idx_remove.append(i)
                        i+=1
                    
                    num_img_day -= len(idx_remove)
                    
                    if len(idx_remove) > 0:
                        
                        img_list_day = np.delete(img_list_day, idx_remove)
                        
                    if num_img_day == 0 :
                        ds1 = xr.Dataset(
                                data_vars={
                                    "interpolation": (("x", "y"), np.zeros_like(xg)*np.nan)
                                    # "individual_maps":(("t", "i", "j"), result)
                                    },
                                coords = {"latitude": (["x","y"], xg),
                                                        "longitude": (["x","y"], yg)
                                    }
                                )
                        ds1.to_netcdf(saveDir+"interpolated_"+DayDir+"_modis"+Res+"km_coarse.nc")
                        # ds1.to_netcdf("interpolated_"+DayDir+"_modis3km_coarse.nc")

                    else : 
                        
                        print('here')
                        print('\n', 'Starting Canny edge algorithm')
                        start_time =  time.time()
                        edge_list_cv, gray_list = canny_edge_opencv(img_list_day, low, high, land_mask)
                        print('Time for canny with OpenCV : ', time.time() - start_time)

                        print('\n', 'Finding the contours')
                        start_time = time.time()
                        polygon_list_time_1, polygon_area_vid = find_contours(edge_list_cv, kernel)
                        print('Time for finding contours : ', time.time() - start_time)
                        
                        print('Removing contours : ')
                        start_time =  time.time()
                        polygon_list_time, poly_area_time\
                            = check_inside_contours(polygon_list_time_1, polygon_area_vid)
                        print('Time for canny with OpenCV : ', time.time() - start_time)
                        print(len(polygon_list_time), num_img_day)
                        start_time = time.time()

                        # print('Now Plotting the run')
                        # plr.plot_edge_finding(gray_list, polygon_list_time, 0, './')
                        #--- Interpolation ---#       
                        print('Image Identification (0 and 1)')
                        
                        if interpolation:
                            result = []
                            for num in range(num_img_day):
                                ice_img = identify_ice_images(num, polygon_list_time, points_ocean, idx_ocean)
                                ice_img_coarse, lat_coarse, lon_coarse = coarse_grain(ice_img, 50, latitude_radar, longitude_radar)
                                print('Identification Done: ', num)
                                result.append(ice_img_coarse)      
                                # result.append(ice_img)            
                            
                            #taking the mean of the frames for the day
                            ice_mean_day = np.nanmean(result, axis=0)
                            points_radar = np.vstack((lat_coarse.flat, lon_coarse.flat)).T
                            # points_radar = np.vstack((latitude_radar.flat, longitude_radar.flat)).T
                            interpolated_ice_modis = griddata(points_radar, ice_mean_day.flat, (xg, yg), method = 'linear')
                            print('Time for total identification : ', time.time() - start_time)
                            
                            if saving :
                                ds1 = xr.Dataset(
                                    data_vars={
                                        "interpolation": (("x", "y"), interpolated_ice_modis)
                                        # "individual_maps":(("t", "i", "j"), result)
                                    },
                                    coords = {"latitude": (["x","y"], xg),
                                                            "longitude": (["x","y"], yg)
                                    }
                                )
                                ds1.to_netcdf(saveDir+"interpolated_"+DayDir+"_modis3km.nc")
                                # ds1.to_netcdf("interpolated_"+DayDir+"_modis3km_coarse.nc")
                        
                            if daily_correlation:
                                interpolated_ice_modis_day = []
                                for num in range(num_img_day):
                                    ice_img = identify_ice_images(num, polygon_list_time, points_ocean, idx_ocean)
                                    ice_img_coarse, lat_coarse, lon_coarse = coarse_grain(ice_img, 50, latitude_radar, longitude_radar)
            
                                    points_radar = np.vstack((lat_coarse.flat, lon_coarse.flat)).T
                                    # points_radar = np.vstack((latitude_radar.flat, longitude_radar.flat)).T
                                    interpolated_ice_modis = griddata(points_radar, ice_img_coarse.flat, (xg, yg), method = 'linear')
                                    interpolated_ice_modis_day.append(interpolated_ice_modis)
                                print('Time for total identification : ', time.time() - start_time)
                                    
                                ds1 = xr.Dataset(
                                    data_vars={
                                        "interpolation_4min": (("t", "x", "y"), interpolated_ice_modis_day)
                                        # "individual_maps":(("t", "i", "j"), result)
                                    },
                                    coords = {"latitude": (["x","y"], xg),
                                            "longitude": (["x","y"], yg)
                                            # "time" : (["t"], num_img_day)
                                    }
                                )
                                ds1.to_netcdf(saveDir+"interpolated_"+DayDir+"_4min_modis"+Res+"km.nc")
                                
                                
                            
    return polygon_list_time

def interpolation_modis(ModisDir, Res, file_coordinates_1km, file_points_1km, file_points_higher, files_coordinates_higherRes, ParamsDirectory, SavingDirectory, saving = True ):
    
    idx_grid_MOD_1km = np.load(ParamsDirectory+file_points_1km, allow_pickle=True).item()['idx_grid']
    idx_point_MOD_1km = np.load(ParamsDirectory+file_points_1km, allow_pickle=True).item()['idx_points']
    idx_i_MOD_1km = np.load(ParamsDirectory+file_points_1km, allow_pickle=True).item()['idx_i']
    idx_j_MOD_1km = np.load(ParamsDirectory+file_points_1km, allow_pickle=True).item()['idx_j']
    
    
    idx_i_MOD = np.load(ParamsDirectory+file_points_higher, allow_pickle=True).item()['idx_i']
    idx_j_MOD = np.load(ParamsDirectory+file_points_higher, allow_pickle=True).item()['idx_j']
    idx_grid_MOD_higher = np.load(ParamsDirectory+file_points_higher, allow_pickle=True).item()['idx_grid']
    
    coordinates = nc.Dataset(file_coordinates_1km, mode='r')
    latitudes_1km  = coordinates['lat'][:][idx_grid_MOD_1km[:, 0], idx_grid_MOD_1km[:, 1]]
    longitudes_1km = coordinates['lon'][:][idx_grid_MOD_1km[:, 0], idx_grid_MOD_1km[:, 1]]

    coordinates_interpolation = nc.Dataset(files_coordinates_higherRes, mode='r')
    latitudes_grid_highRes  = coordinates_interpolation['latitude'][:][idx_grid_MOD_higher[:, 0], idx_grid_MOD_higher[:, 1]]
    longitudes_grid_highRes = coordinates_interpolation['longitude'][:][idx_grid_MOD_higher[:, 0], idx_grid_MOD_higher[:, 1]]
        
    xg, yg = np.reshape(latitudes_grid_highRes, (len(idx_j_MOD), len(idx_i_MOD))), np.reshape(longitudes_grid_highRes, (len(idx_j_MOD), len(idx_i_MOD)))
    
    for YearDir in ['2022', '2023'] : 
        YearDir = str(YearDir)
        try:
            os.mkdir(SavingDirectory+YearDir)
            print("Folder %s created!" % SavingDirectory+YearDir)
        except FileExistsError:
            print("Folder exists")
        #loop through the months
        monthdir = sorted([int(month) for month in os.listdir(ModisDir+YearDir)])
        for MonthDir in monthdir : 
            if MonthDir < 10:
                MonthDir = '0'+str(MonthDir)
            else:
                MonthDir = str(MonthDir)

            month = int(MonthDir)
            # print(ModisDir+YearDir+'/'+MonthDir)
            for files in os.listdir(ModisDir+YearDir+'/'+MonthDir):
                # print(files)
                day = files[-11:-3]
                # MonthDir = MonthDir+'/'
                
                print('Analysing Day:', day)
                saveDir = SavingDirectory+YearDir+'/'+MonthDir+'/'
                try:
                    os.mkdir(saveDir)
                    print("Folder %s created!" % saveDir)
                except FileExistsError:
                    print("Folder exists")
                    
                try:
                    modis_sic_ds = xr.open_dataset(ModisSICDir+YearDir+'/'+MonthDir+'/'+'sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_'+day+'.nc')
                    modis_sic = np.flipud(modis_sic_ds.sic_merged.data).astype(float)/100
                    modis_sic[np.where(modis_sic == np.amax(modis_sic))] = np.nan

                except:
                    modis_sic = np.zeros((4750, 5560))*np.nan
                    
                SIC_modis = np.zeros_like(modis_sic)*np.nan
                SIC_modis[idx_point_MOD_1km[:, 0], idx_point_MOD_1km[:, 1]] = modis_sic[idx_point_MOD_1km[:, 0], idx_point_MOD_1km[:, 1]]
                
                SIC_modis_modisgrid = SIC_modis[idx_grid_MOD_1km[:, 0], idx_grid_MOD_1km[:, 1]]
            
                points_modis = np.vstack((latitudes_1km, longitudes_1km)).T
                interpolated_ice_modis = griddata(points_modis, SIC_modis_modisgrid, (xg, yg), method = 'linear')
                # print(interpolated_ice_modis)
                # print('Time for total identification : ', time.time() - start_time)
                
                if saving :
                    ds1 = xr.Dataset(
                        data_vars={
                            "interpolation": (("x", "y"), interpolated_ice_modis)
                            # "individual_maps":(("t", "i", "j"), result)
                        },
                        coords = {"latitude": (["x","y"], xg),
                                                "longitude": (["x","y"], yg)
                        }
                    )
                    ds1.to_netcdf(saveDir+"interpolated_"+day+"_modis"+Res+".nc")
                    # ds1.to_netcdf("interpolated_"+DayDir+"_modis3km_coarse.nc")
    
    
    return

def extract_timesseries_interpolation(RadIntepDir, ModisSICDir, ParamsDirectory, file_points, resolution):
    
    idx_grid_MOD = np.load(ParamsDirectory+file_points, allow_pickle=True).item()['idx_grid']
    #idx of the points of interest
    idx_point_MOD = np.load(ParamsDirectory+file_points, allow_pickle=True).item()['idx_points']
    idx_i_MOD = np.load(ParamsDirectory+file_points, allow_pickle=True).item()['idx_i']
    idx_j_MOD = np.load(ParamsDirectory+file_points, allow_pickle=True).item()['idx_j']
    
    
    SIC_modis_modisgrid_total, SIC_radar_modisgrid_total = [], []
    months_total, years_total = [], []
    rmse_total, coerfcoeff_total= [], []

    # for YearDir in os.listdir(sorted(RadIntepDir)) : 
    for YearDir in ['2022', '2023'] : 
            YearDir = str(YearDir)
            #loop through the months
            monthdir = sorted([int(month) for month in os.listdir(RadIntepDir+YearDir)])
            for MonthDir in monthdir : 
                if MonthDir < 10:
                    MonthDir = '0'+str(MonthDir)
                else:
                    MonthDir = str(MonthDir)

                month = int(MonthDir)

                for files in os.listdir(RadIntepDir+YearDir+'/'+MonthDir):
                    day = files[13:21]
                    
                    print('Analysing Day:', day)
                    # if day == '20220210':
                    
                    
                    if resolution == 1:
                        try:
                            modis_sic_ds = xr.open_dataset(ModisSICDir+YearDir+'/'+MonthDir+'/'+'sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_'+day+'.nc')
                            modis_sic = np.flipud(modis_sic_ds.sic_merged.data).astype(float)/100
                            modis_sic[np.where(modis_sic == np.amax(modis_sic))] = np.nan
                        except:
                            modis_sic = np.zeros((4750, 5560))*np.nan
                        
                        interpolated_radar_sic_ds = xr.open_dataset(RadIntepDir+YearDir+'/'+MonthDir+'/'+files)
                        interpolated_radar_sic = interpolated_radar_sic_ds.interpolation.data
                        
                        
                        SIC_modis = np.zeros_like(modis_sic)*np.nan
                        SIC_modis[idx_point_MOD[:, 0], idx_point_MOD[:, 1]] = modis_sic[idx_point_MOD[:, 0], idx_point_MOD[:, 1]]
                    
                        SIC_modis_modisgrid = SIC_modis[idx_grid_MOD[:, 0], idx_grid_MOD[:, 1]]
                        SIC_modis_modisgrid = np.reshape(SIC_modis_modisgrid, (len(idx_j_MOD), len(idx_i_MOD)))

                    if resolution > 1:
                        try:
                            modis_sic_ds = xr.open_dataset(RadIntepDir+YearDir+'/'+MonthDir+'/'+"interpolated_"+day+"_modis_radar"+str(resolution)+".nc")
                            SIC_modis_modisgrid = modis_sic_ds.coarse_modis.data
                            interpolated_radar_sic = modis_sic_ds.coarse_radar.data
                            # print(SIC_modis_modisgrid)
                        except:
                            SIC_modis_modisgrid = np.zeros_like(interpolated_radar_sic)*np.nan
                    
                    # SIC_radar_modisgrid = np.zeros_like(SIC_modis)*np.nan
                    # SIC_radar_modisgrid[idx_grid_MOD[:, 0], idx_grid_MOD[:, 1]] = interpolated_radar_sic.flatten()

                    
                    idx_regridded_SIC_modis = ~np.isnan(SIC_modis_modisgrid)
                    
                    SIC_modis_modisgrid_points = SIC_modis_modisgrid[idx_regridded_SIC_modis]
                    SIC_radar_modisgrid_points = interpolated_radar_sic[idx_regridded_SIC_modis]
                    
                    rmse_day = rmse(SIC_modis_modisgrid_points ,SIC_radar_modisgrid_points)
                    #! calculate this with the nans
                    coerfcoeff_day = np.ma.corrcoef(np.ma.masked_invalid(SIC_modis_modisgrid_points), np.ma.masked_invalid(SIC_radar_modisgrid_points))[1, 0]
    
                        
                    SIC_modis_modisgrid_total.extend(SIC_modis_modisgrid_points)
                    SIC_radar_modisgrid_total.extend(SIC_radar_modisgrid_points)
                    
                    rmse_day_list = [rmse_day]*len(SIC_radar_modisgrid_points)
                    rmse_total.extend(rmse_day_list)
                    coerfcoeff_total.append(coerfcoeff_day)
                    
                    month_list = [month]*len(SIC_radar_modisgrid_points)
                    months_total.extend(month_list)
                    
                    year_list = [int(YearDir)]*len(SIC_radar_modisgrid_points)
                    years_total.extend(year_list)
    
    return np.asarray(SIC_modis_modisgrid_total), np.asarray(SIC_radar_modisgrid_total), np.asarray(rmse_total), np.asarray(coerfcoeff_total), np.asarray(months_total), np.asarray(years_total)

def interpolation_coarsegrain(resolution, file_points, Savedir, ModisSICDir, RadIntepDir, ParamsDirectory, saving = True):
    
    idx_grid_MOD = np.load(ParamsDirectory+file_points, allow_pickle=True).item()['idx_grid']
    #idx of the points of interest
    idx_point_MOD = np.load(ParamsDirectory+file_points, allow_pickle=True).item()['idx_points']
    idx_i_MOD = np.load(ParamsDirectory+file_points, allow_pickle=True).item()['idx_i']
    idx_j_MOD = np.load(ParamsDirectory+file_points, allow_pickle=True).item()['idx_j']
    
    # for YearDir in os.listdir(sorted(RadIntepDir)) : 
    for YearDir in ['2022', '2023'] : 
            YearDir = str(YearDir)
            try:
                os.mkdir(Savedir+YearDir)
                print("Folder %s created!" % Savedir+YearDir)
            except FileExistsError:
                print("Folder exists")
            #loop through the months
            monthdir = sorted([int(month) for month in os.listdir(RadIntepDir+YearDir)])
            for MonthDir in monthdir : 
                if MonthDir < 10:
                    MonthDir = '0'+str(MonthDir)
                else:
                    MonthDir = str(MonthDir)

                for files in os.listdir(RadIntepDir+YearDir+'/'+MonthDir):
                    day = files[13:21]
                    saveDir = Savedir+YearDir+'/'+MonthDir+'/'
                    
                    try:
                        os.mkdir(saveDir)
                        print("Folder %s created!" % saveDir)
                    except FileExistsError:
                        print("Folder exists")
                        
                    print('Analysing Day:', day)

                    interpolated_radar_sic_ds = xr.open_dataset(RadIntepDir+YearDir+'/'+MonthDir+'/'+files)
                    interpolated_radar_sic = interpolated_radar_sic_ds.interpolation.data
                    latitudes_radar = interpolated_radar_sic_ds.latitude.data
                    longitudes_radar = interpolated_radar_sic_ds.longitude.data
                    
                    try:
                        modis_sic_ds = xr.open_dataset(ModisSICDir+YearDir+'/'+MonthDir+'/'+'sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_'+day+'.nc')
                        modis_sic = np.flipud(modis_sic_ds.sic_merged.data).astype(float)/100
                        modis_sic[np.where(modis_sic == np.amax(modis_sic))] = np.nan

                    except:
                        modis_sic = np.zeros((4750, 5560))*np.nan

                    SIC_modis = np.zeros_like(modis_sic)*np.nan
                    SIC_modis[idx_point_MOD[:, 0], idx_point_MOD[:, 1]] = modis_sic[idx_point_MOD[:, 0], idx_point_MOD[:, 1]]
                
                    SIC_modis_modisgrid = SIC_modis[idx_grid_MOD[:, 0], idx_grid_MOD[:, 1]]
                    SIC_modis_modisgrid = np.reshape(SIC_modis_modisgrid, (len(idx_j_MOD), len(idx_i_MOD)))

                    radar_coarse, latitude_coarse, longitudes_coarse = coarse_grain(interpolated_radar_sic, resolution, latitudes_radar,  longitudes_radar, np.shape(interpolated_radar_sic))
                    modis_coarse,latitude_coarse, longitudes_coarse = coarse_grain(SIC_modis_modisgrid, resolution, latitudes_radar,  longitudes_radar, np.shape(interpolated_radar_sic))
                    if saving :
                        ds1 = xr.Dataset(
                            data_vars={
                                "coarse_radar": (("x", "y"), radar_coarse), 
                                "coarse_modis": (("x", "y"), modis_coarse)
                                # "individual_maps":(("t", "i", "j"), result)
                            },
                            coords = {"latitude": (["x","y"], latitude_coarse),
                                                    "longitude": (["x","y"], longitudes_coarse)
                            }
                        )
                        ds1.to_netcdf(saveDir+"interpolated_"+day+"_modis_radar"+str(resolution)+".nc")

    return radar_coarse, modis_coarse

def covariance(x, y) : 
    """
    This function calculates the covariance between two arrays

    Inputs:
        x (array): array 1
        y (array): array 2

    Returns the covariance
    """
    return np.nanmean(x*y) - np.nanmean(x)*np.nanmean(y)

def correlation(x, y, cov) : 
    
    """
    This function calculate the correlation coefficient between two variables

    Inputs:
        x (array): array 1
        y (array): array 2
        cov (float): covariance of the two array
    
    Returns:
        correlation (float): correlation coefficient between x and y
    """
    
    #calculating the covariance
    x_x = covariance(x, x)
    y_y = covariance(y, y)
    
    #calculating the correlation coefficient
    correlation = cov/(x_x**(1/2) * y_y**(1/2))
    
    return correlation


