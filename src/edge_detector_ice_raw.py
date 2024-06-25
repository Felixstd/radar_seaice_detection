# %%
import warnings, time, os, sys
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import matplotlib as mpl
import cv2 as cv 
import netCDF4 as nc
import extract_video as vid
import plot_radar as plr
import xarray as xr
import numba
import cmocean


from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression
from shapely import geometry
from shapely.validation import make_valid
from numba import njit, jit
import cartopy.crs as ccrs
# import SIC_SAR
import cartopy.feature as cfeature

"""
0 -> Felix
1 -> Florence Beaudry
2 -> Lizz 
3 -> Bruno
4 -> Ben Ward
5 -> Frederique L.
6 -> Thomas 1
7 -> Antoine
8 -> Thomas 2
9 -> Stephanie
"""

original_stdout = sys.stdout
plt.style.use('FigureStyle.mplstyle')

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
        SIC_calculated -> calculated SIC
        SIC_observations -> observed SIC, in our case it's the analysts' one. 
        
    Output : 
        error -> the RMSE
    '''
    
    error = np.sqrt(np.nanmean((SIC_calculated - SIC_observations)**2))
    
    return error

def mbe(SIC_calculated, SIC_observations) : 
    
    """
    Function used to calculate the Mean Bias error between two datasets. 
    It's used to compare two datasets. It tells us our far one is from the other
    
    RMSE = ((sum((X_i - Y_i))/N)
    
    Input :
        SIC_calculated -> calculated SIC
        SIC_observations -> observed SIC, in our case it's the analysts' one. 
        
    Output : 
        error -> the MBE
    """
    
    error = np.nanmean(SIC_calculated - SIC_observations)
    return error

def canny_edge_opencv(img_list, low, high, land_mask, fill_area = False) : 
    
    """
    Function that applies the canny edge algorithm to a list of radar images. 
    
    To the the value for the land to be, we use the value of two pixels taken in the ocean. 
    If the either one of the two is lower on the gray scale, the land will be blacker. 
    
    Inputs : 
        img_list -> list of images
        low -> lower threshold for the detection algorithm (int)
        high -> highest threshold for the detection algorithm (int)
        land_mask -> Binary mask differientiating the land from the ocean (1 : land, 0 : ocean)
        circle -> coordinates of the circle marking the radar rage (specified in the parameter file) (ints)
        difference_circle -> list containing the off sets for the circle
        fill_area -> Set to true when computing the complete field of view of the radar, otherwise set to false. 
        
    Returns:
        edges_list -> List of canny edge outputs corresponding to the img_list
        gray_list -> same as img_list but the images are now on the gray scale
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
        
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret,bin = cv.threshold(img,5,255,cv.THRESH_BINARY)

        kernel = np.ones((7,7),np.uint8)
        erosion = cv.erode(bin,kernel,iterations = 1)
        

        #Computing the contours
        result_canny = cv.Canny(img, low, high)
        
        result = cv.bitwise_and(result_canny,result_canny,mask = erosion)
        
        # edges_img[result] = 0
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
        img_list -> list containing the edges of every images (output of the canny edge algorithm)
        kernel -> kernel used for the find contours algorithm. 
        recog -> set to True when finding the complete radar range. 
        
    Returns:
        polygon_img_list -> lists of lists containing the Polygons representing the contours in each images in img_list 
        polygon_area_list -> lists of lists containing the Polygon's areas. 
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
            kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5)) #! here
        else : 
            kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel) #! here

        dilate = cv.dilate(img, kernel_ellipse, iterations=1) #dilating the images to get better resolution

        #finding the contours
        (cnts, _) = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #! changed here
        
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
        polygon_list -> list of polygons
        
    Returns:
        polygon_list_valid -> lists of valid polygons
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
        polygon_list -> list of polygons found by the find_contours and canny_edge()
        polygon_area_list -> lists of areas associated to each polygons
        
    Returns:
        polygon_list_checked -> updated list of polygons found by the find_contours and canny_edge()
        polygon_area_list_checked -> updated list of areas
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
            # contains = poly.contains_properly(polygons_v)
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
        polygon_area_vid: lists of the polygons areas
        total_radar_range: area of the total area of the radar range (float)

    Returns:
        tot_area_ice_vid: lists for every images of the total ice area
        tot_area_ow_vid: lists for every images of the total open water area
        concentration_ice: lists for every images of SIC
        concentration_ow: lists for every images of Open water Concentration
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
    land_mask = np.load(MasksDirectory+'land_mask_3.npy', allow_pickle=True).item()['mask']
    
    print('Calulating the total area of the radar range')
    #! changed here
    img_tot_area = '/storage/fstdenis/Barrow_RADAR/RAW_Data/2022/02/20220219/UAFIceRadar_20220219_014400_crop_geo.tif'
    # img_tot_area = '/storage/fstdenis/Barrow_RADAR/AlreadyAnalysed/2022/02/20220219/UAFIceRadar_20220219_014400_crop_geo.tif' 
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
        
    plr.plot_edge_finding(gray_list, polygon_list_time, 0, './')

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

def identification_interpolation_modis(low, high, kernel, file_points, file_coordinates,\
    DataDirectory, MasksDirectory, SavingDirectory, ParamsDirectory, saving = False, interpolation = True, daily_correlation = False) : 
    
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

    print('Reading Mask')
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
                # if DayDir == '20220309':
                if DayDir == '20230526':
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
                        ds1.to_netcdf(saveDir+"interpolated_"+DayDir+"_modis3km_coarse.nc")
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

                        print('Now Plotting the run')
                        plr.plot_edge_finding(gray_list, polygon_list_time, 0, './')
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
                                ds1.to_netcdf(saveDir+"interpolated_"+DayDir+"_modis3km.nc")
                                
                                
                            
    return gray_list, polygon_list_time

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

def plot_SIC_interpolation(SIC_radar_modisgrid, SIC_modis, namefig):
        
        crs_epsg_MOD = ccrs.NorthPolarStereo(central_longitude = SIC_SAR.lon_0, true_scale_latitude = SIC_SAR.lat_ts)
        color_min = cmocean.cm.ice(0)
        fig = plt.figure(figsize = (12, 6))

        ax1 = fig.add_subplot(121, projection = crs_epsg_MOD)
        SIC_radar_modisgrid[np.where(SIC_radar_modisgrid == 0)] = np.nan
        cs = ax1.pcolormesh(SIC_SAR.SIC_MOD_l[1], SIC_SAR.SIC_MOD_l[2], SIC_radar_modisgrid[:-1, :-1], 
                    cmap=cmocean.cm.ice,  transform = crs_epsg_MOD, vmin = 0, vmax = 1)
        # #plotting Utqiagvik location
        ax1.plot(SIC_SAR.lon_utq, SIC_SAR.lat_utq, transform = ccrs.PlateCarree(), color = 'r', \
        marker = '*', label = 'Utqiagvik', alpha = 0.5)
        ax1.plot(*SIC_SAR.polygon_radar_fieldview.exterior.xy, color = 'r', \
        transform = ccrs.PlateCarree(), linewidth = 2, label = 'Radar')                                  
        ax1.set_extent([-157.5, -156, 71.2, 71.4],ccrs.PlateCarree())
        ax1.add_feature(cfeature.LAND, color = 'gray')

        ax1.set_title('Radar')
        ax1.coastlines()
        ax1.set_facecolor(color_min)
        # trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)

        # ax2 = fig.add_subplot(122, projection = crs_epsg_MOD)
        # SIC_radar_modisgrid_coarse[np.where(SIC_radar_modisgrid_coarse == 0)] = np.nan
        # cs = ax2.pcolormesh(SIC_SAR.SIC_MOD_l[1], SIC_SAR.SIC_MOD_l[2], SIC_radar_modisgrid_coarse[:-1, :-1], 
        #             cmap=cmocean.cm.ice,  transform = crs_epsg_MOD, vmin = 0, vmax = 1)
        # # #plotting Utqiagvik location
        # ax2.plot(SIC_SAR.lon_utq, SIC_SAR.lat_utq, transform = ccrs.PlateCarree(), color = 'r', \
        # marker = '*', label = 'Utqiagvik', alpha = 0.5)
        # ax2.plot(*SIC_SAR.polygon_radar_fieldview.exterior.xy, color = 'r', \
        # transform = ccrs.PlateCarree(), linewidth = 2, label = 'Radar')                                  
        # ax2.set_extent([-157.5, -156, 71.2, 71.4],ccrs.PlateCarree())
        # ax2.add_feature(cfeature.LAND, color = 'gray')

        # ax2.set_title('Radar Coarse')
        # ax2.coastlines()
        # ax2.set_facecolor(color_min)
        # trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)


        ax2 = fig.add_subplot(122, projection = crs_epsg_MOD, sharex = ax1, sharey = ax1)
        axes = [ax1, ax2]
        cs = ax2.pcolormesh(SIC_SAR.SIC_MOD_l[1], SIC_SAR.SIC_MOD_l[2], SIC_modis[:-1, :-1], 
                    cmap=cmocean.cm.ice,  transform = crs_epsg_MOD, vmin = 0, vmax = 1)#, edgecolors = 'gray')
        fig.colorbar(cs, ax=axes, label = 'SIC')
        # #plotting Utqiagvik location
        ax2.plot(SIC_SAR.lon_utq, SIC_SAR.lat_utq, transform = ccrs.PlateCarree(), color = 'r', \
        marker = '*', label = 'Utqiagvik', alpha = 0.5)
        ax2.plot(*SIC_SAR.polygon_radar_fieldview.exterior.xy, color = 'r', \
        transform = ccrs.PlateCarree(), linewidth = 2, label = 'Radar')                                  
        ax2.set_extent([-157.5, -156, 71.2, 71.4],ccrs.PlateCarree())
        ax2.add_feature(cfeature.LAND, color = 'gray')

        ax2.set_title('Merged MODIS-AMSR2')
        ax2.coastlines()
        ax2.set_facecolor(color_min)
        fig.tight_layout()
        plt.savefig(namefig, dpi = 500, bbox_inches = 'tight')   

def plot_fig_interpolation(SIC_radar_modisgrid_total, SIC_modis_modisgrid_total,rmse_total, months_total, resolution, eps = 0.3):
    
    idx_rmse_small = np.where(rmse_total < eps)[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 6), sharex = True, sharey = True, layout = 'constrained')
    axes = [ax1, ax2]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid()
        ax.set_aspect('equal')

    ax1.scatter(SIC_radar_modisgrid_total, SIC_modis_modisgrid_total, color = 'b', s = 5)
    ax1.set_title('Complete Time Series')


    ax2.scatter(SIC_radar_modisgrid_total[idx_rmse_small], SIC_modis_modisgrid_total[idx_rmse_small], color = 'b', s = 5)
    ax2.set_title('RMSE <  {}'.format(eps))

    fig.supxlabel('radar SIC')
    fig.supylabel('merged MODIS-AMSR2 SIC')
    fig.suptitle('{}km Resolution'.format(resolution))
    plt.savefig('radar_modis_{}km.png'.format(resolution), dpi = 500, bbox_inches = 'tight')
    
    months = [1, 2, 3, 4, 5, 10, 11, 12]
    fig, axes = plt.subplots(2, 4, sharex = True, sharey = True, figsize = (12, 6))
    count = 0
    for month in months:
        ax = axes.flatten()[count]
        idx_month = np.where(months_total == month)[0]

        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid()
        sc = ax.scatter(SIC_radar_modisgrid_total[idx_month], SIC_modis_modisgrid_total[idx_month], color = 'b', s = 3)
        ax.set_aspect('equal')
        ax.set_title('Month: {}'.format(month))
        count += 1
    # fig.tight_layout()
    fig.supxlabel('radar SIC')
    fig.supylabel('merged MODIS-AMSR2 SIC')
    fig.suptitle('{}km Resolution'.format(resolution))
    plt.savefig('radar_modis_{}km_monthly.png'.format(resolution), dpi = 500, bbox_inches = 'tight')

def histogram(rmse, num_bins, resolution): 
        
    plt.figure(figsize = (8, 8))
    bins=np.linspace(0,1,num_bins) 
    centers=0.5*(bins[1:]+bins[:-1])

    plt.hist(rmse, bins=bins, color = 'blue')
    plt.xlabel('RMSE')
    plt.ylabel('Counts')
    plt.title('{}km Resolution'.format(resolution))
    
    plt.savefig('pdf_rmse_{}km.png'.format(resolution), dpi = 500, bbox_inches = 'tight')

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
    print(x_x, y_y)
    
    #calculating the correlation coefficient
    correlation = cov/(x_x**(1/2) * y_y**(1/2))
    
    return correlation

savedir = '/storage/fstdenis/Barrow_RADAR/IdentTypeIce/edge_detection/run05/'
viddir = '/storage/fstdenis/Barrow_RADAR/Data/RADAR_2014/2014_0311_0320.mp4'
# viddir = '/storage/fstdenis/Barrow_RADAR/RADAR_2023/Utq_20230504to20230506.mp4'

masksDir = "/storage/fstdenis/Barrow_RADAR/ICE_RADAR_MISC/IdentTypeIce/edge_detection/masks/"
DataDir = '/storage/fstdenis/Barrow_RADAR/RAW_Data/'
# DataDir = '/storage/fstdenis/Barrow_RADAR/RAW_Data_Summer/'
AreaDirectory = '/storage/fstdenis/Barrow_RADAR/saved_run/RADAR/Area_Vid/'
# SavingDirectory = '/storage/fstdenis/Barrow_RADAR/saved_run/RADAR/saved_TimeSeries_SIC/'
SavingDirectory = '/storage/fstdenis/Barrow_RADAR/saved_run/RADAR/saved_Summer/'
ParamsDirectory = '/storage/fstdenis/Barrow_RADAR/IdentTypeIce/edge_detection/params/'

unc = False
alg_plot = False
mask = False
canny = False


Datadir = "/storage/fstdenis/Barrow_RADAR/RAW_Data/"
SaveDir = "/storage/fstdenis/Barrow_RADAR/Data_Interpolation_Modis_Coarse/"
RadIntepDir = '/storage/fstdenis/Barrow_RADAR/Data_Interpolation/Data_Interpolation_Modis_Coarse/'
ModisSICDir = '/storage/fstdenis/SIC_MODIS_AMSR2_Daily/'
ParamsDir_MODIS = '/storage/fstdenis/Barrow_RADAR/saved_run/SIC_MODIS/params/'

filespoints_1km, coordinates_1km = 'idx_polygon_MODIS_4.npy', './coordinates_npstere_1km_arctic.nc'

filespoints_3km, coordinates_3km = 'idx_3kmresolution.npy', './NSIDC_EASE/NSIDC0772_LatLon_EASE2_N03km_v1.0.nc'

savedir_radarmodis_2km = '/storage/fstdenis/Barrow_RADAR/Interpolation_Radar_Modis/Radar_Modis_2km/'
savedir_radarmodis_3km = '/storage/fstdenis/Barrow_RADAR/Interpolation_Radar_Modis/Radar_Modis_3km/'
savedir_radarmodis_4km = '/storage/fstdenis/Barrow_RADAR/Interpolation_Radar_Modis/Radar_Modis_4km/'
savedir_radarmodis_5km = '/storage/fstdenis/Barrow_RADAR/Interpolation_Radar_Modis/Radar_Modis_5km/'
savedir_radarmodis_6km = '/storage/fstdenis/Barrow_RADAR/Interpolation_Radar_Modis/Radar_Modis_6km/'
savedir_radarmodis_7km = '/storage/fstdenis/Barrow_RADAR/Interpolation_Radar_Modis/Radar_Modis_7km/'
savedir_radarmodis_8km = '/storage/fstdenis/Barrow_RADAR/Interpolation_Radar_Modis/Radar_Modis_8km/'
savedir_radarmodis_9km = '/storage/fstdenis/Barrow_RADAR/Interpolation_Radar_Modis/Radar_Modis_9km/'
savedir_radarmodis_10km = '/storage/fstdenis/Barrow_RADAR/Interpolation_Radar_Modis/Radar_Modis_10km/'

savedir_3km_radar = '/storage/fstdenis/Barrow_RADAR/Data_Interpolation_Modis_3km/'
savedir_daily_interp = '/storage/fstdenis/Barrow_RADAR/Daily_4min_Interpolation_Radar/'
data_modis_daily = '/storage/fstdenis/Barrow_RADAR/Modis_Visible_Comparison/'
Analysis=0
#%%
if Analysis:
    tiff_file_modis = "./modis_imagery_utq/20220325/"
    tiff_file_radar_2022_03_25 = '/storage/fstdenis/Barrow_RADAR/Modis_Visible_Comparison/2022/03/20220325/'
    # img_radar_2022_03_25 = vid.extract_img_folder(tiff_file_radar_2022_03_25)[0]
    # img_modis_2022_03_25 = vid.extract_img_folder(tiff_file_modis)[0]

    tiff_20220312 = "./modis_imagery_utq/20220312/"
    tiff_file_radar_2022_03_12 = '/storage/fstdenis/Barrow_RADAR/Modis_Visible_Comparison/2022/03/20230312/'
    # img_radar_2022_03_12 = vid.extract_img_folder(tiff_file_radar_2022_03_12)[0]
    # img_modis_2022_03_12 = vid.extract_img_folder(tiff_20220312)[0]

    tiff_20220309 = "./modis_imagery_utq/20220309/"
    tiff_file_radar_2022_03_09 = '/storage/fstdenis/Barrow_RADAR/Modis_Visible_Comparison/2022/03/20220309/'
    img_radar_2022_03_09 = vid.extract_img_folder(tiff_file_radar_2022_03_09)[0]
    img_modis_2022_03_09 = vid.extract_img_folder(tiff_20220309)[0]


    tiff_20220417 = "./modis_imagery_utq/20220417/"
    tiff_file_radar_2022_04_17 = '/storage/fstdenis/Barrow_RADAR/Modis_Visible_Comparison/2022/04/20220417/'
    img_radar_2022_04_17 = vid.extract_img_folder(tiff_file_radar_2022_04_17)[0]
    img_modis_2022_04_17 = vid.extract_img_folder(tiff_20220417)[0]

    tiff_20230526 = "./modis_imagery_utq/20230526/"
    tiff_file_radar_2023_05_26 = '/storage/fstdenis/Barrow_RADAR/Modis_Visible_Comparison/2023/05/20230526/'
    img_radar_2023_05_26 = vid.extract_img_folder(tiff_file_radar_2023_05_26)
    img_modis_2023_05_26 = vid.extract_img_folder(tiff_20230526)[0]

    # latlon_2022328 = xr.open_dataset("latitude_modis_img_20220328.nc")
    latlon_2022325 = xr.open_dataset("Latitude_Longitude_Modis/latitude_modis_img_20220325.nc")
    latlon_2022312 = xr.open_dataset("Latitude_Longitude_Modis/latitude_modis_img_20220312.nc")
    latlon_2022309 = xr.open_dataset("Latitude_Longitude_Modis/latitude_modis_img_20220309.nc")
    latlon_202230417 = xr.open_dataset("Latitude_Longitude_Modis/latitude_modis_img_20220417.nc")

    latlon_202330526 = xr.open_dataset("Latitude_Longitude_Modis/latitude_modis_img_20230526.nc")

    lat_img = latlon_2022325.latitude.data
    lon_img = latlon_2022325.longitude.data

    lat_img_2022312 = latlon_2022312.latitude.data
    lon_img_2022312 = latlon_2022312.longitude.data

    lat_img_2022309 = latlon_2022309.latitude.data
    lon_img_2022309 = latlon_2022309.longitude.data

    lat_img_2022317 = latlon_202230417.latitude.data
    lon_img_2022317 = latlon_202230417.longitude.data

    lat_img_20230526 = latlon_202330526.latitude.data
    lon_img_20230526 = latlon_202330526.longitude.data


    gray_img_2022_03_25, polygons_list_2023_05_26 = identification_interpolation_modis(4, 82, (11,11), filespoints_1km, \
        coordinates_1km, data_modis_daily, masksDir, './', ParamsDir_MODIS, saving = False)

    crs_epsg_MOD = ccrs.NorthPolarStereo(central_longitude = -45, true_scale_latitude = 70)
    color_min = cmocean.cm.ice(0)
    fig = plt.figure(figsize = (8, 4))
    # tr = mpl.transforms.Affine2D().rotate_deg(90)

    ax1 = fig.add_subplot(122, projection = crs_epsg_MOD)
    base = ax1.transData
    rot = mpl.transforms.Affine2D().rotate_deg(5)
    ax1.pcolormesh(lon_img_20230526, lat_img_20230526,img_modis_2023_05_26, cmap = cmocean.cm.ice)#, transform =base+rot)  
    ax1.plot(SIC_SAR.lon_utq, SIC_SAR.lat_utq, color = 'r', \
    marker = '*', label = 'Utqiagvik', alpha = 0.5, transform =ccrs.PlateCarree())
    ax1.plot(*SIC_SAR.polygon_radar_fieldview.exterior.xy, color = 'r', linewidth = 2, label = 'Radar', 
            transform =ccrs.PlateCarree())
    ax1.set_title('22:02 UTC')


    ax2 = fig.add_subplot(121, projection = crs_epsg_MOD)
    # base = ax2.transData
    # ax2.invert_yaxis()
    rot = mpl.transforms.Affine2D().rotate_deg(90)
    ax2.set_title('22:00 UTC')
   
    base = ax2.transData
    # rotated_img = scipy.ndimage.rotate(img_radar_2023_05_26[0], -100)
    ax2.imshow(img_radar_2023_05_26[0], cmap = plt.cm.gray, zorder =1)#, transform = rot + base)
    for polygon in polygons_list_2023_05_26[0] : 
        x,y = polygon.exterior.xy
        x, y = np.asarray(x), np.asarray(y)
    #     print(*polygon.exterior.xy)
        ax2.plot(x, y, color = 'r')#, transform = rot)
    ax2.invert_xaxis()

    # ax2 = fig.add_subplot(313, projection = crs_epsg_MOD)

    # rot = mpl.transforms.Affine2D().rotate_deg(90)
    # ax2.set_title('22:04 UTC')
    # ax2.invert_yaxis()
    # base = ax2.transData
    # # rotated_img = scipy.ndimage.rotate(img_radar_2023_05_26[1], -100)
    # # ax2.imshow(rotated_img, cmap = plt.cm.gray, zorder =1)
    # ax2.imshow(img_radar_2023_05_26[1], cmap = plt.cm.gray, zorder =1)#, transform = rot + base)
    # for polygon in polygons_list_2023_05_26[1] : 
    #     x,y = polygon.exterior.xy
    #     x, y = np.asarray(x), np.asarray(y)
    # #     print(*polygon.exterior.xy)
    #     ax2.plot(x, y, color = 'r')#, transform = rot)
    # ax2.invert_xaxis()

    # fig.suptitle('2023-05-26')

    plt.savefig('modis_radar_2023-05-26.png', dpi = 500, bbox_inches = 'tight')


    # %%

    # %%

    #%% 
    #------- Comparison 1km interpolation to 4 min intervals --------#
    interpolation_4min_ds = xr.open_dataset('/storage/fstdenis/Barrow_RADAR/Daily_4min_Interpolation_Radar/2023/03/interpolated_20230301_4min_modis_1km.nc')
    interp_4min_radar = interpolation_4min_ds.interpolation_4min.data

    modis_sic_ds = xr.open_dataset(ModisSICDir+str(2023)+'/'+'03'+'/'+'sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_'+str(20230301)+'.nc')
    modis_sic = np.flipud(modis_sic_ds.sic_merged.data).astype(float)/100
    modis_sic[np.where(modis_sic == np.amax(modis_sic))] = np.nan

    idx_grid_MOD = np.load(ParamsDir_MODIS+filespoints_1km, allow_pickle=True).item()['idx_grid']
    #idx of the points of interest
    idx_point_MOD = np.load(ParamsDir_MODIS+filespoints_1km, allow_pickle=True).item()['idx_points']
    idx_i_MOD = np.load(ParamsDir_MODIS+filespoints_1km, allow_pickle=True).item()['idx_i']
    idx_j_MOD = np.load(ParamsDir_MODIS+filespoints_1km, allow_pickle=True).item()['idx_j']


    SIC_modis = np.zeros_like(modis_sic)*np.nan
    SIC_modis[idx_point_MOD[:, 0], idx_point_MOD[:, 1]] = modis_sic[idx_point_MOD[:, 0], idx_point_MOD[:, 1]]

    SIC_modis_modisgrid = SIC_modis[idx_grid_MOD[:, 0], idx_grid_MOD[:, 1]]
    SIC_modis_modisgrid = np.reshape(SIC_modis_modisgrid, (len(idx_j_MOD), len(idx_i_MOD)))

    idx_regridded_SIC_modis = ~np.isnan(SIC_modis_modisgrid)

    SIC_modis_modisgrid_points = SIC_modis_modisgrid[idx_regridded_SIC_modis]

    correlation_day = []
    rmse_day = []
    for i in range(len(interp_4min_radar)):
        
        interpolated_radar_sic_i = interp_4min_radar[i]
        SIC_radar_modisgrid_points = interpolated_radar_sic_i[idx_regridded_SIC_modis]
        
        cov_mod_rad = covariance(np.ma.masked_invalid(SIC_modis_modisgrid_points), np.ma.masked_invalid(SIC_radar_modisgrid_points))
        coerfcoeff = correlation(np.ma.masked_invalid(SIC_modis_modisgrid_points), np.ma.masked_invalid(SIC_radar_modisgrid_points), cov_mod_rad)
        
        
        rmse_i = rmse(np.ma.masked_invalid(SIC_modis_modisgrid_points), np.ma.masked_invalid(SIC_radar_modisgrid_points))
        coerfcoeff = np.ma.corrcoef(np.ma.masked_invalid(SIC_modis_modisgrid_points), np.ma.masked_invalid(SIC_radar_modisgrid_points))[1, 0]
        correlation_day.append(coerfcoeff)
        rmse_day.append(rmse_i)

    minutes = np.arange('2023-03-01T00:00', '2023-03-01T23:59',4, dtype='datetime64[m]')
    fmt = mdates.DateFormatter('%H:%M:%S') 
    fig = plt.figure(figsize = (15, 8))
    ax = fig.add_subplot()
    ax.xaxis.set_major_formatter(fmt)

    # ax.xaxis.set_major_formatter(
    #     mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    # fig.autofmt_xdate(rotation = 0, ha = 'center')
    plt.title('2023/03/01')
    plt.plot(minutes, correlation_day, marker = 'o' ,linestyle = '')
    plt.xlabel('Hour of Day')
    plt.ylabel('Correlation Coefficient')

    plt.savefig('correlation_20230301.png', dpi = 500, bbox_inches = 'tight')

    fig = plt.figure(figsize = (15, 8))
    ax = fig.add_subplot()
    ax.xaxis.set_major_formatter(fmt)
    plt.title('2023/03/01')
    # ax.xaxis.set_major_formatter(
    #     mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    # fig.autofmt_xdate(rotation = 0, ha = 'center')
    plt.plot(minutes, rmse_day, marker = 'o' ,linestyle = '')
    plt.savefig('rmse_20230301.png', dpi = 500, bbox_inches = 'tight')


    # %%
    #------- Interpolation to 1km -------#
    #interpolation
    # interpolated_ice_modis, result, ice_mean_day, latitudes_grid, longitudes_grid = \
    #     identification_interpolation_modis(32, 92, (7,7), filespoints_1km, coordinates_1km, Datadir, masksDir, SaveDir, ParamsDir_MODIS, saving = True)
    #analysis

    SIC_modis_modisgrid_total_1km, SIC_radar_modisgrid_total_1km, rmse_total_1km, coerfcoeff_total_1km, months_total_1km, years_total_1km =\
        extract_timesseries_interpolation(RadIntepDir, ModisSICDir, ParamsDir_MODIS, filespoints_1km, 1)

    #------ Interpolation to 2km -------# 
    # %%
    _, _ = interpolation_coarsegrain(2, filespoints_1km, savedir_radarmodis_2km, ModisSICDir, RadIntepDir, ParamsDir_MODIS, saving = True)   
    SIC_modis_modisgrid_total_2km, SIC_radar_modisgrid_total_2km, rmse_total_2km, coerfcoeff_total_2km, months_total_2km, years_total_2km =\
        extract_timesseries_interpolation(savedir_radarmodis_2km, ModisSICDir, ParamsDir_MODIS, filespoints_1km, 2)

    # %%
    #------- Interpolation to 3km -------#
    # SIC_modis_modisgrid_total_3km, SIC_radar_modisgrid_total_3km = interpolation_coarsegrain(3, filespoints_1km, savedir_radarmodis_3km, ModisSICDir, RadIntepDir, ParamsDir_MODIS, saving = True)
    SIC_modis_modisgrid_total_3km, SIC_radar_modisgrid_total_3km, rmse_total_3km, coerfcoeff_total_3km, months_total_3km, years_total_3km =\
        extract_timesseries_interpolation(savedir_radarmodis_3km, ModisSICDir, ParamsDir_MODIS, filespoints_3km, 3)

    # %%
    #------- Interpolation to 4km -------#
    SIC_modis_modisgrid_total_4km, SIC_radar_modisgrid_total_4km = interpolation_coarsegrain(4, filespoints_1km, savedir_radarmodis_4km, ModisSICDir, RadIntepDir, ParamsDir_MODIS, saving = True)
    SIC_modis_modisgrid_total_4km, SIC_radar_modisgrid_total_4km, rmse_total_4km, coerfcoeff_total_4km, months_total_4km, years_total_4km =\
        extract_timesseries_interpolation(savedir_radarmodis_4km, ModisSICDir, ParamsDir_MODIS, filespoints_3km, 4)

    #%%
    #------- Interpolation to 5km -------#
    SIC_modis_modisgrid_total_5km, SIC_radar_modisgrid_total_5km = interpolation_coarsegrain(5, filespoints_1km, savedir_radarmodis_5km, ModisSICDir, RadIntepDir, ParamsDir_MODIS, saving = True)
    SIC_modis_modisgrid_total_5km, SIC_radar_modisgrid_total_5km, rmse_total_5km, coerfcoeff_total_5km, months_total_5km, years_total_5km =\
        extract_timesseries_interpolation(savedir_radarmodis_5km, ModisSICDir, ParamsDir_MODIS, filespoints_3km, 5)

    #%%
    #------- Interpolation to 6km -------#
    SIC_modis_modisgrid_total_6km, SIC_radar_modisgrid_total_6km = interpolation_coarsegrain(6, filespoints_1km, savedir_radarmodis_6km, ModisSICDir, RadIntepDir, ParamsDir_MODIS, saving = True)
    SIC_modis_modisgrid_total_6km, SIC_radar_modisgrid_total_6km, rmse_total_6km, coerfcoeff_total_6km, months_total_6km, years_total_6km =\
        extract_timesseries_interpolation(savedir_radarmodis_6km, ModisSICDir, ParamsDir_MODIS, filespoints_3km, 6)

    #%%
    #------- Interpolation to 7km -------#
    SIC_modis_modisgrid_total_7km, SIC_radar_modisgrid_total_7km = interpolation_coarsegrain(7, filespoints_1km, savedir_radarmodis_7km, ModisSICDir, RadIntepDir, ParamsDir_MODIS, saving = True)
    SIC_modis_modisgrid_total_7km, SIC_radar_modisgrid_total_7km, rmse_total_7km, coerfcoeff_total_7km, months_total_7km, years_total_7km =\
        extract_timesseries_interpolation(savedir_radarmodis_7km, ModisSICDir, ParamsDir_MODIS, filespoints_3km, 7)

    #%%
    #------- Interpolation to 8km -------#
    SIC_modis_modisgrid_total_8km, SIC_radar_modisgrid_total_8km = interpolation_coarsegrain(8, filespoints_1km, savedir_radarmodis_8km, ModisSICDir, RadIntepDir, ParamsDir_MODIS, saving = True)
    SIC_modis_modisgrid_total_8km, SIC_radar_modisgrid_total_8km, rmse_total_8km, coerfcoeff_total_8km, months_total_8km, years_total_8km =\
        extract_timesseries_interpolation(savedir_radarmodis_8km, ModisSICDir, ParamsDir_MODIS, filespoints_3km, 8)

    #%%
    #------- Interpolation to 9km -------#
    SIC_modis_modisgrid_total_9km, SIC_radar_modisgrid_total_9km = interpolation_coarsegrain(9, filespoints_1km, savedir_radarmodis_9km, ModisSICDir, RadIntepDir, ParamsDir_MODIS, saving = True)
    SIC_modis_modisgrid_total_9km, SIC_radar_modisgrid_total_9km, rmse_total_9km, coerfcoeff_total_9km, months_total_9km, years_total_9km =\
        extract_timesseries_interpolation(savedir_radarmodis_9km, ModisSICDir, ParamsDir_MODIS, filespoints_3km, 9)

    #%%
    #------- Interpolation to 10km -------#
    SIC_modis_modisgrid_total_10km, SIC_radar_modisgrid_total_10km = interpolation_coarsegrain(10, filespoints_1km, savedir_radarmodis_10km, ModisSICDir, RadIntepDir, ParamsDir_MODIS, saving = True)
    SIC_modis_modisgrid_total_10km, SIC_radar_modisgrid_total_10km, rmse_total_10km, coerfcoeff_total_10km, months_total_10km, years_total_10km =\
        extract_timesseries_interpolation(savedir_radarmodis_10km, ModisSICDir, ParamsDir_MODIS, filespoints_3km, 10)

    # %%
    #------- plotting -------#
    plot_fig_interpolation(SIC_radar_modisgrid_total_1km, SIC_modis_modisgrid_total_1km,rmse_total_1km, months_total_1km, 1, eps = 0.3)
    plot_fig_interpolation(SIC_radar_modisgrid_total_2km, SIC_modis_modisgrid_total_2km,rmse_total_2km, months_total_2km, 2, eps = 0.3)
    plot_fig_interpolation(SIC_radar_modisgrid_total_3km, SIC_modis_modisgrid_total_3km,rmse_total_3km, months_total_3km, 3, eps = 0.3)
    plot_fig_interpolation(SIC_radar_modisgrid_total_4km, SIC_modis_modisgrid_total_4km,rmse_total_4km, months_total_4km, 4, eps = 0.3)
    plot_fig_interpolation(SIC_radar_modisgrid_total_5km, SIC_modis_modisgrid_total_5km,rmse_total_5km, months_total_5km, 5, eps = 0.3)

    histogram(rmse_total_1km, 51, 1)
    histogram(rmse_total_2km, 51, 2)
    histogram(rmse_total_3km, 51, 3)
    histogram(rmse_total_4km, 51, 4)
    histogram(rmse_total_5km, 51, 5)
    # %%
    rmse_interpolation_resolution = [rmse_total_1km, rmse_total_2km, rmse_total_3km, rmse_total_4km, rmse_total_5km, rmse_total_6km, \
        rmse_total_7km, rmse_total_8km, rmse_total_9km, rmse_total_10km]

    SIC_radar_modisgrid_resolution = [SIC_radar_modisgrid_total_1km, SIC_radar_modisgrid_total_2km, SIC_radar_modisgrid_total_3km, SIC_radar_modisgrid_total_4km, SIC_radar_modisgrid_total_5km, SIC_radar_modisgrid_total_6km, \
        SIC_radar_modisgrid_total_7km, SIC_radar_modisgrid_total_8km, SIC_radar_modisgrid_total_9km, SIC_radar_modisgrid_total_10km]

    SIC_modis_modisgrid_resolution = [SIC_modis_modisgrid_total_1km, SIC_modis_modisgrid_total_2km, SIC_modis_modisgrid_total_3km, SIC_modis_modisgrid_total_4km, SIC_modis_modisgrid_total_5km, SIC_modis_modisgrid_total_6km, \
        SIC_modis_modisgrid_total_7km, SIC_modis_modisgrid_total_8km, SIC_modis_modisgrid_total_9km, SIC_modis_modisgrid_total_10km]

    rmse_filter_total = np.zeros(len(rmse_interpolation_resolution))
    eps = 0.3
    eps_resolution = [0.2, 0.3, 0.3, 0.4, 0.5 ,0.3, 0.4,0.8,0.7,0.8]
    for i in range(len(rmse_interpolation_resolution)):
        rmse_i = rmse_interpolation_resolution[i]
        SIC_radar_modisgrid_resolution_i = SIC_radar_modisgrid_resolution[i]
        SIC_modis_modisgrid_resolution_i = SIC_modis_modisgrid_resolution[i]
        
        idx_rmse_small = np.where(rmse_i < eps_resolution[i])
        rmse_i_filter = rmse(SIC_modis_modisgrid_resolution_i[idx_rmse_small], SIC_radar_modisgrid_resolution_i[idx_rmse_small])
        rmse_filter_total[i] = rmse_i_filter
        
    plt.figure()
    plt.plot(np.arange(1, 11), rmse_filter_total, marker = 'o', color = 'k', linestyle = '')
    plt.xlabel('Resolution (km)')
    plt.ylabel('RMSE')
    plt.savefig('radarmodis_error.png', dpi = 500, bbox_inches = 'tight')


    num_bins = 51
    resolution = np.arange(1, 11)

    fig, axes = plt.subplots(2, 5, sharex = True, figsize = (22, 8))
    bins=np.linspace(0,1,num_bins) 
    centers=0.5*(bins[1:]+bins[:-1])

    for count, ax in enumerate(axes.flat):
        counts, _, _ = ax.hist(rmse_interpolation_resolution[count], bins=bins, color = 'blue')
        ax.set_title('{}km'.format(resolution[count]))
        print(counts)
    fig.supxlabel('RMSE')
    fig.supylabel('Counts')
    fig.tight_layout()
    plt.savefig('pdf_rmse_radarmodis.png'.format(resolution), dpi = 500, bbox_inches = 'tight')



    idx_rmse_small = np.where(rmse_total_5km < 0.4)
    rmse_1km = rmse(SIC_radar_modisgrid_total_5km[idx_rmse_small], SIC_modis_modisgrid_total_5km[idx_rmse_small])
# print(rmse_1km)

#------- Masks --------#
#%%

    
"""

360 time 
"""
    
# #----- Plotting figures for Article -----#
#%%
    




                    # for img in range(num_img_
                    
                    
                    # missing_files = 0
# if missing_files:
#     # savefile_missing(SaveDir+'202202/', '20220228', xg, yg)
#     # savefile_missing(SaveDir+'202203/', '20220301', xg, yg)
#     # savefile_missing(SaveDir+'202203/', '20220302', xg, yg)
#     # savefile_missing(SaveDir+'202302/', '20230220', xg, yg)
#     savefile_missing(SaveDir+'202212/', '20221231', xg, yg)
#     # list_missing = ['20221123', '20221124', '20221125', '20221126', '20221127', '20221128', '20221129']
#     list_missing2 = ['20221217', '20221218', '20221219', '20221220', '20221221' ]
#     for miss in list_missing2:
#         savefile_missing(SaveDir+'202212/', miss, xg, yg)