import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import time
import os
import glob
import rasterio
import xarray as xr


from natsort import natsorted
from shapely import geometry
from shapely.validation import make_valid

def extract_img_folder(folder) : 
    
    img_list = []
    
    for img in natsorted(glob.glob(folder+'/*.tif')):
        
        with rasterio.open(img) as f:
            img = f.read(1)
        
        img_list.append(img)

    return img_list

def read_img(namefile) : 
    
    with rasterio.open(namefile) as f:
            img = f.read(1)
            
    return img

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
    img_tot_area = '/storage/fstdenis/Barrow_RADAR/RAW_Data/2022/02/20220219/UAFIceRadar_20220219_014400_crop_geo.tif'
    img = read_img(img_tot_area)
    
    #applying the detection algorithm
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
                img_list_day = extract_img_folder(DataDirectory+YearDir+'/'+MonthDir+'/'+DayDir)
                
                #testing to see if the images are the good sizes, if not delete 
                i = 0
                idx_remove = []
                #loop through the images
                for img in img_list_day:
                    shape = img.shape
                    #if not good shape
                    if shape != (900, 900):
                        idx_remove.append(i)
                    i+=1
                
                if len(idx_remove) > 0:
                    #delete the unwanted images
                    img_list_day = np.delete(img_list_day, idx_remove)
                
                #save nothing just for the sake of the code
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
                                            
                    if saving:
                        saving_dic = {'Polygons' : polygon_list_time, 'Img' : gray_list, 'Conc_ice' : concentration_ice}
                        np.save(saveDir+'saved_params'+DayDir+'.npy', saving_dic) 
        
    return img_list_day, gray_list, edge_list_cv, polygon_list_time, polygon_list_time_1, concentration_ice

def find_latlon_tif(tiff_file, namefile, saving = False):
    
    with rasterio.open(tiff_file) as src:
        shape_img = np.shape(src)
        
        latitude = np.zeros(shape_img)
        longitude = np.zeros(shape_img)
        
        for i in range(shape_img[0]):
            for j in range(shape_img[1]):
                
                #this gets the value of the lat and lon
                lon, lat = src.xy(i, j)

                latitude[i, j] = lat
                longitude[i, j] = lon

    if saving:
                 
        ds1 = xr.Dataset(
            data_vars={
                "latitude": (( "x", "y"), latitude),
                "longitude": (( "x", "y"), longitude),
            }
        )
        ds1.to_netcdf(namefile)
        
        
        
#those packages can be a pain to install
from osgeo import osr, gdal

def pixel2coord(img_path, x, y):
    """
    Returns latitude/longitude coordinates from pixel x, y coords

    Keyword Args:
    img_path: Text, path to tif image
    x: Pixel x coordinates. For example, if numpy array, this is the column index
    y: Pixel y coordinates. For example, if numpy array, this is the row index
    """
    # Open tif file
    ds = gdal.Open(img_path)

    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjectionRef())

    # create the new coordinate system
    # In this case, we'll use WGS 84
    # This is necessary becuase Planet Imagery is default in UTM (Zone 15). So we want to convert to latitude/longitude
        
    polar_stereo = """PROJCS["WGS 84 / NSIDC Sea Ice Polar Stereographic North",
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]],
    PROJECTION["Polar_Stereographic"],
    PARAMETER["latitude_of_origin",70],
    PARAMETER["central_meridian",-45],
    PARAMETER["false_easting",0],
    PARAMETER["false_northing",0],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AUTHORITY["EPSG","3413"]]"""
    
    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(polar_stereo)

    # create a transform object to convert between coordinate systems
    transform = osr.CoordinateTransformation(old_cs,new_cs) 
    
    gt = ds.GetGeoTransform()

    # GDAL affine transform parameters, According to gdal documentation xoff/yoff are image left corner, a/e are pixel wight/height and b/d is rotation and is zero if image is north up. 
    xoff, a, b, yoff, d, e = gt

    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff

    lat_lon = transform.TransformPoint(xp, yp) 

    xp = lat_lon[0]
    yp = lat_lon[1]
    
    return xp, yp


#exemple to run the last functions
#just need to specify the shape of the img

# latitude = np.zeros((416, 430))
# longitude = np.zeros((416, 430))
# for i in range(416):
#     for j in range(430):
        
#         print(i, j)
#         lat, lon = pixel2coord(tiff_file, i, j)
#         # print(lon)
#         latitude[i, j] = lat
#         longitude[i, j] = lon