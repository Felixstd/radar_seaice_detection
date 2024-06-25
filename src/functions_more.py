import numpy as np
import os, rasterio
import cv2 as cv 
import extract_video as vid
import matplotlib.pyplot as plt
from affine import Affine
from pyproj import Proj, transform
from mpl_toolkits.basemap import Basemap

"""
This file contains function initially produced for the algorithm that may be 
helpful later but not used anymore. 
"""
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


def analysis_opt(max_slope, concentration_vid_algo, concentration_analysers_tot, concentration_analysers_nan) : 
        
    mask = ~np.isnan(concentration_analysers_nan) & ~np.isnan(concentration_vid_algo)
    concentration_vid_algo_nan = concentration_vid_algo[mask]
    
    rmse_conc = np.nanmean(np.sqrt(((concentration_analysers_tot - (concentration_analysers_tot*max_slope))**2)))
    mbe_conc = rmse(concentration_vid_algo, concentration_analysers_tot)

    corr_coef = np.corrcoef(concentration_analysers_tot, concentration_vid_algo)[0][1]
    
    coeffs = [rmse_conc, mbe_conc, corr_coef]
    
    return coeffs, concentration_vid_algo_nan


def random_images_radar(DataDirectory, num) : 
    
    """
    Function used to find a number of random images between every frame available. 
    
    Inputs: 
        DataDirectory: Directory containing every video
        num: number of wanted random frames
    
    Outputs:
        Nothing, but it saves the random files in a folder called /rdm_img/
    
    """
    
    #loop reading the files
    name_files_tot = []
    for DataDir in os.listdir(DataDirectory) : #finding the data directories for 2022 and 2023
        DataDir = str(DataDir)
        
        if (DataDir == 'RADAR_2022') or (DataDir == 'RADAR_2023') : 
            
            for files in os.listdir(DataDirectory+DataDir+'/') : 
                name_files_tot.append(DataDirectory+DataDir+'/'+files) #appending every files
                
    random_idx = np.random.random_integers(0, len(name_files_tot), size = num) #taking number of random files
    
    it = 0
    for idx in random_idx : 
        #loop that for each randomly selected files it takes a random frame
        name_file = name_files_tot[idx]
        _,  _,  _,  _,  _,  _, img_list,  _ = vid.extract_video(name_file)
        
        rdm_idx_img = np.random.random_integers(0, len(img_list), 1)[0]
        
        cv.imwrite(f'./rdm_img/image_{it}.png', img_list[rdm_idx_img]) #saving the random frame
        it += 1
             
def find_land(filename) : 
    
    """
    Function used to differientiate between the land and the ocean in the images. 
    It's based on the fact that the raw images are GEOtiffs and contain information
    about the lat/lon of each pixels in the images. 
    
    It's uses basemaps, rasterio and Affine to compute the pixels' lat/lon to then use the basemap coastline mask
    with full resolution to tell if it's land or ocean. It also saves the mask for latter use because 
    computing the 900x900 pixel type takes a long time. 

    Inputs : 
        filename -> file to make the compute on (doesnt matter which one it is)
        
    Returns:
        land_mask -> binary mask
    """
    
    #set the Basemap class with full resolution
    bm = Basemap(resolution = 'f')
    
    #read the file
    with rasterio.open(filename) as f:
        T0 = f.transform  # upper-left pixel corner affine transform
        p1 = Proj(f.crs) #get the projection information
        img = f.read(1)  # pixel values

    width, height = img.shape 
    # All rows and columns
    cols, rows = np.meshgrid(np.arange(height), np.arange(width))

    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)
    
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T1

    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(rc2en, otypes=[float, float])(rows, cols)

    # Project all longitudes, latitudes
    p2 = Proj(proj='latlong',datum='WGS84')
    longs, lats = transform(p1, p2, eastings, northings)

    #set the land_mask
    land_mask = np.zeros((height, width))
    
    #loop through every pixel
    it = 0
    for i in range(height) : 
        for j in range(width) : 
            
            print('iterations : ', it)
            lat, lon = lats[j, i], longs[j, i] #taking the lats and lons
            
            land = bm.is_land(lon, lat) #testing ocean or land
             
            land_mask[j, i] = int(land) #setting the mask pixel value
            
            it+=1
    
    #saving the mask
    dict_mask = {'mask' : land_mask}
    np.save('/storage/fstdenis/Barrow_RADAR/IdentTypeIce/edge_detection/masks/land_mask_2.npy', dict_mask)
            
            
    img[np.where(land_mask == 1)] = 0
    plt.figure()
    plt.imshow(img)
    plt.savefig('test.png')       
         
    return land_mask

def change_colors(img, height, width, a, b,r, difference_circle, fill_value) : 
    
    """
    Function used to change the pixel number used to black out the unwanted part in the radar images outside of the
    radar range.
    
    Input :
        img -> the frame to change the colors in ((height, width), array)
        height -> height of the img (int)
        width -> width of the img (int)
        a, b, r -> coordinates of the circle marking the radar rage (specified in the parameter file) (ints)
        difference_circle -> list containing the off sets for the circle
        fill_value -> value used to erase the unwanted part
    
    Ouput : 
        img -> initial input image with the radar range outside blacked out
    """
    
    delta_a, delta_b, delta_r = difference_circle
    
    #Main loop
    for i in range(height) : 
        for j in range(width) : 
            
            #condition that the pixel is outside
            if (i - (b+delta_b))**2 + (j-(a+delta_a))**2 >= (r+delta_r)**2 : 
                
                img[i, j] = fill_value
                
    return img



#------- Code for figures that are not used ---------#
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(12, 12))

# ax1.plot(days, SIC_2022_mean, color='b', marker='o', markersize=2, label='Radar (2022)')
# ax1.plot(days, SIC_Modis_2022, color='r', marker='o', markersize=2, label='Merged (2022)')
# ax1.plot(days, SIC_CDR_2022, color='g', marker='o', markersize=2, label='CDR (2022)')
# ax1.plot(days_2023, SIC_2023_mean, color='black', linestyle='--', marker='o', markersize=2, label='Radar (2023)')
# ax1.plot(days_2023, SIC_Modis_2023, color='gray', linestyle='--', marker='o', markersize=2, label='Merged (2023)')
# ax1.plot(days_2023, SIC_CDR_2023, color='violet', linestyle='--', marker='o', markersize=2, label='CDR (2023)')
# ax1.annotate('a)', xy=(ax2.get_xlim()[0], ax2.get_ylim()[1]), weight='bold')

# ax2.set_title(f'{size_MA_week*2+1} days Moving Average')
# ax2.plot(days, SIC_MovAve_Week_2022, color='b', marker='o', markersize=2, label='Radar (2022)')
# ax2.plot(days_2023, SIC_MovAve_Week_2023, color='black', linestyle='--', marker='o', markersize=2, label='Radar (2023)')
# ax2.plot(days, SIC_MovAve_Week_CDR_2022, color='g', marker='o', markersize=2, label='CDR (2022)')
# ax2.plot(days_2023, SIC_MovAve_Week_CDR_2023, color='violet', linestyle='--', marker='o', markersize=2, label='CDR (2023)')
# ax2.plot(days, SIC_MovAve_Week_Modis_2022, color='r', marker='o', markersize=2, label='Merged (2022)')
# ax2.plot(days_2023, SIC_MovAve_Week_Modis_2023, color='gray', linestyle='--', marker='o', markersize=2, label='Merged (2023)')

# ax2.annotate('b)', xy=(ax2.get_xlim()[0], ax2.get_ylim()[1]), weight='bold')

# lines, labels = ax2.get_legend_handles_labels()
# labels = ['Radar (2022)', 'Radar (2023)', 'CDR (2022)', 'CDR (2023)', 'Merged (2022)', 'Merged (2023)']
# fig.legend(lines, labels, loc='center', ncol=int(len(labels)/2), bbox_to_anchor=(0.5, 0.5), bbox_transform=fig.transFigure)
# fig.supylabel('Sea Ice Concentration')
# plt.xlabel('Days of the year')
# plt.savefig(SaveFig+'timeseries_RADSAT.png', dpi=500, bbox_inches='tight')


        
        # if year <= 2014 : 
        #     start_frame, frame_step, gaus_filter, kernel_size, sigma, low, high, circle, \
        #         difference_circle, area, plotting_cont, saving, check_cont = read.read_parameters_edge(ParamsDirectory+'init_params_edge_2014.txt')
            
        #     circle = circ.detect_cricle(first_2014_minus, circle) 
        #     a,b,r = circle
            
        #     print('Reading Mask')
        #     land_mask = loadmat(MasksDirectory+'mask_2020_down.mat')['Mask']
            
        # else : 
        
        
    # idx_max_3 = np.unravel_index(np.argmin(slope_minus1_3, axis=None), slope_total_3.shape)
    # idx_max_5 = np.unravel_index(np.argmin(slope_minus1_5, axis=None), slope_total_5.shape)
    # idx_max_7 = np.unravel_index(np.argmin(slope_minus1_7, axis=None), slope_total_7.shape)
    # idx_max_13 = np.unravel_index(np.argmin(slope_minus1_13, axis=None), slope_total_13.shape)
    
    # idx_min_intercept3 = np.unravel_index(np.argmin(abs(intercept_total_3), axis=None), slope_total_3.shape)
    # idx_min_intercept5 = np.unravel_index(np.argmin(abs(intercept_total_5), axis=None), slope_total_5.shape)
    # idx_min_intercept7 = np.unravel_index(np.argmin(abs(intercept_total_7), axis=None), slope_total_7.shape)
    # idx_min_intercept13 = np.unravel_index(np.argmin(abs(intercept_total_13), axis=None), slope_total_13.shape)
    
        # max_slope_3 = slope_total_3[idx_max_3]
    # max_slope_5 = slope_total_5[idx_max_5]
    # max_slope_7 = slope_total_7[idx_max_7]
    # max_slope_13 = slope_total_13[idx_max_13]