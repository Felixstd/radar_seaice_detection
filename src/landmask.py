import numpy as np
import cv2 as cv

def make_landmask(img_namefile) : 
    
    """
    Function used to make a binary land mask from the SIR_mask_Utqiagvik image from Andy Mahoney (UAF). 
    This function is much more convenient to use than the other one since it takes a lot of less time to
    process.  
    
    
    Pixel values : 
    
        0 (black): outside 6 nautical mile range
        32 (dark gray): land
        64 (lighter gray): Elson lagoon waters (typically flat ice and no radar returns)
        255 (white): ocean/ice in range of the radar
    """
    
    #reading the img
    img_mask = cv.imread(img_namefile, cv.IMREAD_GRAYSCALE)

    land_mask = np.zeros_like(img_mask)
    land_mask[np.where(img_mask != 255)] = 1
    
    dict_mask = {'mask' : land_mask}
    np.save('/storage/fstdenis/Barrow_RADAR/IdentTypeIce/edge_detection/masks/land_mask_3.npy', dict_mask)
    
    return land_mask

mask = make_landmask('./SIR_mask_Utqiagvik.png')