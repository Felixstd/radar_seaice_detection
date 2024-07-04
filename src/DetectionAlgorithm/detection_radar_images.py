import edge_detector_ice_raw as edi

"""
This script is used to run the detection algorithm over the whole time series. 
You need to specifiy the low and high thresholds, and the kernel that were 
optimized (if needed). 
"""


DataDir = '/storage/fstdenis/Barrow_RADAR/RAW_Data/'
masksDir = "/storage/fstdenis/Barrow_RADAR/ICE_RADAR_MISC/IdentTypeIce/edge_detection/masks/"

noBorder_WithoutFog = 0
Border = 0
noBorder_WithFog = 1

if noBorder_WithFog:
    
    low_threshold_opt = 37
    high_threshold_opt = 82
    kernel_opt = (13, 13)
    saving_Directory_timeseries = '/storage/fstdenis/Barrow_RADAR/saved_run_WithoutBorder_withFog/'

if noBorder_WithoutFog:
    
    low_threshold_opt = 32
    high_threshold_opt = 92
    kernel_opt = (7, 7)
    saving_Directory_timeseries = '/storage/fstdenis/Barrow_RADAR/saved_run_Withoutborders/'

if noBorder_WithoutFog:
    
    low_threshold_opt = 4
    high_threshold_opt = 82
    kernel_opt = (11, 11)
    saving_Directory_timeseries = '/storage/fstdenis/Barrow_RADAR/saved_run/RADAR/saved_TimeSeries_SIC/'



img_list, gray_list, edge_list_cv, polygon_list_end, polygon_list, concentration_ice = \
    edi.identification_ice_ow(low_threshold_opt, high_threshold_opt, kernel_opt, DataDir, masksDir, saving_Directory_timeseries, plotting = False, saving = True)


