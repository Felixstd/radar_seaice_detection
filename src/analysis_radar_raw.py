import os, sys, warnings
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as mtransforms
import numpy.ma as ma
import matplotlib.dates as mdates


original_stdout = sys.stdout
# 
SMALL_SIZE = 8
MEDIUM_SIZE = 15
BIGGER_SIZE = 15

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)
warnings.filterwarnings("ignore")

"""
My data is starting on the 1 january 00:00
"""
def timeseries_radar(namefile, SavedDir, SICDir):
    
    #--- Making the time series for 2022 ---#
    SIC_2022 = []
    SIC_2023 = []

    #looping through the years
    for yeardir in os.listdir(SavedDir):
        counter = 0
        year = int(yeardir)
        
        #looping through the day files
        for dayfile in sorted(os.listdir(SavedDir + yeardir)):

            for files_concentration in os.listdir(SavedDir + yeardir +'/'+dayfile):
                print('Reading File : ', counter, files_concentration)
                concentration_ice = np.load(SavedDir + yeardir +'/'+dayfile+'/'+files_concentration, allow_pickle=True).item()['Conc_ice']
                
                #if not the good size
                lenght_concentration_ice = len(concentration_ice)

                #padding with nans if no data associated
                if lenght_concentration_ice < 360:
                    concentration_ice = np.append(concentration_ice, np.ones(360 - lenght_concentration_ice)*np.nan)
                    
                if lenght_concentration_ice > 360:
                    concentration_ice = concentration_ice[:360]
            
                if year == 2023: 
                    SIC_2023.extend(concentration_ice)
                    
                if year == 2022:
                    SIC_2022.extend(concentration_ice)
                    
                counter+=1
                    

    SIC = {'2022': SIC_2022, '2023': SIC_2023}
    np.save( SICDir+namefile, SIC)

def daily_average_radar(SIC, frames_per_day, start_frame):
    """
    This function is used to compute the daily average of the sea ice concentration.
    It averages the full 4 minutes snapshot into 24 hours one. 

    Args:
        SIC (array): SIC concentration at each 4 minutes
        frames_per_day (float): number of images per day
        start_frame (float): start of the time series

    Returns:
        SIC_mean_series (array): array containing the daily averaged SIC
    """
    
    #initializing
    SIC = np.asarray(SIC)
    days_int = np.arange(start_frame, len(SIC) + frames_per_day, frames_per_day, dtype=int)
    N = len(days_int) - 1

    #getting rid of the bad data
    SIC[np.where((SIC > 1) & (SIC < 0))] = np.nan
    
    # computing the mean    
    SIC_mean_series = np.zeros(N)
    for it in range(N):
        SIC_mean_series[it] = np.nanmean(SIC[days_int[it]:days_int[it + 1]])

    return SIC_mean_series

def timeseries_MODIS(SavedModis, filename):
    
    days_before_june = 30 * 2 + 31 * 2 + 27
    days_between = 30 * 2 + 31 * 3 - 12
    missing_data = np.zeros(days_between)
    missing_data[np.where(missing_data == 0)] = np.nan

    #loading the SIC and STD data
    SIC_Modis = np.load(SavedModis + '/' + filename, allow_pickle=True).item()['SIC']
    STD_Modis = np.load(SavedModis + '/' + filename, allow_pickle=True).item()['STD']
    SIC_Modis_min = np.load(SavedModis + '/' + filename, allow_pickle=True).item()['min']
    SIC_Modis_max = np.load(SavedModis + '/' + filename, allow_pickle=True).item()['max']

    #Replacing missing days by nans
    SIC_Modis = np.asarray(SIC_Modis)
    STD_Modis_2 = np.append(STD_Modis[:days_before_june], missing_data)
    STD_Modis_total = np.append(STD_Modis_2, SIC_Modis[days_before_june:])
    
    SIC_Modis = np.asarray(SIC_Modis)
    SIC_Modis_2 = np.append(SIC_Modis[:days_before_june], missing_data)
    SIC_Modis_total = np.append(SIC_Modis_2, SIC_Modis[days_before_june:])
    
    SIC_Modis_min = np.asarray(SIC_Modis_min)
    SIC_Modis_min_2 = np.append(SIC_Modis_min[:days_before_june], missing_data)
    SIC_Modis_min_total = np.append(SIC_Modis_min_2, SIC_Modis_min[days_before_june:])
    
    SIC_Modis_max = np.asarray(SIC_Modis_max)
    SIC_Modis_max_2 = np.append(SIC_Modis_max[:days_before_june], missing_data)
    SIC_Modis_max_total = np.append(SIC_Modis_max_2, SIC_Modis_max[days_before_june:])

    return SIC_Modis_total, SIC_Modis_min_total, SIC_Modis_max_total, STD_Modis_total

def timeseries_CDR(SavedCDR, filename):
    """
    This function is used to load the time series associated with the CDR data

    Args:
        SavedCDR (str): directory name where the data is saved
        filename (str): filename 

    Returns:
        SIC_CDR (array): Full SIC
        SIC_CDR_min (array): Min associated with the SIC
        SIC_CDR_max (array): Max associated with the SIC
        
    """

    #loading all of the data
    SIC_CDR = np.load(SavedCDR + '/' + filename, allow_pickle=True).item()['SIC']
    STD_CDR = np.load(SavedCDR + '/' + filename, allow_pickle=True).item()['STD']
    SIC_CDR_min = np.load(SavedCDR + '/' + filename, allow_pickle=True).item()['min']
    SIC_CDR_max = np.load(SavedCDR + '/' + filename, allow_pickle=True).item()['max']   

    SIC_CDR = np.asarray(SIC_CDR)

    return SIC_CDR, SIC_CDR_min, SIC_CDR_max, STD_CDR

def moving_average(SIC, size):
    """
    Function used to compute the running mean associated with a certain time series. 
    Here, for this computation, we are taking the 'size' points before and after each grid points. 

    Args:
        SIC (array): array of the sea ice concentration
        size (int): size of the running average

    Returns:
        moving_average_SIC (array): array containing the moving average
    """
    #initializing
    N = len(SIC) - 2*size
    moving_average_SIC = np.zeros(N)

    #looping in the data
    j = 0
    for i in range(size, N + size):
        #taking the mean associated with size
        moving_average_SIC[j] = np.nanmean(SIC[(i - size):(i + size + 1)])
        j+=1

    moving_average_SIC = np.append(np.append(SIC[:size], moving_average_SIC), SIC[-size:])
    
    return np.asarray(moving_average_SIC)

def reglin(SIC_1, SIC_2):
    """
    This function is used to do a linear regression between 2 time series.
    But, we need to mask the arrays such that the nan values are not counted. 


    Args:
        SIC_1 (array): Array (1) of SIC for a given period 
        SIC_2 (array): Array (2) of SIC for a given period 

    Returns:
        lin_reg (array): fitted parameters of the linear regression
    """
    #masking the array
    nanMasks = ~np.isnan(SIC_1) & ~np.isnan(SIC_2)
    #doing the linear regression
    lin_reg = stats.linregress(SIC_1[nanMasks], SIC_2[nanMasks])

    return lin_reg


def error_SIC(SIC_calculated, SIC_observations) : 
    
    rsme = np.sqrt(np.nanmean(((SIC_calculated - SIC_observations)**2)))
    mean_bias_error = np.nanmean(SIC_calculated - SIC_observations)
    return rsme, mean_bias_error
    

def analysis_SIC(SIC_Modis, SIC_CDR, SIC_Radar, size_MA_Week, size_MA_Month) : 
    
    SIC_tot = [SIC_Modis, SIC_CDR, SIC_Radar]
    #----Modis Loop -----#
    
    it = 0
    SIC_MA_week_tot = []
    SIC_MA_Month_tot = []
    
    for Prod in SIC_tot : 
        # print(Prod)
        for SIC in Prod : 
            SIC_MA_week = moving_average(SIC, size_MA_Week)
            SIC_MA_Month  = moving_average(SIC, size_MA_Month)
            
            SIC_MA_week_tot.append(SIC_MA_week)
            SIC_MA_Month_tot.append(SIC_MA_Month)
            
    SIC_MA_Modis = SIC_MA_week_tot[0:len(SIC_Modis)] + SIC_MA_Month_tot[0:len(SIC_Modis)]
    SIC_MA_CDR = SIC_MA_week_tot[len(SIC_Modis):len(SIC_CDR)+len(SIC_Modis)] + SIC_MA_Month_tot[len(SIC_Modis):len(SIC_CDR)+len(SIC_Modis)]
    SIC_MA_Radar = SIC_MA_week_tot[len(SIC_CDR)+len(SIC_Modis):] + SIC_MA_Month_tot[len(SIC_CDR)+len(SIC_Modis):]
    
    return SIC_MA_Radar, SIC_MA_CDR, SIC_MA_Modis


"""
The sea ice concentration with the borders is stored into SIC_radar_raw.npy

The new SIC without the border is stored into the SIC_radar_noborders_1.npy




At the end of the time series, you need to add a 20220629 file bwith nothing in it.

"""

SavedDir = '/storage/fstdenis/Barrow_RADAR/saved_run'

SavedCDR = SavedDir + '/SIC_CDR/params/'
SavedMODIS = SavedDir + '/SIC_MODIS/params/'

SICDir = '/storage/fstdenis/Barrow_RADAR/saved_run/RADAR/saved_SIC/'

start_frame = 14
interval = 5
frames_per_day = 24 * 60 / 4
size_MA_week = 3
size_MA_Month = 15

# summer = True
summer = False

# timeseries_radar(SavedRadar, SICDir+'SIC_radar_raw.npy')
# 
# --------- Loading SIC for 2022 and for 2023 ----------#

Added_Fog = 1
NoAdded_Fog = 0
Border = 0

    
if Border: 
    sic_radar_file = 'SIC_radar_raw.npy'
    figDir = './Results_Borders/'
    SavedRadar = '/storage/fstdenis/Barrow_RADAR/saved_run/RADAR/saved_TimeSeries_SIC/'
    # timeseries_radar(sic_radar_file, SavedRadar, SICDir)
    

if NoAdded_Fog:
    sic_radar_file = 'SIC_radar_noborders_1.npy'
    figDir = './Results_NoBorders_WithoutAddedFog/'
    SavedRadar = '/storage/fstdenis/Barrow_RADAR/saved_run_Withoutborders/'
    # timeseries_radar(sic_radar_file, SavedRadar, SICDir)
    
if Added_Fog:
    sic_radar_file = 'SIC_radar_noborders_Fog.npy'
    figDir = './Results_NoBorders_WithAddedFog/'
    SavedRadar = '/storage/fstdenis/Barrow_RADAR/saved_run_WithoutBorder_withFog/'
    timeseries_radar(sic_radar_file, SavedRadar, SICDir)



SIC_2022_tot = np.load(SICDir+sic_radar_file, allow_pickle=True).item()['2022']
SIC_2023_tot = np.load(SICDir+sic_radar_file, allow_pickle=True).item()['2023']


SIC_CDR_2022, SIC_CDR_min_2022, SIC_CDR_max_2022, STD_CDR_2022 = timeseries_CDR(SavedCDR, '2022_CDR_3.npy')
SIC_CDR_2023, SIC_CDR_min_2023, SIC_CDR_max_2023, STD_CDR_2023 = timeseries_CDR(SavedCDR, '2023_CDR_3.npy')

SIC_Modis_2022, SIC_Modis_min_2022, SIC_Modis_max_2022, STD_Modis_2022 = timeseries_MODIS(SavedMODIS, '2022_MODIS_2.npy')

SIC_Modis_2023, SIC_Modis_min_2023, SIC_Modis_max_2023, STD_Modis_2023 = np.asarray(np.load(SavedMODIS+'2023_MODIS_2.npy', allow_pickle=True).item()['SIC']),\
    np.asarray(np.load(SavedMODIS+'2023_MODIS.npy', allow_pickle=True).item()['min']), \
        np.asarray(np.load(SavedMODIS+'2023_MODIS.npy', allow_pickle=True).item()['max']), np.asarray(np.load(SavedMODIS+'2023_MODIS_2.npy', allow_pickle=True).item()['STD'])


nan_modis_idx = np.isnan(SIC_Modis_2022)

if summer : 
    print('without summer for CDR')
    SIC_CDR_2022[nan_modis_idx] = np.nan

SIC_2022_mean = daily_average_radar(SIC_2022_tot, frames_per_day, 0)
SIC_2023_mean = daily_average_radar(SIC_2023_tot, frames_per_day, 0)

SIC_2223_mean = np.append(SIC_2022_mean, SIC_2023_mean)
SIC_Modis_2223 = np.append(SIC_Modis_2022, SIC_Modis_2023)
SIC_CDR_2223 = np.append(SIC_CDR_2022, SIC_CDR_2023)

STD_CDR_2223 = np.append(STD_CDR_2022, STD_CDR_2023)
STD_Modis_2223 = np.append(STD_Modis_2022, STD_Modis_2023)

days = np.arange(1, 1 + len(SIC_2022_mean))
days_2023 = np.arange(1, 1 + len(SIC_2023_mean))
days_2223 = np.append(days, days_2023)
days_combined = np.arange(1, 1+len(SIC_2223_mean))

SIC_Radar = [SIC_2022_mean, SIC_2023_mean]

SIC_Modis = [SIC_Modis_2022, SIC_Modis_2023, STD_Modis_2022, STD_Modis_2023]

SIC_CDR = [SIC_CDR_2022, SIC_CDR_2023, STD_CDR_2022, STD_CDR_2023]


#--- Moving Averages ---#
SIC_MA_Radar, SIC_MA_CDR, SIC_MA_Modis = analysis_SIC(SIC_Modis, SIC_CDR, SIC_Radar, size_MA_week, size_MA_Month)

SIC_MovAve_Week_2022, SIC_MovAve_Week_2023, SIC_MovAve_Month_2022, SIC_MovAve_Month_2023 = SIC_MA_Radar

SIC_MovAve_Week_Modis_2022, SIC_MovAve_Week_Modis_2023, STD_MA_Week_Modis_2022,STD_MA_Week_Modis_2023, SIC_MovAve_Month_Modis_2022, SIC_MovAve_Month_Modis_2023, \
        STD_MA_Month_Modis_2022,STD_MA_Month_Modis_2023 = SIC_MA_Modis
    
SIC_MovAve_Week_CDR_2022, SIC_MovAve_Week_CDR_2023,  \
     STD_MA_Week_CDR_2022, STD_MA_Week_CDR_2023,SIC_MovAve_Month_CDR_2022, SIC_MovAve_Month_CDR_2023, STD_MA_Month_CDR_2022,STD_MA_Month_CDR_2023 = SIC_MA_CDR   



SIC_MovAve_Week_2223 = np.append(SIC_MovAve_Week_2022, SIC_MovAve_Week_2023)
SIC_MovAve_Month_2223 = np.append(SIC_MovAve_Month_2022, SIC_MovAve_Month_2023)


SIC_MovAve_Week_Modis_2223 = np.append(SIC_MovAve_Week_Modis_2022, SIC_MovAve_Week_Modis_2023)
SIC_MovAve_Month_Modis_2223 = np.append(SIC_MovAve_Month_Modis_2022, SIC_MovAve_Month_Modis_2023)
STD_MA_Week_Modis_2223 = np.append(STD_MA_Week_Modis_2022, STD_MA_Week_Modis_2023)
STD_MA_Month_Modis_2223 = np.append(STD_MA_Month_Modis_2022, STD_MA_Month_Modis_2023)

SIC_MovAve_Week_CDR_2223 = np.append(SIC_MovAve_Week_CDR_2022, SIC_MovAve_Week_CDR_2023)
SIC_MovAve_Month_CDR_2223 = np.append(SIC_MovAve_Month_CDR_2022, SIC_MovAve_Month_CDR_2023)
STD_MA_Week_CDR_2223 = np.append(STD_MA_Week_CDR_2022, STD_MA_Week_CDR_2023)
STD_MA_Month_CDR_2223 = np.append(STD_MA_Month_CDR_2022, STD_MA_Month_CDR_2023)


#---------- Correlation computation ----------#

SIC_Modis_Daily_Minus_MA = SIC_Modis_2223 - SIC_MovAve_Month_Modis_2223
SIC_CDR_Daily_Minus_MA = SIC_CDR_2223 - SIC_MovAve_Month_CDR_2223
SIC_RADAR_Daily_Minus_MA = SIC_2223_mean - SIC_MovAve_Month_2223


SIC_Modis_Weekly_Minus_MA = SIC_MovAve_Week_Modis_2223 - SIC_MovAve_Month_Modis_2223
SIC_CDR_Weekly_Minus_MA = SIC_MovAve_Week_CDR_2223 - SIC_MovAve_Month_CDR_2223
SIC_RADAR_Weekly_Minus_MA = SIC_MovAve_Week_2223 - SIC_MovAve_Month_2223


correlation_Daily_Modis_Radar = ma.corrcoef(ma.masked_invalid(SIC_RADAR_Daily_Minus_MA), ma.masked_invalid(SIC_Modis_Daily_Minus_MA))
correlation_Daily_CDR_Radar = ma.corrcoef(ma.masked_invalid(SIC_RADAR_Daily_Minus_MA), ma.masked_invalid(SIC_CDR_Daily_Minus_MA))

correlation_weekly_Modis_Radar = ma.corrcoef(ma.masked_invalid(SIC_RADAR_Weekly_Minus_MA), ma.masked_invalid(SIC_Modis_Weekly_Minus_MA))
correlation_weekly_CDR_Radar = ma.corrcoef(ma.masked_invalid(SIC_RADAR_Weekly_Minus_MA), ma.masked_invalid(SIC_CDR_Weekly_Minus_MA))

# --------- MODIS analysis ---------#

lin_reg_MODIS = reglin(SIC_2022_mean, SIC_Modis_2022)
lin_reg_MODIS_2023 = reglin(SIC_2023_mean, SIC_Modis_2023)
lin_reg_MODIS_2223 = reglin(SIC_2223_mean, SIC_Modis_2223)


# --- MODIS moving average ---#

lin_reg_MODIS_MA = reglin(SIC_MovAve_Week_2022, SIC_MovAve_Week_Modis_2022)
lin_reg_MODIS_MA_2023 = reglin(SIC_MovAve_Week_2023, SIC_MovAve_Week_Modis_2023)
lin_reg_MODIS_MA_2223 = reglin(SIC_MovAve_Week_2223, SIC_MovAve_Week_Modis_2223)

lin_reg_MODIS_MA_Month = reglin(SIC_MovAve_Month_2022, SIC_MovAve_Month_Modis_2022)
lin_reg_MODIS_MA_2023_Month = reglin(SIC_MovAve_Month_2023, SIC_MovAve_Month_Modis_2023)
lin_reg_MODIS_MA_2223_Month = reglin(SIC_MovAve_Month_2223, SIC_MovAve_Month_Modis_2223)

# --------- CDR analysis ----------#

lin_reg_CDR = reglin(SIC_2022_mean, SIC_CDR_2022)
lin_reg_CDR_2023 = reglin(SIC_2023_mean, SIC_CDR_2023)
lin_reg_CDR_2223 = reglin(SIC_2223_mean, SIC_CDR_2223)

# --- CDR moving average ---#
lin_reg_CDR_MA = reglin(SIC_MovAve_Week_2022, SIC_MovAve_Week_CDR_2022)
lin_reg_CDR_MA_2023 = reglin(SIC_MovAve_Week_2023, SIC_MovAve_Week_CDR_2023)
lin_reg_CDR_MA_2223 = reglin(SIC_MovAve_Week_2223, SIC_MovAve_Week_CDR_2223)

lin_reg_CDR_MA_Month = reglin(SIC_MovAve_Month_2022, SIC_MovAve_Month_CDR_2022)
lin_reg_CDR_MA_2023_Month = reglin(SIC_MovAve_Month_2023, SIC_MovAve_Month_CDR_2023)
lin_reg_CDR_MA_2223_Month = reglin(SIC_MovAve_Month_2223, SIC_MovAve_Month_CDR_2223)

linreg_CDR_2022 = SIC_MovAve_Week_2022 * lin_reg_CDR_MA.slope + lin_reg_CDR_MA.intercept
linreg_CDR_2022[np.where(linreg_CDR_2022 > 1)] = np.nan

# #------------- RMSE -------------#
#Calculating the RMSE 
linreg_CDR_2223_Week_MA = SIC_MovAve_Week_2223 * lin_reg_CDR_MA_2223.slope + lin_reg_CDR_MA_2223.intercept
linreg_CDR_2223_Week_MA[np.where(linreg_CDR_2223_Week_MA > 1)] = np.nan

linreg_CDR_2223_Week = SIC_2223_mean * lin_reg_CDR_2223.slope + lin_reg_CDR_2223.intercept
linreg_CDR_2223_Week[np.where(linreg_CDR_2223_Week > 1)] = np.nan

linreg_CDR_2223_Month_MA = SIC_MovAve_Month_2223 * lin_reg_CDR_MA_2223_Month.slope + lin_reg_CDR_MA_2223_Month.intercept
linreg_CDR_2223_Month_MA[np.where(linreg_CDR_2223_Month_MA > 1)] = np.nan

Lin_Reg_2223_Modis = SIC_2223_mean * lin_reg_MODIS_2223.slope + lin_reg_MODIS_2223.intercept

lin_reg_MODIS_MA_2223_tot = SIC_MovAve_Week_2223 * lin_reg_MODIS_MA_2223.slope + lin_reg_MODIS_MA_2223.intercept
lin_reg_MODIS_MA_2223_tot[np.where(lin_reg_MODIS_MA_2223_tot > 1)] = np.nan

lin_reg_MODIS_MA_2223_tot_Month = SIC_MovAve_Month_2223 * lin_reg_MODIS_MA_2223_Month.slope + lin_reg_MODIS_MA_2223_Month.intercept
lin_reg_MODIS_MA_2223_tot_Month[np.where(lin_reg_MODIS_MA_2223_tot_Month > 1)] = np.nan


# rmse_Modis_2223, _      = error_SIC(SIC_Modis_2223, Lin_Reg_2223_Modis)
# rmse_Modis_Week_2223, _ = error_SIC(SIC_MovAve_Week_Modis_2223, lin_reg_MODIS_MA_2223_tot)
# rmse_Modis_Month_2223,_ = error_SIC(SIC_MovAve_Month_Modis_2223, lin_reg_MODIS_MA_2223_tot_Month)
# rmse_CDR_2223,_       = error_SIC(SIC_CDR_2223, linreg_CDR_2223_Week)
# rmse_CDR_Week_2223,_   = error_SIC(SIC_MovAve_Week_CDR_2223, linreg_CDR_2223_Week_MA)
# rmse_CDR_Month_2223,_   = error_SIC(SIC_MovAve_Month_CDR_2223, linreg_CDR_2223_Month_MA)

rmse_Modis_2223, mbe_Modis_2223     = error_SIC(SIC_2223_mean, SIC_Modis_2223)
rmse_Modis_Week_2223, mbe_Modis_Week_2223 = error_SIC(SIC_MovAve_Week_2223, SIC_MovAve_Week_Modis_2223)
rmse_Modis_Month_2223, mbe_Modis_Month_2223 = error_SIC(SIC_MovAve_Month_2223, SIC_MovAve_Month_Modis_2223)
rmse_CDR_2223, mbe_CDR_2223       = error_SIC(SIC_2223_mean, SIC_CDR_2223)
rmse_CDR_Week_2223, mbe_CDR_Week_2223 = error_SIC(SIC_MovAve_Week_2223, SIC_MovAve_Week_CDR_2223)
rmse_CDR_Month_2223, mbe_CDR_Month_2223 = error_SIC(SIC_MovAve_Month_2223, SIC_MovAve_Month_CDR_2223)


#----------- Printing statements -----------#
print('Writing in file')
with open(figDir+'results_RADAR.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print('Without Summer : ', summer, '\n')
    print('R^2 and slope values for the linear regression (MODIS) (2223)', correlation_Daily_Modis_Radar, lin_reg_MODIS_2223.slope) 
    print(f'R^2 and slope values for the linear regression of the {size_MA_week*2+1} days moving average (MODIS) (2223)', correlation_weekly_Modis_Radar, lin_reg_MODIS_MA_2223.slope)
    print(f'R^2 and slope values for the linear regression of the {size_MA_Month*2+1} days moving average (MODIS) (2223)', lin_reg_MODIS_MA_2223_Month.rvalue ** 2, lin_reg_MODIS_MA_2223_Month.slope, '\n')
    
    print('R^2 and slope values  for the linear regression (CDR) (2223)', correlation_Daily_CDR_Radar, lin_reg_CDR_2223.slope)
    print(f'R^2 and slope values  for the linear regression of the {size_MA_week*2+1} days moving average (CDR) (2223)', correlation_weekly_CDR_Radar, lin_reg_CDR_MA_2223.slope)
    print(f'R^2 and slope values  for the linear regression of the {size_MA_Month*2+1} days moving average (CDR) (2223)', lin_reg_CDR_MA_2223_Month.rvalue ** 2, lin_reg_CDR_MA_2223_Month.slope, '\n')

    print(f'RMSE for linear regression of of the (CDR) (2223)', rmse_CDR_2223)
    print(f'RMSE for linear regression of of the {size_MA_week*2+1} days moving average (CDR) (2223)', rmse_CDR_Week_2223)
    print(f'RMSE for linear regression of of the {size_MA_Month*2+1} days moving average (CDR) (2223)', rmse_CDR_Month_2223)
    print(f'RMSE for linear regression of of the (Modis) (2223)', rmse_Modis_2223)
    print(f'RMSE for linear regression of of the {size_MA_week*2+1} days moving average (Modis) (2223)', rmse_Modis_Week_2223)
    print(f'RMSE for linear regression of of the {size_MA_Month*2+1} days moving average (Modis) (2223)', rmse_Modis_Month_2223, '\n')
    
    print(f'Mean Bias Error of the (CDR) (2223)', mbe_CDR_2223)
    print(f'Mean Bias Error of the {size_MA_week*2+1} days moving average (CDR) (2223)', mbe_CDR_Week_2223)
    print(f'Mean Bias Error of the {size_MA_Month*2+1} days moving average (CDR) (2223)', mbe_CDR_Month_2223)
    print(f'Mean Bias Error of the (Modis) (2223)', mbe_Modis_2223)
    print(f'Mean Bias Error of the {size_MA_week*2+1} days moving average (Modis) (2223)', mbe_Modis_Week_2223)
    print(f'Mean Bias Error of the {size_MA_Month*2+1} days moving average (Modis) (2223)', mbe_Modis_Month_2223)
sys.stdout = original_stdout # Reset the standard output to its original value
#----------- Figures -----------#
# --- Total Time Series ---#
error_CDR_2223_min, error_CDR_2223_max = SIC_CDR_2223 - STD_CDR_2223, SIC_CDR_2223 + STD_CDR_2223
error_CDR_2223_min[error_CDR_2223_min < 0], error_CDR_2223_max[error_CDR_2223_max > 1] = np.nan, np.nan

error_Modis_2223_min, error_Modis_2223_max = SIC_Modis_2223 - STD_Modis_2223, SIC_Modis_2223 + STD_Modis_2223
error_Modis_2223_min[error_Modis_2223_min < 0], error_Modis_2223_max[error_Modis_2223_max > 1] = np.nan, np.nan


# #------ Fig.2 -------#
days = np.arange('2022-01', '2023-06', dtype='datetime64[D]')
fmt = mdates.DateFormatter('%Y-%m')

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot()
# plt.xticks(rotation=90)
ax.set_title('Daily')
# ax.minorticks_on()
ax.grid(which='major')
ax.plot(days, SIC_2223_mean, color='b', marker='o', markersize=2, label='Radar')
ax.plot(days, SIC_Modis_2223, color='r', marker='o', markersize=2, label='Modis-AMSR2')
ax.plot(days, SIC_CDR_2223, color='g', marker='o', markersize=2, label=' CDR')
ax.fill_between(days, error_CDR_2223_min, error_CDR_2223_max, color = 'g', alpha = 0.5)
ax.fill_between(days, error_Modis_2223_min, error_Modis_2223_max, color = 'r', alpha = 0.5)
ax.set_xlabel('Day of year')
ax.set_ylabel('Sea Ice Concentration')
ax.xaxis.set_major_formatter(fmt)

# ax.xaxis.set_major_formatter(
#     mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
fig.autofmt_xdate(rotation = 0, ha = 'center')
# plt.legend()
plt.savefig(figDir+'radar_dailySIC2.png', dpi = 500, bbox_inches = 'tight')





fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(12, 12))
trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)

#--- Daily Data ---#


error_CDR_2223_min_Week, error_CDR_2223_max_Week = SIC_MovAve_Week_CDR_2223 - STD_MA_Week_CDR_2223, SIC_MovAve_Week_CDR_2223 + STD_MA_Week_CDR_2223
error_CDR_2223_min_Week[error_CDR_2223_min_Week < 0], error_CDR_2223_max_Week[error_CDR_2223_max_Week > 1] = np.nan, np.nan

error_Modis_2223_min_Week, error_Modis_2223_max_Week = SIC_MovAve_Week_Modis_2223 - STD_MA_Week_Modis_2223, SIC_MovAve_Week_Modis_2223 + STD_MA_Week_Modis_2223
error_Modis_2223_min_Week[error_Modis_2223_min_Week < 0], error_Modis_2223_max_Week[error_Modis_2223_max_Week > 1] = np.nan, np.nan

error_CDR_2223_min_Month, error_CDR_2223_max_Month = SIC_MovAve_Month_CDR_2223 - STD_MA_Month_CDR_2223, SIC_MovAve_Month_CDR_2223 + STD_MA_Month_CDR_2223
error_CDR_2223_min_Month[error_CDR_2223_min_Month < 0], error_CDR_2223_max_Month[error_CDR_2223_max_Month > 1] = np.nan, np.nan

error_Modis_2223_min_Month, error_Modis_2223_max_Month = SIC_MovAve_Month_Modis_2223 - STD_MA_Month_Modis_2223, SIC_MovAve_Month_Modis_2223 + STD_MA_Month_Modis_2223
error_Modis_2223_min_Month[error_Modis_2223_min_Month < 0], error_Modis_2223_max_Month[error_Modis_2223_max_Month > 1] = np.nan, np.nan

ax1.set_title('Daily')
ax1.minorticks_on()
ax1.grid(which='major')
ax1.plot(days_combined, SIC_2223_mean, color='b', marker='o', markersize=2, label='Radar')
ax1.plot(days_combined, SIC_Modis_2223, color='r', marker='o', markersize=2, label='Modis-AMSR2')
ax1.plot(days_combined, SIC_CDR_2223, color='g', marker='o', markersize=2, label=' CDR')
ax1.fill_between(days_combined, error_CDR_2223_min, error_CDR_2223_max, color = 'g', alpha = 0.5)
ax1.fill_between(days_combined, error_Modis_2223_min, error_Modis_2223_max, color = 'r', alpha = 0.5)
ax1.text(0.0, .98, 'a)', transform=ax1.transAxes + trans,
        fontsize='large', va='bottom')

#--- 1 week running mean ---#
ax2.set_title(f'{size_MA_week*2+1} days Running Mean')
ax2.minorticks_on()
ax2.grid(which='major')
ax2.plot(days_combined, SIC_MovAve_Week_2223, color='b', marker='o', markersize=2, label='Radar')
ax2.plot(days_combined, SIC_MovAve_Week_Modis_2223, color='r', marker='o', markersize=2, label='Modis-AMSR2')
ax2.plot(days_combined, SIC_MovAve_Week_CDR_2223, color='g', marker='o', markersize=2, label='CDR')
ax2.fill_between(days_combined, error_CDR_2223_min_Week, error_CDR_2223_max_Week, color = 'g', alpha = 0.5)
ax2.fill_between(days_combined, error_Modis_2223_min_Week, error_Modis_2223_max_Week, color = 'r', alpha = 0.5)
ax2.text(0.0, .98, 'b)', transform=ax2.transAxes + trans,
        fontsize='large', va='bottom')

#--- Monthly running mean ---#
ax3.set_title(f'{size_MA_Month*2+1} days Running Mean')
ax3.minorticks_on()
ax3.grid(which='major')
ax3.plot(days_combined, SIC_MovAve_Month_2223, color='b', marker='o', markersize=2, label='Radar')
ax3.plot(days_combined, SIC_MovAve_Month_Modis_2223, color='r', marker='o', markersize=2, label='Modis-AMSR2')
ax3.plot(days_combined, SIC_MovAve_Month_CDR_2223, color='g', marker='o', markersize=2, label='CDR')
ax3.fill_between(days_combined, error_CDR_2223_min_Month, error_CDR_2223_max_Month, color = 'g', alpha = 0.5)
ax3.fill_between(days_combined, error_Modis_2223_min_Month, error_Modis_2223_max_Month, color = 'r', alpha = 0.5)
ax3.text(0.0, .98, 'c)', transform=ax3.transAxes + trans,
        fontsize='large', va='bottom')

lines, labels = ax2.get_legend_handles_labels()
labels = ['Radar', 'MODIS-AMSR2', 'CDR']
fig.legend(lines, labels, loc='lower center', ncol=int(len(labels)), bbox_to_anchor = (0.5, -0.065),bbox_transform=fig.transFigure)
fig.supylabel('Sea Ice Concentration')
plt.tight_layout(h_pad=2.0)
plt.xlabel('Julian Days')
plt.savefig(figDir+'timeseries_RADSAT_tot.png', dpi=500, bbox_inches='tight')


# -- Moving Average --#

linreg_CDR = SIC_MovAve_Week_2022 * lin_reg_CDR_MA.slope + lin_reg_CDR_MA.intercept
linreg_CDR[np.where(linreg_CDR > 1)] = np.nan

#! here

plt.clf()
# #------- Fig. 4 --------#

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6.5),sharex=True, sharey=True)

trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
axes = [ax1, ax2]



sc = ax1.scatter(SIC_2223_mean, SIC_Modis_2223, c=days_2223, cmap='inferno')
ax1.set_ylabel('Merged MODIS-AMSR2 SIC')
ax1.text(0.0, .98, 'a)', transform=ax1.transAxes + trans,
        fontsize='large', va='bottom')


ax2.scatter(SIC_2223_mean, SIC_MovAve_Week_CDR_2223, c=days_2223, cmap='inferno')
ax2.set_ylabel('CDR SIC')
ax2.text(0.0, .98, 'b)', transform=ax2.transAxes + trans,
        fontsize='large', va='bottom')

cax = fig.add_axes([ax2.get_position().x1+0.1,ax2.get_position().y0-0.01,0.02,ax2.get_position().y1-0.08])
cbar = fig.colorbar(sc, cax=cax, label = 'Days of the year')

for ax in axes : 
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid()
    ax.axis('square')
    
fig.tight_layout()
fig.suptitle('Daily')
fig.supxlabel('Radar SIC')

plt.savefig(figDir+'Comparison_dailySIC.png', dpi = 500, bbox_inches = 'tight')

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize = (14, 10) ,sharex=True, sharey=True)#, layout = 'constrained'
trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
axes = [ax1, ax2, ax3, ax4, ax5, ax6]

#Starting with daily SIC
sc = ax1.scatter(SIC_2223_mean, SIC_Modis_2223, c=days_2223, cmap='inferno')
ax1.set_ylabel('Merged MODIS-AMSR2 SIC')
ax1.text(0.0, .98, 'a)', transform=ax1.transAxes + trans,
        fontsize='large', va='bottom')
ax1.set_title('Daily')

ax4.scatter(SIC_2223_mean, SIC_MovAve_Week_CDR_2223, c=days_2223, cmap='inferno')
ax4.set_ylabel('CDR SIC')
ax4.text(0.0, .98, 'b)', transform=ax4.transAxes + trans,
        fontsize='large', va='bottom')

ax2.scatter(SIC_MovAve_Week_2223, SIC_MovAve_Week_Modis_2223, c=days_2223, cmap='inferno')
ax2.text(0.0, .98, 'c)', transform=ax2.transAxes + trans,
        fontsize='large', va='bottom')
ax2.set_title(f'{size_MA_week*2+1} Days Running Mean')

ax5.scatter(SIC_MovAve_Week_2223, SIC_MovAve_Week_CDR_2223, c=days_2223, cmap='inferno')
ax5.text(0.0, .98, 'd)', transform=ax5.transAxes + trans,
        fontsize='large', va='bottom')
ax5.set_xlabel('Radar SIC')

ax3.scatter(SIC_MovAve_Month_2223, SIC_MovAve_Month_Modis_2223, c=days_2223, cmap='inferno')
ax3.text(0.0, .98, 'e)', transform=ax3.transAxes + trans,
        fontsize='large', va='bottom')
ax3.set_title(f'{size_MA_Month*2+1} Days Running Mean')

ax6.scatter(SIC_MovAve_Month_2223, SIC_MovAve_Month_CDR_2223, c=days_2223, cmap='inferno')
ax6.text(0.0, .98, 'f)', transform=ax6.transAxes + trans,
        fontsize='large', va='bottom')

cax = fig.add_axes([ax3.get_position().x1+0.1,ax6.get_position().y0-.05,0.02,ax6.get_position().y1+0.45])
cbar = fig.colorbar(sc, cax=cax, label = 'Days of the year')

for ax in axes : 
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid()
    ax.axis('square')
    
fig.tight_layout()
plt.savefig(figDir+'Comparison_SIC.png', dpi = 500, bbox_inches = 'tight')

#----- Figures for Presentation ------#

# lines, labels = ax2.get_legend_handles_labels()
# labels = ['Lin. Reg. (2022-2023)']

# fig.legend(lines, labels, loc='center', ncol=len(labels),  bbox_transform=fig.transFigure, bbox_to_anchor=(0.42, 0.51))
# plt.savefig('test.png', dpi=500)

# mpl.rcParams['axes.spines.right'] = False
# mpl.rcParams['axes.spines.top'] = False
# plt.figure(figsize = (12, 6))
# plt.annotate('CDR', (0, 0.5), color = 'g', fontsize = 'large', fontweight = 'bold')
# plt.annotate('Radar', (0, 0.45), color = 'b', fontsize = 'large', fontweight = 'bold')
# plt.annotate('Modis-AMSR ', (0, 0.40), color = 'r', fontsize = 'large', fontweight = 'bold')
# plt.plot(days_combined, SIC_MovAve_Week_2223, color='b', marker='o', markersize=2, label='Radar')
# plt.plot(days_combined, SIC_MovAve_Week_Modis_2223, color='r', marker='o', markersize=2, label='Modis-AMSR2')
# plt.plot(days_combined, SIC_MovAve_Week_CDR_2223, color='g', marker='o', markersize=2, label='CDR')
# # plt.fill_between(days_combined, SIC_MovAve_Week_CDR_min_2223, SIC_MovAve_Week_CDR_max_2223, alpha = 0.5, color = 'g'


days = np.arange('2022-01', '2023-06', dtype='datetime64[D]')
fmt = mdates.DateFormatter('%m/%d')

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot()


ax.plot(days, SIC_2223_mean, color='b', marker='o', markersize=2, label='Radar')
ax.plot(days, SIC_Modis_2223, color='r', marker='o', markersize=2, label='Modis-AMSR2')
ax.plot(days, SIC_CDR_2223, color='g', marker='o', markersize=2, label=' CDR')
ax.fill_between(days, error_CDR_2223_min, error_CDR_2223_max, color = 'g', alpha = 0.5)
ax.fill_between(days, error_Modis_2223_min, error_Modis_2223_max, color = 'r', alpha = 0.5)
ax.set_xlabel('Day of year')
ax.set_ylabel('Sea Ice Concentration')

ax.set_xlim(200+30*4+52*365, 200+30*4+4*30+52*365)
# ax.text(0.4, 0.5,'CDR', color = 'g', fontsize = 'medium', fontweight = 'bold', transform=ax.transAxes)
# ax.text(0.4, 0.45, 'Radar', color = 'b', fontsize = 'medium', fontweight = 'bold', transform=ax.transAxes)
# ax.text(0.4, 0.40, 'Modis-AMSR2 ', color = 'r', fontsize = 'medium', fontweight = 'bold', transform=ax.transAxes)
ax.xaxis.set_major_formatter(fmt)

# ax.xaxis.set_major_formatter(
#     mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
fig.autofmt_xdate(rotation = 45, ha = 'center')

plt.savefig(figDir+'radar_dailySIC_presentation_winter2023.png', dpi = 500, bbox_inches = 'tight')








