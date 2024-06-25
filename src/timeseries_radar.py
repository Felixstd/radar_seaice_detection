import os
import numpy as np


"""
This file is used to make the time series of the radar
sea ice concentration for 2022 and 2023. It reads all of the 
data files and combine them into 2 arrays. 
"""

# SavedDir = '/storage/fstdenis/Barrow_RADAR/saved_run'
SavedDir = '/storage/fstdenis/Barrow_RADAR/saved_run_Withoutborders/'
# SavedRadar = SavedDir + '/RADAR/saved_TimeSeries_SIC/'
SICDir = '/storage/fstdenis/Barrow_RADAR/saved_run/RADAR/saved_SIC/'


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