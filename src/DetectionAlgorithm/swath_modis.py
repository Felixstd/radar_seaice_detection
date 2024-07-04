#%%

import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
import xarray as xr
import cmocean
import cartopy.crs as ccrs

from pyhdf import SD  # Import the HDF library
# Reading the file:

    

#%%

SwathDir = '/storage/fstdenis/Barrow_RADAR/Individual_Swaths_Modis/20220325/'
file =  SwathDir+'MOD02QKM.A2022084.2155.061.2022091163622.hdf'
# print(rxr.open_rasterio(file))/


hdf=SD.SD(file)

# Get lat and lon info
lat = hdf.select('Latitude')
latitude = lat[:]
min_lat=latitude.min()
max_lat=latitude.max()
lon = hdf.select('Longitude')
longitude = lon[:]
min_lon=longitude.min()
max_lon=longitude.max()

sds=hdf.select('EV_250_RefSB')

attributes=sds.attributes()

print(attributes)


# scale_factor=attributes['scale_factor']
#get valid range for AOD SDS
range=sds.getrange()
min_range=min(range)
max_range=max(range)

#get SDS data
data=sds.get()
#get data within valid range
valid_data=data.ravel()
valid_data=[x for x in valid_data if x>=min_range]
valid_data=[x for x in valid_data if x<=max_range]
valid_data=np.asarray(valid_data)
#scale the valid data
# valid_data=valid_data*scale_factor


#%%
print(valid_data)
crs_epsg_MOD = ccrs.NorthPolarStereo(central_longitude = -45, true_scale_latitude = 70)

# path = "MOD14.A2021085.0315.006.2021085095713.hdf"
# file = SD(file, SDC.READ)
# # Printing all the datasets names
# datasets_dic = file.datasets()
# for idx,sds in enumerate(datasets_dic.keys()):
#     print (idx,sds)
    
    
# sds_obj = file.select('EV_250_RefSB') # select sds  
# latitude = file.select('Latitude')
# longitude = file.select('Longitude')

# data = sds_obj.get() 
# # print(data)
# lat, lon = latitude.get(), longitude.get()
# print(data.shape)


# band_250m = datasets_dic['EV_250_RefSB']
# # print(data)
# plt.figure()
# plt.imshow(data[1], cmap = cmocean.cm.ice)
# plt.savefig('test_modis.png', dpi = 500, bbox_inches = 'tight')
    
# with xr.open_dataset(file) as file_hdf:
#     modis_pre = file_hdf.read(1)
# 
# ds = gd.Open('HDF4_SDS:UNKNOWN:"MOD021KM.A2013048.0750.hdf":6')
# data = ds.ReadAsArray()
# ds = None

# fig, ax = plt.subplots(figsize=(6,6))

# ax.imshow(data[0,:,:], cmap=plt.cm.Greys, vmin=1000, vmax=6000)
# modis_pre = rxr.open_rasterio(file, masked=True)
# 
# print(modis_pre)

# %%
