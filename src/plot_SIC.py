import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


coordinates = nc.Dataset('./coordinates_npstere_1km_arctic.nc', mode='r')
MOD = nc.Dataset('/storage/fstdenis/SIC_MODIS_AMSR2_Daily/2022/'+'sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_20221221.nc', mode = 'r')

print(MOD)
SIC = MOD['sic_merged'][:]
SIC_x = MOD['x']
SIC_y = MOD['y']

crs_epsg_MOD = ccrs.NorthPolarStereo(central_longitude = 315, true_scale_latitude = 70)

plt.figure()
ax = plt.axes(projection = crs_epsg_MOD)
cs = plt.pcolormesh(SIC_x, SIC_y, np.flipud(SIC[:-1, :-1]), 
                    cmap=plt.cm.magma,  transform = crs_epsg_MOD, alpha = 0.5)
ax.coastlines()
ax.add_feature(cfeature.LAND, color = 'peru')
ax.set_extent([-157.5, -156, 71.2, 71.4],ccrs.PlateCarree())
plt.savefig('SIC_test.png')

