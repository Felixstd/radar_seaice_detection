import os 
import warnings
import numpy as np
import netCDF4 as nc
import xarray as xr
import plot_radar as plr

from pykml import parser
from shapely import geometry
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
"""
The latitude and longitude of the bank building in Utiqiagvik are respectively : 
71.319770, -156.706460
"""

#--- Data for the projection of MODIS---#
a = 6378273.0
b = 6356889.449
lat_ts = 70.0
lat_0 = 90.0
lon_0 = 315.0
x_0 = 2656249.8497792133
y_0 = 2493750.112961679
proj = 'npstere'

#--- Data for the projection of CDR ---#
proj_CDR = 'npstere'
lat_0_CDR = 90
lat_ts_CDR = 70
lon_0_CDR = -45
k_CDR = 1
x_0_CDR = 0
y_0_CDR = 0
a_CDR = 6378273
b_CDR = 6356889.449


def readKML(kml_filename) : 
    
    """
    A KML file was produced using https://www.freemaptools.com/radius-around-point.htm. It creates a circle around whatever location 
    that you choose. So here, a circle of arounf 12 kilometers was produced around the bank building in utqiagvik to simulate the 
    radar field of view. 
    
    A shapely polygon is created with the latitudes and the longitudes of the circle. 
    """
    kml_file = 'rad_around_utq.kml'

    with open(kml_filename) as f :
        doc = parser.parse(f)
        
    root = doc.getroot()

    #finding the coordinates 
    coords = root.Document.Placemark.Polygon.outerBoundaryIs.LinearRing.coordinates.text
    coords = coords.split(',0') #splitting the string to get each coordinates

    #loop to get the latitudes and longitudes of the circle.
    latitudes_field_view = []
    longitudes_field_view = []
    Polygon_radar_pts = []
    for coor in coords[:-1] : 
        coor = coor.split(',')
        longitude = float(coor[0])
        latitude = float(coor[1])
        
        latitudes_field_view.append(latitude)
        longitudes_field_view.append(longitude)
        Polygon_radar_pts.append([longitude, latitude])
        
    #creating the polygon
    polygon_radar_fieldview = geometry.Polygon(Polygon_radar_pts)
    
    return latitudes_field_view, longitudes_field_view, polygon_radar_fieldview

def finding_interestSIC_point(latitudes, longitudes, polygon_radar) :
    
    width, height = np.shape(latitudes)
    
    idx_point_SIC = []
    for i in range(width) : 
        for j in range(height) : 
            
            latitude = latitudes[i, j]
            longitude = longitudes[i, j]
            
            if polygon_radar.contains(geometry.Point((longitude,latitude))) : 
                idx_point_SIC.append([i, j])
                
    return idx_point_SIC

def SIC_Modis(SICDir, ParamsDir, PlotDir, latitudes, longitudes, idx_interest_SIC, polygon, lon_0, lat_ts, lon_utq, lat_utq, plotting = False) :
    
    weeks = np.arange(0, 365, 7)

    for dir in os.listdir(SICDir) : 
        
        if (int(dir) == 2022) or (int(dir) == 2023) : 
        
            print('Now Analysing SIC from : ', dir)
            
            dir = str(dir)
            directory_SIC = SICDir+dir+'/'
            
            it = 0
            week_count = 0
            average_SIC_year = []
            average_STD_year = []
            SIC_min_year = []
            SIC_max_year = []
            for file in os.listdir(directory_SIC) : 
                
                day_analysing = file[-11:-3]
                day_formatted = day_analysing[:4]+'_'+day_analysing[4:6]+'_'+day_analysing[6:]
                
                print('Now analysing : ', day_formatted)

                SIC_Arctic_Day = nc.Dataset(directory_SIC+file, mode = 'r')

                SIC_x, SIC_y = SIC_Arctic_Day['x'][:], SIC_Arctic_Day['y'][:]

                SIC_arctic = np.flipud(SIC_Arctic_Day['sic_merged'][:].astype(float))/100
                SIC_arctic[np.where(SIC_arctic == np.amax(SIC_arctic))] = np.nan #the continent are 120, so removing those
                
                Unc_Arctic = np.flipud(SIC_Arctic_Day['unc_sic_merged'][:].astype(float))/100
                Unc_Arctic[np.where(Unc_Arctic == np.amax(Unc_Arctic))] = np.nan

                # dict_day = {'x' : SIC_x, 'y' : SIC_y, 'SIC' : SIC_arctic}
                # np.save(day_formatted+'_MOD'+'.npy', dict_day)

                latitudes_radar = []
                longitudes_radar = []
                SIC_Radar = []
                STD_Radar = []
                for idx in idx_interest_SIC : 

                    latitudes_radar.append(latitudes[idx[0], idx[1]])
                    longitudes_radar.append(longitudes[idx[0], idx[1]])
                    SIC_Radar.append(SIC_arctic[idx[0], idx[1]])
                    STD_Radar.append(Unc_Arctic[idx[0], idx[1]])
                # print(STD_Radar)
                    
                SIC_Radar = np.asarray(SIC_Radar)
                SIC_Radar = SIC_Radar[np.where(SIC_Radar >= 0)[0]]
                
                min_SIC_Radar = np.amin(SIC_Radar)
                max_SIC_Radar = np.amax(SIC_Radar)
                
                STD_Radar = np.asarray(STD_Radar)
                
                average_SIC_radar = np.mean(SIC_Radar)
                average_STD_radar = np.nanmean(STD_Radar)
                
                if plotting : 
                    
                    plr.plot_SIC_Modis(SIC_arctic, SIC_x, SIC_y, polygon, lon_0, lat_ts, lon_utq, lat_utq, 'Modis_Merged'+day_formatted, PlotDir)
                
                average_SIC_year.append(average_SIC_radar)
                average_STD_year.append(average_STD_radar)
                SIC_max_year.append(max_SIC_Radar)
                SIC_min_year.append(min_SIC_Radar)
                
            dict_ICE = {'SIC' : average_SIC_year, 'STD' : average_STD_year, 'min' : SIC_min_year, 'max' : SIC_max_year}
            
            np.save(ParamsDir+str(dir)+'_MODIS_2.npy', dict_ICE)
            
                # if it > 0 :
                #     if it % 7 == 0 : 
                #         average_week = average_SIC_year[weeks[week_count]:weeks[week_count+1]]
                #         week_dict = {'SIC' : average_week}
                #         np.save(ParamsDir+dir+'week'+str(week_count)+'.npy', week_dict)
                #         plr.plot_SIC(average_week, str(week_count), dir, 'Modis', PlotDir)
                #         week_count +=1
                    
                # it += 1
    
    return 

def SIC_CDR(SICDir, ParamsDir, PlotDir, latitudes, longitudes, idx_interest_SIC, polygon, lon_0, lat_ts, lon_utq, lat_utq, plotting = False) : 
    
    weeks = np.arange(0, 365, 7)
    
    for dir in os.listdir(SICDir) : 
        
        dir = int(dir)
        if (dir == 2022) or (dir == 2023) : 
        
            print('Now Analysing SIC from : ', dir)
            
            dir_str = str(dir)
            directory_SIC = SICDir+dir_str+'/'
            
            it = 0
            week_count = 0
            average_SIC_year = []
            average_STD_year = []
            max_SIC_year = []
            min_SIC_year = []
            for file in os.listdir(directory_SIC) : 
                
                file_splitted = file.split('_')
                
                if dir < 2023 : 
                    day_analysing = file_splitted[4]
                
                else : 
                    day_analysing = file_splitted[5]
                
                day_formatted = day_analysing[:4]+'_'+day_analysing[4:6]+'_'+day_analysing[6:]
                
                print('Now analysing : ', day_formatted)
                
                SIC_Arctic_Day = nc.Dataset(directory_SIC+file, mode = 'r')
                # print(SIC_Arctic_Day)
                
                SIC_x, SIC_y = SIC_Arctic_Day['xgrid'][:], SIC_Arctic_Day['ygrid'][:]

                SIC_arctic = SIC_Arctic_Day['cdr_seaice_conc'][0]
                
                STD_SIC = SIC_Arctic_Day['stdev_of_cdr_seaice_conc'][0]

                latitudes_radar = []
                longitudes_radar = []
                SIC_Radar = []
                SIC_STD = []
                
                for idx in idx_interest_SIC : 

                    latitudes_radar.append(latitudes[idx[0], idx[1]])
                    longitudes_radar.append(longitudes[idx[0], idx[1]])
                    SIC_Radar.append(SIC_arctic[idx[0], idx[1]])
                    SIC_STD.append(STD_SIC[idx[0], idx[1]])
                    
                SIC_Radar = np.asarray(SIC_Radar)
                SIC_Radar = SIC_Radar[np.where(SIC_Radar >= 0)[0]]
                
                SIC_STD = np.asarray(SIC_STD)
                SIC_STD = SIC_STD[~np.isnan(SIC_STD)]
                
                average_SIC_radar = np.mean(SIC_Radar)
                max_SIC_radar = np.amax(SIC_Radar)
                min_SIC_radar = np.min(SIC_Radar)
                average_STD_radar = np.mean(SIC_STD)
                
                average_SIC_year.append(average_SIC_radar)
                average_STD_year.append(average_STD_radar)
                max_SIC_year.append(max_SIC_radar)
                min_SIC_year.append(min_SIC_radar)
                
                it += 1
                
            dict_ICE = {'SIC' : average_SIC_year, 'max' : max_SIC_year, 'min' : min_SIC_year, 'STD' : average_STD_year}
            np.save(ParamsDir+str(dir)+'_CDR_3.npy', dict_ICE)
            
def find_neighbors_utq(utq_point, latitudes, longitudes, radius) : 
    
    idx_j_point = utq_point[1]
    idx_i_point = utq_point[0]
    
    latitudes_neighbors = []
    longitudes_neighbors = []
    for j in range(idx_j_point - 1 - radius, idx_j_point + radius) : 
        for i in range(idx_i_point - 1 - radius, idx_i_point + radius) : 
            
            latitudes_neighbors.append(latitudes[i, j])
            longitudes_neighbors.append(longitudes[i, j])
            
    return longitudes_neighbors, latitudes_neighbors

SICDir_MODIS = '/storage/fstdenis/SIC_MODIS_AMSR2_Daily/'
ParamsDir_MODIS = '/storage/fstdenis/Barrow_RADAR/saved_run/SIC_MODIS/params/'
PlotDir_MODIS = '/aos/home/fstdenis/ICE_RADAR/Figures_SIC/MODIS/'

SICDir_CDR = '/storage/fstdenis/SIC_CDR_Daily/'
ParamsDir_CDR = '/storage/fstdenis/Barrow_RADAR/saved_run/SIC_CDR/params/'
PlotDir_CDR = '/aos/home/fstdenis/ICE_RADAR/Figures_SIC/CDR/'

coordinates_file = 'coordinates_npstere_1km_arctic.nc'
coordinates_file_CDR = 'G02202-cdr-ancillary-nh.nc'
kml_file = 'rad_around_utq.kml'
kml_file_CDR = 'rad_around_utq_CDR_3.kml'


lat_utq, lon_utq = 71.319770, -156.706460

#--- Reading the kml file for the radar field of view ---#
print('Reading the KML file for the radar field of view')
latitudes_field_view, longitudes_field_view, polygon_radar_fieldview = readKML(kml_file)
latitudes_field_view_CDR, longitudes_field_view_CDR, polygon_radar_fieldview_CDR = readKML(kml_file_CDR)

print('Analysing MODIS SIC')

# # print('Reading the coordinates file, taking lat and lon')
coordinates = nc.Dataset(coordinates_file, mode='r')

latitudes = coordinates['lat'][:]
longitudes = coordinates['lon'][:]

idx_grid_SIC = np.load(ParamsDir_MODIS+'idx_polygon_MODIS_3.npy', allow_pickle=True).item()['idx_grid']

latitudes_grid  = latitudes[idx_grid_SIC[:, 0], idx_grid_SIC[:, 1]]
longitudes_grid = latitudes[idx_grid_SIC[:, 0], idx_grid_SIC[:, 1]]





# latitudes_arr   = latitudes_grid[:, 0]
# longitudes_arr = longitudes_grid[0, :]
# print(nc.Dataset(SICDir_MODIS+'2022/'+'sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_20221221.nc', mode = 'r'))
point_MODIS = 0
if point_MODIS:
    MOD = np.flipud(nc.Dataset(SICDir_MODIS+'2022/'+'sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_20221221.nc', mode = 'r')['sic_merged'][:])
    idx_point_SIC = finding_interestSIC_point(latitudes, longitudes, polygon_radar_fieldview)
    idx_point_SIC = np.array(idx_point_SIC)
    idx_point_SIC[:, 0] = idx_point_SIC[:, 0]-1
    i = 0
    del_idx = []
    for idx in idx_point_SIC:
        sic_value =  MOD[idx[0], idx[1]]

        if sic_value == 127:
            del_idx.append(i)
        
        i+=1
    idx_point_SIC = np.delete(idx_point_SIC, del_idx, axis = 0)
    point = {'Idx' : idx_point_SIC}
    np.save('idx_polygon_MODIS_2.npy', point)

idx_point_SIC = np.load('idx_polygon_MODIS_3.npy', allow_pickle=True).item()['idx_points']
# SIC_Modis(SICDir_MODIS, ParamsDir_MODIS, PlotDir_MODIS, latitudes, longitudes, idx_point_SIC, polygon_radar_fieldview, \
#         lon_0, lat_ts, lon_utq, lat_utq, plotting = False)


point_interest = False
if point_interest:
    idx_i = np.arange(3237, 3258)
    idx_j = np.arange(748, 770)
    point_i, point_j = np.meshgrid(idx_i, idx_j)
    idx_point_grid = np.vstack((point_i.flatten(), point_j.flatten())).T
    point = {'idx_grid' : idx_point_grid, 'idx_points': idx_point_SIC, 'idx_i' : idx_i, 'idx_j' : idx_j}
    np.save(ParamsDir_MODIS+'idx_polygon_MODIS_4.npy', point)
    
    

# MOD = np.flipud(nc.Dataset(SICDir_MODIS+'2022/'+'sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_20221221.nc', mode = 'r')['sic_merged'][:])
# SIC_modis_2 = np.zeros_like(MOD)*np.nan
# SIC_modis_2[idx_point_SIC[:, 0], idx_point_SIC[:, 1]] = MOD[idx_point_SIC[:, 0], idx_point_SIC[:, 1]]

# SIC_modis_radar = SIC_modis_2[idx_point_grid[:, 0], idx_point_grid[:, 1]]
# SIC_modis_radar = np.reshape(SIC_modis_radar, (len(idx_point_grid[:, 0]), len(idx_j)))


#---- EASE GRID ----# 

# coordinates_3km = xr.open_dataset('./NSIDC_EASE/NSIDC0772_LatLon_EASE2_N03km_v1.0.nc', mode = 'r')

# latitudes_3km = coordinates_3km.latitude.data
# longitudes_3km = coordinates_3km.longitude.data
# x = coordinates_3km.x.data
# y= coordinates_3km.y.data


# point_interest_3km = finding_interestSIC_point(latitudes_3km, longitudes_3km, polygon_radar_fieldview)
# idx_point_interest_3km = np.array(point_interest_3km)
# point = {'idx_interest' : idx_point_interest_3km}
# np.save('idx_3kmresolution.npy', point)




# idx_point_3km = np.load('idx_3kmresolution.npy', allow_pickle=True).item()['idx_interest']


idx_i = np.arange(2361, 2368)
# idx_j = np.arange(2723, 2730)
# point_i, point_j = np.meshgrid(idx_i, idx_j)
# idx_point_grid = np.vstack((point_i.flatten(), point_j.flatten())).T
# crs_epsg_3km= ccrs.LambertAzimuthalEqualArea(central_longitude=0.0, central_latitude=90.0, false_easting=0.0, false_northing=0.0)

# sic_3km = np.zeros_like(latitudes_3km)
# sic_3km[idx_point_grid[:, 0], idx_point_grid[:, 1]] = 1

# point = {'idx_grid' : idx_point_grid, 'idx_points': idx_point_3km, 'idx_i' : idx_i, 'idx_j' : idx_j}
# np.save('idx_3kmresolution.npy', point)

# plt.figure()
# ax1 = plt.axes(projection = crs_epsg_3km)
# cs = ax1.pcolormesh(x, y, sic_3km, 
#                     cmap = plt.cm.magma , transform = crs_epsg_3km, edgecolors = 'gray', hatch = '/', alpha = 0.5)
# #plotting Utqiagvik location
# ax1.plot(lon_utq, lat_utq, transform = ccrs.PlateCarree(), color = 'r', \
#     marker = '*', label = 'Utqiagvik')
# ax1.plot(*polygon_radar_fieldview.exterior.xy, color = 'r', \
#     transform = ccrs.PlateCarree(), linewidth = 2, label = 'Radar')                                  
# ax1.set_extent([-157.5, -156, 71.2, 71.4],ccrs.PlateCarree())
# ax1.add_feature(cfeature.LAND, color = 'peru')
# ax1.coastlines()
# plt.savefig('3km_grid.png', dpi = 500)


# print(point_interest_3km)
#--- Testing for CDR ---#

print('Analysing CDR SIC')
print('Reading the coordinates file, taking lat and lon')

coordinates_CDR = nc.Dataset(coordinates_file_CDR, mode = 'r')

latitudes_CDR = coordinates_CDR['latitude']
longitudes_CDR = coordinates_CDR['longitude']

point_CDR = 0
if point_CDR:

    idx_point_SIC_CDR = finding_interestSIC_point(latitudes_CDR, longitudes_CDR, polygon_radar_fieldview_CDR)
    idx_point_SIC_CDR = np.array(idx_point_SIC_CDR)
    idx_point_SIC_CDR[:, 0] = idx_point_SIC_CDR[:, 0]-1
    idx_point_SIC_CDR = np.delete(idx_point_SIC_CDR, [2,1], axis=0)
    point = {'Idx' : idx_point_SIC_CDR}
    np.save(ParamsDir_CDR+'idx_polygon_CDR_3.npy', point)

idx_point_SIC_CDR = np.load(ParamsDir_CDR+'idx_polygon_CDR_3.npy', allow_pickle=True).item()['Idx']
# SIC_CDR(SICDir_CDR, ParamsDir_CDR, PlotDir_CDR, latitudes_CDR, longitudes_CDR, idx_point_SIC_CDR, polygon_radar_fieldview, \
#     lon_0_CDR, lat_ts_CDR, lon_utq, lat_utq, plotting = False)


# # --- Plotting ---#
plotting = 0
if plotting:
    CDR = np.load('/storage/fstdenis/Barrow_RADAR/ICE_RADAR_MISC/Figures_SIC/2022_01_02_CDR.npy', allow_pickle=True).item()
    MOD = np.load('/storage/fstdenis/Barrow_RADAR/ICE_RADAR_MISC/Figures_SIC/2022_01_02_MOD.npy', allow_pickle=True).item()

    SIC_CDR, SIC_CDR_x, SIC_CDR_y = CDR['SIC'], CDR['x'], CDR['y']
    SIC_MOD, SIC_MOD_x, SIC_MOD_y = MOD['SIC'], MOD['x'], MOD['y']

    # SIC_MOD = np.flipud(nc.Dataset(SICDir_MODIS+'2022/'+'sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_20221221.nc', mode = 'r')['sic_merged'][:])   

    SIC_MOD_l = [SIC_MOD, SIC_MOD_x, SIC_MOD_y]
    SIC_CDR_l = [SIC_CDR, SIC_CDR_x, SIC_CDR_y]

    proj_CDR_l = [lon_0_CDR, lat_ts_CDR] 
    proj_MOD_l = [lon_0, lat_ts] 
    # plr.plot_SAT(SIC_MOD_l, SIC_CDR_l, polygon_radar_fieldview, proj_CDR_l, proj_MOD_l, lon_utq, 
    #             lat_utq, idx_point_SIC_CDR, idx_point_SIC, './')

# plr.plot_SIC_Modis(SIC_CDR, SIC_CDR_x, SIC_CDR_y, polygon_radar_fieldview, lon_0_CDR, lat_ts_CDR, lon_utq, lat_utq, 'test', './')