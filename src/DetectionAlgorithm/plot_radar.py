import cv2
import matplotlib.pyplot as plt
import numpy as np
import extract_video as ev
from shapely import geometry
from scipy.io import loadmat
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms
SMALL_SIZE = 8
MEDIUM_SIZE = 18
BIGGER_SIZE = 15

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)


def plot_lags(pearson_coefficients_lags, max_cor_idx, savedir, count) : 
    
    plt.figure()
    plt.axhline(0, linestyle = '--', color = 'k', linewidth = 1)
    plt.axvline(0, linestyle = '--', color = 'k', linewidth = 1)
    plt.plot(max_cor_idx[0]-21, max_cor_idx[1]-21, marker = 'x', color = 'pink')
    im2 = plt.imshow(pearson_coefficients_lags, extent=[-22,22,-22,22], cmap = 'jet')
    plt.xlabel('x lag (pixel)')
    plt.ylabel('y lag (pixel)')
    plt.colorbar(im2, label = 'pearson r')
    if count < 10 : 
        plt.savefig(savedir+'test_pearson_0{}.png'.format(count), dpi = 500)
    else :
        plt.savefig(savedir+'test_pearson_{}.png'.format(count), dpi = 500)
    plt.close()
    
def plot_typepixel(type_ice_vid, circle, img_list, savedir, start) : 
    
    # iter = 0
    # for it in range(start, end):
    
    
    # fig, ax = plt.subplots()
    # for it in range(0, len(type_ice_vid)) : 
    it = 0
    for type_ice in type_ice_vid :    
        # type_ice = type_ice_vid[it]
        
        a, b, r = circle
        
        type_ice_idx = np.where(type_ice != 0)
        type_ice_non_zero_idx = np.array((type_ice_idx[0], type_ice_idx[1])).T
        
        # with np.printoptions(threshold = np.inf) : 
        #     print(type_ice_non_zero_idx)
        
        plt.figure()
        plt.imshow(img_list[start + it], cmap = plt.cm.gray)
        for idx in type_ice_non_zero_idx : 
            # print(idx)
            
            type = type_ice[idx[0], idx[1]]
            # if idx[1] == 245 and idx[0] ==  75 : 
            #     print('Now test')
            
            if (idx[0] - a)**2 + (idx[1] - b)**2 <= r**2 :
                
            
                if type == 1 :
                    # if idx[1] == 245 and idx[0] ==  75 : 
                    #     # print(type)
                    # print('here 1')
                    plt.plot(idx[0], idx[1], marker = 'o', color = 'g', markersize = 1)

                elif type == 2 : 
                #     if idx[1] == 245 and idx[0] ==  75 : 
                #         print(type)
                #         print('here')
                    # print('here 2')
                    plt.plot(idx[0], idx[1], marker = 'o', color = 'r', markersize = 1)
                
                elif type == 3 : 
                    # print('here 3')
                    plt.plot(idx[0], idx[1], marker = 'o', color = 'b', markersize = 1)
            
        if start+it < 10 :    
            plt.savefig(savedir+'test_color_0{}.png'.format(start + it), dpi = 500, bbox_inches = 'tight')
        else : 
            plt.savefig(savedir+'test_color_{}.png'.format(start + it), dpi = 500, bbox_inches = 'tight')
        plt.close()
        
        it += 1
        
def plot_average(type_average_vid, circle, viddir, savedir) :
    
    # canvas_noice = ev.extract_first_frame(viddir)
    for it in range(0, len(type_average_vid)) : 
        
        type_average = type_average_vid[it]
        a, b, r = circle
        
        type_ice_idx = np.where(type_average != 0)
        type_ice_non_zero_idx = np.array((type_ice_idx[0], type_ice_idx[1])).T
        
        
        plt.figure()
        # plt.imshow(canvas_noice, cmap = plt.cm.gray)
        for idx in type_ice_non_zero_idx : 
            
            type = type_average[idx[1], idx[0]]
            
            if (idx[0] - b)**2 + (idx[1] - a)**2 <= r**2 :
                
            
                if type == 1 :
                    if (idx[1] == 75) and (idx[0] == 245) : 
                        print('wrong')
                    plt.plot(idx[1], idx[0], marker = 'o', color = 'g', markersize = 1)

                if type == 2 : 
                    plt.plot(idx[1], idx[0], marker = 'o', color = 'r', markersize = 1)
                
                if type == 3 : 
                    plt.plot(idx[1], idx[0], marker = 'o', color = 'b', markersize = 1)
                
                if type == 4 : 
                    plt.plot(idx[1], idx[0], marker = 'o', color = 'yellow', markersize = 1)
                    
                if type == 5 : 
                    plt.plot(idx[1], idx[0], marker = 'o', color = 'pink', markersize = 1)
                    
        if it < 10 : 
            plt.savefig(savedir+'average_0{}.png'.format(it), dpi = 500, bbox_inches = 'tight')
            
        else : 
            plt.savefig(savedir+'average_{}.png'.format(it), dpi = 500, bbox_inches = 'tight')
    
def plot_average_timeseries(area_ow_vid, area_still_vid, area_moved_vid, savedir, timeseries) : 
    
    time_steps = np.arange(0, len(area_ow_vid))
    
    plt.figure()
    
    plt.plot(time_steps, area_ow_vid/1e6, linestyle = '-', marker = 'o', color = 'b', label = r'Moved $\rightarrow$ At rest')
    plt.plot(time_steps, area_still_vid/1e6, linestyle = '-', marker = 'o', color = 'r', label = 'At Rest')
    plt.plot(time_steps, area_moved_vid/1e6, linestyle = '-', marker = 'o', color = 'g', label = 'Moving')
    
    plt.xlabel('Time Steps')
    plt.ylabel(r'Area (km$^2$)')

    plt.legend()
    plt.grid(linestyle = '--')
    plt.title(timeseries)
    plt.savefig(savedir+'timeseries_area.png', dpi = 500, bbox_inches = 'tight')
    
def plot_typeice(type_ice_vid, img_list, start_frame, circle, savedir) : 
    
    a, b, r = circle    
    
    for it in range(len(type_ice_vid)) : 
        
        type_ice = type_ice_vid[it]
        
        type_ice_idx = np.where(type_ice != 0)
        type_ice_non_zero_idx = np.array((type_ice_idx[0], type_ice_idx[1])).T
        
        
        plt.figure()
        plt.imshow(img_list[start_frame + it], cmap = plt.cm.gray)
        for idx in type_ice_non_zero_idx : 
            
            type = type_ice[idx[0], idx[1]]
            
            if (idx[0] - b)**2 + (idx[1] - a)**2 <= r**2 :
                
            
                if type == 1 :
                    plt.plot(idx[0], idx[1], marker = 'o', color = 'b', markersize = 1)

                if type == 2 : 
                    plt.plot(idx[0], idx[1], marker = 'o', color = 'r', markersize = 1)
    
        if (it+start_frame) < 0 :
            plt.savefig(savedir+'type_ice_0{}.png'.format(it+start_frame), dpi = 500)
        else :
            plt.savefig(savedir+'type_ice_{}.png'.format(it+start_frame), dpi = 500)
       
def plot_ice_timeseries(area_ice_vid, area_ow_vid, savedir, timeseries) : 
    
    
    time_steps = np.arange(0, len(area_ow_vid))
    
    plt.figure()
    
    plt.plot(time_steps, area_ow_vid/1e6, linestyle = '-', marker = 'o', color = 'b', label = r'Open-Water')
    plt.plot(time_steps, area_ice_vid/1e6, linestyle = '-', marker = 'o', color = 'r', label = 'Ice')
    
    plt.xlabel('Time Steps')
    plt.ylabel(r'Area (km$^2$)')

    plt.legend()
    plt.grid(linestyle = '--')
    plt.title(timeseries)
    plt.savefig(savedir+'timeseries_ice_area.png', dpi = 500, bbox_inches = 'tight')
    
def plot_edge_finding(img_list, polygon_list, start_frame, savedir) : 
    
    for it in range(len(img_list)) : 
        plt.figure()
        print('plotting : ', it)
        
        img = img_list[it]
        polygons = polygon_list[it]
        plt.imshow(img, cmap = plt.cm.gray)
        plt.axis('off')
        
        for polygon in polygons : 
            plt.plot(*polygon.exterior.xy, color = 'r')
            
        if (it+start_frame) < 10 :
            plt.savefig(savedir+'edge_ice_0{}.png'.format(it+start_frame), dpi = 500, bbox_inches = 'tight')
        else : 
             plt.savefig(savedir+'edge_ice_{}.png'.format(it+start_frame), dpi = 500, bbox_inches = 'tight')
        plt.close()


def plot_area_ice(tot_ice_vid, tot_ow_vid, concentration_ice, concentration_ow, title, savedir) : 
    
    time = np.arange(len(tot_ice_vid))
    
    plt.figure()
    plt.plot(time, tot_ice_vid, color = 'r', marker = 'o', label = 'Sea Ice')
    plt.plot(time, tot_ow_vid, color = 'b', marker = 'o', label = 'Open Water')
    plt.xlabel('Time Steps')
    plt.ylabel('Area (no units)')
    plt.title(title)
    plt.legend()
    plt.savefig(savedir+'area_timeseries.png', dpi = 500, bbox_inches = 'tight')
    
    plt.figure()
    plt.plot(time, concentration_ice, color = 'r', marker = 'o', label = 'Sea Ice')
    plt.plot(time, concentration_ow, color = 'b', marker = 'o', label = 'Open Water')
    plt.xlabel('Time Steps')
    plt.ylabel('Concentration')
    plt.title(title)
    plt.legend()
    plt.savefig(savedir+'conc_timeseries.png', dpi = 500, bbox_inches = 'tight')
    
    
def plot_SIC_Modis(SIC_arctic, SIC_x, SIC_y, polygon_radar_fieldview, lon_0, lat_ts, lon_utq, lat_utq, day_analysing, savedir) : 
    
    crs_epsg = ccrs.NorthPolarStereo(central_longitude = lon_0, true_scale_latitude = lat_ts)

    fig = plt.figure(figsize=[10, 10])
    ax = plt.axes(projection = crs_epsg)

    #adding grid lines for lat/lon
    gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
    # set the map projection and associated boundaries
    cs = ax.pcolorfast(SIC_x, SIC_y, SIC_arctic[:-1, :-1], 
                    cmap=plt.cm.viridis,  transform = crs_epsg)
    #plotting Utqiagvik location
    ax.plot(lon_utq, lat_utq, transform = ccrs.PlateCarree(), color = 'r', marker = '*', markersize = 2, label = 'Utqiagvik')
    ax.plot(*polygon_radar_fieldview.exterior.xy, color = 'b', transform = ccrs.PlateCarree(), linewidth = 0.5, label = 'Radar')                                  
    ax.set_extent([-157.5, -156, 71.1, 71.4],ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, color = 'peru')
    cbar = fig.colorbar(cs, ax = ax, location = 'bottom', shrink = 0.8, pad = 0.05)
    cbar.set_label('Sea Ice Concentration (%)')
    plt.title(day_analysing)
    ax.coastlines()
    plt.legend()   
    plt.savefig(savedir+'SIC_UTQ_'+day_analysing+'.png', dpi = 500, bbox_inches = 'tight')

    fig = plt.figure(figsize=[10, 10])
    ax = plt.axes(projection = crs_epsg)
    #adding grid lines for lat/lon
    gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
    # set the map projection and associated boundaries
    cs = ax.pcolorfast(SIC_x, SIC_y, SIC_arctic[:-1, :-1], 
                    cmap=plt.cm.viridis,  transform = crs_epsg)
    #plotting Utqiagvik location
    ax.plot(lon_utq, lat_utq, transform = ccrs.PlateCarree(), color = 'r', marker = '*', markersize = 2, label = 'Utqiagvik')
    ax.plot(*polygon_radar_fieldview.exterior.xy, color = 'b', transform = ccrs.PlateCarree(), linewidth = 0.5, label = 'Radar')
    ax.add_feature(cfeature.LAND, color = 'peru')
    cbar = fig.colorbar(cs, ax = ax, location = 'bottom', shrink = 0.8, pad = 0.05)
    cbar.set_label('Sea Ice Concentration (%)')
    plt.title(day_analysing)
    ax.coastlines()
    plt.legend()   
    plt.savefig(savedir+'SIC_'+day_analysing+'.png', dpi = 500, bbox_inches = 'tight')
    
def plot_SIC(average_week, week, year, type_SIC, savedir) : 
    
    days = np.arange(1, 8)
    
    plt.figure()
    plt.plot(days, average_week, color = 'r', marker = 'o')
    plt.xlabel('Days')
    plt.ylabel('Ice Concentration')
    plt.title(type_SIC+' week '+week+', '+year)
    plt.savefig(savedir+'SIC_'+year+'_'+week+'.png', dpi = 500, bbox_inches = 'tight')
    
    
def plot_SAT(SIC_MOD, SIC_CDR, polygon_radar_fieldview, proj_CDR, proj_MOD, lon_utq, 
             lat_utq, idx_interest_CDR, idx_interest_MODIS, savedir) : 
    
    
    lon_0_CDR, lat_ts_CDR = proj_CDR
    lon_0_MOD, lat_ts_MOD = proj_MOD
    
    crs_epsg_CDR = ccrs.NorthPolarStereo(central_longitude = lon_0_CDR, true_scale_latitude = lat_ts_CDR)
    crs_epsg_MOD = ccrs.NorthPolarStereo(central_longitude = lon_0_MOD, true_scale_latitude = lat_ts_MOD)

    fig = plt.figure(figsize = (10, 10))
    
    #--- CDR ---#
    ax1 = fig.add_subplot(121, projection = crs_epsg_CDR)

    idx_interest_CDR = np.array(idx_interest_CDR)
    SIC_CDR[0][:-1,:-1] = np.nan
    SIC_CDR[0][idx_interest_CDR[:, 0], idx_interest_CDR[:, 1]] = 100
    
    cs = ax1.pcolormesh(SIC_CDR[1], SIC_CDR[2], SIC_CDR[0][:-1,:-1], 
                        cmap = plt.cm.magma , transform = crs_epsg_CDR, edgecolors = 'gray', hatch = '/', alpha = 0.5)
    #plotting Utqiagvik location
    ax1.plot(lon_utq, lat_utq, transform = ccrs.PlateCarree(), color = 'r', \
        marker = '*', label = 'Utqiagvik')
    ax1.plot(*polygon_radar_fieldview.exterior.xy, color = 'r', \
        transform = ccrs.PlateCarree(), linewidth = 2, label = 'Radar')                                  
    ax1.set_extent([-157.5, -156, 71.2, 71.4],ccrs.PlateCarree())
    ax1.add_feature(cfeature.LAND, color = 'peru')
    ax1.set_title('CDR')
    ax1.coastlines()
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax1.text(0.0, .98, 'a)', transform=ax1.transAxes + trans,
            fontsize='medium', va='bottom')

    idx_interest_MODIS = np.array(idx_interest_MODIS)
    
    SIC_MOD[0][:-1, :-1] = np.nan
    SIC_MOD[0][idx_interest_MODIS[:, 0], idx_interest_MODIS[:, 1]] = 100
    ax2 = fig.add_subplot(122, projection = crs_epsg_MOD)

    cs = ax2.pcolormesh(SIC_MOD[1], SIC_MOD[2], SIC_MOD[0][:-1, :-1], 
                    cmap=plt.cm.magma,  transform = crs_epsg_MOD,  edgecolors = 'gray', alpha = 0.5)
    # #plotting Utqiagvik location
    ax2.plot(lon_utq, lat_utq, transform = ccrs.PlateCarree(), color = 'r', \
        marker = '*', label = 'Utqiagvik')
    ax2.plot(*polygon_radar_fieldview.exterior.xy, color = 'r', \
        transform = ccrs.PlateCarree(), linewidth = 2, label = 'Radar')                                  
    ax2.set_extent([-157.5, -156, 71.2, 71.4],ccrs.PlateCarree())
    ax2.add_feature(cfeature.LAND, color = 'peru')

    ax2.set_title('Merged MODIS-AMSR2')
    ax2.coastlines()
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax2.text(0.0, .98, 'b)', transform=ax2.transAxes + trans,
            fontsize='medium', va='bottom')

    
    plt.savefig(savedir+'SIC_zoomed.png', dpi = 500, bbox_inches = 'tight')
    
    
    fig = plt.figure(figsize = (10, 10))
    
    #--- CDR ---#
    ax1 = fig.add_subplot(121, projection = crs_epsg_CDR)

    idx_interest_CDR = np.array(idx_interest_CDR)
    SIC_CDR[0][:-1,:-1] = np.nan
    SIC_CDR[0][idx_interest_CDR[:, 0], idx_interest_CDR[:, 1]] = 100
    
    cs = ax1.pcolormesh(SIC_CDR[1], SIC_CDR[2], SIC_CDR[0][:-1,:-1], 
                        cmap = plt.cm.magma , transform = crs_epsg_CDR, edgecolors = 'gray', hatch = '/', alpha = 0.5)
    #plotting Utqiagvik location
    ax1.plot(lon_utq, lat_utq, transform = ccrs.PlateCarree(), color = 'r', \
        marker = '*', label = 'Utqiagvik', markersize = 8)
    ax1.plot(*polygon_radar_fieldview.exterior.xy, color = 'r', \
        transform = ccrs.PlateCarree(), linewidth = 2, label = 'Radar')                                  
    ax1.set_extent([-157.5, -156, 71.2, 71.4],ccrs.PlateCarree())
    ax1.add_feature(cfeature.LAND, color = 'peru')
    ax1.set_title('CDR - 25km')
    ax1.coastlines()
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    # ax1.text(0.0, .98, 'a)', transform=ax1.transAxes + trans,
    #         fontsize='medium', va='bottom')

    idx_interest_MODIS = np.array(idx_interest_MODIS)
    
    SIC_MOD[0][:-1, :-1] = np.nan
    SIC_MOD[0][idx_interest_MODIS[:, 0], idx_interest_MODIS[:, 1]] = 100
    ax2 = fig.add_subplot(122, projection = crs_epsg_MOD)

    cs = ax2.pcolormesh(SIC_MOD[1], SIC_MOD[2], SIC_MOD[0][:-1, :-1], 
                    cmap=plt.cm.magma,  transform = crs_epsg_MOD,  edgecolors = 'gray', alpha = 0.5)
    # #plotting Utqiagvik location
    ax2.plot(lon_utq, lat_utq, transform = ccrs.PlateCarree(), color = 'r', \
        marker = '*', label = 'Utqiagvik', markersize = 8)
    ax2.plot(*polygon_radar_fieldview.exterior.xy, color = 'r', \
        transform = ccrs.PlateCarree(), linewidth = 2, label = 'Radar')                                  
    ax2.set_extent([-157.5, -156, 71.2, 71.4],ccrs.PlateCarree())
    ax2.add_feature(cfeature.LAND, color = 'peru')

    ax2.set_title('Merged MODIS-AMSR2 - 1km')
    ax2.coastlines()
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    # ax2.text(0.0, .98, 'b)', transform=ax2.transAxes + trans,
    #         fontsize='medium', va='bottom')

    
    plt.savefig(savedir+'SIC_zoomed_presentation.png', dpi = 500, bbox_inches = 'tight')
    
   
   
def plot_analysts(concentration_analysers, concentration_vid_algo, max_slope, intercept, analysts, namefig) : 
    
    Lin_Reg = concentration_analysers*max_slope + intercept
    # Lin_Reg7_nan = concentration_analysers*LinReg_7_nan[0].coef_ 
    cmap = plt.get_cmap('RdBu', np.max(analysts) - np.min(analysts) + 1)
    fig, ax = plt.subplots()
    plt.ylim(0, 1)
    plt.xlim(0, 1)

    sc = plt.scatter(concentration_analysers, concentration_vid_algo, marker = 'o', c = analysts, cmap=cmap, vmin=np.min(analysts) - 0.5, 
                        vmax=np.max(analysts) + 0.5)
    plt.plot(concentration_analysers, Lin_Reg, color = 'green')
    plt.xlabel('Analysts SIC')
    plt.ylabel('Radar SIC')
    plt.grid()
    props = dict(boxstyle='round', facecolor = 'green')

    if round(intercept, 3) < 0:
        plt.text(0.45, 0.15, f"$y = {round(max_slope, 3)}x {round(intercept, 3)}$", transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    else:
        plt.text(0.45, 0.15, f"$y = {round(max_slope, 3)}x + {round(intercept, 3)}$", transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    props = dict(boxstyle='round', facecolor = 'slategray')

    cbar = plt.colorbar(sc, ticks=np.arange(np.min(analysts), np.max(analysts)+1))
    cbar.set_label('Analysts')

    plt.savefig(namefig, dpi = 500, bbox_inches = 'tight')
    
    # fig, ax = plt.subplots()
    # plt.ylim(0, 1)
    # plt.xlim(0, 1)
    # plt.grid()
    # sc = plt.scatter(concentration_analysers, concentration_vid_algo, marker = 'o', c = analysts, cmap=cmap, vmin=np.min(analysts) - 0.5, 
    #                     vmax=np.max(analysts) + 0.5)
    # plt.plot(concentration_analysers, Lin_Reg, color = 'green')
    # plt.xlabel('Analysts SIC')
    # plt.ylabel('Radar SIC')

    # cbar = plt.colorbar(sc, ticks=np.arange(np.min(analysts), np.max(analysts)+1))
    # cbar.set_label('Analysts')

    # plt.savefig(namefig, dpi = 500, bbox_inches = 'tight')
    
def plot_optimizing(X, Y, param_3, param_5, param_7, param_13, label, namefig) : 
    
    plt.figure(figsize = (12, 12))
    ax = plt.axes(projection='3d')
    ax.view_init(azim=25, elev = 30)
    surf = ax.plot_surface(X, Y, param_3, cmap='inferno_r', label = 'Kernel = (3,3)',  edgecolor = 'gray')
    surf2 = ax.plot_surface(X, Y, param_5, cmap='inferno_r', label = 'Kernel = (5,5)',  edgecolor = 'k')
    surf3 = ax.plot_surface(X, Y, param_7, cmap='inferno_r', label = 'Kernel = (7,7)',  edgecolor = 'darkcyan')
    surf4 = ax.plot_surface(X, Y, param_13, cmap='inferno_r', label = 'Kernel = (13,13)',  edgecolor = 'red')
    surf._facecolors2d = surf._edgecolor3d
    surf._edgecolors2d = surf._edgecolor3d 
    surf2._facecolors2d = surf2._edgecolor3d
    surf2._edgecolors2d = surf2._edgecolor3d 
    surf3._facecolors2d = surf3._edgecolor3d
    surf3._edgecolors2d = surf3._edgecolor3d 
    surf4._facecolors2d = surf4._edgecolor3d
    surf4._edgecolors2d = surf4._edgecolor3d 
    ax.set_xlabel('$T_{Low}$')
    ax.set_ylabel('$T_{High}$')
    cbar = plt.colorbar(surf, ax = ax, shrink = 0.8)
    cbar.set_label(label)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)  
    plt.show()
    plt.savefig(namefig, dpi = 500, bbox_inches = 'tight')
    

 
# viddir = '/storage/fstdenis/Barrow_RADAR/RADAR_2014/2014_0221_0228.mp4'
# viddir = '/storage/fstdenis/Barrow_RADAR/RADAR_2010/2010_1021_1031.mpg'
# video, num_frames, height, width, fps, nom_gray_list, img_list = ev.extract_video(viddir)

# print(np.min(nom_gray_list[567]))
# idx_neg = np.where(nom_gray_list[567] < (.+np.min(nom_gray_list[567])))
# idx_neg = np.array((idx_neg[0], idx_neg[1])).T
# print(idx_neg)
# plt.figure()
# im = plt.imshow(nom_gray_list[567], cmap = plt.cm.gray)
# for idx in idx_neg : 
#     print(idx)
#     plt.plot(idx[1], idx[0], marker = 'o', color = 'r', markersize = 1)
# plt.colorbar(im)
# plt.savefig('test.png', dpi = 500)

# data_radar = loadmat('/storage/fstdenis/Barrow_RADAR/IdentTypeIce/run27/VelocityField_MCC.mat')

# type_ice_vid = data_radar['Type']
# circle = data_radar['Circle'][0]
# img_list = data_radar['img']
# start, end = data_radar['Time'][0]
# print(type_ice_vid)

# plot_icetype(type_ice_vid, circle, img_list, './', start)

# cv2.imwrite('utqiagvik_noice.png', img_list[0][:-32])



# print(height, width)

# print(len(windows), len(next_windows))
    
# plotting_pearson = loadmat('plotting_lags.mat')
# pearson_vid = plotting_pearson['lags']
# delta_xy_pixel = plotting_pearson['pixel'][0][0]
# windows = plotting_pearson['windows']
# next_windows = plotting_pearson['next_windows']
# nom_gray_list = plotting_pearson['img']

# plt.figure()

# plt.plot((windows[150][0][0], windows[150][0][0]), (windows[150][1][0], windows[150][1][1]), color = 'r', linestyle = '--', label = 'Template')
# plt.plot((windows[150][0][1], windows[150][0][1]), ((windows[150][1][0], windows[150][1][1])), color = 'r', linestyle = '--')
# plt.plot((windows[150][0][0], windows[150][0][1]), ((windows[150][1][1], windows[150][1][1])), color = 'r', linestyle = '--')
# plt.plot((windows[150][0][0], windows[150][0][1]), ((windows[150][1][0], windows[150][1][0])), color = 'r', linestyle = '--')

# plt.plot((next_windows[150][0][0], next_windows[150][0][0]), (next_windows[150][1][0], next_windows[150][1][1]), color = 'b', linestyle = '--', label = 'Search')
# plt.plot((next_windows[150][0][1], next_windows[150][0][1]), ((next_windows[150][1][0], next_windows[150][1][1])), color = 'b', linestyle = '--')
# plt.plot((next_windows[150][0][0], next_windows[150][0][1]), ((next_windows[150][1][1], next_windows[150][1][1])), color = 'b', linestyle = '--')
# plt.plot((next_windows[150][0][0], next_windows[150][0][1]), ((next_windows[150][1][0], next_windows[150][1][0])), color = 'b', linestyle = '--')


# plt.legend()
# plt.imshow(nom_gray_list[666], cmap = plt.cm.gray)
# plt.savefig('plot_666.png', dpi = 500, bbox_inches = 'tight')
# plt.close()

# plt.figure()

# # plt.plot((windows[150][0][0], windows[150][0][0]), (windows[150][1][0], windows[150][1][1]), color = 'r', linestyle = '--', label = 'Template')
# # plt.plot((windows[150][0][1], windows[150][0][1]), ((windows[150][1][0], windows[150][1][1])), color = 'r', linestyle = '--')
# # plt.plot((windows[150][0][0], windows[150][0][1]), ((windows[150][1][1], windows[150][1][1])), color = 'r', linestyle = '--')
# # plt.plot((windows[150][0][0], windows[150][0][1]), ((windows[150][1][0], windows[150][1][0])), color = 'r', linestyle = '--')

# plt.plot((next_windows[150][0][0], next_windows[150][0][0]), (next_windows[150][1][0], next_windows[150][1][1]), color = 'b', linestyle = '--', label = 'Search')
# plt.plot((next_windows[150][0][1], next_windows[150][0][1]), ((next_windows[150][1][0], next_windows[150][1][1])), color = 'b', linestyle = '--')
# plt.plot((next_windows[150][0][0], next_windows[150][0][1]), ((next_windows[150][1][1], next_windows[150][1][1])), color = 'b', linestyle = '--')
# plt.plot((next_windows[150][0][0], next_windows[150][0][1]), ((next_windows[150][1][0], next_windows[150][1][0])), color = 'b', linestyle = '--')


# # plt.legend()
# plt.imshow(nom_gray_list[667], cmap = plt.cm.gray)
# plt.savefig('plot_667.png', dpi = 500, bbox_inches = 'tight')
# plt.close()

# print(pearson_vid.shape)
# print(delta_xy_pixel)

# plt.figure()
# plt.axhline(0, linestyle = '--', color = 'k', linewidth = 1)
# plt.axvline(0, linestyle = '--', color = 'k', linewidth = 1)
# # plt.plot(max_cor_idx[0]-21, max_cor_idx[1]-21, marker = 'x', color = 'pink')
# im2 = plt.imshow(pearson_vid[100][150], extent=[-delta_xy_pixel, delta_xy_pixel, -delta_xy_pixel, delta_xy_pixel], \
#     cmap = 'viridis')
# plt.xlabel('x lag (pixel)')
# plt.ylabel('y lag (pixel)')
# plt.colorbar(im2, label = 'pearson r')
# plt.savefig('lags_ppt.png', dpi = 500, bbox_inches = 'tight')
# plt.close()

        
        


#         for coord in ice_still_coord : 
            
#             if (coord[0] - b)**2 + (coord[1] - a)**2 <= r**2 : 
#                 plt.plot(coord[1], coord[0], marker = 'o', color = 'r', markersize = 1)
        
#         for coord in ice_move_coord :
        #     if (coord[0] - b)**2 + (coord[1] - a)**2 <= r**2 : 
        #         plt.plot(coord[1], coord[0], marker = 'o', color = 'g', markersize = 1)
        
        # if it > 0 :
        #     for coord in moved_coord : 
        #         if (coord[0] - b)**2 + (coord[1] - a)**2 <= r**2 : 
        #             plt.plot(coord[1], coord[0], marker = 'o', color = 'b', markersize = 1)
        
        # if it > 1 :
        #     for coord in openwater_coord : 
        #         # if len(coord[0]) != 1 : 
        #         # print(coord)
        #         if (len(coord) != 0) and ((coord[0] - b)**2 + (coord[1] - a)**2 <= r**2) : 
        #             plt.plot(coord[1], coord[0], marker = 'o', color = 'yellow', markersize = 1)

        # if it > 2  : 
        #     # print(ow_still_vid)
        #     for coord in ow_still_coord : 
        #         # if (len(coord) != 0) : 
        #         #     if  (len(coord) == 1) :
        #         #         coord = coord[0]
        #         #         if (coord[0] - b)**2 + (coord[1] - a)**2 <= r**2 : 
        #         #             plt.plot(coord[1], coord[0], marker = 'o', color = 'pink', markersize = 1)
                
        #         #     else :
        #                 # for coor in coord :               
        #         if (coord[0] - b)**2 + (coord[1] - a)**2 <= r**2 : 
        #             plt.plot(coord[1], coord[0], marker = 'o', color = 'pink', markersize = 1)
