import os 
from scipy.io import savemat


def read_parameters(filename) : 
    
    """
    All the parameters for the analysis of a video are written in a specific 
    file in a certain path. This function reads that file and extracts the 
    parameters for the run.

    Args:
        filename -> name of the file

    Parameters: 
    
        Frame_Step (int) -> How much frame of a video will be analysed.
        Strart_Frame (int) -> Which frame of the video the analysis starts
        Window1 (int) -> Start of the first window
        Window2 (int) -> Start of the second window
        Window_step (int) -> How much pixels in between windows
        CC_Thresold (float) -> Specified threshold for the correlation coefficient cutoff
        Reciprocal_filtering (bool) -> Use of the reciprocal filtering algorithm
        Neighbor_filtering (bool) -> Use of the nearest neighbor algorithm
        Averaging (bool) -> Is the analys averaged for a specified time
        Saving (bool) -> Is the run saved in a file
        Plotting (bool) -> Is the analysis plotted 
                
    """
    
    #Reading the file
    lines = open(filename).readlines()
    
    for line in lines : 
        
        inline = line.split()
        
        if inline[0] == 'Frame_Step' : 
            frame_step = int(inline[1])
            
        if inline[0] == 'Start_Frame' : 
            start_step = int(inline[1])
            
        if inline[0] == 'Window1' : 
            start_window1 = int(inline[1])
            
        if inline[0] == 'Window2' : 
            start_window2 = int(inline[1])
        
        if inline[0] == 'Window_Step' : 
            window_step = int(inline[1])
            
        if inline[0] == 'Reciprocal_Filtering' : 
            reciprocal = bool(int(inline[1]))
            
        if inline[0] == 'Neighbor_Filtering' : 
            neighbor = bool(int(inline[1]))
            
        if inline[0] == 'CC_Treshold' : 
            cc_thresh = float(inline[1])
            
            
        if inline[0] == 'Averaging' : 
            average = bool(int(inline[1]))
            
        if inline[0] == 'Time_Area' : 
            area = bool(int(inline[1]))
            
        if inline[0] == 'Plotting_Average' : 
            plot_average = bool(int(inline[1]))
            
        if inline[0] == 'Saving' : 
            saving = bool(int(inline[1]))
        
        if inline[0] == 'Plotting' : 
            plotting = bool(int(inline[1]))


    #Printing the parameters
    print('Frame Step : ', frame_step)
    print('Starting at frame : ', start_step)
    print('Starting window 1 at ', start_window1)
    print('Starting window 2 at ', start_window2)
    print('Window Step : ', window_step)
    print('Reciprocal filtering : ', reciprocal)
    print('Minimum threshold for CC : ', cc_thresh)
    print('Avering the pixel types : ', average)
    print('Plotting the averaging : ', plot_average)
    print('Neighbor filtering : ', neighbor)
    print('Calculating the area of pixel type for the given time series : ', area)
    print('Saving the run : ', saving)
    print('Plotting the run : ', plotting, '\n')
            
    return frame_step, start_step, start_window1, start_window2, window_step, cc_thresh, reciprocal, average, area, saving, plotting, \
        plot_average, neighbor
        

def read_parameters_edge(filename) : 
    
    lines = open(filename).readlines()
    
    for line in lines : 
        
        inline = line.split()
        
        if inline[0] == 'Frame_Step' : 
            frame_step = int(inline[1])
            
        if inline[0] == 'Start_Frame' : 
            start_frame = int(inline[1])
            
        if inline[0] == 'Kernel_Size' : 
            kernel_size = int(inline[1])
            
        if inline[0] == 'Sigma' : 
            sigma = int(inline[1])
            
        if inline[0] == 'Low_threshold' : 
            low = int(inline[1])  
        
        if inline[0] == 'High_Threshold' : 
            high = int(inline[1])
            
        if inline[0] == 'Delta_a' : 
            delta_a = int(inline[1]) 
            
        if inline[0] == 'Delta_b' : 
            delta_b = int(inline[1])
            
        if inline[0] == 'Delta_r' : 
            delta_r = int(inline[1])
            
        if inline[0] == 'Circle_Finding' : 
            circle = int(inline[1])
            
        if inline[0] == 'Gaussian_Filtering' : 
            gaus_filter = bool(int(inline[1]))
        
        if inline[0] == 'Calc_Area' : 
            area = bool(int(inline[1]))
            
        if inline[0] == 'Checking_Contours' : 
            check_cont = bool(int(inline[1]))
            
        if inline[0] == 'Plotting_Contours' : 
            plotting_cont = bool(int(inline[1]))
            
        if inline[0] == 'Saving' : 
            saving = bool(int(inline[1]))
                
    difference_cirle = [delta_a, delta_b, delta_r]       

    print('Starting at frame : ', start_frame)
    print('Frame step : ', frame_step)
    print('Gaussian Filtering : ', gaus_filter)
    print('Kernel Size for gaussian filtering : ', kernel_size)
    print('Sigma for gaussian filtering : ', sigma)
    print('Lower threshold for canny edge detection : ', low)
    print('Higher threshold for canny edge detection : ', high)
    print('Threshold for finding the field of view of the radar : ', circle)
    print('Difference for centering the detected cricle : ', difference_cirle)
    print('Calculating the area of sea ice and open-water', area)
    print('Checking if the contours contain each other : ', check_cont)
    print('Plotting the run : ', plotting_cont)
    print('Saving the run ', saving)
    
    
    return start_frame, frame_step, gaus_filter, kernel_size, sigma, \
        low, high, circle, difference_cirle, area, plotting_cont, saving, check_cont
        
def saving_edge(polygon_list, polygon_area_list, savedir) : 
    
    polygon_vid = {'Poly' : polygon_list, 'Area' : polygon_area_list}
    savemat(savedir+'analysis_polygons.mat', polygon_vid)