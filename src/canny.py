import numpy as np
import cv2 as cv 
import scipy.ndimage as nimg

from numba import njit


"""
This file is the homemade CANNY edge algorithm. 

"""

@njit
def neighbors(edge_list, threshold) : 
    
    height, width = np.shape(edge_list[0])
    
    img_filled_list = []
    for img in edge_list : 
        
        img_filled = np.copy(img)
        for i in range(1, height - 1) : 
            for j in range(1, width -1) : 
                
                neighbors_img = np.array([img[i-1, j], img[i+1, j], img[i, j-1], img[i, j+1],\
                    img[i+1, i-1], img[i-1, j-1], img[i+1, j-1], img[i-1, j+1]])#, img[i-2, j], img[i+2, j], img[i, j-2], img[i, j+2],\
                    # img[i+2, i-2], img[i-2, j-2], img[i+2, j-2], img[i-2, j+2]])
                
                if img[i, j] != 255 : 
                    
                    max_neighbors = np.count_nonzero(neighbors_img == 255)
                    if max_neighbors > threshold : 
                        img_filled[i, j] = 255
            
        img_filled_list.append(img_filled)
    
    return img_filled_list

# @njit
def gaussian_filtering(img_list, size, sigma) :
    
    print('Gaussian filtering', '\n')
    #--- Defining the gaussian kernel ---#
    #size of the kernel
    size = size //2
    
    #grid
    x, y = np.mgrid[-size:size+1, -size:size+1]
    beta = 1/(2 * np.pi * sigma**2)
    #gaussian kernel
    gaussian_kernel =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * beta
    
    #--- Applying the gaussian filter to blur the images ---# 
    
    blurred_img_list = []
    
    for gray_img in img_list : 
        #convolve the kernel to the gray image
        img_blurred = nimg.filters.convolve(gray_img, gaussian_kernel)
        blurred_img_list.append(img_blurred)
    
    return blurred_img_list

@njit
def sobel_filtering(img_list) :
    
    print('Applying the sobel filter')
    #defining the sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    # sobel_x = np.array([[2, 2, 4, 2, 2], [1, 1, 2, 1, 1], [0, 0, 0, 0, 0], [-1, -1, -2, -1, -1], [-2, -2, -4, -2, -2]])
    # sobel_y = np.array([[2, 1, 0, -1, -2], [2, 1, 0, -1, -2], [4, 2, 0, -2, -4], [2, 1, 0, -1, -2], [2, 1, 0, -1, -2]])
    
    height, width = np.shape(img_list[0])  # we need to know the shape of the input grayscale image
    
    
    #--- Calculating the gradient of the whole list of images
    gradient_imgs_list = []
    orientation_gradient_imgs_list = []
    it = 0
    for img in img_list : 
        print('Computing the image gradient : ', it)
        
        gradient_image = np.zeros((height, width))
        gradient_x = np.zeros_like(gradient_image)
        gradient_y = np.zeros_like(gradient_image)
        orientations_grad = np.zeros_like(gradient_image)
        
        idx = 0
        for i in range(height - 2)  : 
            for j in range(width - 2 - idx ) : 
                
                #convolve with the sobel kernels to find the gradiants 
                grad_x = np.sum(np.multiply(sobel_x, img[i:i+3, j:j+3]))
                grad_y = np.sum(np.multiply(sobel_y, img[i:i+3, j:j+3]))
                
                #find the orientation
                angle = np.arctan2(grad_y, grad_x)
                
                if angle < 0  :
                    angle += np.pi 
                    
                gradient_x[i, j] = grad_x
                gradient_y[i, j] = grad_y
                orientations_grad[i, j] = angle
            idx += 1
        
        #compute the gradient norm
        gradient_image = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_norm = gradient_image * 255 / np.max(gradient_image)
        
        
        gradient_imgs_list.append(gradient_norm)
        orientation_gradient_imgs_list.append(orientations_grad)
        it += 1
       
        
    return gradient_imgs_list, orientation_gradient_imgs_list

@njit
def non_maximal_suppression(grad_list, orientation_list) :
    """https://github.com/StefanPitur/Edge-detection---Canny-detector/blob/master/canny.py
    """
    print('\n', 'Non maximal suppression')
    height, width = np.shape(grad_list[0])
    time = len(grad_list)
    
    suppressed_list = []
    
    for it in range(0, time) : 
        print('Non max suppression : ', it)
        suppressed_pixel = np.zeros((height, width))
        grad_it = grad_list[it]
        orientation_it = orientation_list[it]

        idx = 0
        #loop through the image
        for i in range(height - 1) : 
            for j in range(width - 1 - idx) : 
                
                #--- Find the orientation and compare both values ---#
                if (0 <= orientation_it[i, j] < np.pi/8) or (0 <= orientation_it[i, j] < 7*np.pi/8) : 
                    neighbour = max(grad_it[i, j-1], grad_it[i, j+1])
                    
                elif np.pi/8 <= orientation_it[i, j] < 3*np.pi/8 : 
                    neighbour = max(grad_it[i-1, j-1], grad_it[i+1, j+1])
                    
                elif 3*np.pi/8 <= orientation_it[i, j] < 5*np.pi/8 : 
                    neighbour = max(grad_it[i-1, j], grad_it[i+1, j])
                    
                else : 
                    neighbour = max(grad_it[i+1, j-1], grad_it[i-1, j+1])
                    
                if grad_it[i, j] >= neighbour : 
                    suppressed_pixel[i, j] = grad_it[i, j]
            idx += 1     
        suppressed_list.append(suppressed_pixel)
        
    
    return suppressed_list

@njit
def double_threshold_hysterisis(suppressed_list, low, high) : 
    
    print('\n', 'Applying the double threshold hysterisis : ')
    weak = 50
    strong = 255
    
    height, width = np.shape(suppressed_list[0])
    time = len(suppressed_list)
    edge_list = []
    
    #loop through the list
    for it in range(time) : 
        print('Double threshold hysterisis : ', it)
        suppressed_img = suppressed_list[it]
        edge_img = np.zeros_like(suppressed_img)
        
        #first threshold
        for i in range(height) : 
            for j in range(width) : 
                suppressed_v = suppressed_img[i, j]
                
                if (suppressed_v > low) and (suppressed_v <= high) : 
                    edge_img[i,j] = weak
                    
                elif suppressed_v > high : 
                    edge_img[i,j] = strong

        #find the neighbours
        for i in range(1, height-1) : 
            for j in range(1, width-1) : 
                
                if edge_img[i, j] == weak : 
                    neighbours = np.array([edge_img[i-1, j], edge_img[i+1, j], edge_img[i, j+1], edge_img[i, j-1], \
                        edge_img[i+1, j+1], edge_img[i-1, j-1], edge_img[i+1, j-1], edge_img[i-1, j+1]])
                    
                    idx_strong_neighbours = np.where(neighbours == strong) 
                    
                    if len(idx_strong_neighbours) > 0 : 
                        edge_img[i, j] = strong
                        
                    else : 
                        edge_img[i, j] = 0

        edge_img /= 255
        
        edge_list.append(edge_img)
        
    return edge_list

def canny_edge(img_list, size, sigma, low, high) :
    
    """
    Function used to apply the canny edge algorithm to a radar image
    Its only applying the algorithm to the first half of the image because 
    the second one doesnt interest us (there's no ice/water in it). 

    Args:
        img_list (matrix list): list of normalized and grayed radar images 
        size (int): size of the kernel for the gaussian filtering 
        sigma (float)): standard deviation of the gaussian for the filtering
        low (int): lower threshold for the double threshold hysterisis
        high (int): higher threshold for the double threshold hysterisis

    Returns:
        list of grids: 
            edge_list: Grids with the processed edges
            list_blurred: list of the blurred out images
            gradients_imgs: list of the gradients of the images
            orientations_imgs: list of the orientations of the gradient images
            suppressed_imgs: list with the thinned out edges
    """
    #applying the gaussian filter
    list_blurred = gaussian_filtering(img_list, size, sigma)
    
    #computing the gradient
    gradients_imgs, orientations_imgs = sobel_filtering(list_blurred)
    
    #applying the non maximal suppression algorithm
    suppressed_imgs = non_maximal_suppression(gradients_imgs, orientations_imgs)
    
    #applying the double threshold hysterisis algorithm
    edge_list = double_threshold_hysterisis(suppressed_imgs, low, high)
    
    return edge_list, list_blurred, gradients_imgs, orientations_imgs, suppressed_imgs
