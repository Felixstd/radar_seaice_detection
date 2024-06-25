import numpy as np
import cv2 
import glob
from natsort import natsorted
import rasterio
import xarray as xr
# from geotiff import GeoTiff

# from rasterio.warp import transformy

# from affine import Affine
from pyproj import Proj, transform
# from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


def convert2gray(img_list) : 
    
    """
    Function used to convert an image to gray scale
    """

    gray_list = []
    
    for frame in img_list :
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_list.append(gray)
    
    return gray_list

def normalize_img(img_list):
    
    """
    Function used to normalize an image to have an average of 0
    and a standard deviation of 1
    
    The images need to be in gray scale
    """
    norm_list = []
    for img in img_list : 
        
        mean = np.mean(img)
        std = np.std(img, ddof=1)
        norm_img = (img - mean)/std
        
        norm = (img - np.min(img))/(np.max(img) - np.min(img))
        
        norm_list.append(norm_img)
    
    return norm_list


def video_informations(video) :
    
    """
    Function used to get the frame number, the width, the height of the frame in a video
    """

    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    return num_frames, height, width, fps


def extract_video(viddir) : 
    
    
    video = cv2.VideoCapture(viddir)
    num_frames, height, width, fps = video_informations(video)
    
    img_list = [] #list containing every image of the video

    print('Reading video')
    for i in range(num_frames) : 
        ret, img = video.read() #read frame
        
        if ret : 
            img_list.append(img) #append to list
        
    print('Done reading', '\n')


    gray_list = convert2gray(img_list)
    nom_gray_list = normalize_img(gray_list)
    
    blurred_gray_list = []
    for im in nom_gray_list : 
        blurred_gray_list.append(cv2.GaussianBlur(im, (3,3), 0))
    
    return video, num_frames, height, width, fps, nom_gray_list, img_list, blurred_gray_list

def extract_first_frame(viddir) : 
    
    video = cv2.VideoCapture(viddir)
    ret, img = video.read()
    
    return img

def extract_img_folder(folder) : 
    
    img_list = []
    
    for img in natsorted(glob.glob(folder+'/*.tif')):
        print(img)
        with rasterio.open(img) as f:
            img = f.read(1)
        
        img_list.append(img)

    return img_list

def extract_imgs(folder) : 
    
    img_list = []
    for img in natsorted(glob.glob(folder+'/*.jpg')):
        # print(img)
        img_read = cv2.imread(img)
        
        #need to change this for the SIC from the analyst images
        gray = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
        img_list.append(gray)

    return img_list

def read_img(namefile) : 
    
    with rasterio.open(namefile) as f:
            img = f.read(1)
            
    return img



# img_list = extract_img_folder('/storage/fstdenis/Barrow_RADAR/RAW_Data_Test/2022/01/20220101/')
# tiff_file = "/storage/fstdenis/Barrow_RADAR/RAW_Data_Test/2022/01/20220101/UAFIceRadar_20220101_001600_crop_geo.tif"

tiff_file = "./modis_imagery_utq/20230526/snapshot-2023-05-26T00_00_00Z.tif"
# img = read_img(tiff_file)

# latitude = np.zeros((416, 430))
# longitude = np.zeros((416, 430))

# latitude = np.zeros((416, 430))
# longitude = np.zeros((416, 430))
latlon = 0

def find_latlon_tif(tiff_file, namefile, saving = False):
    
    
    with rasterio.open(tiff_file) as src:
        shape_img = np.shape(src)
        
        latitude = np.zeros(shape_img)
        longitude = np.zeros(shape_img)
        
        for i in range(shape_img[0]):
            for j in range(shape_img[1]):
                
                lon, lat = src.xy(i, j)

                latitude[i, j] = lat
                longitude[i, j] = lon

    if saving:
                 
        ds1 = xr.Dataset(
            data_vars={
                "latitude": (( "x", "y"), latitude),
                "longitude": (( "x", "y"), longitude),
            }
        )
        ds1.to_netcdf(namefile)
    
    
if latlon :
    with rasterio.open(tiff_file) as src:
        latitude = np.zeros(np.shape(src))
        longitude = np.zeros(np.shape(src))
        for i in range(np.shape(src)[0]):
            for j in range(np.shape(src)[1]):
                
                print(i, j)
                lon, lat = src.xy(i, j)
                # print(lon)
                latitude[i, j] = lat
                longitude[i, j] = lon

                    
    ds1 = xr.Dataset(
        data_vars={
            "latitude": (( "x", "y"), latitude),
            "longitude": (( "x", "y"), longitude),
        }
                # "time" : (["t"], num_img_day)
    )
    ds1.to_netcdf("latitude_modis_img_20230526.nc")
            

# geo_tiff = GeoTiff(tiff_file, crs_code = 4326)
# print(geo_tiff)
# img = img_list[0]
# i=5
# j=6
# # in the as_crs coords
# print(geo_tiff.get_wgs_84_coords(i, j))


# get the existing coordinate system
pixellatlon = 0
if pixellatlon:
    from osgeo import osr, gdal
    import numpy as np
    def pixel2coord(img_path, x, y):
        """
        Returns latitude/longitude coordinates from pixel x, y coords

        Keyword Args:
        img_path: Text, path to tif image
        x: Pixel x coordinates. For example, if numpy array, this is the column index
        y: Pixel y coordinates. For example, if numpy array, this is the row index
        """
        # Open tif file
        ds = gdal.Open(img_path)

        old_cs = osr.SpatialReference()
        old_cs.ImportFromWkt(ds.GetProjectionRef())

        # create the new coordinate system
        # In this case, we'll use WGS 84
        # This is necessary becuase Planet Imagery is default in UTM (Zone 15). So we want to convert to latitude/longitude
        # polar_stereo =     """GEOGCS["Hughes 1980",
        #         DATUM["Hughes_1980",
        #             SPHEROID["Hughes 1980",6378273,298.279411123064,
        #                 AUTHORITY["EPSG","7058"]],
        #             AUTHORITY["EPSG","1359"]],
        #         PRIMEM["Greenwich",0,
        #             AUTHORITY["EPSG","8901"]],
        #         UNIT["degree",0.0174532925199433,
        #             AUTHORITY["EPSG","9122"]],
        #         AUTHORITY["EPSG","10345"]],
        #     PROJECTION["Polar_Stereographic"],
        #     PARAMETER["latitude_of_origin",70],
        #     PARAMETER["central_meridian",-45],
        #     PARAMETER["false_easting",0],
        #     PARAMETER["false_northing",0],
        #     UNIT["metre",1,
        #         AUTHORITY["EPSG","9001"]],
        #     AUTHORITY["EPSG","3411"]]"""
            
        polar_stereo = """PROJCS["WGS 84 / NSIDC Sea Ice Polar Stereographic North",
        GEOGCS["WGS 84",
            DATUM["WGS_1984",
                SPHEROID["WGS 84",6378137,298.257223563,
                    AUTHORITY["EPSG","7030"]],
                AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.0174532925199433,
                AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4326"]],
        PROJECTION["Polar_Stereographic"],
        PARAMETER["latitude_of_origin",70],
        PARAMETER["central_meridian",-45],
        PARAMETER["false_easting",0],
        PARAMETER["false_northing",0],
        UNIT["metre",1,
            AUTHORITY["EPSG","9001"]],
        AUTHORITY["EPSG","3413"]]"""
        
        new_cs = osr.SpatialReference()
        new_cs.ImportFromWkt(polar_stereo)

        # create a transform object to convert between coordinate systems
        transform = osr.CoordinateTransformation(old_cs,new_cs) 
        
        gt = ds.GetGeoTransform()

        # GDAL affine transform parameters, According to gdal documentation xoff/yoff are image left corner, a/e are pixel wight/height and b/d is rotation and is zero if image is north up. 
        xoff, a, b, yoff, d, e = gt

        xp = a * x + b * y + xoff
        yp = d * x + e * y + yoff

        lat_lon = transform.TransformPoint(xp, yp) 
        print(lat_lon)

        xp = lat_lon[0]
        yp = lat_lon[1]
        
        return xp, yp

    latitude = np.zeros((416, 430))
    longitude = np.zeros((416, 430))
    # shape = np.shape(img)


    for i in range(416):
        for j in range(430):
            
            print(i, j)
            lat, lon = pixel2coord(tiff_file, i, j)
            # print(lon)
            latitude[i, j] = lat
            longitude[i, j] = lon
            
    # latlon = {'latitude' : latitude, 'longtitude': longitude}
    # np.save('latlon_modis_visible.npy', latlon)
