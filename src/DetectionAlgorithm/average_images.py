import numpy as np
import matplotlib.pyplot
import matplotlib.pyplot as plt
import os
import extract_video as vid

DataDir = '/storage/fstdenis/Barrow_RADAR/RAW_Data/'
masksDir = "/storage/fstdenis/Barrow_RADAR/IdentTypeIce/edge_detection/masks/"

i=1
# while i == 1:
#     for YearDir in os.listdir(DataDir) : 
#         YearDir = str(YearDir)
        
#         #loop through the months
#         for MonthDir in os.listdir(DataDir+YearDir) : 
#             MonthDir = str(MonthDir)
#             print(MonthDir)
            #loop through the days
            
YearDir = '2023'
MonthDir = '02'
DayDir = '20230209'
for DayDir in os.listdir(DataDir+'2023'+'/'+'02') : 
    DayDir = str(DayDir)
    print(DayDir)
    img_list_day = vid.extract_img_folder(DataDir+YearDir+'/'+MonthDir+'/'+DayDir)

img_list_day_avrg = np.mean(img_list_day, axis = 0)

# print(img_list_day.shape)
print(img_list_day_avrg.shape)
            
                
                # break
            
plt.figure()
plt.imshow(img_list_day_avrg)
plt.savefig('test_img.png', dpi = 500)
# print(img_list_day[0].shape)
print(img_list_day_avrg.shape)