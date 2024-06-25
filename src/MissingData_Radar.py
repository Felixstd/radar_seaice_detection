import numpy as np
import matplotlib.pyplot as plt
import extract_video as vid

def add_nan_end(filename, num) : 
    
    SIC = np.load(filename, allow_pickle=True).item()['Conc_ice']
    
    New_SIC = list(SIC) + num*[np.nan]

    dict = {'Conc_ice' : New_SIC}
    
    np.save(filename, dict)
    
def remove_start_add_nan_end(filename, num_start, num_end) : 
    SIC = np.load(filename, allow_pickle=True).item()['Conc_ice']
    
    New_SIC = list(SIC)[num_start:] + num_end*[np.nan]
    
    dict = {'Conc_ice' : New_SIC}
    
    np.save(filename, dict)
    
    
def missing_days(new_file, num) : 

    SIC = np.zeros(num)
    SIC[np.where(SIC == 0)] = np.nan
    
    dict = {'Conc_ice' : SIC}
    np.save(new_file, dict)
    
def combine2files_addNan(file1, file2, new_file, num_start1, num_start2, num_nan, num_nan2) : 
    
    SIC1 = np.load(file1, allow_pickle=True).item()['Conc_ice']
    SIC2 = np.load(file2, allow_pickle=True).item()['Conc_ice']
    
    if num_nan > 0 :
    
        SIC1_trunc = list(SIC1)[num_start1:] + num_nan*[np.nan]
    
    else : 
        SIC1_trunc = list(SIC1)[num_start1:]
        
    if num_start2 == 0 : 
        SIC2_trunc = list(SIC2)

    else :
        SIC2_trunc = list(SIC2)[num_start2:] + num_nan2*[np.nan]
        
    SIC = SIC1_trunc + SIC2_trunc
    
    dict = {'Conc_ice' : SIC}
    np.save(new_file, dict)
    
def remove_start_file(new_file, file2rem, num_start) : 
    
    SIC = np.load(file2rem, allow_pickle=True).item()['Conc_ice']
    SIC_trunc = list(SIC)[num_start:]
    
    
    dict = {'Conc_ice' : SIC_trunc}
    np.save(new_file, dict)
    
def remove_end_file(filename, num_end) : 
    
    SIC = np.load(filename, allow_pickle=True).item()['Conc_ice']
    SIC_trunc = list(SIC)[:-num_end]
    
    
    dict = {'Conc_ice' : SIC_trunc}
    np.save(filename, dict)

def add_nan_start(filename, newfile, num_start) : 
    
    SIC = np.load(filename, allow_pickle=True).item()['Conc_ice']
    
    SIC_added = num_start*[np.nan] + list(SIC)
    dict = {'Conc_ice' : SIC_added}
    np.save(newfile, dict)

def newfile_onlyNans(newfile, num_nans) : 
    
    SIC = np.zeros(num_nans)
    SIC[np.where(SIC == 0)] = np.nan
    
    dict = {'Conc_ice' : SIC}
    np.save(newfile, dict)
    
def combine2files(file1, file2, newfile, numstart1, numend1, numstart2, numend2) : 
    SIC1 = list(np.load(file1, allow_pickle=True).item()['Conc_ice'])
    SIC2 = list(np.load(file2, allow_pickle=True).item()['Conc_ice'])
    
    if numend1 == 0 : 
        SIC_1_trunc = SIC1[numstart1:]
    else :
        SIC_1_trunc = SIC1[numstart1:numend1]
    
    if numend2 == 0 : 
    
        SIC_2_trunc = SIC2[numstart2:]
        
    else : 
        SIC_2_trunc = SIC2[numstart2:numend2]
    
    SIC = SIC_1_trunc+SIC_2_trunc
    
    dict = {'Conc_ice' : SIC}
    np.save(newfile, dict)

    
    
    
    
    
###----------- For 2022 -----------###
Datadir = '/storage/fstdenis/Barrow_RADAR/saved_run/RADAR/saved_params_6/2022/'

# filename1 = Datadir+'saved_params20220127to20220129.npy'
# add_nan_end(filename1, 1)

# filename2 = Datadir+'saved_params20220313to20220315.npy'
# add_nan_end(filename2,int(1 + 3*60/4))

# new_file = Datadir+'saved_params20220220to20220306.npy'
# missing_days(new_file, 183 + 16*360 + 314)

# filename3 = Datadir+'saved_params20220517to20220519.npy'
# remove_start_add_nan_end(filename3, 1, 1)

# filename4 = Datadir+'saved_params20220524to20220526.npy'
# filename5 = Datadir+'saved_params20220529to20220531.npy'
# newfile = Datadir+'saved_params20220526to20220531.npy'

# combine2files_addNan(filename4, filename5, newfile, 498, 0, int(4 + 9*60/4 + 360 + 14*60/4 + 3), 0)
# # # _, _, _, _, _, _, img, _ = vid.extract_video('/storage/fstdenis/Barrow_RADAR/Data/RADAR_2022/20220524to20220526.mp4')
# # # plt.figure()
# # # plt.imshow(img[498])
# # # plt.savefig('test_img.png')

# filename6 = Datadir+'saved_params20220530to20220601.npy'
# newfile2 = Datadir+'saved_params20220601to20220601.npy'

# remove_start_file(newfile2, filename6, 934)
# remove_start_add_nan_end(newfile2, 0, 1)

# # # _, _, _, _, _, _, img, _ = vid.extract_video('/storage/fstdenis/Barrow_RADAR/Data/RADAR_2022/20220530to20220601.mp4')
# # # plt.figure()
# # # plt.imshow(img[934])
# # # plt.savefig('test_img.png')

# filename7 = Datadir+'saved_params20220620to20220622.npy'
# remove_end_file(filename7, -1)

# filename8 = Datadir+'saved_params20220623to20220625.npy'
# add_nan_end(filename8, 1)

# filename9 = Datadir+'saved_params20220726to20220728.npy'
# remove_end_file(filename9, 1)

# filename10 = Datadir+'saved_params20220802to20220804.npy'
# newfile3 = Datadir+'saved_params20220801to20220804.npy'
# add_nan_start(filename10, newfile3, int(2 + 17*60/4 + 4))

# filename11 = Datadir+'saved_params20220830to20220901.npy'
# filename12 = Datadir+'saved_params20220902to20220904.npy'
# newfile4 = Datadir+'saved_params20220901to20220904.npy'

# combine2files_addNan(filename11, filename12, newfile4, 720, 0, 0, 0)
# # # _, _, _, _, _, _, img, _ = vid.extract_video('/storage/fstdenis/Barrow_RADAR/Data/RADAR_2022/20220830to20220901.mp4')
# # # plt.figure()
# # # plt.imshow(img[720])
# # # plt.savefig('test_img.png')


# filename12 = Datadir+'saved_params20220908to20220910.npy'
# remove_start_file(filename12, filename12, 102)
# # # _, _, _, _, _, _, img, _ = vid.extract_video('/storage/fstdenis/Barrow_RADAR/Data/RADAR_2022/20220908to20220910.mp4')
# # # plt.figure()
# # # plt.imshow(img[102])
# # # plt.savefig('test_img.png')

# newfile5 = Datadir+'saved_params20220911to20221016.npy'
# newfile_onlyNans(newfile5, int(56/4 +7*60/4 + 22*360 + 17*360 + 14*60/4 + 5 + 12*360))

# filename13 = Datadir+'saved_params20221030to20221101.npy'
# filename14 = Datadir+'saved_params20221102to20221104.npy'
# newfile6 = Datadir+'saved_params20221101to20221104.npy'
# combine2files_addNan(filename13, filename14, newfile6, 718, 0, 0, 0)
# # _, _, _, _, _, _, img, _ = vid.extract_video('/storage/fstdenis/Barrow_RADAR/Data/RADAR_2022/20221030to20221101.mp4')
# # plt.figure()
# # plt.imshow(img[718])
# # plt.savefig('test_img.png')

# filename15 = Datadir+'saved_params20220109to20220111.npy' #360
# filename16 = Datadir+'saved_params20220111to20220113.npy' #360
# newfile7 = Datadir+'saved_params20220110to20220113.npy'
# combine2files_addNan(filename15, filename16, newfile7, 360, 360, 0, 0)


# # _, _, _, _, _, _, img, _ = vid.extract_video('/storage/fstdenis/Barrow_RADAR/Data/Missing_2022/20220111to20220113.mp4')
# # plt.figure()
# # plt.imshow(img[360])
# # plt.savefig('test_img.png')

# newfile8 = Datadir+'saved_params20220114to20220114.npy'
# filename17 = Datadir+'saved_params20220113to20220114.npy'
# remove_start_file(newfile8, filename17, 360)

###----------- For 2023 -----------###

Datadir_2023 =  '/storage/fstdenis/Barrow_RADAR/saved_run/RADAR/saved_params_6/2023/'


filenames = ['saved_params20230102to20230104.npy', 'saved_params20230105to20230107.npy', 'saved_params20230108to20230110.npy', \
            'saved_params20230114to20230116.npy', 'saved_params20230129to20230131.npy', 'saved_params20230205to20230207.npy', \
            'saved_params20230208to20230210.npy','saved_params20230211to20230213.npy', 'saved_params20230217to20230219.npy']
remove_start_num = [362, 360, 360, 358, 360, 360, 360, 359, 65]


for i in range(len(filenames)) : 
    remove_start_file(Datadir_2023+filenames[i], Datadir_2023+filenames[i], remove_start_num[i])

filename1 = Datadir_2023+'saved_params20230217to20230219.npy'
add_nan_end(filename1, int(36/4+3*60/4 + 48/4))


# filename2 = Datadir_2023+'saved_params20230310to20230313.npy'
# newfile2 = Datadir_2023+'saved_params20230311to20230313.npy'
# remove_start_file(newfile2, filename2, 15)
# add_nan_end(newfile2, 1)


# filename3 = Datadir_2023+'saved_params20230328to20230330.npy'
# filename4 = Datadir_2023+'saved_params20230329to20230331.npy'
# newfile3 = Datadir_2023+'saved_params20230328to20230331.npy'
# combine2files_addNan(filename3, filename4, newfile3, 0, 720, 0, 0)

# filename5 = Datadir_2023+'saved_params20230528to20230530.npy'
# filename6 = Datadir_2023+'saved_params20230529to20230531.npy'
# newfile4 = Datadir_2023+'saved_params20230528to20230531.npy'
# combine2files_addNan(filename5, filename6, newfile4, 0, 720, 0, 6*360)

# filename7 = Datadir_2023+'saved_params20230227to20230301.npy'
# filename8 = Datadir_2023+'saved_params20230302to20230304.npy'
# newfile5  = Datadir_2023+'saved_params20230229to20230304.npy'
# combine2files(filename7, filename8, newfile5, 720, 0, 0, 0)

# _, _, _, _, _, _, img, _ = vid.extract_video('/storage/fstdenis/Barrow_RADAR/Data/RADAR_2023/20230227to20230301.mp4')
# plt.figure()
# plt.imshow(img[360*2])
# plt.savefig('test_img.png')

filename2 = Datadir_2023+'saved_params20230117to20230119.npy'
newfile2 = Datadir_2023+'saved_params20230117to20230119.npy'
remove_start_file(newfile2, filename2, 360)


filename2 = Datadir_2023+'saved_params20230120to20230122.npy'
newfile2 = Datadir_2023+'saved_params20230120to20230122.npy'
remove_start_file(newfile2, filename2, 360)


filename2 = Datadir_2023+'saved_params20230123to20230125.npy'
newfile2 = Datadir_2023+'saved_params20230123to20230125.npy'
remove_start_file(newfile2, filename2, 360)

filename2 = Datadir_2023+'saved_params20230126to20230128.npy'
newfile2 = Datadir_2023+'saved_params20230126to20230128.npy'
remove_start_file(newfile2, filename2, 360)
