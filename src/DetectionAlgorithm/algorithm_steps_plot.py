import numpy as np
import extract_video as vid
import edge_detector_ice_raw as edi
import matplotlib.pyplot as plt

masksDir = "/storage/fstdenis/Barrow_RADAR/ICE_RADAR_MISC/IdentTypeIce/edge_detection/masks/"


land_mask = np.load(masksDir+'land_mask_3.npy', allow_pickle=True).item()['mask']
img = vid.read_img('./Algorithm_Figure/2022/03/20220311/UAFIceRadar_20220311_174800_crop_geo.tif')
img_copy = np.copy(img)
img[np.where(land_mask == 1)] = 0

# edge_list, list_blurred, gradients_imgs, orientations_imgs, suppressed_imgs = canny.canny_edge([img], 5, 1, 84, 84)

img_list, gray_list, edge_list_cv, polygon_list_end, polygon_list, concentration_ice = \
    edi.identification_ice_ow(4, 82, (11,11), './Algorithm_Figure/', masksDir, './')

gray = gray_list[0]
edges = edge_list_cv[0]
polygons = polygon_list[0]
polygons_tot = polygon_list_end[0]


fig = plt.figure(figsize = (14, 10))

ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)

# # ax1 = fig.add_subplot(321)
ax1.imshow(img_copy, cmap = plt.cm.gray)
ax1.set_axis_off()
ax1.annotate('a)', xy=(ax1.get_xlim()[0], ax1.get_ylim()[1]), weight = 'bold')

# ax2 = fig.add_subplot(322)
ax2.imshow(img, cmap = plt.cm.gray)
ax2.set_axis_off()
ax2.annotate('b)', xy=(ax2.get_xlim()[0], ax2.get_ylim()[1]), weight = 'bold')

# ax3 = fig.add_subplot(323)
ax3.imshow(edges, cmap = 'gray')
ax3.annotate('c)', xy=(ax3.get_xlim()[0], ax3.get_ylim()[1]), weight = 'bold')
ax3.set_axis_off()

# ax4 = fig.add_subplot(324)
for polygon in polygons  :
    ax4.plot(*polygon.exterior.xy, color = 'r')
ax4.imshow(gray, cmap = plt.cm.gray) 
ax4.annotate('d)', xy=(ax4.get_xlim()[0], ax4.get_ylim()[1]), weight = 'bold')
ax4.set_axis_off()
# 

# ax5 = fig.add_subplot(3,2,5)
ax5.imshow(img_copy, cmap = plt.cm.gray)
ax5.set_axis_off()
for polygon in polygons_tot : 
    ax5.plot(*polygon.exterior.xy, color = 'r')
ax5.annotate('e)', xy=(ax5.get_xlim()[0], ax5.get_ylim()[1]), weight = 'bold') 
# plt.subplots_adjust(wspace=0.1,
#                     hspace=0.4)   
fig.subplots_adjust(hspace=0.05)
# plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig('Test_2022.png', dpi = 500, bbox_inches = 'tight')


plt.figure()
plt.imshow(img_copy, cmap = plt.cm.gray)
plt.axis('off')
plt.savefig('step1.png', dpi = 500, bbox_inches = 'tight')

plt.figure()
plt.imshow(img, cmap = plt.cm.gray)
plt.axis('off')
plt.savefig('step2.png', dpi = 500, bbox_inches = 'tight')

plt.figure()
plt.imshow(edges, cmap = plt.cm.gray)
plt.axis('off')
plt.savefig('step3.png', dpi = 500, bbox_inches = 'tight')

plt.figure()
plt.imshow(gray, cmap = plt.cm.gray)
for polygon in polygons  :
    plt.plot(*polygon.exterior.xy, color = 'r')
plt.axis('off')
plt.savefig('step4.png', dpi = 500, bbox_inches = 'tight')

plt.figure()
plt.imshow(img_copy, cmap = plt.cm.gray)
for polygon in polygons_tot  :
    plt.plot(*polygon.exterior.xy, color = 'r')
plt.axis('off')
plt.savefig('step5.png', dpi = 500, bbox_inches = 'tight')