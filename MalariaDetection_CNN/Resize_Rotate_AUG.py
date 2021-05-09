import numpy as np
import glob, os
from PIL import Image
from tqdm import tqdm

src = "E://kaggle_malaria_detection//cell_images//Parasitized//"
#src = "E://kaggle_malaria_detection//cell_images//Uninfected//"
#src = "E://kaggle_malaria_detection//crops//res_120_copy//Parasitized//"
#src = "E://kaggle_malaria_detection//crops//res_120_copy//Uninfected//"
dst = "E://kaggle_malaria_detection//crops//res_120//Parasitized//"
#dst = "E://kaggle_malaria_detection//crops//res_120//Uninfected//"

h,w = 120, 120
r = [90,180,270] #not taking 45,135... bc it's quite a work
q = 0
print('src:' + src)
print('dst:' + dst)
# for image in tqdm(glob.glob(src+"//*.png")):
#     img = Image.open(image)
#     resize_img = img.resize((h,w), Image.ANTIALIAS)
#     resize_img.save(dst + '//' + str(q) + ".png" )
#     q+=1

# for image in tqdm(glob.glob(src+"*.png")):
#     img = Image.open(image)
#     rotate90 = img.rotate(90)
#     rotate180 = img.rotate(180)
#     rotate270 = img.rotate(270)
#     rotate90.save(dst + str(r[0]) + '_' + str(q) + ".png" )
#     rotate180.save(dst + str(r[1]) + '_' + str(q) + ".png" )
#     rotate270.save(dst + str(r[2]) + '_' + str(q) + ".png" )
#     q+=1
