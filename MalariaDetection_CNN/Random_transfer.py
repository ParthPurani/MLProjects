'''
this code transfers random images from train set to test set
'''
import os
import random
import shutil
import glob
from tqdm import tqdm


def rand_gen(source):

  src = source

  files = os.listdir(src)
  list_files = []
  for i in range(int(len(files)/6)):
    index = random.randrange(0, len(files))
    list_files.append(index)
  rand_files=[]
  for i in list_files:
    rand_files.append(files[i])

  return rand_files
print(__doc__)
src_0 =  "E://kaggle_malaria_detection//crops//res_120_train//Parasitized//"
src_1 =  "E://kaggle_malaria_detection//crops//res_120_train//Uninfected//"

dst_0 = "E://kaggle_malaria_detection//crops//res_120_test//Parasitized//"
dst_1 = "E://kaggle_malaria_detection//crops//res_120_test//Uninfected//"

rand_files_0 = rand_gen(src_0)
for img in tqdm(rand_files_0):
  shutil.copy(os.path.join(src_0, img), dst_0)
print('done for 0')

rand_files_1 = rand_gen(src_1)
for img in tqdm(rand_files_1):
  shutil.copy(os.path.join(src_1, img), dst_1)
print('done for 1')
