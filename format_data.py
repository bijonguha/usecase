# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 23:07:37 2021

@author: z00490ds
"""

##Reformat Data

import glob
import random

im_dir = 'D:/INTERVIEWS/stryker/Pytorch-UNet/data/DAVIS2017-master/JPEGImages'
an_dir = 'D:/INTERVIEWS/stryker/Pytorch-UNet/data/DAVIS2017-master/Annotations'

im_files = glob.glob(im_dir + '/**/*.jpg', recursive=True)
an_files = glob.glob(an_dir + '/**/*.png', recursive=True)

im_files = [i.split('/DAVIS2017-master/')[1] for i in im_files]
an_files = [i.split('/DAVIS2017-master/')[1] for i in an_files]

conc = [str('%s %s\n') %(i,j) for i,j in zip(im_files, an_files)]

random.shuffle(conc)

tr_len = int(0.80 * len(conc))
va_len = len(conc) - int(0.80 * len(conc))

file1 = open("train.txt","w")#write mode 
file1.writelines(conc[:tr_len]) 
file1.close()

file2 = open("val.txt","w")#write mode 
file2.writelines(conc[tr_len:]) 
file2.close()

file3 = open("trainval.txt","w")#write mode 
file3.writelines(conc) 
file3.close()