#!/usr/bin/python

import cPickle
import csv
import os
import numpy as np
from skimage import io
from skimage import transform
from PIL import Image

from multiprocessing.pool import ThreadPool

def loadClassName(filename):
  class_name = []
  with open(filename, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    i = 0
    for row in spamreader:
      class_name.append(row[1])
      i = i+1
      if i >= 1000:
        break

  image_name = []
  for dirpath, dirnames, filenames in os.walk('/home/hhwu/ImageNet/train/'):
    print "dirpath: ", dirpath
    print "dirnames: ", dirnames
    print "The number of files: %d" % len(filenames)

    image_name = filenames

  return class_name, image_name



def computeMean100K(threadName, dirpath, image_name, start_idx):
  mean = np.zeros((224,224,3))
  counter = 0

  print "Thread: %s starting!" % threadName

  if start_idx + 100000 < len(image_name):
    image_list = image_name[start_idx:start_idx+100000]
  else:
    image_list = image_name[start_idx:]

  i = 0
  for f in image_list:
    absfile = os.path.join(dirpath, f)
    target_img = io.imread(absfile)


    #print target_img
    #print absfile
    if i % 100 == 0:
      print "%s: processing %d images..." % (threadName, i)

    i = i + 1
    ###############################
    #       shape[0]: height      #
    #       shape[1]: width       #
    ###############################

    #Grayscale Img and convert it to RGB
    if len(target_img.shape) == 2:
      RGB_img = np.zeros((target_img.shape[0],target_img.shape[1],3))
      RGB_img[:,:,0] = target_img
      RGB_img[:,:,1] = target_img
      RGB_img[:,:,2] = target_img

      target_img = RGB_img
      counter = counter + 1


    #print target_img.shape
    #print absfile
    if target_img.shape[0] < target_img.shape[1]:
      width = int(target_img.shape[1]*256/target_img.shape[0])
      #if width < 224:
      #  width = 224

      offset = int((width-224)/2)

      target_img = transform.resize(target_img, (256,width,3))
      target_img = target_img[16:240, offset:224+offset, :]
    else:
      height = int(target_img.shape[0]*256/target_img.shape[1])
      #if height < 224:
      #  height = 224

      offset = int((height-224)/2)

      target_img = transform.resize(target_img, (height,256,3))
      target_img = target_img[offset:224+offset, 16:240, :]

    mean = mean + target_img

  #print "(%s)BW num: %d" % (threadName, counter)
  #print "(%s)len: %d" % (threadName, len(image_list))
  #print target_img.shape
  #print 256*target_img
  #io.imsave("%s%s" % (threadName, '.jpeg'), int(255*target_img))
  return mean/len(image_list)
   


if __name__ == '__main__':
  class_name, image_name  = loadClassName('synset.csv')
  
  #print class_name
  #print len(class_name) 
  
  num_images = 1281167
  batch_size = 64
  dirpath = '/home/hhwu/ImageNet/train/'
  
  
  batch_idx = np.random.randint(0,num_images,batch_size)
  for i in batch_idx:
    absfile = os.path.join(dirpath, image_name[i])

 

  #computeMean100K(dirpath, image_name)


  pool = ThreadPool(processes=13)
  print "Multi-threads begin!"

  async_result_1 = pool.apply_async(computeMean100K, ("Thread-1", dirpath, image_name, 0))
  async_result_2 = pool.apply_async(computeMean100K, ("Thread-2", dirpath, image_name, 100000))
  async_result_3 = pool.apply_async(computeMean100K, ("Thread-3", dirpath, image_name, 200000))
  async_result_4 = pool.apply_async(computeMean100K, ("Thread-4", dirpath, image_name, 300000))
  async_result_5 = pool.apply_async(computeMean100K, ("Thread-5", dirpath, image_name, 400000))
  async_result_6 = pool.apply_async(computeMean100K, ("Thread-6", dirpath, image_name, 500000))
  async_result_7 = pool.apply_async(computeMean100K, ("Thread-7", dirpath, image_name, 600000))
  async_result_8 = pool.apply_async(computeMean100K, ("Thread-8", dirpath, image_name, 700000))
  async_result_9 = pool.apply_async(computeMean100K, ("Thread-9", dirpath, image_name, 800000))
  async_result_10 = pool.apply_async(computeMean100K, ("Thread-10", dirpath, image_name, 900000))
  async_result_11 = pool.apply_async(computeMean100K, ("Thread-11", dirpath, image_name, 1000000))
  async_result_12 = pool.apply_async(computeMean100K, ("Thread-12", dirpath, image_name, 1100000))
  async_result_13 = pool.apply_async(computeMean100K, ("Thread-13", dirpath, image_name, 1200000))
   
  return_val_1 = async_result_1.get()
  return_val_2 = async_result_2.get()
  return_val_3 = async_result_3.get()
  return_val_4 = async_result_4.get()
  return_val_5 = async_result_5.get()
  return_val_6 = async_result_6.get()
  return_val_7 = async_result_7.get()
  return_val_8 = async_result_8.get()
  return_val_9 = async_result_9.get()
  return_val_10 = async_result_10.get()
  return_val_11 = async_result_11.get()
  return_val_12 = async_result_12.get()
  return_val_13 = async_result_13.get()

  mean = (return_val_1+return_val_2+return_val_3+return_val_4+return_val_5+return_val_6+return_val_7+return_val_8+
          return_val_9+return_val_10+return_val_11+return_val_12+return_val_13)/13.0
         

  ouf = open('mean.bin', 'w')
  cPickle.dump(mean, ouf, 1)
  ouf.close()

  #fo = open('mean.bin', 'rb')
  #test_file = cPickle.load(fo)
  #fo.close()

  #print test_file
  #print len(test_file)
 
#  ###############################
#  #       shape[0]: height      #
#  #       shape[1]: width       #
#  ###############################
#  print absfile  
#  test_img = io.imread(absfile)
#  print test_img.shape
#  if test_img.shape[0] < test_img.shape[1]:
#    width = int(test_img.shape[1]*256/test_img.shape[0])
#    offset = int((width-224)/2)
#
#    test_img = transform.resize(test_img, (256,width,3))
#    test_img = test_img[16:240, offset:224+offset, :]
#  else:
#    height = int(test_img.shape[0]*256/test_img.shape[1])
#    test_img = transform.resize(test_img, (height,256,3))
#    test_img = test_img[offset:224+offset, 16:240, :]
#
#
#  print test_img.shape
#  io.imsave("%s%s" % ("test_img_", '.jpeg'), test_img)

  #img = Image.open(absfile)
  #print img.size
  #print img.shape


  #for f in image_name:
  #  absfile = os.path.join(dirpath, f)
  #  test_img = io.imread(absfile)

  #  if test_img.shape[0] < test_img.shape[1]:
  #    print "FUCKING DPP"
