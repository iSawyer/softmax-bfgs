# -*- coding: utf-8 -*-


import os, struct
from array import array


train_data_path = "data/train-images-idx3-ubyte"
train_label_path = "data/train-labels-idx1-ubyte"
test_data_path = "data/t10k-images-idx3-ubyte"
test_label_path = "data/t10k-labels-idx1-ubyte"


def process_data(data_path,label_path,state = "train"):
    fd = open(label_path,'rb')
    magic, size = struct.unpack(">II", fd.read(8))
    labels = array("B",fd.read())
    fd.close()
    
    fd = open(data_path,'rb')
    magic,size,rows,cols = struct.unpack(">IIII", fd.read(16))
    image_data = array("B", fd.read())
    images = []
    for i in xrange(size):
        images.append([0]*rows*cols)
    for i in xrange(size):
        images[i][:] = image_data[i*rows*cols : (i+1)*rows*cols]
    
    
    file_name1 = state + "_data"
    file_name2 = state + "_label"
    file_name3 = state + "_info"
    fw = open(file_name1,'w')
    for i in range(size):
        for j in range(rows*cols):
            fw.write(str(images[i][j]) + ' ')
    fw.close()
    fw = open(file_name2,'w')
    for i in range(size):
        fw.write(str(labels[i]) + ' ')
    fw.close()

    fw = open(file_name3,'w')
    fw.write("sample number:" + str(size) + '\t' + "sample size:" + str(rows*cols))
    fw.close()




    

process_data(train_data_path,train_label_path)
process_data(test_data_path, test_label_path, "test")
