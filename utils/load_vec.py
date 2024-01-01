import numpy as np
from bresenham import bresenham
import torchvision.transforms as transforms
import torch.utils.data as data
import lmdb
import pickle
import random
import scipy.ndimage
from PIL import Image
import torchvision.transforms.functional as F
import torch
import os

def to_Five_Point(sketch_points, max_seq_length):
    len_seq = len(sketch_points[:, 0])
    new_seq = np.zeros((max_seq_length, 5))
    new_seq[0:len_seq, :2] = sketch_points[:, :2]
    new_seq[0:len_seq, 3] = sketch_points[:, 2]
    new_seq[0:len_seq, 2] = 1 - new_seq[0:len_seq, 3]
    new_seq[(len_seq - 1):, 4] = 1
    new_seq[(len_seq - 1), 2:4] = 0
    new_seq = np.concatenate((np.zeros((1, 5)), new_seq), axis=0)
    return new_seq, len_seq

def mydrawPNG(vector_image, Side = 256):

    raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)
    initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
    stroke_bbox = []
    pixel_length = 0

    for i in range(0, len(vector_image)):
        if i > 0:
            if vector_image[i - 1, 2] == 1:
                initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

        cordList = list(bresenham(initX, initY, int(vector_image[i, 0]), int(vector_image[i, 1])))
        pixel_length += len(cordList)

        for cord in cordList:
            if (cord[0] > 0 and cord[1] > 0) and (cord[0] < Side and cord[1] < Side):
                raster_image[cord[1], cord[0]] = 255.0
        initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

    raster_image = scipy.ndimage.binary_dilation(raster_image) * 255.0

    return raster_image, stroke_bbox


def preprocess(sketch_points, side = 256.0):
    sketch_points = sketch_points.astype(np.float64)
    sketch_points[:, :2] = sketch_points[:, :2] / np.array([800, 800])
    sketch_points[:,:2] = sketch_points[:,:2] * side
    sketch_points = np.round(sketch_points)
    return sketch_points

def rasterize_Sketch(sketch_points):
    sketch_points = preprocess(sketch_points)
    raster_images, stroke_bbox = mydrawPNG(sketch_points)
    return raster_images,  sketch_points


def mydrawPNG_from_list(vector_image, Side = 256):

    raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)

    for stroke in vector_image:
        initX, initY = int(stroke[0, 0]), int(stroke[0, 1])

        for i_pos in range(1, len(stroke)):
            cordList = list(bresenham(initX, initY, int(stroke[i_pos, 0]), int(stroke[i_pos, 1])))
            for cord in cordList:
                if (cord[0] > 0 and cord[1] > 0) and (cord[0] <= Side and cord[1] <= Side):
                    raster_image[cord[1], cord[0]] = 255.0
                else:
                    print('error', cord)
                    print(vector_image)
                    print(stroke)

                    print(initX, initY, int(stroke[i_pos, 0]), int(stroke[i_pos, 1]))
                    exit()
            initX, initY =  int(stroke[i_pos, 0]), int(stroke[i_pos, 1])

    raster_image = scipy.ndimage.binary_dilation(raster_image) * 255.0

    return Image.fromarray(raster_image).convert('RGB')

def get_transform(type):
    transform_list = []
    if type == 'Train':
        transform_list.extend([transforms.Resize(256)])
    elif type == 'Test':
        transform_list.extend([transforms.Resize(256)])
    transform_list.extend([transforms.ToTensor()])
    return transforms.Compose(transform_list)

        
class Dataset_Quickdraw(data.Dataset):
    def __init__(self, root, mode, raster_only = False):
        self.txn = False
        self.root = root
        with open(root+"QuickDraw/QuickDraw_Keys.pickle", "rb") as handle:
            self.Train_keys, self.Valid_keys, self.Test_keys = pickle.load(handle)

        get_all_classes, all_samples = [], []
        for x in self.Train_keys:
            get_all_classes.append(x.split('_')[0])
            all_samples.append(x)
        get_all_classes = list(set(get_all_classes))
        get_all_classes.sort()

        self.classnames=get_all_classes

        #################################################################
        #########  MAPPING Dictionary ###################################
        self.num2name, self.name2num = {}, {}
        for num, val in enumerate(get_all_classes):
            self.num2name[num] = val
            self.name2num[val] = num

        print('Total Training Sample {}'.format(len(self.Train_keys)))
        print('Total Testing Sample {}'.format(len(self.Test_keys)))


        self.train_transform = get_transform('Train')
        self.test_transform = get_transform('Test')

        self.mode = mode
        self.raster_only = raster_only

        

    def opener(self):
        # self.hp = hp
        self.TrainData_ENV = []
        self.TestData_ENV = []


        if self.mode == 'Train':
            self.TrainData_ENV = lmdb.open(self.root+"QuickDraw/QuickDraw_TrainData",  max_readers=1,readonly=True, lock=False, readahead=False, meminit=False)
    
        elif self.mode == 'Test':
            self.TestData_ENV = lmdb.open(self.root+"QuickDraw/QuickDraw_TestData",  max_readers=1,readonly=True, lock=False, readahead=False, meminit=False)

    def __getitem__(self, item):
        if not self.txn:
            self.opener()

        if self.mode == 'Train':
            with self.TrainData_ENV.begin(write=False) as txn:
                sketch_path = self.Train_keys[item]
                sample = txn.get(sketch_path.encode("ascii"))
                sketch_points = np.frombuffer(sample).reshape(-1, 3).copy()
                stroke_list = np.split(sketch_points[:, :2], np.where(sketch_points[:, 2])[0] + 1, axis=0)[:-1]
                sketch_img = mydrawPNG_from_list(stroke_list)

            
            sketch_img = self.train_transform(sketch_img)
            if not self.raster_only:
                sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path, 'sketch_points': sketch_points,
                        'sketch_label': self.name2num[sketch_path.split('_')[0]]}
            else:
                sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path, 'sketch_label': self.name2num[sketch_path.split('_')[0]]}


        elif self.mode == 'Test':

            with self.TestData_ENV.begin(write=False) as txn:
                sketch_path = self.Test_keys[item]
                sample = txn.get(sketch_path.encode())
                sketch_points = np.frombuffer(sample).reshape(-1, 3).copy()
                stroke_list = np.split(sketch_points[:, :2], np.where(sketch_points[:, 2])[0] + 1, axis=0)[:-1]
                sketch_img = mydrawPNG_from_list(stroke_list)

                n_flip = random.random()
                sketch_img = self.train_transform(sketch_img)


                if not self.raster_only:
                    sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path, 'sketch_points': sketch_points,
                            'sketch_label': self.name2num[sketch_path.split('_')[0]]}
                else:
                    sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path, 'sketch_label': self.name2num[sketch_path.split('_')[0]]}
        return sample

    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_keys)
        elif self.mode == 'Test':
            return len(self.Test_keys)
        
if __name__ == '__main__':
    dset = Dataset_Quickdraw("/home/sketchx/Datasets/", 'Train')
    item = dset.__getitem__(0)
    pts = torch.tensor(item['sketch_points'])
    print(pts.shape)
    torch.save(pts, 'test_vec.pt')
    

