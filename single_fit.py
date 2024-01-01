import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import requests
from bresenham import bresenham
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb

from utils.convert_utils import to_stroke3
from utils.load_vec import Dataset_Quickdraw, mydrawPNG_from_list
from utils.loss import Dist_wrapper
from utils.model import DeepSDF

from tqdm import tqdm



class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules():
    specs = {"LearningRateSchedule" : [
    {
      "Type" : "Step",
      "Initial" : 1e-4,
      "Interval" : 500,
      "Factor" : 0.5
    },
    {
      "Type" : "Step",
      "Initial" : 1e-3,
      "Interval" : 500,
      "Factor" : 0.5
    }]}

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules

def get_step(stroke_len, gt_pens):
    t_ = []
    del_t = []
    ctr = 0
    i_in = 0
    i_out = 0
    
    for t in range(stroke_len):
        t_.extend([ctr])
        del_t.extend([1])
        i_out = t
        ctr+=1
        if gt_pens[t] == 1:        
            i_out = t
            t_[i_in:i_out+1] = [(m-t_[i_in])/(i_out+1-i_in) for m in t_[i_in:i_out+1]]
            del_t[i_in:i_out+1] = [m/(i_out+1-i_in) for m in del_t[i_in:i_out+1]]
            i_in = t+1
        
    t_model = torch.tensor(range(stroke_len)).float()/stroke_len
    del_t = torch.tensor([1-t_model[-1].item()]).float().repeat(stroke_len)
    t_in = positionalencoding1d(t_model.cuda()).cuda()
    del_t = positionalencoding1d(del_t.cuda()).cuda()
    
    return t_in, del_t



def GetSpacedElements(array, numElems = 4):
    out = array[np.round(np.linspace(0, len(array)-1, numElems)).astype(int)]
    return out


def preprocess_data(data, num_points=20):
    data_xy = data
    
    initX, initY = int(data_xy[0,0]), int(data_xy[0,1])
    stroke_bbox = []
    pixel_length = 0
    new_list = []
    orig_list = []
    stroke_list = []
    for i in range(0, len(data_xy)):
        if i > 0:
            if data_xy[i - 1, 2] == 1:
                new_pts = GetSpacedElements(np.array(new_list), num_points)
                stroke_list.extend(new_pts)
                
                new_list = []
                orig_list = []
                initX, initY = int(data_xy[i, 0]), int(data_xy[i, 1])

        cordList = list(bresenham(initX, initY, int(data_xy[i, 0]), int(data_xy[i, 1])))
        pixel_length += len(cordList)
        orig_list.append([int(data_xy[i, 0]), int(data_xy[i, 1])])
        for cord in cordList:
            if (cord[0] > 0 and cord[1] > 0) and (cord[0] < 256 and cord[1] < 256):
                new_list.append(cord)

        initX, initY = int(data_xy[i, 0]), int(data_xy[i, 1])
    
    new_pts = GetSpacedElements(np.array(new_list), num_points)
    stroke_list.extend(new_pts)
 
    stroke_list = np.array(stroke_list)
    pens = ([0]*19 + [1])*int(len(stroke_list)/20)
    pens = np.expand_dims(pens, axis=1)
    out_ = np.concatenate((stroke_list, pens), axis=-1)
    return out_

def positionalencoding1d(t, d_model = 8):
    # return t.unsqueeze(1)
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    t = t * 200
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(t.shape[0],d_model)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model))).cuda()
    t = t.unsqueeze(1).repeat(1,div_term.shape[0])
    div_term = div_term.unsqueeze(0).repeat(t.shape[0],1)
    pe[:,0::2] = torch.sin(t * div_term)
    pe[:,1::2] = torch.cos(t * div_term)

    return pe.cuda()


def create_args():
    parser = argparse.ArgumentParser(description='Sample Args')
    parser.add_argument('--lamda', type=float, default=0.9, help='Value of lamda')
    parser.add_argument('--dset', type=str, default='quickdraw', help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='samples_out_single', help='Output directory')
    parser.add_argument('--category', type=str, default='cat', help='Category name')
    parser.add_argument('--latent_size', type=int, default=512, help='Latent size')
    parser.add_argument('--sample_idx', type=int, default=1, help='Sample index') # Not implemented
    parser.add_argument('--log_interval', type=int, default=200, help='Log interval')
    parser.add_argument('--num_iterations', type=int, default=10000, help='Number of iterations')

    args = parser.parse_args()
    return args

def sketch_point_Inp(sketch,name, dir_name, invert_yaxis=True):
    
    sketch = sketch.astype(np.int64)
    
    stroke_list = np.split(sketch[:, :2], np.where(sketch[:, 2])[0] + 1, axis=0)[:-1]
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.plasma(np.linspace(0,1,len(stroke_list)+2)))
    for stroke in stroke_list:
        stroke = stroke[:, :2].astype(np.int64)
        plt.plot(stroke[:, 0], stroke[:, 1], '.', linestyle='--', dashes=(2, 0.4), linewidth=3.0, markersize=3)#0, cmap = "reds")
    if invert_yaxis:
        plt.gca().invert_yaxis();
    plt.axis('off')
    plt.savefig(f'./{dir_name}/{name}.png', bbox_inches='tight',
                pad_inches=0.3, dpi=100)
    plt.cla()
    plt.clf()


if __name__ == "__main__":

    args = create_args()
    dir_name = args.output_dir
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    category_name = args.category
    lamda = args.lamda
    latent_size = args.latent_size
    
    root = "/vol/research/hmriCV2/Datasets/"
    data_all = []

    if args.dset == "quickdraw":
        with open("qd_full.pkl", "rb") as f:
                random_indices = pickle.load(f)

        dset = Dataset_Quickdraw(root, 'Train', raster_only=False)

        data_ = dset.__getitem__(
            random_indices[args.category][args.sample_idx]
            )
        if sum(data_["sketch_points"][:,2]) <= 4:
            """ Do not process less than 4 strokes"""
            print("Skipping sample as less than 4 strokes")
            # exit()
                
        data_all.append(torch.from_numpy(preprocess_data(data_['sketch_points'])))



    elif args.dset == "fscoco":
            root_fs = os.path.join(root, "fscoco","vector_sketches")
            subdir = os.listdir(root_fs)
            subdir = [os.path.join(root_fs,i) for i in subdir]
            files = []
            for sub in tqdm(subdir):
                for file in os.listdir(sub):
                    files.append(os.path.join(sub,file))
            file_ = files[args.sample_idx]
            try:
                data_all.append(torch.from_numpy(preprocess_data(np.load(file_))).cuda())
            except Exception as e:
                print(e)
            
        
    elif args.dset == "sketchy":
        f_names = []
        

        root_sk = os.path.join(root, "sketches")
        
        for category in os.listdir(root_sk):
            category_txt = os.path.join(root_sk, category, "checked.txt")
            with open(category_txt, "r") as f:
                lines = f.readlines()
            lines = [i.strip() for i in lines]
            for idx in lines:
                file = os.path.join(root_sk, category, idx+".svg")
                f_names.append(file)
                
        if args.category!="all":
            f_train = [i for i in f_names if args.category in i]
            f_train = [f_train[args.sample_idx]]
        else:
            f_train = [f_names[args.sample_idx]]
        

        img_all = []
        data_all = []
        file_all = []
        for i,file in enumerate(tqdm(f_train)):
            stroke = torch.tensor(preprocess_data(to_stroke3(file)))
            if sum(stroke[:,2]) <= 4:
                print("Skipping sample as less than 4 strokes")
                # exit()
            data_all.append(stroke)
    

    model = DeepSDF(latent_size=latent_size, num_tensors=4).cuda()
    params = sum(p.numel() for p in model.parameters())
    print("Number of Parameters {}".format(params))
    print("Number of Samples {}".format(len(data_all)))

    lat_vecs = torch.nn.Embedding(len(data_all), latent_size, max_norm=1.0).cuda()
    lr_schedules = get_learning_rate_schedules()
    optimizer_all = torch.optim.Adam(
        [
            {
                "params": model.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    criterion = Dist_wrapper(grid_size=64)

    for i in tqdm(range(args.num_iterations)):
        model.train()
        for s_n in range(len(data_all)):
            data = data_all[s_n]
            batch_vec = lat_vecs(torch.tensor([s_n]).long().cuda())
            stroke_len = len(data)
            gt_coords = data[:, :2] / 256
            gt_coords = gt_coords.float()
            gt_pens = data[:, 2].unsqueeze(1).float()

            stroke_len_rand = torch.randint(int(stroke_len / 3), int(stroke_len * 3), (1,)).item()
            num_strokes_rand = int(sum(gt_pens) * stroke_len_rand / stroke_len)
            stroke_len_rand = int(num_strokes_rand * 20)
            stroke_division = int(stroke_len_rand / num_strokes_rand)
            stroke_arr = torch.zeros(stroke_len_rand).long() + int(num_strokes_rand) - 1
            pens_arr = torch.zeros(stroke_len_rand)

            for m in range(num_strokes_rand - 1):
                stroke_arr[m * stroke_division:(m + 1) * stroke_division] = m
                pens_arr[(m + 1) * stroke_division - 1] = 1

            pens_arr[-1] = 1
            gamma_curr = 250

            assert sum(pens_arr) == num_strokes_rand

            stroke_arr = stroke_arr / num_strokes_rand
            num_stroke_in = positionalencoding1d(stroke_arr.float().cuda())
            num_stroke_full = positionalencoding1d(1 - stroke_arr[-1].repeat(len(stroke_arr)).float().cuda())

            t_in, del_t = get_step(stroke_len_rand, pens_arr)
            input_ = torch.cat([t_in, del_t, num_stroke_in, num_stroke_full, batch_vec.repeat(stroke_len_rand, 1)], dim=1)
            pred_coords = model(input_)

            pred_pens = pens_arr.unsqueeze(1).cuda()
            pred = torch.cat((pred_coords, pred_pens), dim=1)
            gt = torch.cat((gt_coords, gt_pens), dim=1)
            loss = criterion.forward_single(pred, gt.cuda(), gamma=gamma_curr)

            loss_sum = loss

            num_strokes = int(sum(gt_pens))
            stroke_division = int(stroke_len / num_strokes)
            stroke_arr = torch.zeros(stroke_len).long() + int(num_strokes) - 1

            ctr = 0
            for m in range(len(gt_pens)):
                stroke_arr[m] = ctr
                if gt_pens[m] == 1:
                    ctr += 1

            pens_arr = gt_pens.squeeze()
            stroke_arr = stroke_arr / num_strokes
            num_strokes_in = positionalencoding1d(stroke_arr.float().cuda())
            num_strokes_full = positionalencoding1d(1 - stroke_arr[-1].repeat(len(stroke_arr)).float().cuda())

            t_in, del_t = get_step(stroke_len, gt_pens)
            input_ = torch.cat([t_in, del_t, num_strokes_in, num_strokes_full, batch_vec.repeat(stroke_len, 1)], dim=1)
            pred_coords = model(input_)
            pred_pens = pens_arr.unsqueeze(1).cuda()

            loss_ = {"coord_loss": nn.MSELoss()(pred_coords, gt_coords.cuda()).mean(), "pen_loss": torch.tensor(0).cuda()}
            loss_sum['coord_loss'] = loss_['coord_loss']
            loss_sum['pen_loss'] = loss_['pen_loss']

            loss_sum = (loss_sum['coord_loss']) * lamda + (loss_sum['dist_loss']) * (1 - lamda)
            loss_sum.backward()
            optimizer_all.step()
            optimizer_all.zero_grad()

        if i % args.log_interval == 0:
            print('Iteration: {}, Dist_loss {} Coord_loss {} '.format(i, loss['dist_loss'].item(), loss['coord_loss'].item() + loss['pen_loss'].item()))
            with torch.no_grad():
                try:
                    os.mkdir('./{}/{}'.format(dir_name, category_name))
                except:
                    pass
                idx = 0
                with torch.no_grad():
                    data = data_all[idx]
                    batch_vec = lat_vecs(torch.tensor([idx]).long().cuda())
                    stroke_len = len(data)

                    gt_coords = data[:, :2] / 256
                    gt_pens = data[:, 2].long().unsqueeze(1)
                    num_strokes = int(sum(gt_pens))
                    stroke_division = int(stroke_len / num_strokes)
                    stroke_arr = torch.zeros(stroke_len).long() + int(num_strokes) - 1
                    ctr = 0
                    for m in range(len(gt_pens)):
                        stroke_arr[m] = ctr
                        if gt_pens[m] == 1:
                            ctr += 1
                    pens_arr = gt_pens.squeeze()
                    stroke_arr = stroke_arr / num_strokes

                    num_strokes_in = positionalencoding1d(stroke_arr.float().cuda())
                    num_strokes_full = positionalencoding1d(1 - stroke_arr[-1].repeat(len(stroke_arr)).float().cuda())

                    t_in, del_t = get_step(stroke_len, gt_pens.squeeze())
                    input_ = torch.cat([t_in, del_t, num_strokes_in, num_strokes_full, batch_vec.repeat(stroke_len, 1)], dim=1)
                    pred_coords = model(input_)
                    pred_pens = gt_pens.cuda()

                    gt_points = torch.cat((gt_coords * 256, gt_pens), dim=1).cpu().numpy()
                    sketch_point_Inp(gt_points, '{}/gt'.format(category_name), dir_name)

                    sketch_points = torch.cat((pred_coords * 256, pred_pens), dim=1).cpu().numpy()
                    print("Error {}".format(nn.MSELoss(reduce="mean")(pred_coords, gt_coords.cuda())))

                    sketch_point_Inp(sketch_points, '{}/pred_lamda_{}'.format(category_name, lamda), dir_name)
