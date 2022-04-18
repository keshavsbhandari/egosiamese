from torchvision.transforms import *
import numpy as np
import PIL
from PIL import Image
from equilib import equi2equi as randRotate
from py360convert import e2c as getCubes
import py360convert as projector
import torch
from random import choice
from flow_vis import flow_to_color
import torch.nn as nn
from time import time
import sys
import multiprocessing.pool as pool
import torchvision.transforms as T
import random
from imageio import imread
import torch.distributed as dist
import pickle

R = lambda :random.choice((-random.random(), random.random()))

import torch.distributed as dist

def writeToPickle(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def readFromPickle(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def sync_tensor_across_gpus(t):
    # t needs to have dim 0 for troch.cat below. 
    # if not, you need to prepare it.
    if t is None:
        return None
    group = dist.group.WORLD
    group_size = torch.distributed.get_world_size(group)
    gather_t_tensor = [torch.zeros_like(t) for _ in
                       range(group_size)]
    dist.all_gather(gather_t_tensor, t)  # this works with nccl backend when tensors need to be on gpu. 
   # for gloo and mpi backends, tensors need to be on cpu. also this works single machine with 
   # multiple   gpus. for multiple nodes, you should use dist.all_gather_multigpu. both have the 
   # same definition... see [here](https://pytorch.org/docs/stable/distributed.html). 
   #  somewhere in the same page, it was mentioned that dist.all_gather_multigpu is more for 
   # multi-nodes. still dont see the benefit of all_gather_multigpu. the provided working case in 
   # the doc is  vague... 
    return torch.cat(gather_t_tensor, dim=0)

def readImg2Np(path):
    return imread(path).transpose(2,0,1)

def readImg2Tensor(path):
    return T.ToTensor()(Image.open(path))

def parallelImgRead(img_path_list):    
    p = pool.Pool(len(img_path_list))
    imgs = p.map(readImg2Tensor, img_path_list)
    p.close()
    p.join()
    return torch.cat(imgs,0)

def readFlo2Np(path):
    return np.load(path).transpose(2,0,1)

def readFlo2Tensor(path):
    return torch.from_numpy(np.load(path)).permute(2,0,1)

def parallelFloRead(flo_path_list):    
    p = pool.Pool(len(flo_path_list))
    flos = p.map(readFlo2Tensor, flo_path_list)
    p.close()
    p.join()
    return torch.cat(flos,0)

def imgAugmentation(img_path_list, rots):
    imgs = parallelImgRead(img_path_list)
    rotimgs = randRotate(imgs, rots=rots)
    return rotimgs

def floAugmentation(flo_path_list, rots):
    flos = parallelFloRead(flo_path_list)
    rmap, rotflos = maprange(flos)
    rotflos = randRotate(rotflos, rots=rots)
    rmap, rotflos = maprange(rotflos, **rmap)
    return rotflos

def floImgAugmentation(img_path_list = None, flo_path_list = None, return_focus = False, H = 320, W = 640):
    rots = {'pitch':R(),'roll':R(),'yaw':R()}
    sample = {}
    if img_path_list:
        sample['imgs'] = imgAugmentation(img_path_list, rots)
    if flo_path_list:
        sample['flos'] = floAugmentation(flo_path_list, rots)
    if return_focus:
        sample['focus'] = randRotate(np.load('focus.npy')[None,:], rots = rots)
    
    return sample

def stampTime(x):
    return f"{time()}_{x}"

def stopExec():
    stop = False
    with open('Extras/stopflag.txt', 'r') as f:
        if f.read() == 'stop':
            stop = True
    if stop:    
        with open('Extras/stopflag.txt', 'w') as f:
            f.write('continue')
        if stop:
            torch.cuda.empty_cache()
            sys.exit()

def simLoss(p1,p2,z1,z2):
  criterion = nn.CosineSimilarity(dim=1)
  simloss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
  return simloss
  

def count_parameters(model):
    """COUNT THE MDOEL TRAINABLE PARAMETERS

    Args:
            model (_type_): torch model

    Returns:
            _type_: integer
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def maprange(x, minfrom=None, maxfrom=None, minto=0, maxto=1, check = False):
    """map x to minfrom,maxfrom range to minto,maxto range
    Args:
            x (_type_): _description_
            minfrom (_type_, optional): _description_. Defaults to None.
            maxfrom (_type_, optional): _description_. Defaults to None.
            minto (int, optional): _description_. Defaults to 0.
            maxto (int, optional): _description_. Defaults to 1.
            check (bool, optional): _description_. Defaults to False.

    Returns:
            _type_: _description_
    """
    minfrom = x.min() if minfrom is None else minfrom
    maxfrom = x.max() if maxfrom is None else maxfrom
    if check:
        if minfrom > x.min() or maxfrom < x.max():
            x = x.clip(minfrom, maxfrom)
    reverse_map = dict(minfrom = minto, maxfrom = maxto, minto = minfrom, maxto = maxfrom)
    return reverse_map, minto + ((maxto - minto)*(x - minfrom))/(maxfrom - minfrom)

def getPanoRowsImg(img, face_w = 64, pitch = None, yaw = None, roll = None, return_rot_args = False):
    if not isinstance(img, PIL.Image.Image):
        raise Exception("img should be in PIL format")
    
    def getPitchYawRoll(p,y,r):
        C = lambda :choice(np.linspace(-np.pi,np.pi,180))
        p = C() if p is None else p
        y = C() if y is None else y
        r = C() if r is None else r
        return p,y,r
    
    def rotate(img, p, y, r):
        arr = ToTensor()(img)
        arr = randRotate(arr, 
                         rots = {'pitch':p,
                                 'yaw':y, 
                                 'roll':r})
        arr = ToPILImage()(arr)
        arr = np.asarray(arr)
        return arr
    
    def cubify(arr):
        cubes = getCubes(arr, face_w = face_w, cube_format='dict')
        cat = lambda *x,d:np.concatenate(x,d)
        row1 = cat(cubes['F'], np.fliplr(cubes['R']),np.fliplr(cubes['B']), d = 1)
        row2 = cat(np.flipud(np.rot90(cubes['U'],2)), np.rot90(cubes['L']),cubes['D'], d = 1)
        arr = cat(row1, row2, d = 0)
        return arr
    
    pitch, yaw, roll = getPitchYawRoll(pitch, yaw, roll)
    arr = rotate(img, pitch, yaw, roll)
    arr = cubify(arr)
    if return_rot_args:
        return Image.fromarray(arr), dict(pitch = pitch, roll = roll, yaw = yaw)
    return Image.fromarray(arr)

def getPanoRowsFlow(flow, face_w = 64, pitch = None, yaw = None, roll = None, return_rot_args = False):
    # flow.shape = HW2

    if not isinstance(flow, np.ndarray):
        raise Exception("flow should be an numpy ndarray")

    def getPitchYawRoll(p,y,r):
        C = lambda :choice(np.linspace(-np.pi,np.pi,180))
        p = C() if p is None else p        
        y = C() if y is None else y
        r = C() if r is None else r
        return p,y,r
    
    def rotate(flow, p, y, r):
        maxfrom, minfrom, arr = maprange(flow)
        arr = torch.from_numpy(arr).permute(2,0,1)
        arr = randRotate(arr, 
                         rots = {'pitch':p, 
                                 'yaw':y, 
                                 'roll':r})
        arr = arr.permute(1,2,0).numpy()
        return maxfrom, minfrom, arr
    
    def cubify(arr, mx, mn):
        cubes = getCubes(arr, face_w = face_w, cube_format='dict')
        cat = lambda *x,d:np.concatenate(x,d)
        row1 = cat(cubes['F'], np.fliplr(cubes['R']),np.fliplr(cubes['B']), d = 1)
        row2 = cat(np.flipud(np.rot90(cubes['U'],2)), np.rot90(cubes['L']),cubes['D'], d = 1)
        arr = cat(row1, row2, d = 0)
        _, _, arr = maprange(arr, maxto = mx, minto = mn)
        return arr

    pitch, yaw, roll = getPitchYawRoll(pitch, yaw, roll)
    maxfrom, minfrom, arr = rotate(flow, pitch, yaw, roll)
    arr = cubify(arr, maxfrom, minfrom)

    if return_rot_args:
        return arr, dict(pitch = pitch, roll = roll, yaw = yaw)
    return arr