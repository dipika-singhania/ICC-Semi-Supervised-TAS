import importlib
import os, sys
from sklearn.cluster import KMeans
from utility.perform_linear import get_linear_acc
from utility.dump_features import DumpFeatures
import warnings
import copy
import glob
import pandas as pd
warnings.filterwarnings('ignore')

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import C2F_TCN
from utils import calculate_mof
from postprocess import PostProcess
from dataset import Breakfast, collate_fn_override
import torch.nn.functional as F

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--split_number', type=int, required=False)
my_parser.add_argument('--semi_per', type=float, required=True)
my_parser.add_argument('--cudad', type=str)
# my_parser.add_argument('--noaug', action='store_true')
# my_parser.add_argument('--nolps', action='store_true')
my_parser.add_argument('--transposed', action='store_true')
my_parser.add_argument('--loadtext', action='store_true')
my_parser.add_argument('--base_dir', type=str, default="/mnt/data/ar-datasets/dipika/breakfast/ms_tcn/data/breakfast/")
my_parser.add_argument('--dataset_name', type=str, default="breakfast")
my_parser.add_argument('--wd', type=float, required=False)
my_parser.add_argument('--lr', type=float, required=False)
my_parser.add_argument('--chunk_size', type=int, required=False)
my_parser.add_argument('--max_frames_per_video', type=int, required=False)
my_parser.add_argument('--weights', type=str, required=False)
my_parser.add_argument('--features_file_name', type=str, required=False)
my_parser.add_argument('--feature_size', type=int, required=False)
my_parser.add_argument('--epochs', type=int, default=500, required=False)
my_parser.add_argument('--avg_actions', type=int, required=False)
my_parser.add_argument('--data_per_low', type=float, required=False)
my_parser.add_argument('--data_per_high', type=float, required=False)
my_parser.add_argument('--ftype', default='i3d', type=str)
my_parser.add_argument('--outft', type=int, default=256)
my_parser.add_argument('--unsuper', action='store_true')
args = my_parser.parse_args()

seed = 42

# Ensure deterministic behavior
def set_seed():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
set_seed()

# Device argsuration
os.environ['CUDA_VISIBLE_DEVICES']=args.cudad
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_best_model(args):
    return torch.load(args.output_dir + '/best_' + args.dataset_name + '_unet.wt')

def load_avgbest_model(args):
    return torch.load(args.output_dir + '/avgbest_' + args.dataset_name + '_unet.wt')

def make(args):
    # Make the data
    validation_data_files = open(args.test_split_file).read().split("\n")[0:-1]
    testdat = get_data(args, validation_data_files, train=False)
    test_loader = make_loader(testdat, batch_size=args.batch_size, train=False)

    # Make the model
    model = get_model(args).to(device)
    
    num_params = sum([p.numel() for p in model.parameters()])
    print("Number of parameters = ", num_params/1e6, " million")

    # postprocessor declaration
    postprocessor = PostProcess(args)
    postprocessor = postprocessor.to(device)
    
    return model, test_loader, postprocessor


def get_data(args, split_file_list, train=True, pseudo_data=False):
    if train is True:
        fold='train'
    else:
        fold='val'
    dataset = Breakfast(args, pseudo_data, fold=fold, list_data=split_file_list)
    return dataset


def make_loader(dataset, batch_size, train=True):
    def _init_fn(worker_id):
        np.random.seed(int(seed))
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=train,
                                         pin_memory=True, num_workers=7, collate_fn=collate_fn_override,
                                         worker_init_fn=_init_fn)
    return loader


def get_model(args):
    set_seed()
    return C2F_TCN(n_channels=args.feature_size, n_classes=args.num_class, n_features=args.outft)


def test(model, test_loader, postprocessors, args):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for i, item in enumerate(test_loader):
            samples = item[0].to(device).permute(0, 2, 1)
            count = item[1].to(device)
            labels = item[2].to(device)
            src_mask = torch.arange(labels.shape[1], device=labels.device)[None, :] < count[:, None]
            src_mask = src_mask.to(device)

            if args.dataset_name == 'breakfast':
               activity_labels = np.array([name.split('_')[-1] for name in item[5]])
            elif args.dataset_name == '50salads':
               activity_labels = None 
            elif args.dataset_name == 'gtea':
               activity_labels = None 

            # Forward pass ➡
            projection_out, pred_out = model(samples, args.weights)
            
            pred = torch.argmax(pred_out, dim=1)
            correct += float(torch.sum((pred == labels) * src_mask).item())
            total += float(torch.sum(src_mask).item())
            postprocessors(pred_out, item[5], labels, count)
            
        # Add postprocessing and check the outcomes
        path = os.path.join(args.output_dir, "predict_" + args.dataset_name)
        if not os.path.exists(path):
            os.mkdir(path)
        postprocessors.dump_to_directory(path)
        final_edit_score, map_v, overlap_scores = calculate_mof(args.ground_truth_files_dir, path, args.back_gd)
        postprocessors.start()
        acc = 100.0 * correct / total
        print(f"Accuracy of the model on the {total} " +
              f"test images: {acc}%")
        return overlap_scores, final_edit_score, map_v * 100.0

all_splits = []
if args.split_number is None:
    if args.dataset_name != "50salads":
        all_splits = [1, 2, 3, 4]
    else:
        all_splits = [1, 2, 3, 4, 5]
else:
    all_splits.append(args.split_number)


over_10 = []
over_25 = []
over_50 = []
edit_arr = []
mof_arr = []

import copy
for split in all_splits:
    origargs = copy.deepcopy(args)
    args.output_dir = args.base_dir + "results/full_C2FTCN_split{}_unsuper{}_per{}".format(args.split_number, args.unsuper, args.semi_per)
    
    if args.wd is not None:
        args.output_dir=args.output_dir + "_wd{:.5f}".format(args.wd)
    
    if args.lr is not None:
        args.output_dir=args.output_dir + "_lr{:.6f}".format(args.lr)
    
    if args.chunk_size is not None:
        args.output_dir=args.output_dir + "_chunk{}".format(args.chunk_size)
    
    if args.max_frames_per_video is not None:
        args.output_dir=args.output_dir + "_maxf{}".format(args.max_frames_per_video)
    
    if args.data_per_low is not None:
        args.output_dir=args.output_dir + "_dl{:.4f}".format(args.data_per_low)
    
    if args.data_per_high is not None:
        args.output_dir = args.output_dir + "_dh{:.4f}".format(args.data_per_high)
    
    if args.weights is not None:
        args.output_dir = args.output_dir + "_wts{}".format(args.weights.replace(',','-'))
        args.weights = list(map(int, args.weights.split(",")))
        print("Weights being used is ", args.weights)
    
    if args.feature_size is not None:
        args.output_dir = args.output_dir + "_ft_size{}".format(args.feature_size)
    else:
        args.feature_size = 2048
    
    if args.avg_actions is not None:
        args.output_dir = args.output_dir + "_avdiv{}".format(args.avg_actions)
    
    if args.dataset_name == "50salads":
        if args.chunk_size is None:
            args.chunk_size = 20
        if args.max_frames_per_video is None:
            args.max_frames_per_video = 960
        if args.lr is None:
            args.lr = 3e-4
        if args.wd is None:
            args.wd = 1e-3
        args.batch_size = 50
        args.num_class = 19
        args.back_gd = ['action_start', 'action_end']
        if args.weights is None:
            args.weights = [1, 1, 1, 1, 1, 1]
        if args.data_per_low is None:
            args.data_per_low = 0.00200
        if args.data_per_high is None:
            args.data_per_high = 0.0300
        args.high_level_act_loss = False
        if args.avg_actions is None:
            args.avg_actions = 70
    elif args.dataset_name == "breakfast":
        if args.chunk_size is None:
            args.chunk_size = 10
        if args.max_frames_per_video is None:
            args.max_frames_per_video = 600
        if args.lr is None:
            args.lr = 1e-4
        if args.wd is None:
            args.wd = 3e-3
        args.batch_size = 100
        args.num_class = 48
        args.back_gd = ['SIL']
        if args.weights is None:
            args.weights = [1, 1, 1, 1, 1, 1]
        if args.data_per_low is None:
            args.data_per_low = 0.0050
        if args.data_per_high is None:
            args.data_per_high = 0.0300
        args.high_level_act_loss = True
        if args.avg_actions is None:
            args.avg_actions = 20
    elif args.dataset_name == "gtea":
        if args.chunk_size is None:
            args.chunk_size = 4
        if args.max_frames_per_video is None:
            args.max_frames_per_video = 600
        if args.lr is None:
            args.lr = 5e-4
        if args.wd is None:
            args.wd = 3e-4
        args.batch_size = 25
        args.num_class = 11
        args.back_gd = ['background']
        if args.weights is None:
            args.weights = [1, 1, 1, 1, 1, 1]
        if args.data_per_low is None:
            args.data_per_low = 0.0050
        if args.data_per_high is None:
            args.data_per_high = 0.0200
        args.high_level_act_loss = False
        if args.avg_actions is None:
            args.avg_actions = 20
    
    
    args.output_dir = args.output_dir + "/"
    print("printing in output dir = ", args.output_dir)
    
    args.test_split_file = args.base_dir + "/splits/test.split{}.bundle".format(split)
    if args.features_file_name is None:
        args.features_file_name = args.base_dir + "/features/"
    args.ground_truth_files_dir = args.base_dir + "/groundTruth/"
    args.label_id_csv = args.base_dir + "mapping.csv"
    model, test_loader, postprocessor = make(args)
    model.load_state_dict(load_best_model(args))
    over, edit, mof = test(model, test_loader,  postprocessor, args)
    over_10.append(over[0])
    over_25.append(over[1])
    over_50.append(over[2])
    edit_arr.append(edit)
    mof_arr.append(mof)
    args = copy.deepcopy(origargs)

over_10_avg = np.mean(np.array(over_10))
over_25_avg = np.mean(np.array(over_25))
over_50_avg = np.mean(np.array(over_50))
edit_avg = np.mean(np.array(edit_arr))
mof_avg = np.mean(np.array(mof_arr))
print(f"F1@10, f1@25, f1@50, edit, mof : {over_10_avg:.1f} & {over_25_avg:.1f} & {over_50_avg:.1f} & {edit_avg:.1f} & {mof_avg:.1f}") 


