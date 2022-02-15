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
import time

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

seed = 42
my_parser = argparse.ArgumentParser()
my_parser.add_argument('--split_number', type=int, required=True, help="Split for which results needs to be updated")
my_parser.add_argument('--semi_per', type=float, required=True, help="Percentage of semi-supervised data to be used as trainingdata")
my_parser.add_argument('--output_dir', type=str, required=True, help="Output directory where the programs outputs (checkpoints, logs etc.) must be stored")
my_parser.add_argument('--base_dir', type=str, required=True)
my_parser.add_argument('--model_wt', type=str, required=True)
my_parser.add_argument('--dataset_name', type=str, required=False)
my_parser.add_argument('--wd', type=float, required=False)
my_parser.add_argument('--lr_unsuper', type=float, required=False)
my_parser.add_argument('--lr_proj', type=float, required=False)
my_parser.add_argument('--lr_main', type=float, required=False)
my_parser.add_argument('--gamma_proj', type=float, required=False)
my_parser.add_argument('--gamma_main', type=float, required=False)
my_parser.add_argument('--epochs_unsuper', type=int, required=False)
my_parser.add_argument('--mse', action='store_true')
my_parser.add_argument(
  "--steps_proj",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=int,
  default=[600],  # default if nothing is provided
)
my_parser.add_argument(
  "--steps_main",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=int,
  default=[600, 1200],  # default if nothing is provided
)
my_parser.add_argument('--chunk_size', type=int, required=False)
my_parser.add_argument('--max_frames_per_video', type=int, required=False)
my_parser.add_argument('--weights', type=str, required=False)
my_parser.add_argument('--features_file_name', type=str, required=False)
my_parser.add_argument('--feature_size', type=int, default=2048, required=False)
my_parser.add_argument('--epochs', type=int, required=False)
my_parser.add_argument('--num_samples_frames', type=int, required=False)
my_parser.add_argument('--epsilon', type=float, required=False)
my_parser.add_argument('--delta', type=float, required=False)
my_parser.add_argument('--ftype', default='i3d', type=str)
my_parser.add_argument('--outft', type=int, default=256)
my_parser.add_argument('--no_unsuper', action='store_true')
my_parser.add_argument('--perdata', type=int, default=100)
my_parser.add_argument('--iter_num', type=int, nargs="*")
my_parser.add_argument('--getOutDir', action='store_true')
my_parser.add_argument('--train_split', type=str, required=False, help="File to be used for unsupervised feature learning")
my_parser.add_argument('--cudad', type=str, required=False)
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



def get_out_dir(args, iter_num):

    args.output_dir = args.output_dir + "/semi_iter{}_split{}_per{}".format(iter_num, args.split_number, args.semi_per)

    if args.wd is not None:
        args.output_dir = args.output_dir + "_wd{:.5f}".format(args.wd)

    if args.chunk_size is not None:
        args.output_dir = args.output_dir + "_chunk{}".format(args.chunk_size)

    if args.max_frames_per_video is not None:
        args.output_dir = args.output_dir + "_maxf{}".format(args.max_frames_per_video)


    if args.delta is not None:
        args.output_dir = args.output_dir + "_dh{:.4f}".format(args.delta)

    if args.weights is not None:
        args.output_dir = args.output_dir + "_wts{}".format(args.weights.replace(',','-'))
        args.weights = list(map(int, args.weights.split(",")))
        print("Weights being used is ", args.weights)

    if args.epsilon is not None:
        args.output_dir = args.output_dir + "_epsilon{:.4f}".format(args.epsilon)

    if args.feature_size != 2048:
        args.output_dir = args.output_dir + "_ft_size{}".format(args.feature_size)

    if args.num_samples_frames is not None:
        args.output_dir = args.output_dir + "_avdiv{}".format(args.num_samples_frames)

    if args.epochs_unsuper:
        args.output_dir = args.output_dir + f"_epu{args.epochs_unsuper}"

    if args.dataset_name == "50salads":
        if args.epochs_unsuper is None:
            if args.semi_per == 0.05:
                args.epochs_unsuper = 100
            else:
                args.epochs_unsuper = 200
        if args.epochs is None:
            args.epochs = 1800
        if args.lr_unsuper is None:
            args.lr_unsuper = 3e-4
        if args.lr_proj is None:
            args.lr_proj = 1e-2
        if args.lr_main is None:
            args.lr_main = 1e-5
        if args.gamma_proj is None:
            args.gamma_proj = 0.1
        if args.gamma_main is None:
            args.gamma_main = 5
        args.steps_proj = [600]
        args.steps_main = [600, 1200]
        if args.chunk_size is None:
            args.chunk_size = 20
        if args.max_frames_per_video is None:
            args.max_frames_per_video = 960
        # if args.lr is None:
        #     args.lr = 3e-4
        if args.wd is None:
            args.wd = 1e-3
        args.batch_size = 50
        args.num_class = 19
        args.back_gd = ['action_start', 'action_end']
        if args.weights is None:
            args.weights = [1., 1., 1., 1., 1., 1.]
        if args.epsilon is None:
            args.epsilon = 0.05
        if args.delta is None:
            args.delta = 0.05
        args.high_level_act_loss = False
        if args.num_samples_frames is None:
            args.num_samples_frames = 70
    elif args.dataset_name == "breakfast":
        if args.epochs_unsuper is None:
            args.epochs_unsuper = 100
        if args.epochs is None:
            args.epochs = 600
        if args.lr_proj is None:
            if args.semi_per == 0.05:
                args.lr_proj = 1e-1
            else:
                args.lr_proj = 1e-2
        if args.lr_main is None:
            if args.semi_per == 0.05:
                args.lr_main = 3e-6
            else:
                args.lr_main = 1e-5
        if args.gamma_proj is None:
            args.gamma_proj = 0.1
        if args.gamma_main is None:
            args.gamma_main = 2

        args.steps_proj = [700]
        args.steps_main = [700]
        if args.lr_unsuper is None:
            args.lr_unsuper = 1e-4
        if args.chunk_size is None:
            args.chunk_size = 10
        if args.max_frames_per_video is None:
            args.max_frames_per_video = 600
        # if args.lr is None:
        #     args.lr = 1e-4
        if args.wd is None:
            args.wd = 3e-3
        args.batch_size = 100
        args.num_class = 48
        args.back_gd = ['SIL']
        if args.weights is None:
            args.weights = [1., 1., 1., 1., 1., 1.]
        if args.epsilon is None:
            args.epsilon = 0.03
        if args.delta is None:
            args.delta = 0.03
        args.high_level_act_loss = True
        if args.num_samples_frames is None:
            args.num_samples_frames = 20
    elif args.dataset_name == "gtea":
        if args.epochs_unsuper is None:
            args.epochs_unsuper = 100
        if args.lr_main is None:
            args.lr_main = 1e-5
        if args.lr_proj is None:
            args.lr_proj = 1e-2
        if args.gamma_main is None:
            args.gamma_main = 5
        if args.gamma_proj is None:
            args.gamma_proj = 0.5
        args.epochs = 2000
        args.steps_proj = [800, 1500]
        args.steps_main = [800, 1500]
        if args.chunk_size is None:
            args.chunk_size = 4
        if args.max_frames_per_video is None:
            args.max_frames_per_video = 600
        if args.lr_unsuper is None:
            args.lr_unsuper = 5e-4
        if args.wd is None:
            args.wd = 3e-4
        args.batch_size = 25
        args.num_class = 11
        args.back_gd = ['background']
        if args.weights is None:
            args.weights = [1., 1., 1., 1., 1., 1.]
        if args.epsilon is None:
            args.epsilon = 0.02
        if args.delta is None:
            args.delta = 0.02
        args.high_level_act_loss = False
        if args.num_samples_frames is None:
            args.num_samples_frames = 20

    step_p_str = "_".join(map(str, args.steps_proj))
    step_m_str = "_".join(map(str, args.steps_main))
    optim_sche_format = f"lrp_{args.lr_proj}_lrm_{args.lr_main}_gp_{args.gamma_proj}_gm_{args.gamma_main}_sp_{step_p_str}_sm_{step_m_str}"
    args.output_dir = args.output_dir + optim_sche_format

    args.output_dir = args.output_dir + "/"
    print("printing in output dir = ", args.output_dir)
    if args.getOutDir:
        import sys
        sys.exit(1)

    args.project_name="{}-split{}".format(args.dataset_name, args.split_number)
    if args.semi_per >= 1:
        args.train_split_file = args.base_dir + "/splits/train.split{}.bundle".format(args.split_number)
    else:
        args.train_split_file = args.base_dir + "/semi_supervised/train.split{}_amt{}.bundle".format(args.split_number, args.semi_per)
    print("train split file name = ", args.train_split_file)
        args.unsupervised_train_split_file = args.base_dir + "/splits/train.split{}.bundle".format(args.train_split)

    args.test_split_file = args.base_dir + "/splits/test.split{}.bundle".format(args.split_number)
    if args.features_file_name is None:
        args.features_file_name = args.base_dir + "/features/"
    args.ground_truth_files_dir = args.base_dir + "/groundTruth/"
    args.label_id_csv = args.base_dir + "mapping.csv"
    args.all_files = args.base_dir + "/splits/all_files.txt"
    args.base_test_split_file = args.base_dir + "/splits/test.split{}.bundle"
    return args

def dump_actual_true_data(args):
    if not os.path.exists(args.pseudo_labels_dir):
        os.mkdir(args.pseudo_labels_dir)
    os.system('rm -rf ' + args.pseudo_labels_dir + "/*txt")
    labeled_data_files = open(args.train_split_file).read().split("\n")[0:-1]
    for ele in labeled_data_files:
        # print('cp ' + args.ground_truth_files_dir + ele + " "  + args.pseudo_labels_dir)
        os.system('cp ' + args.ground_truth_files_dir + ele + " "  + args.pseudo_labels_dir)

    new_files = glob.glob(args.pseudo_labels_dir + "/*.txt")
    print(f"Dumped {len(new_files)} into {args.pseudo_labels_dir} directory")


def get_label_idcsv(args):
    df = pd.read_csv(args.label_id_csv)
    label_id_to_label_name = {}
    label_name_to_label_id_dict = {}
    for i, ele in df.iterrows():
        label_id_to_label_name[ele.label_id] = ele.label_name
        label_name_to_label_id_dict[ele.label_name] = ele.label_id
    return label_id_to_label_name, label_name_to_label_id_dict

def dump_pseudo_labels(video_id, video_value, label_id_to_label_name, args):
    pred_value = video_value[0]
    video_path = os.path.join(args.ground_truth_files_dir, video_id + ".txt")
    with open(video_path, 'r') as f:
        recog_content = f.read().split('\n')[0:-1]  # framelevel recognition is in 6-th line of file
        f.close()

    label_name_arr = [label_id_to_label_name[i.item()] for i in pred_value]
    new_label_name_expanded = [] # np.empty(len(recog_content), dtype=np.object_)
    for i, ele in enumerate(label_name_arr):
        st = i * args.chunk_size
        end = st + args.chunk_size
        if end > len(recog_content):
            end = len(recog_content)
        for j in range(st, end):
            new_label_name_expanded.append(ele)
        if len(new_label_name_expanded) >= len(recog_content):
            break

    out_path = os.path.join(args.pseudo_labels_dir, video_id + ".txt")
    with open(out_path, "w") as fp:
        fp.write("\n".join(new_label_name_expanded))
        fp.write("\n")

def get_unlabbeled_data_and_dump_pseudo_labels(args, model, device):
    label_id_to_label_name, _ = get_label_idcsv(args)
    model.eval()

    all_files_data = open(args.all_files).read().split("\n")[0:-1]
    train_file_dumped = open(args.train_split_file).read().split("\n")[0:-1]
    full_minus_train_dataset = list(set(all_files_data) - set(train_file_dumped))
    unlabeled_dataset = get_data(args, full_minus_train_dataset, train=False, pseudo_data=False)
    unlabeled_dataset_loader = make_loader(unlabeled_dataset, batch_size=args.batch_size, train=False)
    
    results_dict = {}
    
    for i, unlabelled_item in enumerate(unlabeled_dataset_loader):
        unlabelled_data_features = unlabelled_item[0].to(device).permute(0, 2, 1)
        unlabelled_data_count = unlabelled_item[1]
        
        unlabelled_data_output = model(unlabelled_data_features, args.weights)
        unlabelled_output_probabilities = torch.softmax(unlabelled_data_output[1], dim=1)
        argmax_prob_values = torch.argmax(unlabelled_output_probabilities, dim=1).squeeze().detach().cpu().numpy()
        
        for output_prob, vn, count in zip(unlabelled_output_probabilities, unlabelled_item[5], unlabelled_data_count):
            max_prob_out = torch.max(output_prob[:, :count], dim=0)[0].squeeze().detach().cpu().numpy()
            output_pred = torch.argmax(output_prob[:, :count], dim=0).squeeze().detach().cpu().numpy()
            new_min = np.mean(max_prob_out)
            
            if vn in results_dict:
                prev_pred, prev_count, prev_min = results_dict[vn]
                output_pred = np.concatenate([prev_pred, output_pred])
                count = count + prev_count
                new_min = np.mean([prev_min, new_min])
            
            results_dict[vn] = [output_pred, count, new_min]
    
    sort_video_values = sorted(results_dict.items(), key=lambda x: x[1][2], reverse=True)

    videos_labelled = []
    high_level_dict = {}
    # per_high_level_act_budget = config.budget / 10
    videos_added = 0
    for i in sort_video_values:
        videos_added += 1
        dump_pseudo_labels(i[0], i[1], label_id_to_label_name, args)
        videos_labelled.append(i[0] + ".txt")

    new_files = glob.glob(args.pseudo_labels_dir + "/*.txt")
    print(f"Contains {len(new_files)} into {args.pseudo_labels_dir} directory")
    

def model_pipeline():
    origargs = my_parser.parse_args()
    if origargs.iter_num is None:
        origargs.iter_num = [1, 2, 3, 4]
    else:
        origargs.iter_num = origargs.iter_num

    if origargs.dataset_name is None:
        origargs.dataset_name = origargs.base_dir.split("/")[-2]
        print(f"Picked up last directory name to be dataset name {origargs.dataset_name}")

    if not os.path.exists(origargs.output_dir):
        os.mkdir(origargs.output_dir)
        print(f"Created the directory {origargs.output_dir}")

    if origargs.dataset_name == "breakfast":
        origargs.num_class = 48
    elif origargs.dataset_name == "50salads":
        origargs.num_class = 19
    elif origargs.dataset_name == "gtea":
        origargs.num_class = 11

    # Device argsuration
    if origargs.cudad is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = origargs.cudad

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(origargs).to(device)
    model.load_state_dict(torch.load(origargs.model_wt))
    print(f"Loaded model with successfully from path {origargs.model_wt}")

    for iter_n in origargs.iter_num: 
        hyperparameters = copy.deepcopy(origargs)
        set_seed()
        hyperparameters = get_out_dir(hyperparameters, iter_n)
        if not os.path.exists(hyperparameters.output_dir):
            os.mkdir(hyperparameters.output_dir)

        args = hyperparameters

        train_loader, test_loader, criterion, optimizer_group, postprocessor, scheduler_group = make(args, model, device)
        # print(model)
        model = train(device, model, train_loader, criterion, optimizer_group[:2], args, test_loader, postprocessor, scheduler_group, unsupervised=False)
        model.load_state_dict(load_best_model(args))
        acc, avg_acc = test(device, model, test_loader, criterion, postprocessor, args, args.epochs + 1, '', False)

        if iter_n == origargs.iter_num[-1]:
            break

        # Create unsupervised directory and dump current 5% data and rest model evaluation data
        args.pseudo_labels_dir = os.path.join(args.output_dir, 'pseudo_labels_dir') + "/"
        dump_actual_true_data(args)
        get_unlabbeled_data_and_dump_pseudo_labels(args, model, device)

        # Train the unsupervised model 

        all_files_data = open(args.all_files).read().split("\n")[0:-1]

        unsuper_traindataset = get_data(args, all_files_data, train=True, pseudo_data=True)
        unsuper_trainloader = make_loader(unsuper_traindataset, batch_size=args.batch_size, train=True)
        unsuper_testdataset = get_data(args, all_files_data, train=False, pseudo_data=False)
        unsuper_testloader = make_loader(unsuper_testdataset, batch_size=args.batch_size, train=False)

        model = train(device, model, unsuper_trainloader, criterion, [optimizer_group[2]], args, unsuper_testloader, None, None, unsupervised=True)


    return model

def load_best_model(args):
    return torch.load(args.output_dir + '/best_' + args.dataset_name + '_c2ftcn.wt')

def make(args, model, device):
    # Make the data
    all_train_data_files = open(args.train_split_file).read().split("\n")[0:-1]
    if len(all_train_data_files[-1]) <= 1:
        all_train_data_files = all_train_data_files[0:-1]
        # print(all_train_data_files)
    print("Length of files picked up for semi-supervised training is ", len(all_train_data_files))
    validation_data_files = open(args.test_split_file).read().split("\n")[0:-1]
    print("Length of files picked up for semi-supervised validation is ", len(validation_data_files))

    train, test = get_data(args, all_train_data_files, train=True), get_data(args, validation_data_files, train=False)
    train_loader = make_loader(train, batch_size=args.batch_size, train=True)
    test_loader = make_loader(test, batch_size=args.batch_size, train=False)

    # Make the model
    # model = get_model(args).to(device)
    
    # num_params = sum([p.numel() for p in model.parameters()])
    # print("Number of parameters = ", num_params/1e6, " million")

    # Make the loss and optimizer
    criterion = get_criterion(args)

    set_params_for_proj = set(model.outc0.parameters()) | set(model.outc1.parameters()) | \
                        set(model.outc2.parameters()) | set(model.outc3.parameters()) | \
                        set(model.outc4.parameters()) | set(model.outc5.parameters())

    set_params_main_model = set(model.parameters()) - set_params_for_proj

    optimizer_group = [torch.optim.Adam(list(set_params_for_proj), lr=args.lr_proj, weight_decay=args.wd),
                       torch.optim.Adam(list(set_params_main_model), lr=args.lr_main, weight_decay=args.wd),
                       torch.optim.Adam(list(set_params_main_model), lr=args.lr_unsuper, weight_decay=args.wd)]
    scheduler_group = [torch.optim.lr_scheduler.MultiStepLR(optimizer_group[0], milestones=args.steps_proj, gamma=args.gamma_proj),
                        torch.optim.lr_scheduler.MultiStepLR(optimizer_group[1], milestones=args.steps_main, gamma=args.gamma_main)
                        ]
    # scheduler_group = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_group[0], factor=0.1, verbose=True),
    #                     torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_group[1], factor=2, verbose=True)]
    
    # postprocessor declaration
    postprocessor = PostProcess(args)
    postprocessor = postprocessor.to(device)
    
    return train_loader, test_loader, criterion, optimizer_group, postprocessor, scheduler_group


class CriterionClass(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.args = args
        self.mse = nn.MSELoss(reduction='none')
#        self.sigmoid = nn.Sigmoid()
#        self.bce = nn.BCELoss()

    def get_mse_loss(self, count, outp1):

        src_mask = torch.arange(outp1.shape[2], device=outp1.device)[None, :] < count[:, None]
        src_mask = src_mask.to(torch.float32).to(outp1.device).unsqueeze(1)
        
        mse_loss = 0.15 * torch.mean(torch.clamp(self.mse(outp1[:, :, 1:],
                                                          outp1.detach()[:, :, :-1]), 
                                                 min=0, max=16) * src_mask[:, :, 1:])
        return {'mse_loss': mse_loss}
    
    def get_unsupervised_losses(self, count, outp1, activity_labels, actual_labels):
        vid_ids = []
        f1 = []
        t1 = []
        label_info = []
        
        feature_activity = []
        maxpool_features = []
       
        bsize = outp1.shape[0] 
        for j in range(bsize): 
            vidlen = count[j]
            
            sel_frames_current = torch.linspace(0, vidlen, self.args.num_samples_frames,  dtype=int)
            idx = []
            for kkl in range(len(sel_frames_current) - 1):
                cur_start = sel_frames_current[kkl]
                cur_end   = sel_frames_current[kkl + 1]
                list_frames = list(range(cur_start, cur_end + 1))
                idx.append(np.random.choice(list_frames, 1)[0])
            
            idx = torch.tensor(idx).type(torch.long).to(outp1.device)
            idx = torch.clamp(idx, 0, vidlen - 1)


            # Sampling of second set of frames from surroundings epsilon
            vlow = 1   # To prevent value 0 in variable low
            vhigh = int(np.ceil(self.args.epsilon * vidlen.item()))

            if vhigh <= vlow:
                vhigh = vlow + 2
            offset = torch.randint(low=vlow, 
                                   high=vhigh,
                                   size=(len(idx),)).type(torch.long).to(outp1.device)
            previdx = torch.clamp(idx - offset, 0, vidlen - 1)
           
            # Now adding all frames togather 
            f1.append(outp1[j].permute(1,0)[idx, :])
            f1.append(outp1[j].permute(1,0)[previdx, :])
           
            if activity_labels is not None: 
                feature_activity.extend([activity_labels[j]] * len(idx) * 2)
            else:
                feature_activity = None
            
            label_info.append(actual_labels[j][idx])
            label_info.append(actual_labels[j][previdx])
            
            vid_ids.extend([j] * len(idx))
            vid_ids.extend([j] * len(previdx))
            
            idx = idx / vidlen.to(dtype=torch.float32, device=vidlen.device)
            previdx = previdx / vidlen.to(dtype=torch.float32, device=vidlen.device)
            
            t1.extend(idx.detach().cpu().numpy().tolist())
            t1.extend(previdx.detach().cpu().numpy().tolist())
            
            maxpool_features.append(torch.max(outp1[j,:,:vidlen], dim=-1)[0])
           
        # Gathering all features togather  
        vid_ids = torch.tensor(vid_ids).numpy()
        t1 = np.array(t1)
        f1 = torch.cat(f1, dim=0)
        label_info = torch.cat(label_info, dim=0).numpy()
    
        if feature_activity is not None:
            feature_activity = np.array(feature_activity)

        sim_f1 = (f1 @ f1.data.T)
        f11 = torch.exp(sim_f1 / 0.1)

        if feature_activity is None:
            pos_weight_mat = torch.tensor((vid_ids[:, None] == vid_ids[None, :]) & \
                                          (np.abs(t1[:, None] - t1[None, :]) <= self.args.delta) & \
                                          (label_info[:, None] == label_info[None, :]))
            negative_samples_minus = torch.tensor((vid_ids[:, None] == vid_ids[None, :]) & \
                                                  (np.abs(t1[:, None] - t1[None, :]) > self.args.delta) & \
                                                  (label_info[:, None] == label_info[None, :])).type(torch.float32).to(outp1.device)
            pos_weight_mat = pos_weight_mat | torch.tensor((vid_ids[:, None] != vid_ids[None, :]) &\
                                                           (label_info[:, None] == label_info[None, :]))
        else: 
            pos_weight_mat = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                          (np.abs(t1[:, None] - t1[None, :]) <= self.args.delta) & \
                                          (label_info[:, None] == label_info[None, :]))

            negative_samples_minus = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                                  (np.abs(t1[:, None] - t1[None, :]) > self.args.delta) & \
                                                  (label_info[:, None] == label_info[None, :])).type(torch.float32).to(outp1.device)

          
        I = torch.eye(len(pos_weight_mat)).to(outp1.device)
        
        pos_weight_mat = (pos_weight_mat).type(torch.float32).to(outp1.device) - I
        not_same_activity = 1 - pos_weight_mat - I - negative_samples_minus
        countpos = torch.sum(pos_weight_mat)
        if countpos == 0:
            print("Feature level contrast no positive is found")
            feature_contrast_loss = 0
       
        else: 
            featsim_pos = pos_weight_mat * f11
            max_val = torch.max(not_same_activity * f11, dim=1, keepdim=True)[0]
            acc = torch.sum(featsim_pos > max_val) / countpos
            featsim_negsum = torch.sum(not_same_activity * f11, dim=1)
            
            simprob = (featsim_pos / (featsim_negsum + featsim_pos)) + not_same_activity + I + negative_samples_minus
            
            feature_contrast_loss = -torch.sum(torch.log(simprob)) / countpos

        #### Activity level contrastive learning ####
        if self.args.high_level_act_loss is True:
            maxpool_features = torch.stack(maxpool_features)
            maxpool_features = maxpool_features / torch.norm(maxpool_features, dim=1, keepdim=True)
            maxpool_featsim = torch.exp(maxpool_features @ maxpool_features.T / 0.1)
            same_activity = torch.tensor(np.array(activity_labels)[:,None]==np.array(activity_labels)[None,:])
            I = torch.eye(len(same_activity)).to(outp1.device)
            same_activity = (same_activity).type(torch.float32).to(outp1.device) - I
            not_same_activity = 1 - same_activity - I
            countpos = torch.sum(same_activity)
            if countpos == 0:
                print("Video level contrast has no same pairs")
                video_level_contrast = 0
            else:
                maxpool_featsim_pos = same_activity * maxpool_featsim
                maxpool_featsim_negsum = torch.sum(not_same_activity * maxpool_featsim, dim=1)
                simprob = maxpool_featsim_pos/(maxpool_featsim_negsum + maxpool_featsim_pos) + not_same_activity
                video_level_contrast = torch.sum(-torch.log(simprob + I)) / countpos
        else:
            video_level_contrast = 0

        unsupervised_loss = feature_contrast_loss + video_level_contrast
        usupervised_dict_loss = {'contrastive_loss': unsupervised_loss, 'feature_contrast_loss': feature_contrast_loss,
                                 'video_level_contrast': video_level_contrast, 'contrast_feature_acc': acc} 
        return usupervised_dict_loss

    def forward(self, count, projection, prediction, labels, activity_labels, input_f, unsupervised):
        if unsupervised is False:
            ce_loss = self.ce(prediction, labels)

            loss = ce_loss 
            loss_dict = {'ce_loss': ce_loss}
            if self.args.mse:
                mse_loss_dict = self.get_mse_loss(count, prediction)
                loss = loss + mse_loss_dict['mse_loss']
                loss_dict.update(mse_loss_dict)
        else:
            loss = 0
            loss_dict = {}

        unsupervised_loss_dict = self.get_unsupervised_losses(count, projection, activity_labels, labels.detach().cpu())
        loss_dict.update(unsupervised_loss_dict)
        loss = loss + unsupervised_loss_dict['contrastive_loss']


        loss_dict.update({'full_loss': loss})
        return loss_dict

def get_criterion(args):
    return CriterionClass(args)

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


def unsupervised_test(args, model, test_loader, device):
    set_seed()
    dump_dir = args.output_dir + "/features_dump/"
    dump_featres = DumpFeatures(args)
    dump_featres.forward(model, test_loader, device, dump_dir, wts=args.weights)

    acc, all_result = get_linear_acc(args.label_id_csv, dump_dir, args.ground_truth_files_dir, args.perdata, 
                                     args.base_test_split_file, args.chunk_size, False, False)
    # acc = test(model, test_loader, criterion, postprocessor, args, args.epochs, '')
    print_str = f"Best Results:Linear f1@10, f1@25, f1@50, edit, MoF = " + \
          f"{all_result[1]:.1f} & {all_result[2]:.1f} & {all_result[3]:.1f} & {all_result[0]:.1f} & {acc:.1f}\n"
    print(print_str)

    with open(args.output_dir + "/run_summary.txt", "a+") as fp:
        fp.write(print_str)
    return acc, all_result


def train(device, model, loader, criterion, optimizer_group, args, test_loader, postprocessor, scheduler_group, unsupervised):

    if unsupervised == True:
        epochs = args.epochs_unsuper
        print_epochs = 300
        prefix = "unsuper"
    else:
        epochs = args.epochs
        print_epochs = 20
        prefix = ""

    best_acc = 0
    avg_best_acc = 0
    accs = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        for i, item in enumerate(loader):
            samples = item[0].to(device).permute(0, 2, 1)
            count = item[1].to(device)
            labels = item[2].to(device)

            if args.dataset_name == 'breakfast':
               activity_labels = np.array([name.split('_')[-1] for name in item[5]])
            elif args.dataset_name == '50salads':
               activity_labels = None 
            elif args.dataset_name == 'gtea':
               activity_labels = None 

            # Forward pass ➡
            projection_out, pred_out = model(samples, args.weights)
            
            loss_dict = criterion(count, projection_out, pred_out, labels, activity_labels, item[0], unsupervised)
            loss = loss_dict['full_loss']

            # Backward pass ⬅
            for optim in optimizer_group:
                optim.zero_grad()

            loss.backward()

            # Step with optimizer
            for optim in optimizer_group:
                optim.step()

            train_log(loss_dict, epoch)


        if scheduler_group is not None:
            # Step for scheduler
            for sch in scheduler_group:
                sch.step()

        if epoch % print_epochs == 0:

            if unsupervised is True:
               acc, _ = unsupervised_test(args, model, test_loader, device)
            else: 
               acc, avg_acc = test(device, model, test_loader, criterion, postprocessor, args, epoch, prefix, False)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), args.output_dir + "/" + prefix + 'best_' + args.dataset_name + '_c2ftcn.wt')

            accs.append(acc)
        torch.save(model.state_dict(), args.output_dir + "/" + prefix + 'last_' + args.dataset_name + '_c2ftcn.wt')
        accs.sort(reverse=True)
        print(f'Best accuracies till now -> {" ".join(["%.2f"%item for item in accs[:3]])}')
    return model


def train_log(loss_dict, epoch):
    final_dict = {"epoch": epoch}
    final_dict.update(loss_dict)
    print(f"Loss after " + str(epoch).zfill(5) + f" examples: {loss_dict['full_loss']:.3f}")

def test(device, model, test_loader, criterion, postprocessors, args, epoch, dump_prefix, unsupervised):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        avg_loss = []
        avg_total_loss = []
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
            
            loss_dict = criterion(count, projection_out, pred_out, labels, activity_labels, item[0], unsupervised)
            loss = loss_dict['ce_loss']
            avg_loss.append(loss_dict['ce_loss'])
            avg_total_loss.append(loss_dict['full_loss'])
            
            pred = torch.argmax(pred_out, dim=1)
            correct += float(torch.sum((pred == labels) * src_mask).item())
            total += float(torch.sum(src_mask).item())
            postprocessors(pred_out, item[5], labels, count)
            
        # Add postprocessing and check the outcomes
        path = os.path.join(args.output_dir, dump_prefix + "predict_" + args.dataset_name)
        if not os.path.exists(path):
            os.mkdir(path)
        postprocessors.dump_to_directory(path)
        final_edit_score, map_v, overlap_scores = calculate_mof(args.ground_truth_files_dir, path, args.back_gd)
        postprocessors.start()
        acc = 100.0 * correct / total
        print(f"Accuracy of the model on the {total} " +
              f"test images: {acc}%")
        
        final_dict = {"test_accuracy": 100.0 * correct / total}
        final_dict.update({"ce_test_loss": sum(avg_loss) / len(avg_loss)})
        final_dict.update({"total_test_loss": sum(avg_total_loss) / len(avg_total_loss)})
        final_dict.update({"test_actual_acc": map_v})
        final_dict.update({"test_edit_score": final_edit_score})
        final_dict.update({"f1@50": overlap_scores[-1]})
        with open(args.output_dir + "/results_file.txt", "a+") as fp:
            print_string = "Epoch={}: {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}\n".format(epoch, overlap_scores[0], overlap_scores[1], 
                                                overlap_scores[2], final_edit_score, map_v)
            print(print_string)
            fp.write(print_string)

        if epoch == (args.epochs + 1):
            with open(args.output_dir + "/" + dump_prefix + "final_results_file.txt", "a+") as fp:
                fp.write("Epoch={}: {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}\n".format(epoch, overlap_scores[0], overlap_scores[1], 
                                                    overlap_scores[2], final_edit_score, map_v))
                

    # Save the model in the exchangeable ONNX format
#     torch.onnx.export(model, "model.onnx")
    avg_score = (map_v + final_edit_score) / 2
    return map_v, avg_score

start_time = time.time()
model = model_pipeline()

end_time = time.time()
duration = end_time - start_time 

mins = duration / 60 
print("Time taken to complete 4 iteration ", mins, " mins")
