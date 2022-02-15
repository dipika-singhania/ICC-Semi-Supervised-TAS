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
my_parser.add_argument('--split_number', type=int, required=True)
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


if args.split_number is None:
    args.split_number = "full"

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
        args.avg_actions = 30
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

args.project_name="{}-split{}".format(args.dataset_name, args.split_number)
if args.semi_per >= 1:
    args.train_split_file = args.base_dir + "/splits/train.split{}.bundle".format(args.split_number)
else:
    args.train_split_file = args.base_dir + "/semi_supervised/train.split{}_amt{}.bundle".format(args.split_number, args.semi_per)

args.test_split_file = args.base_dir + "/splits/test.split{}.bundle".format(args.split_number)
if args.features_file_name is None:
    args.features_file_name = args.base_dir + "/features/"
args.ground_truth_files_dir = args.base_dir + "/groundTruth/"
args.label_id_csv = args.base_dir + "mapping.csv"


def model_pipeline(hyperparameters):
    if not os.path.exists(hyperparameters.output_dir):
        os.mkdir(hyperparameters.output_dir)
    args = hyperparameters
    model, train_loader, test_loader, criterion, optimizer, postprocessor = make(args)
    # print(model)

    # and use them to train the model
    train(model, train_loader, criterion, optimizer, args, test_loader, postprocessor)

    # and test its final performance
    # model.load_state_dict(load_avgbest_model(args))
    # acc = test(model, test_loader, criterion, postprocessor, args, args.epochs, 'avg')
    model.load_state_dict(load_best_model(args))
    acc, avg_acc = test(model, test_loader, criterion, postprocessor, args, args.epochs + 1, '')

    return model

def load_best_model(args):
    return torch.load(args.output_dir + '/best_' + args.dataset_name + '_unet.wt')

def load_avgbest_model(args):
    return torch.load(args.output_dir + '/avgbest_' + args.dataset_name + '_unet.wt')

def make(args):
    # Make the data
    all_train_data_files = open(args.train_split_file).read().split("\n")[0:-1]
    validation_data_files = open(args.test_split_file).read().split("\n")[0:-1]
    train, test = get_data(args, all_train_data_files, train=True), get_data(args, validation_data_files, train=False)
    train_loader = make_loader(train, batch_size=args.batch_size, train=True)
    test_loader = make_loader(test, batch_size=args.batch_size, train=False)

    # Make the model
    model = get_model(args).to(device)
    
    num_params = sum([p.numel() for p in model.parameters()])
    print("Number of parameters = ", num_params/1e6, " million")

    # Make the loss and optimizer
    criterion = get_criterion(args)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # postprocessor declaration
    postprocessor = PostProcess(args)
    postprocessor = postprocessor.to(device)
    
    return model, train_loader, test_loader, criterion, optimizer, postprocessor


class CriterionClass(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
#        self.mse = nn.MSELoss(reduction='none')
#        self.sigmoid = nn.Sigmoid()
#        self.bce = nn.BCELoss()
    
    def get_unsupervised_losses(self, count, outp1, activity_labels, actual_labels):
        vid_ids = []
        f1 = []
        t1 = []
        label_info = []
        
        feature_activity = []
        maxpool_features = []
       
        bsize = outp1.shape[0] 
        for j in range(bsize): 
            num_clusters = args.avg_actions
            vidlen = count[j]
            
            sel_frames_current = torch.linspace(0, vidlen, num_clusters,  dtype=int)
            idx = []
            for kkl in range(len(sel_frames_current) - 1):
                cur_start = sel_frames_current[kkl]
                cur_end   = sel_frames_current[kkl + 1]
                list_frames = list(range(cur_start, cur_end + 1))
                idx.append(np.random.choice(list_frames, 1)[0])
            
            idx = torch.tensor(idx).type(torch.long).to(device)
            idx = torch.clamp(idx, 0, vidlen - 1)

            offset = torch.randint(low=int(args.data_per_low * vidlen.item()), 
                                   high=int(np.ceil(args.data_per_high * vidlen.item())),
                                   size=(len(idx),)).type(torch.long).to(device)
            previdx = torch.clamp(idx - offset, 0, vidlen - 1)
            
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
                                          (np.abs(t1[:, None] - t1[None, :]) <= args.data_per_high) & \
                                          (label_info[:, None] == label_info[None, :]))
            negative_samples_minus = torch.tensor((vid_ids[:, None] == vid_ids[None, :]) & \
                                                  (np.abs(t1[:, None] - t1[None, :]) > args.data_per_high) & \
                                                  (label_info[:, None] == label_info[None, :])).type(torch.float32).to(device)
            pos_weight_mat = pos_weight_mat | torch.tensor((vid_ids[:, None] != vid_ids[None, :]) &\
                                                           (label_info[:, None] == label_info[None, :]))
        else: 
            pos_weight_mat = torch.tensor((vid_ids[:, None] == vid_ids[None, :]) & \
                                           (np.abs(t1[:, None] - t1[None, :]) <= args.data_per_high) & \
                                           (label_info[:, None] == label_info[None, :]))
            negative_samples_minus = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                                  (np.abs(t1[:, None] - t1[None, :]) > args.data_per_high) & \
                                                  (label_info[:, None] == label_info[None, :]))
            pos_weight_mat = pos_weight_mat | torch.tensor((feature_activity[:, None] == feature_activity[None, :]) &\
                                                           (vid_ids[:, None] != vid_ids[None, :]) &\
                                                           (label_info[:, None] == label_info[None, :]))
            negative_samples_minus = negative_samples_minus | torch.tensor((feature_activity[:, None] != feature_activity[None, :]) &\
                                                                            (vid_ids[:, None] != vid_ids[None, :]) &\
                                                                            (label_info[:, None] == label_info[None, :]))
            negative_samples_minus = negative_samples_minus.type(torch.float32).to(device)
          
        I = torch.eye(len(pos_weight_mat)).to(device)
        
        pos_weight_mat = (pos_weight_mat).type(torch.float32).to(device) - I
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
        if args.high_level_act_loss is True:
            maxpool_features = torch.stack(maxpool_features)
            maxpool_features = maxpool_features / torch.norm(maxpool_features, dim=1, keepdim=True)
            maxpool_featsim = torch.exp(maxpool_features @ maxpool_features.T / 0.1)
            same_activity = torch.tensor(np.array(activity_labels)[:,None]==np.array(activity_labels)[None,:])
            I = torch.eye(len(same_activity)).to(device)
            same_activity = (same_activity).type(torch.float32).to(device) - I
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

    def forward(self, count, projection, prediction, labels, activity_labels, input_f):
        ce_loss = self.ce(prediction, labels)

        loss = ce_loss 
        loss_dict = {'ce_loss': ce_loss}
        if args.unsuper is True:
            unsupervised_loss_dict = self.get_unsupervised_losses(count, projection, activity_labels, labels.detach().cpu())
            loss_dict.update(unsupervised_loss_dict)
            loss = loss + unsupervised_loss_dict['contrastive_loss']
        loss_dict.update({'full_loss':loss})
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


def train(model, loader, criterion, optimizer, args, test_loader, postprocessor):
    total_batches = len(loader) * args.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    best_acc = 0
    avg_best_acc = 0
    accs = []
    
    for epoch in range(1, args.epochs + 1):
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
            
            loss_dict = criterion(count, projection_out, pred_out, labels, activity_labels, item[0])
            loss = loss_dict['full_loss']

            # Backward pass ⬅
            optimizer.zero_grad()
            loss.backward()

            # Step with optimizer
            optimizer.step()

            example_ct +=  len(item[2])
            batch_ct += 1

            train_log(loss_dict, example_ct, epoch)

        if epoch % 5 == 0:

            acc, avg_acc = test(model, test_loader, criterion, postprocessor, args, epoch, '')
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), args.output_dir + '/best_' + args.dataset_name + '_unet.wt')

            accs.append(acc)
        torch.save(model.state_dict(), args.output_dir + '/last_' + args.dataset_name + '_unet.wt')
        accs.sort(reverse=True)
        print(f'Best accuracies till now -> {" ".join(["%.2f"%item for item in accs[:3]])}')


def train_log(loss_dict, example_ct, epoch):
    final_dict = {"epoch": epoch}
    final_dict.update(loss_dict)
    print(f"Loss after " + str(epoch).zfill(5) + f" examples: {loss_dict['full_loss']:.3f}")

def test(model, test_loader, criterion, postprocessors, args, epoch, dump_prefix):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        avg_loss = []
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
            elif args.dataset_name == "gtea":
               activity_labels = None

            # Forward pass ➡
            projection_out, pred_out = model(samples, args.weights)
            
            loss_dict = criterion(count, projection_out, pred_out, labels, activity_labels, item[0])
            loss = loss_dict['full_loss']
            avg_loss.append(loss)
            
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
        acc =100.0 * correct / total
        print(f"Accuracy of the model on the {total} " +
              f"test images: {acc}%")
        
        final_dict = {"test_accuracy": 100.0 * correct / total}
        final_dict.update({"test_loss": sum(avg_loss) / len(avg_loss)})
        final_dict.update({"test_actual_acc": map_v})
        final_dict.update({"test_edit_score": final_edit_score})
        final_dict.update({"f1@50": overlap_scores[-1]})
        with open(args.output_dir + "/results_file.txt", "a+") as fp:
            print_string = "Epoch={}: {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}\n".format(epoch, overlap_scores[0], overlap_scores[1], 
                                                overlap_scores[2], final_edit_score, map_v)
            print(print_string)
            fp.write(print_string)

        if epoch == args.epochs:
            with open(args.output_dir + "/" + dump_prefix + "final_results_file.txt", "a+") as fp:
                fp.write("Epoch={}: {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}\n".format(epoch, overlap_scores[0], overlap_scores[1], 
                                                    overlap_scores[2], final_edit_score, map_v))
                

    avg_score = (map_v + final_edit_score) / 2
    return map_v, avg_score

print(args)
model = model_pipeline(args)
