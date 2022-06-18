import importlib
import os, sys
from sklearn.cluster import KMeans
from utility.perform_linear import get_linear_acc, get_linear_acc_on_split
from utility.dump_features import DumpFeatures
import warnings
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

seed = 42

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--output_dir', type=str, required=True, help="Output directory where the programs outputs (checkpoints, logs etc.) must be stored")
my_parser.add_argument('--base_dir', type=str, help="Base directory where dataset's all files like features, split, ground truth is present")
my_parser.add_argument('--wd', type=float, required=False)
my_parser.add_argument('--lr', type=float, required=False)
my_parser.add_argument('--chunk_size', type=int, required=False)
my_parser.add_argument('--max_frames_per_video', type=int, required=False)
my_parser.add_argument('--weights', type=str, required=False)
my_parser.add_argument('--features_file_name', type=str, required=False)
my_parser.add_argument('--feature_size', type=int, required=False)
my_parser.add_argument('--clustCount', type=int, required=False)
my_parser.add_argument('--epsilon', type=float, required=False)
my_parser.add_argument('--tau', default=0.1, type=float)
my_parser.add_argument('--epochs', type=int, default=100, required=False)
my_parser.add_argument('--num_samples_frames', type=int, required=False)
my_parser.add_argument('--delta', type=float, required=False)
my_parser.add_argument('--outft', type=int, default=256)
my_parser.add_argument('--nohigh', action='store_true')
my_parser.add_argument('--perdata', type=int, default=100, help="Linear evaluation with amount of percentage data")
my_parser.add_argument('--getOutDir', action="store_true", help="Run program only to get where model checkpoint is stored")
my_parser.add_argument('--eval_only', action="store_true", help="Run program only to get the linear evaluation scores")
my_parser.add_argument('--no_time', action="store_true", help="Run program with no time-proximity condition")
my_parser.add_argument('--val_split', type=int, required=False, help="By default it learns on all splits and evaluates on all splits")
my_parser.add_argument('--train_split', type=str, required=False, help="By default it is all splits except val_split")
my_parser.add_argument('--cudad', type=str, required=False, help="Specify the cuda number in string which the program needs to be run on")
my_parser.add_argument('--dataset_name', type=str, required=False, help="If last directory name of base is not dataset name, then specify dataset name 50salads, breakfast or gtea")
args = my_parser.parse_args()


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

if args.dataset_name is None:
    args.dataset_name = args.base_dir.split("/")[-2]
    print(f"Picked up last directory name to be dataset name {args.dataset_name}")


# Device argsuration
if args.cudad is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cudad

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if args.val_split is None:
    args.val_split = "full"

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
    print(f"Created the directory {args.output_dir}")
args.output_dir = args.output_dir + "/unsupervised_{}_split{}".format("C2FTCN", args.val_split)

if args.train_split is not None:
    args.output_dir = args.output_dir + f"_ts{args.train_split}"
else:
    args.train_split = f"{args.val_split}"

if args.wd is not None:
    args.output_dir = args.output_dir + "_wd{:.5f}".format(args.wd)

if args.lr is not None:
    args.output_dir = args.output_dir + "_lr{:.6f}".format(args.lr)

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

if args.feature_size is not None:
    args.output_dir = args.output_dir + "_ft_size{}".format(args.feature_size)
else:
    args.feature_size = 2048

if args.num_samples_frames is not None:
    args.output_dir = args.output_dir + "_avdiv{}".format(args.num_samples_frames)

if args.clustCount is not None:
    args.output_dir = args.output_dir + "_clusC{}".format(args.clustCount)

if args.dataset_name == "50salads":
    args.epsilon_l = 0.002
    if args.epsilon is None:
        args.epsilon = 0.05
    if args.clustCount is None:
        args.clustCount = 40
    if args.chunk_size is None:
        args.chunk_size = 20
    if args.max_frames_per_video is None:
        args.max_frames_per_video = 960
    if args.lr is None:
        args.lr = 1e-3
    if args.wd is None:
        args.wd = 1e-3
    args.batch_size = 50
    args.num_class = 19
    args.back_gd = ['action_start', 'action_end']
    if args.weights is None:
        args.weights = [1, 1, 1, 1, 1, 1]
    if args.delta is None:
        args.delta = 0.5
    args.high_level_act_loss = False
    if args.num_samples_frames is None:
        args.num_samples_frames = 80
elif args.dataset_name == "breakfast":
    args.epsilon_l = 0.005
    if args.epsilon is None:
        args.epsilon = 0.03
    if args.clustCount is None:
        args.clustCount = 100
    if args.chunk_size is None:
        args.chunk_size = 10
    if args.max_frames_per_video is None:
        args.max_frames_per_video = 600
    if args.lr is None:
        args.lr = 1e-4
    if args.wd is None:
        args.wd = 3e-3
    args.batch_size = 50
    args.num_class = 48
    args.back_gd = ['SIL']
    if args.weights is None:
        args.weights = [1, 1, 1, 1, 1, 1]
    if args.delta is None:
        args.delta = 0.03
    if args.nohigh:
        args.high_level_act_loss = False
        args.output_dir = args.output_dir + "_noactloss_"
    else:
        args.high_level_act_loss = True
    if args.num_samples_frames is None:
        args.num_samples_frames = 20
elif args.dataset_name == "gtea":
    args.epsilon_l = 0.005
    if args.epsilon is None:
        args.epsilon = 0.02
    if args.clustCount is None:
        args.clustCount = 30
    if args.chunk_size is None:
        args.chunk_size = 4
    if args.max_frames_per_video is None:
        args.max_frames_per_video = 600
    if args.lr is None:
        args.lr = 1e-3
    if args.wd is None:
        args.wd = 3e-4
    args.batch_size = 25
    args.num_class = 11
    args.back_gd = ['background']
    if args.weights is None:
        args.weights = [1, 1, 1, 1, 1, 1]
    if args.delta is None:
        args.delta = 0.02
    args.high_level_act_loss = False
    if args.num_samples_frames is None:
        args.num_samples_frames = 20

args.output_dir = args.output_dir + "/"
print("printing in output dir = ", args.output_dir)
if args.getOutDir is True:
    sys.exit(1)

args.project_name="{}-split{}".format(args.dataset_name, args.val_split)
if args.val_split != "full":
    args.train_split_file = args.base_dir + "splits/train.split{}.bundle".format(args.train_split)
    print("Picking the training file from ", args.train_split_file)
else:
    args.train_split_file = args.base_dir + "splits/all_files.txt"
    print("Picking the training file from ", args.train_split_file)

args.base_test_split_file = args.base_dir + "splits/test.split{}.bundle"
args.test_split_file = args.base_dir + "splits/all_files.txt"
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

    if not args.eval_only:
        # and use them to train the model
        train(model, train_loader, criterion, optimizer, args, test_loader, postprocessor)

    # and test its final performance
    model.load_state_dict(load_best_model(args))

    set_seed()
    dump_dir = args.output_dir + "/features_dump/"
    dump_featres = DumpFeatures(args)
    dump_featres.forward(model, test_loader, device, dump_dir, wts=args.weights)

    if args.val_split != "full":
        val_split_file = args.base_test_split_file.format(args.val_split)
        acc, all_result = get_linear_acc_on_split(val_split_file, args.train_split_file, args.label_id_csv, dump_dir, args.ground_truth_files_dir, 
                                                  args.perdata, args.chunk_size, False, False)
    else:
        acc, all_result = get_linear_acc(args.label_id_csv, dump_dir, args.ground_truth_files_dir, args.perdata, 
                                         args.base_test_split_file, args.chunk_size, False, False)
    print_str = f"Best Results:Linear f1@10, f1@25, f1@50, edit, MoF = " + \
          f"{all_result[1]:.1f} & {all_result[2]:.1f} & {all_result[3]:.1f} & {all_result[0]:.1f} & {acc:.1f}\n"
    print(print_str)

    with open(args.output_dir + "/run_summary.txt", "a+") as fp:
        fp.write(print_str)
    print(f'final_test_acc_avg:{acc:.2f}')

    return model

def load_best_model(args):
    return torch.load(args.output_dir + '/best_' + args.dataset_name + '_c2f_tcn.wt')

def make(args):
    # Make the data
    all_train_data_files = open(args.train_split_file).read().split("\n")[0:-1]
    validation_data_files = open(args.test_split_file).read().split("\n")[0:-1]
    train, test = get_data(args, all_train_data_files, train=True), get_data(args, validation_data_files, train=False)
    train_loader = make_loader(args, train, batch_size=args.batch_size, train=True)
    test_loader = make_loader(args, test, batch_size=args.batch_size, train=False)

    # Make the model
    model = get_model(args).to(device)
    
    num_params = sum([p.numel() for p in model.parameters()])
    print("Number of parameters = ", num_params/1e6, " million")

    # Make the loss and optimizer
    criterion = get_criterion(args)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    # postprocessor declaration
    postprocessor = PostProcess(args)
    postprocessor = postprocessor.to(device)
    
    return model, train_loader, test_loader, criterion, optimizer, postprocessor


class CriterionClass(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss()
        self.args = args
    
    def get_unsupervised_losses(self, count, outp1, activity_labels, input_f):
        vid_ids = []
        f1 = []
        t1 = []
        i3d_f = []
        
        feature_activity = []
        maxpool_features = []
       
        bsize = outp1.shape[0] 
        for j in range(bsize): 

            # Sampling of first K frames
            vidlen = count[j]
            sel_frames_current = torch.linspace(0, vidlen, args.num_samples_frames,  dtype=int)
            idx = []
            for kkl in range(len(sel_frames_current) - 1):
                cur_start = sel_frames_current[kkl]
                cur_end   = sel_frames_current[kkl + 1]
                list_frames = list(range(cur_start, cur_end + 1))
                idx.append(np.random.choice(list_frames, 1)[0])
            
            idx = torch.tensor(idx).type(torch.long).to(device)
            idx = torch.clamp(idx, 0, vidlen - 1)

            # Sampling of second set of frames from surroundings epsilon
            # vlow = 1   # To prevent value 0 in variable low
            vlow = int(np.ceil(args.epsilon_l * vidlen.item()))
            vhigh = int(np.ceil(args.epsilon * vidlen.item()))

            if vhigh <= vlow:
                vhigh = vlow + 2
            offset = torch.randint(low=vlow, 
                                   high=vhigh,
                                   size=(len(idx),)).type(torch.long).to(device)
            previdx = torch.clamp(idx - offset, 0, vidlen - 1)
           
            # Now adding all frames togather 
            f1.append(outp1[j].permute(1,0)[idx, :])
            f1.append(outp1[j].permute(1,0)[previdx, :])
           
            if activity_labels is not None: 
                feature_activity.extend([activity_labels[j]] * len(idx) * 2)
            else:
                feature_activity = None
            
            i3d_f.append(input_f[j][idx, :])
            i3d_f.append(input_f[j][previdx, :])
            
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
        i3d_f = torch.cat(i3d_f, dim=0)
    
        # Getting label_info from semnatic information
        clust = KMeans(n_clusters=args.clustCount)
        label_info = clust.fit_predict(i3d_f.numpy())

        if feature_activity is not None:
            feature_activity = np.array(feature_activity)

        sim_f1 = (f1 @ f1.data.T)
        f11 = torch.exp(sim_f1 / args.tau)

        if feature_activity is None:  # For 50salads and GTEA where there is no high level activity defined
            if self.args.no_time:
                pos_weight_mat = torch.tensor((label_info[:, None] == label_info[None, :]))
                negative_samples_minus = 0
            else:
                pos_weight_mat = torch.tensor((vid_ids[:, None] == vid_ids[None, :]) & \
                                              (np.abs(t1[:, None] - t1[None, :]) <= args.delta) & \
                                               (label_info[:, None] == label_info[None, :]))
                negative_samples_minus = torch.tensor((vid_ids[:, None] == vid_ids[None, :]) & \
                                                      (np.abs(t1[:, None] - t1[None, :]) > args.delta) & \
                                                      (label_info[:, None] == label_info[None, :])).type(torch.float32).to(device)
                pos_weight_mat = pos_weight_mat | torch.tensor((vid_ids[:, None] != vid_ids[None, :]) &\
                                                               (label_info[:, None] == label_info[None, :]))
        else:                    # For datasets like Breakfast where high level activity is known
            if self.args.no_time:
                pos_weight_mat = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                              (label_info[:, None] == label_info[None, :]))
                negative_samples_minus = torch.tensor((feature_activity[:, None] != feature_activity[None, :]) & \
                                                        (label_info[:, None] == label_info[None, :])).type(torch.float32).to(device)
            else:
                pos_weight_mat = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                              (np.abs(t1[:, None] - t1[None, :]) <= args.delta) & \
                                              (label_info[:, None] == label_info[None, :]))
                negative_samples_minus = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                                      (np.abs(t1[:, None] - t1[None, :]) > args.delta) & \
                                                      (label_info[:, None] == label_info[None, :])).type(torch.float32).to(device)

          
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
        unsupervised_loss_dict = self.get_unsupervised_losses(count, projection, activity_labels, input_f)
        loss = unsupervised_loss_dict['contrastive_loss']
        loss_dict = {'full_loss':loss}
        loss_dict.update(unsupervised_loss_dict)
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


def make_loader(args, dataset, batch_size, train=True):
    def _init_fn(worker_id):
        np.random.seed(int(seed))
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=train,
                                         pin_memory=False, num_workers=0, collate_fn=collate_fn_override,
                                         worker_init_fn=_init_fn)
    return loader


def get_model(args):
    set_seed()
    return C2F_TCN(n_channels=args.feature_size, n_classes=args.num_class, n_features=args.outft)


def train(model, loader, criterion, optimizer, args, test_loader, postprocessor):
    # wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * args.epochs
    best_acc = 0
    avg_best_acc = 0
    accs = []
    
    for epoch in range(0, args.epochs + 1):
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

            train_log(loss_dict, epoch)

        if epoch % 5 == 0:

            set_seed()
            dump_dir = args.output_dir + "/features_dump/"
            dump_featres = DumpFeatures(args)
            dump_featres.forward(model, test_loader, device, dump_dir, wts=args.weights)

            if args.val_split != "full":
                val_split_file = args.base_test_split_file.format(args.val_split)
                acc, all_result = get_linear_acc_on_split(val_split_file, args.train_split_file, args.label_id_csv, dump_dir, 
                                                          args.ground_truth_files_dir, args.perdata, args.chunk_size,
                                                           False, False)
            else:
                acc, all_result = get_linear_acc(args.label_id_csv, dump_dir, args.ground_truth_files_dir, args.perdata, 
                                                 args.base_test_split_file, args.chunk_size, False, False)

            if acc >= best_acc:
                best_acc = acc
                torch.save(model.state_dict(), args.output_dir + '/best_' + args.dataset_name + '_c2f_tcn.wt')
                torch.save(model.state_dict(), args.output_dir + f'/{epoch}ep__' + args.dataset_name + '_c2f_tcn.wt')

            print_str = f"Epoch{epoch}:Linear f1@10, f1@25, f1@50, edit, MoF = " + \
                  f"{all_result[1]:.1f} & {all_result[2]:.1f} & {all_result[3]:.1f} & {all_result[0]:.1f} & {acc:.1f}, best={best_acc:.1f}\n"
            print(print_str)

            with open(args.output_dir + "/run_summary.txt", "a+") as fp:
                fp.write(print_str)

            accs.append(acc)
        torch.save(model.state_dict(), args.output_dir + '/last_' + args.dataset_name + '_c2f_tcn.wt')
        accs.sort(reverse=True)
        # scheduler.step()
        # wandb.log({'avgbest_test_acc': avg_best_acc}, epoch)
        print(f'Best accuracies till now -> {" ".join(["%.2f"%item for item in accs[:3]])}')


def train_log(loss_dict, epoch):
    final_dict = {"epoch": epoch}
    final_dict.update(loss_dict)
    print(f"Loss after " + str(epoch).zfill(5) + f" examples: {loss_dict['full_loss']:.3f}")

model = model_pipeline(args)
