import importlib
import os, sys
from sklearn.cluster import KMeans
from utility.perform_linear import get_linear_acc
from utility.dump_features import DumpFeatures
import warnings
warnings.filterwarnings('ignore')
import copy
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
my_parser.add_argument('--base_dir', type=str, required=True)
my_parser.add_argument('--semi_per', type=float, required=False)
my_parser.add_argument('--output_dir', type=str, required=False)
my_parser.add_argument('--iter_num', type=int, required=False)
my_parser.add_argument('--split_number', type=int, required=False)
my_parser.add_argument('--model_wt', type=str, required=False)
my_parser.add_argument('--dataset_name', type=str)
my_parser.add_argument('--wd', type=float, required=False)
my_parser.add_argument('--lr_unsuper', type=float, required=False)
my_parser.add_argument('--lr_proj', type=float, required=False)
my_parser.add_argument('--lr_main', type=float, required=False)
my_parser.add_argument('--gamma_proj', type=float, required=False)
my_parser.add_argument('--gamma_main', type=float, required=False)
my_parser.add_argument('--epochs_unsuper', type=int, required=False)
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
my_parser.add_argument('--feature_size', type=int, required=False)
my_parser.add_argument('--epochs', type=int, required=False)
my_parser.add_argument('--num_samples_frames', type=int, required=False)
my_parser.add_argument('--epsilon', type=float, required=False)
my_parser.add_argument('--delta', type=float, required=False)
my_parser.add_argument('--ftype', default='i3d', type=str)
my_parser.add_argument('--outft', type=int, default=256)
my_parser.add_argument('--no_unsuper', action='store_true')
my_parser.add_argument('--perdata', type=int, default=100)
my_parser.add_argument('--cudad', type=str)
args = my_parser.parse_args()

if args.dataset_name is None:
    args.dataset_name = args.base_dir.split("/")[-2]
    print(f"Picked up last directory name to be dataset name {args.dataset_name}")

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
if args.cudad is not None:
    os.environ['CUDA_VISIBLE_DEVICES']=args.cudad
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.model_wt:
    if not args.split_number:
        print("Input --split_number <split> to specify which test split you want to run the model wt on")
        sys.exit(1)
    if not args.output_dir:
        args.output_dir = os.path.dirname(args.model_wt)
else:
    if not args.iter_num:
        args.iter_num = 4
    if not args.semi_per:
        print("Input --semi_per to specify of which semi supervised percentage model need to be picked up")
        sys.exit(1)

def load_best_model(args):
    print("Loading model ", args.output_dir + '/best_' + args.dataset_name + '_c2ftcn.wt')
    return torch.load(args.output_dir + '/best_' + args.dataset_name + '_c2ftcn.wt')

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
            elif args.dataset_name == "gtea":
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
        return  overlap_scores, final_edit_score, map_v


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

for split in all_splits:
    origargs = copy.deepcopy(args)
    if not args.model_wt:
        if args.iter_num:
            args.output_dir = args.output_dir + "/icc{}_semi{}_split{}".format(args.iter_num, args.semi_per, split)

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
            args.output_dir = args.output_dir + "_dl{:.4f}".format(args.epsilon)

        if args.feature_size is not None:
            args.output_dir = args.output_dir + "_ft_size{}".format(args.feature_size)

        if args.num_samples_frames is not None:
            args.output_dir = args.output_dir + "_avdiv{}".format(args.num_samples_frames)

        if args.epochs_unsuper:
            args.output_dir = args.output_dir + f"_epu{args.epochs_unsuper}"
    else:
        print(f"Output directory is {args.output_dir}")
        print(f"Model checkpoint to be picked is {args.model_wt}")

    if args.feature_size is None:
        args.feature_size = 2048

    if args.dataset_name == "50salads":
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
            args.delta = 0.5
        args.high_level_act_loss = False
        if args.num_samples_frames is None:
            args.num_samples_frames = 80
    elif args.dataset_name == "breakfast":
        if args.lr_proj is None:
            args.lr_proj = 1e-1
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
        if args.wd is None:
            args.wd = 3e-3
        args.batch_size = 50
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

    # step_p_str = "_".join(map(str, args.steps_proj))
    # step_m_str = "_".join(map(str, args.steps_main))
    # optim_sche_format = f"lrp_{args.lr_proj}_lrm_{args.lr_main}_gp_{args.gamma_proj}_gm_{args.gamma_main}_sp_{step_p_str}_sm_{step_m_str}"
    # args.output_dir = args.output_dir + optim_sche_format

    args.output_dir = args.output_dir + "/"
    print("printing in output dir = ", args.output_dir)

    args.project_name = "{}-split{}".format(args.dataset_name, split)

    args.test_split_file = args.base_dir + "/splits/test.split{}.bundle".format(split)
    if args.features_file_name is None:
        args.features_file_name = args.base_dir + "/features/"
    args.ground_truth_files_dir = args.base_dir + "/groundTruth/"
    args.label_id_csv = args.base_dir + "mapping.csv"
    args.all_files = args.base_dir + "/splits/all_files.txt"
    args.base_test_split_file = args.base_dir + "/splits/test.split{}.bundle"

    # Actual code for doing the validation
    model, test_loader, postprocessor = make(args)
 
    if args.model_wt: 
        model.load_state_dict(torch.load(args.model_wt))
    else:
        model.load_state_dict(load_best_model(args))

    over, edit, mof = test(model, test_loader,  postprocessor, args)

    # Variable store the multiple splits values in array
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



