import numpy as np
import pandas as pd
import argparse
from sklearn.linear_model import LogisticRegression
import glob
import os
from utils import get_all_scores

def get_gathered_features_labels_videoids(ground_truth_files_dir, dump_dir, label_name_to_label_id_dict, transposed, loadtext):
    # Gathering the features for linear prediction 
    features_arr_concat = []
    labels_arr_concat = []
    video_id_arr = []
    list_filenames = glob.glob(ground_truth_files_dir + "*.txt")
    
    for i, filename in enumerate(list_filenames):
        gd_truth_labels_for_video = open(filename, 'r').read().split("\n")[0: -1]
        gd_truth_labels_for_video = np.array([label_name_to_label_id_dict[ele] for ele in gd_truth_labels_for_video])
        video_id = filename.split("/")[-1].split(".txt")[0]
       
        if loadtext is True: 
            sample_numpy_data_arr = os.path.join(dump_dir, video_id + '.txt')
            numpy_data_arr = np.loadtxt(sample_numpy_data_arr)
        else: 
            sample_numpy_data_arr = os.path.join(dump_dir, video_id + '.npy')
            numpy_data_arr = np.load(sample_numpy_data_arr)

        if transposed is True:
            numpy_data_arr = numpy_data_arr.T

        numpy_data_arr = numpy_data_arr[:len(gd_truth_labels_for_video)]
    
        features_arr_concat.append(numpy_data_arr)
        labels_arr_concat.append(gd_truth_labels_for_video)
        video_id_arr.extend([video_id + ".txt"] * len(gd_truth_labels_for_video))
    
    features_arr_concat = np.concatenate(features_arr_concat, axis=0)
    labels_arr_concat = np.concatenate(labels_arr_concat, axis=0)
    video_id_arr = np.array(video_id_arr)

    return {'features_arr_concat': features_arr_concat, 'labels_arr_concat': labels_arr_concat, 'video_id_arr': video_id_arr}


def get_label_name_to_id_dict(label_id_name_csv):
    df = pd.read_csv(label_id_name_csv)
    label_id_to_label_name = {}
    label_name_to_label_id_dict = {}
    for i, ele in df.iterrows():
        label_id_to_label_name[ele.label_id] = ele.label_name
        label_name_to_label_id_dict[ele.label_name] = ele.label_id
    return {'label_name_to_label_id_dict': label_name_to_label_id_dict, 
            'label_id_to_label_name_dict': label_id_to_label_name}


def perform_linear_ona_split(test_split_file, train_split_file, feat_labels_videoids_dict, percentage_data, gapused, reg_v=1.0):
    # base_test_split_file = '/mnt/data/ar-datasets/dipika/breakfast/ms_tcn/data/breakfast/splits/test.split{}.bundle'
    datasetname = None
    if "50salads" in test_split_file:
        end = 6
        bg_list = ['action_start', 'action_end']
        datasetname = "50salads"
    elif "gtea" in test_split_file:
        end = 5
        bg_list = ['background']
        datasetname = "gtea"
    else:
        end = 5
        bg_list = ['SIL']
        datasetname = "breakfast"


    features_arr_concat = feat_labels_videoids_dict['features_arr_concat']
    labels_arr_concat = feat_labels_videoids_dict['labels_arr_concat']
    video_id_arr = feat_labels_videoids_dict['video_id_arr']

    acc_arr = []
    # base_test_split_file = '/mnt/data/ar-datasets/dipika/breakfast/ms_tcn/data/breakfast/splits/test.split{}.bundle'
        
    test_split_files_names = open(test_split_file).read().split("\n")[0:-1]
    train_split_files_names = open(train_split_file).read().split("\n")[0:-1]
    
    split_train_features = []
    split_train_labels = []
    
    split_val_features = []
    split_val_labels = []
    
    for feat, label, vid in zip(features_arr_concat, labels_arr_concat, video_id_arr):
        if vid in test_split_files_names:
            split_val_features.append(feat)
            split_val_labels.append(label)
        elif vid in train_split_files_names:
            split_train_features.append(feat)
            split_train_labels.append(label)
            
    split_train_features = np.stack(split_train_features, axis=0)[::gapused, :]
    split_train_labels = np.array(split_train_labels)[::gapused]    
    split_val_features = np.stack(split_val_features, axis=0)[::gapused, :]  
    split_val_labels = np.array(split_val_labels)[::gapused] 
    
    ten_per_train = int(percentage_data/100.0 * split_train_features.shape[0])
    split_train_features = split_train_features[:ten_per_train, :]
    split_train_labels = split_train_labels[:ten_per_train]
    
    print("shape of train features ", split_train_features.shape)
    print("shape of val features ", split_val_features.shape)
    print("Shape of train labels ", split_train_labels.shape)
    print("Shape of val labels ", split_val_labels.shape)
    
    reg = LogisticRegression(C=reg_v).fit(split_train_features, split_train_labels[:, None])
    train_score = reg.score(split_train_features, split_train_labels[:, None])
    val_score = reg.score(split_val_features, split_val_labels[:, None])

    preds = reg.predict(split_val_features)
    all_scores = get_all_scores(preds, split_val_labels, bg_list)        
    
    print(f"For split {test_split_file}, train acc = {train_score * 100.0}, val acc = {val_score * 100}.")
    
    return val_score * 100.0, [all_scores[3], all_scores[0], all_scores[1], all_scores[2]]


def perform_linear(feat_labels_videoids_dict, base_test_split_file, percentage_data, gapused, reg_v=1.0, no_split=False):
    features_arr_concat = feat_labels_videoids_dict['features_arr_concat']
    labels_arr_concat = feat_labels_videoids_dict['labels_arr_concat']
    video_id_arr = feat_labels_videoids_dict['video_id_arr']

    acc_arr = []
    edit_scor_arr = []
    f1_10_arr = []
    f1_25_arr = []
    f1_50_arr = []
    # base_test_split_file = '/mnt/data/ar-datasets/dipika/breakfast/ms_tcn/data/breakfast/splits/test.split{}.bundle'
    datasetname = None
    if "50salads" in base_test_split_file:
        end = 6
        bg_list = ['action_start', 'action_end']
        datasetname = "50salads"
    elif "gtea" in base_test_split_file:
        end = 5
        bg_list = ['background']
        datasetname = "gtea"
    else:
        end = 5
        bg_list = ['SIL']
        datasetname = "breakfast"

    if no_split is True:
        
        reg = LogisticRegression(C=reg_v).fit(features_arr_concat, labels_arr_concat)
        train_score = reg.score(features_arr_concat, labels_arr_concat)
        # preds = reg.predict(features_arr_concat)
        # all_scores = get_all_scores(preds, labels_arr_concat, bg_list)       
        return train_score, None 

    actl_score = []
    for split in range(1, end):
        
        test_split_files = open(base_test_split_file.format(split)).read().split("\n")[0:-1]
        
        split_train_features = []
        split_train_labels = []
        
        split_val_features = []
        split_val_labels = []
        # if datasetname == "breakfast":
        #     action_dict_val_feat = {'coffee' : [], 'cereals' : [], 'tea' : [], 'milk' : [], 'juice' : [],
        #                'sandwich' : [], 'scrambledegg' : [], 'friedegg' : [], 'salat' : [], 'pancake' : []}
        #     action_dict_train_feat = {'coffee' : [], 'cereals' : [], 'tea' : [], 'milk' : [], 'juice' : [],
        #                'sandwich' : [], 'scrambledegg' : [], 'friedegg' : [], 'salat' : [], 'pancake' : []}
        #     action_dict_val_label = {'coffee' : [], 'cereals' : [], 'tea' : [], 'milk' : [], 'juice' : [],
        #                'sandwich' : [], 'scrambledegg' : [], 'friedegg' : [], 'salat' : [], 'pancake' : []}
        #     action_dict_train_label = {'coffee' : [], 'cereals' : [], 'tea' : [], 'milk' : [], 'juice' : [],
        #                'sandwich' : [], 'scrambledegg' : [], 'friedegg' : [], 'salat' : [], 'pancake' : []}
        # else:
        action_dict_val_feat = None
        action_dict_train_feat = None
        action_dict_val_label = None
        action_dict_train_label = None
        
        
        for feat, label, vid in zip(features_arr_concat, labels_arr_concat, video_id_arr):
            activity = vid.split("_")[-1].split(".")[0]

            if vid in test_split_files:
                split_val_features.append(feat)
                split_val_labels.append(label)
                if action_dict_val_feat is not None and action_dict_val_label is not None:
                    action_dict_val_feat[activity].append(feat)
                    action_dict_val_label[activity].append(label)
            else:
                split_train_features.append(feat)
                split_train_labels.append(label)
                if action_dict_train_feat is not None and action_dict_train_label is not None:
                    action_dict_train_feat[activity].append(feat)
                    action_dict_train_label[activity].append(label)

        actl_labels = []
        actl_preds = []
        if action_dict_val_feat is not None and action_dict_val_label is not None \
           and action_dict_train_feat is not None and action_dict_train_label is not None:
            for action in action_dict_val_feat.keys():
                tf = np.stack(action_dict_train_feat[action], axis=0)[::gapused, :]
                tl = np.array(action_dict_train_label[action])[::gapused]
                vf = np.stack(action_dict_val_feat[action], axis=0)[::gapused, :]
                vl = np.array(action_dict_val_label[action])[::gapused]
                reg = LogisticRegression(C=reg_v).fit(tf, tl[:, None])
                val_score = reg.score(vf, vl[:, None])
                preds = reg.predict(vf)
                actl_preds.append(preds)
                actl_labels.append(vl)
                print(f"Average score for activity {action} is {val_score * 100}")
                
        if len(actl_labels) > 0 and len(actl_preds) > 0:
            actl_labels = np.concatenate(actl_labels)
            actl_preds = np.concatenate(actl_preds)
            mof = np.sum(actl_preds == actl_labels) / len(actl_preds)
            actl_score.append(mof * 100)

        split_train_features = np.stack(split_train_features, axis=0)[::gapused, :]
        split_train_labels = np.array(split_train_labels)[::gapused]    
        split_val_features = np.stack(split_val_features, axis=0)[::gapused, :]  
        split_val_labels = np.array(split_val_labels)[::gapused] 
        
        ten_per_train = int(percentage_data/100.0 * split_train_features.shape[0])
        split_train_features = split_train_features[:ten_per_train, :]
        split_train_labels = split_train_labels[:ten_per_train]
        
        print("shape of train features ", split_train_features.shape)
        print("shape of val features ", split_val_features.shape)
        print("Shape of train labels ", split_train_labels.shape)
        print("Shape of val labels ", split_val_labels.shape)
        
        reg = LogisticRegression(C=reg_v).fit(split_train_features, split_train_labels[:, None])
        train_score = reg.score(split_train_features, split_train_labels[:, None])
        val_score = reg.score(split_val_features, split_val_labels[:, None])
        acc_arr.append(val_score * 100)

        preds = reg.predict(split_val_features)
        all_scores = get_all_scores(preds, split_val_labels, bg_list)        
        f1_10_arr.append(all_scores[0])
        f1_25_arr.append(all_scores[1])
        f1_50_arr.append(all_scores[2])
        edit_scor_arr.append(all_scores[3])
        
        print(f"For split {split}, train acc = {train_score * 100.0}, val acc = {val_score * 100}.")
    
    print("All splits linear accuracy = ", np.mean(np.array(acc_arr)))
    print("All splits all activity level average= ", np.mean(np.array(actl_score)))
    return np.mean(np.array(acc_arr)), [np.mean(np.array(edit_scor_arr)), np.mean(np.array(f1_10_arr)), np.mean(np.array(f1_25_arr)), 
            np.mean(np.array(f1_50_arr)), acc_arr, edit_scor_arr, f1_10_arr, f1_25_arr, f1_50_arr]


def get_linear_acc(label_id_name_csv, feature_dump_dir, ground_truth_dir, percentage_data, base_test_split_file, 
                   gapused, transposed=False, loadtxt=False, no_splits=False):
    labels_name_id_dict = get_label_name_to_id_dict(label_id_name_csv)

    feat_labels_videoids_dict = get_gathered_features_labels_videoids(ground_truth_dir, feature_dump_dir, 
                                                                      labels_name_id_dict['label_name_to_label_id_dict'], transposed, loadtxt) 

    final_result, result_arr = perform_linear(feat_labels_videoids_dict, base_test_split_file, percentage_data, gapused, 1, no_splits)
    return final_result, result_arr
    

def get_linear_acc_on_split(test_file, train_file, label_id_name_csv, feature_dump_dir, ground_truth_dir, percentage_data, 
                            gapused, transposed=False, loadtxt=False):
    labels_name_id_dict = get_label_name_to_id_dict(label_id_name_csv)

    feat_labels_videoids_dict = get_gathered_features_labels_videoids(ground_truth_dir, feature_dump_dir, 
                                                                      labels_name_id_dict['label_name_to_label_id_dict'], transposed, loadtxt) 

    final_result, all_scores  = perform_linear_ona_split(test_file, train_file, feat_labels_videoids_dict,
                                                         percentage_data, gapused)
    return final_result, all_scores

def parse_arguments():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--split', type=int, required=True)
    my_parser.add_argument('--model_path', type=str, default='unet_model.new_enemble_big_model_wo_dac')
    my_parser.add_argument('--cudad', type=str)
    my_parser.add_argument('--noaug', action='store_true')
    my_parser.add_argument('--nolps', action='store_true')
    my_parser.add_argument('--base_dir', type=str, default="/mnt/data/ar-datasets/dipika/breakfast/ms_tcn/data/breakfast/")
    my_parser.add_argument('--dataset_name', type=str, default="breakfast")
    my_parser.add_argument('--wd', type=float, required=False)
    my_parser.add_argument('--lr', type=float, required=False)
    my_parser.add_argument('--chunk_size', type=int, required=False)
    my_parser.add_argument('--max_frames', type=int, required=False)
    my_parser.add_argument('--weights', type=str, required=False)
    my_parser.add_argument('--ft_file', type=str, required=False)
    my_parser.add_argument('--ft_size', type=int, required=False)
    arguments = my_parser.parse_args()
    return arguments

if __name__ == '__main__':
    arguments = parse_arguments()
    main(*arguments)


