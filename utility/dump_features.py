import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
import glob
import sys
from collections import defaultdict

class DumpFeatures():
    def __init__(self, args): # fold_file_name, results_json, ignore_label, fps, threshold, chunk_size):

        self.results_dict = dict()
        self.chunk_size = args.chunk_size
        self.args = args
        self.count = 0

    def start(self):
        self.results_dict = dict()
        self.count = 0

    def dump_output_predictions_to_file(self, video_id_dict, output_files_dump_dir):

        for key, features_list in video_id_dict.items():
            features = open(self.args.ground_truth_files_dir + key + '.txt').read().split("\n")[0:-1]
            len_required = len(features)
            out_file_name = output_files_dump_dir + key + '.npy'
            
            new_label_name_expanded = [] # np.empty(len(recog_content), dtype=np.object_)
            for i, ele in enumerate(features_list):
                st = i * self.chunk_size
                end = st + self.chunk_size
                if end > len_required:
                    end = len_required
                for j in range(st, end):
                    new_label_name_expanded.append(ele)
                if len(new_label_name_expanded) >= len_required:
                    break

            while len(new_label_name_expanded) < len_required:
                print("Need to append")
                new_label_name_expanded.append(new_label_name_expanded[-1])

            new_label_name_expanded = np.stack(new_label_name_expanded)
            with open(out_file_name, "wb") as fp:
                np.save(fp, new_label_name_expanded)
    
    def perform_kmeans(self, num_clusters, pred_features):
        model_kmeans = KMeans(n_clusters=num_clusters)
        out_labels_clusters = model_kmeans.fit_predict(pred_features)
        return out_labels_clusters
    
    def forward(self, model, valloader, device, outdir, wts=None):
        output_files_dump_dir = outdir
        if not os.path.exists(output_files_dump_dir):
            os.mkdir(output_files_dump_dir)
        
        all_out_features = None
        all_gd_labels = None
        last_vid_id = None
        last_vid_count = None
        true_f = 0
        total = 0
        avg_acc = []
        all_vid_ids = []
        vid_id_dict = {}
        for i, item in enumerate(valloader):
            vid = item[0].to(device).permute(0, 2, 1)
            count = item[1].cpu().numpy()
            labels = item[2].cpu().numpy()
            video_ids = item[5]
            with torch.no_grad():
                if wts is not None:
                    features_last_list = model(vid, wts)
                else:
                    features_last_list = model(vid)

                features_last = features_last_list[0]

                # features_last = features_last / torch.norm(features_last, dim=1, keepdim=True)
                features = features_last.permute(0, 2, 1).cpu().numpy()

                for one_ele, count_ele, vid_id in zip(features, count, video_ids):
                    val_features = one_ele[:count_ele, :]
                    if vid_id in vid_id_dict:
                        # print("Concatenating")
                        old_features = vid_id_dict[vid_id]
                        vid_id_dict[vid_id] = np.concatenate([old_features, val_features], axis=0)
                    else:
                        vid_id_dict[vid_id] = val_features

        self.dump_output_predictions_to_file(vid_id_dict, output_files_dump_dir)
        return 
