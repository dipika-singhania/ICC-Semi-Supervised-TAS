# python runcodes/iterative_semi_supervised_eval.py --base_dir <data_dir>/<datasetname>/ --model_wt <checkpoint> --split_number <test_split_number> --cuda <gpu_number>
python runcodes/iterative_semi_supervised_eval.py --base_dir ../mstcn_data/50salads/ --model_wt ../mstcn_data/50salads/results/ICC_SS_TAS/semi_iter4semi_split1_per0.05/best_50salads_c2ftcn.wt --split_number 1 --cuda 3
