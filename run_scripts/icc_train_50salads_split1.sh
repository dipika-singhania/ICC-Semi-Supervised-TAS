# python runcodes/iterative_semi_supervised.py --split_number <split_number> --semi_per <semi_supervised_percentage> --base_dir <data_directory>/<datasetname>/ --output_dir <output_directory_to_dump_model_checkpoints_logs> --model_wt <unsupervised_representation_modelwt> --cuda <gpu_number>
python runcodes/iterative_semi_supervised.py --split_number 1 --semi_per 0.05 --output_dir ../mstcn_data/50salads/results/ICC_SS_TAS/ --base_dir ../mstcn_data/50salads/ --model_wt ../mstcn_data/50salads/results/ICC_SS_TAS/ulC2FTCN_splitfull_notimeFalse/best_50salads_c2f_tcn.wt --cuda 3
