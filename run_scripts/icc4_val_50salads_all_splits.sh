# python runcodes/iterative_semi_supervised_eval.py --semi_per <semi_supervised_percentage> --base_dir <data_directory>/<datasetname>/ --output_dir <dir_to_dump_checkpoints_logs>  --iter_num <iteration_number_of_ICC> --cuda <cuda_device_number>
python runcodes/iterative_semi_supervised_eval.py --semi_per 0.1 --output_dir ../mstcn_data/50salads/results/ICC_SS_TAS/ --base_dir ../mstcn_data/50salads/ --iter_num 4 --cuda 3