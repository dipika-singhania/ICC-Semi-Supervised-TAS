# ICC-Semi-Supervised-TAS
[Iterative Contrast-Classify For Semi-supervised Temporal Action Segmentation](https://arxiv.org/abs/2112.01402) accepted to be presented in AAAI-2022.

### Data download and directory structure:

The I3D features, ground-truth and test split files are similar used to [MSTCN++](https://github.com/yabufarha/ms-tcn). 
In the mstcn_data, download additional files, checkpoints and semi-supervised splits can be downloaded from [drive](https://drive.google.com/drive/folders/1ArYPctLZZKfjicEf5nl4LJrY9xxFc6wU?usp=sharing) . 
Specifically, this drive link contains all necessary data in required directory structure except breakfast I3D feature files which can be downloaded from MSTCN++ data directory.

The data directory is arranged in following structure

- mstcn_data
   - mapping.csv
   - dataset_name
   - groundTruth
   - splits
   - semi_supervised
   - results
        - ICC_SS_TAS
            - unsupervised_checkpoint_files
            - ICC_checkpoints_files

### Run Scripts
The various scripts to run the unsupervised training, evaluation or ICC training and evaluation is provided in run_scripts folder with 50salads dataset as example.
Change the dataset_name,  to run on a different dataset.

#### Train Unsupervised Contrastive Representation Learning
    ##### python runcodes/unsupervised_traineval.py --base_dir mstcn_data/<dataset_name>/ --output_dir <output_directory_to_dump_modelcheckpoint_logs>
    Example:
    python runcodes/unsupervised_traineval.py --output_dir mstcn_data/50salads/results/ICC_SS_TAS/ --base_dir mstcn_data/50salads/


#### Evaluate Unsupervised Contrastive Representation Learning
    ##### python runcodes/unsupervised_traineval.py --base_dir mstcn_data/<dataset_name>/ --output_dir <output_directory_to_dump_modelcheckpoint_logs> --eval
    Example:
    python runcodes/unsupervised_traineval.py --output_dir mstcn_data/50salads/results/ICC_SS_TAS/ --base_dir mstcn_data/50salads/ --eval

#### Train ICC for 4 iterations for particular split
    ##### python runcodes/iterative_semi_supervised.py --split_number <split_number> --semi_per <semi_supervised_percentage> --base_dir mstcn_data/<datasetname>/ --output_dir <output_directory_to_dump_model_checkpoints_logs> --model_wt <unsupervised_representation_modelwt> --cuda <gpu_number>
    python runcodes/iterative_semi_supervised.py --split_number 1 --semi_per 0.05 --output_dir mstcn_data/50salads/results/ICC_SS_TAS/ --base_dir mstcn_data/50salads/ --model_wt mstcn_data/50salads/results/ICC_SS_TAS/unsupervised_C2FTCN_splitfull/best_50salads_c2f_tcn.wt --cuda 3

#### Evaluate 4th iteration results of ICC algorithm for all splits
    ###### python runcodes/iterative_semi_supervised_eval.py --semi_per <semi_supervised_percentage> --base_dir mstcn_data/<datasetname>/ --output_dir <dir_to_dump_checkpoints_logs>  --iter_num <iteration_number_of_ICC> --cuda <cuda_device_number>
    python runcodes/iterative_semi_supervised_eval.py --semi_per 0.1 --output_dir mstcn_data/50salads/results/ICC_SS_TAS/ --base_dir mstcn_data/50salads/ --iter_num 4 --cuda 3


### Citation:

If you use the code, please cite

    D Singhania, R Rahaman, A Yao
    Coarse to fine multi-resolution temporal convolutional network.
    arXiv preprint 

    D Singhania, R Rahaman, A Yao
    Iterative Frame-Level Representation Learning And Classification For Semi-Supervised Temporal Action Segmentation
    arXiv preprint 
