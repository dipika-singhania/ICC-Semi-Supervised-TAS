# ICC-Semi-Supervised-TAS
[Iterative Contrast-Classify For Semi-supervised Temporal Action Segmentation](https://arxiv.org/abs/2112.01402) accepted to be presented in AAAI-2022.
Code for semi supervsion version of ‘C2F-TCN: A Framework for Semi- and Fully-Supervised Temporal Action Segmentation’ [link](https://ieeexplore.ieee.org/abstract/document/10147035) published in TPAMI-2023.

### Data download and directory structure:

The I3D features, ground-truth and test split files are similar used to [MSTCN++](https://github.com/yabufarha/ms-tcn). 
In the mstcn_data, download additional files, checkpoints and semi-supervised splits can be downloaded from [drive](https://drive.google.com/drive/folders/1ArYPctLZZKfjicEf5nl4LJrY9xxFc6wU?usp=sharing) . It also contains the checkpoints files for unsupervised pretraining, semi-supervised checkpoints.
Specifically, this drive link contains all necessary data in required directory structure except breakfast I3D feature files which can be downloaded from MSTCN++ data directory.

The data directory is arranged in following structure

- mstcn_data
   - mapping.csv
   - dataset_name
   - groundTruth
   - splits
   - semi_supervised (It contains semi-supervised, 5%, 10% or 40% selection of samples.)
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
    Example:
    python runcodes/iterative_semi_supervised.py --split_number 1 --semi_per 0.05 --output_dir mstcn_data/50salads/results/ICC_SS_TAS/ --base_dir mstcn_data/50salads/ --model_wt mstcn_data/50salads/results/ICC_SS_TAS/unsupervised_C2FTCN_splitfull/best_50salads_c2f_tcn.wt --cuda 3

#### Evaluate results of ICC algorithm for all splits
    ###### python runcodes/iterative_semi_supervised_eval.py --semi_per <semi_supervised_percentage> --base_dir mstcn_data/<datasetname>/ --output_dir <dir_to_dump_checkpoints_logs> --cuda <cuda_device_number>
    Example:
    python runcodes/iterative_semi_supervised_eval.py --semi_per 0.1 --output_dir mstcn_data/50salads/results/ICC_SS_TAS/ --base_dir mstcn_data/50salads/ --cuda 3

We have also provided semi-supervised selection of training data and for 50saladsa and GTEA all unsupervised and semi-supervised all iterations and splits checkpoints for 5% and 10% data of [drive](https://drive.google.com/drive/folders/1ArYPctLZZKfjicEf5nl4LJrY9xxFc6wU?usp=sharing). These checkpoints correspond to 1 single selection of semi-supervised training set of 5%, 10%. However, results reported in the paper is mean of 5 different selections of semi-supervised training set. So these checkpints result though closely match with reported results, will not match in exact number. For most scores it will give higher or equivalent results.

#### Command to evaluate the final checkpints of 50salads provided checkpoints inside ICC_SS_TAS folder in mstcn_data

    python runcodes/iterative_semi_supervised_eval.py --semi_per 0.05 --output_dir mstcn_data/50salads/results/ICC_SS_TAS/ --base_dir mstcn_data/50salads/ --cuda 6
    python runcodes/iterative_semi_supervised_eval.py --semi_per 0.1 --output_dir mstcn_data/50salads/results/ICC_SS_TAS/ --base_dir mstcn_data/50salads/ --cuda 6
    
    
#### Command to evaluate the final checkpints of GTEA provided checkpoints inside ICC_SS_TAS folder in mstcn_data

    python runcodes/iterative_semi_supervised_eval.py --semi_per 0.05 --output_dir mstcn_data/gtea/results/ICC_SS_TAS/ --base_dir mstcn_data/gtea/ --cuda 6
    python runcodes/iterative_semi_supervised_eval.py --semi_per 0.1 --output_dir mstcn_data/gtea/results/ICC_SS_TAS/ --base_dir mstcn_data/gtea/ --cuda 6

### Citation:

If you use the code, please cite

    D Singhania, R Rahaman, A Yao
    Coarse to fine multi-resolution temporal convolutional network.
    arXiv preprint 

    @inproceedings{singhania2022iterative,
     title={Iterative Contrast-Classify For Semi-supervised Temporal Action Segmentation},
     author={Singhania, Dipika and Rahaman, Rahul and Yao, Angela},
     booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
     volume={36},
     number={2},
     pages={2262--2270},
     year={2022}
   }

   @ARTICLE{10147035,
     author={Singhania, Dipika and Rahaman, Rahul and Yao, Angela},
     journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
     title={C2F-TCN: A Framework for Semi- and Fully-Supervised Temporal Action Segmentation}, 
     year={2023},
     pages={1-18},
     doi={10.1109/TPAMI.2023.3284080}}
