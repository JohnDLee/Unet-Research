1. Install Python Dependancies in requirements.sh

2. Navigate to the directory ../Unet_research/

Note: () means the parameter is a directory/file and can be changed
3. Generate augmentations: 

python3 unet_code/base_model_tests/create_augmentations.py -seed 1234 \
                                    -tn 500 \
                                    -tv 20 \
                                    -dest (augmented_data)


4. Generate a base model from unet_code/base_model_tests/training.py like so:

mkdir results # needed to store results

python unet_code/base_model_tests/training.py \
                -mode train \
                -data_path (augmented_data) \
                -save_path (results/base_model) \
                -num_epochs 50 \
                -train_batch 1 \
                -val_batch 1 \
                -momentum .99 \
                -lr .001 \
                -block_size 7 \
                -max_drop_prob .15 \
                -dropblock_steps 1500 \
                -seed 1234 \
                --gpus 1 \
                --gradient_clip_val .5 \
                --check_val_every_n_epoch 1 \
                --profiler simple \
                --detect_anomaly

# gpus can be changed in number or removed altogethor (not recommended)
# lr will be automatically tuned
# dropblock is automatically scheduled

5. If tests would be run on the base model, use test mode:

python unet_code/base_model_tests/training.py \
                -mode test \
                -model_path (results/base_model/model_info/(your_checkpoint)) \
                -data_path (augmented_data) \
                -save_path (results/base_model/test_statistics) \
                -seed 1234 \
                --gpus 1 \
                --gradient_clip_val .5 \
                --check_val_every_n_epoch 1 \
                --profiler simple \
                --detect_anomaly

# checkpoint names may be different

6. Run Rotational Uncertainty:

python unet_code/uncertainty_tests/Rotational_Uncertainty.py \
                -model_path (results/base_model/model_info/your_model_ckpt) \
                -data_path (augmented_data) \
                -save_path (results/base_model/rotational_uncertainty) \
                -save_num 25 \
                -seed 1234 \
                --gpus 1 \
                --profiler simple \
                --detect_anomaly

7. Run Dropblock Uncertainty:

Dependent Dropblock:

python unet_code/uncertainty_tests/Dropblock_Uncertainty.py \
                -model_path (results/base_model/model_info/your_model_ckpt) \
                -data_path (augmented_data) \
                -save_path (results/base_model/dropblock_uncertainty_d) \
                -block_size 7 \
                -drop_prob .15 \
                -iter_num 1000 \
                -save_num 25 \
                -seed 1234 \
                --gpus 1 \
                --profiler simple \
                --detect_anomaly

Independent DropBlock:

python unet_code/uncertainty_tests/Dropblock_Uncertainty.py \
                -model_path (results/base_model/model_info/your_model_ckpt) \
                -data_path (augmented_data) \
                -save_path (results/base_model/dropblock_uncertainty_i) \
                -block_size 7 \
                -drop_prob .15 \
                -independent_drop \
                -iter_num 1000 \
                -save_num 25 \
                -seed 1234 \
                --gpus 1 \
                --profiler simple \
                --detect_anomaly

8. Evaluate using Evaluate_Uncertainty.ipynb. There are interactable plots.





