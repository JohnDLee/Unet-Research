Run setup_requirements.sh first to install any python libraries

Run create_augmentations.py using python3 to generate augmented images. They will be located in a folder called final_data

Run optimize.py [num_trials] to use optuna library to optimize model parameters

Run training.py to utilize optuna optimized parameters to train model and retrieve data.

