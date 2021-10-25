Run setup_requirements.sh first to install any python libraries

Run create_augmentations.py using python3 to generate augmented images. They will be located in a folder called final_data

Pipeline 1

Run optimize.py [num_trials] to use optuna library to optimize model parameters
Run training.py to utilize optuna optimized parameters to train model and retrieve data.


Pipeline 2
Run generate_ini.py to generate an ini file and fill in parameters
Run training.py and pass the ini file to train model with user params

