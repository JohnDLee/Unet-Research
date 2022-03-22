

import os

if __name__=='__main__':

    root = 'results/DRIVE'
    models = 'BM-1 BM-2 BM-3 LF-1 LF-2 LF-3 LF-4 LF-5 LF-6 MF-1 MF-2'.split()
    for model_name in models:
        model_folder = os.path.join(root, model_name)
        
        
        os.system(f"rm -rf {model_folder}")

    # check version
    os.system(f"qsub unet_code/base_model_tests/training_RED.sh DRIVE/BM-1 1")
    os.system(f"qsub unet_code/base_model_tests/training_RED.sh DRIVE/BM-2 {4/14}")
    os.system(f"qsub unet_code/base_model_tests/training_RED.sh DRIVE/BM-3 {2/14}")
        
    os.system(f"qsub unet_code/multi-fidelity/MF_training_UNI.sh DRIVE/MF-1")
    os.system(f"qsub unet_code/multi-fidelity/MF_training_RAT.sh DRIVE/MF-2")
    os.system(f"qsub unet_code/multi-fidelity/MF_training_RSZ_RAT.sh DRIVE/MF-3")
    
    os.system(f"qsub unet_code/multi-fidelity/LF_training_HFT.sh DRIVE/LF-1 256 1")
    os.system(f"qsub unet_code/multi-fidelity/LF_training_HFT.sh DRIVE/LF-3 128 1")
    os.system(f"qsub unet_code/multi-fidelity/LF_training_HFT.sh DRIVE/LF-5 256 {4/14}")
    
    os.system(f"qsub unet_code/multi-fidelity/LF_training_LFT.sh DRIVE/LF-2 256 1")
    os.system(f"qsub unet_code/multi-fidelity/LF_training_LFT.sh DRIVE/LF-4 128 1")
    os.system(f"qsub unet_code/multi-fidelity/LF_training_LFT.sh DRIVE/LF-6 256 {4/14}")