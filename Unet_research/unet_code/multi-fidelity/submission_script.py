
import os

if __name__=='__main__':

    root = 'results/DRIVE'
    for model_name in os.listdir(root):
        model_folder = os.path.join(root, model_name)
        
        # checkpoint
        checkpoint = os.path.join(model_folder, 'model_info', os.listdir(os.path.join(model_folder, 'model_info'))[0])
        
        os.system(f"rm -rf {os.path.join(model_folder, 'statistics_fixed')}")
        os.system(f"rm -rf {os.path.join(model_folder, 'statistics')}")

        # check version
        if model_name.startswith('BM'):
            os.system(f"qsub unet_code/base_model_tests/testing_RED.sh {checkpoint} {os.path.join(model_folder, 'statistics')}")
            
        if model_name == 'MF-1':
            os.system(f"qsub unet_code/multi-fidelity/MF_testing_UNI.sh {checkpoint} {os.path.join(model_folder, 'statistics')}")
        if model_name == 'MF-2':
            os.system(f"qsub unet_code/multi-fidelity/MF_testing_RAT.sh {checkpoint} {os.path.join(model_folder, 'statistics')}")

        if model_name == 'LF-1':
            os.system(f"qsub unet_code/multi-fidelity/LF_testing_HFT.sh {checkpoint} {os.path.join(model_folder, 'statistics')} 256")
        elif model_name == 'LF-3':
            os.system(f"qsub unet_code/multi-fidelity/LF_testing_HFT.sh {checkpoint} {os.path.join(model_folder, 'statistics')} 128")
        elif model_name == 'LF-5':
            os.system(f"qsub unet_code/multi-fidelity/LF_testing_HFT.sh {checkpoint} {os.path.join(model_folder, 'statistics')} 256")

        if model_name == 'LF-2':
            os.system(f"qsub unet_code/multi-fidelity/LF_testing_LFT.sh {checkpoint} {os.path.join(model_folder, 'statistics')} 256")
        elif model_name == 'LF-4':
            os.system(f"qsub unet_code/multi-fidelity/LF_testing_LFT.sh {checkpoint} {os.path.join(model_folder, 'statistics')} 128")
        elif model_name == 'LF-6':
            os.system(f"qsub unet_code/multi-fidelity/LF_testing_LFT.sh {checkpoint} {os.path.join(model_folder, 'statistics')} 256")
