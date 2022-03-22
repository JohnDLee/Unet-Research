
import os

if __name__=='__main__':

    root = 'results/DRIVE'
    for model_name in os.listdir(root):
        model_folder = os.path.join(root, model_name)
        
        os.system(f"rm -rf {os.path.join(model_folder, 'rotation_uncertainty')}")
        os.system(f"rm -rf {os.path.join(model_folder, 'dropblock_uncertainty')}")

        # checkpoint
        checkpoint = os.path.join(model_folder, 'model_info', os.listdir(os.path.join(model_folder, 'model_info'))[0])
        resize = -1
        if model_name in 'LF-2 LF-6'.split():
            resize = 256
        elif model_name == 'LF-4':
            resize = 128
        # submit a dropblock
        os.system(f"qsub unet_code/uncertainty_tests/dropblock_i.sh {checkpoint} {os.path.join(model_folder, 'dropblock_uncertainty')} {resize}")
        # submit a rotation
        os.system(f"qsub unet_code/uncertainty_tests/rotation.sh {checkpoint} {os.path.join(model_folder, 'rotation_uncertainty')} {resize}")