import os

os.system("qsub unet_code/create_density.sh create_density_CV.py")
os.system("qsub unet_code/create_density.sh create_density_DID.py")
os.system("qsub unet_code/create_density.sh create_density_STD.py")