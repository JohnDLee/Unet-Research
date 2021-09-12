import os


metrics = 'metrics'
for test_type in os.listdir(metrics):

    num_tests = len(os.listdir(os.path.join(metrics, test_type)))
    bash_file = ["#!bin/csh\n",
                 "\n",
                 "#$ -q gpu\n",
                 "#$ -l gpu_card=1\n",
                 f"#$ -N batch_run_{test_type}\n",
                 f"#$ -t {num_tests}\n",
                 "\n",
                 "module load python\n",
                 "setenv OMP_NUM_THREADS 1\n"
                 "\n"
                 f"python training.py metrics/{test_type}/test$SGE_TASK_ID/ params.ini\n"
                ]
    
    with open(f'batch_run.sh', 'w') as brsh:
        brsh.writelines(bash_file)
    
    os.system("qsub batch_run.sh")

os.system("rm batch_run.sh")