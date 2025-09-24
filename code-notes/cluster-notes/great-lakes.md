---
layout: default
title: Great Lakes Cluster
parent: Cluster Notes
nav_order: 8
---

# Great Lakes Cluster

In order to get access, you have to request HPC login credentials from the IT department of your institution. In [Getting Started](https://its.umich.edu/advanced-research-computing/high-performance-computing/great-lakes/getting-started), we can find the [HPC User login](https://its.umich.edu/advanced-research-computing/high-performance-computing/login). This link allows you to request access to great lakes.

Once you get access, you can login using your uofm uniqname+password combo.

## Starting out

1. Get into UofM Network (via VPN or on-campus).
2. login via ssh

```bash
ssh [uniqname]@greatlakes.arc-ts.umich.edu
```

You'll find yourself in your home directory, which is located at `/home/[uniqname]`.

Keep in mind that you are in the login node. This is a place to view job results and submit new jobs. It cannot be used to run application workloads. 

We will mainly run sbatch and srun to work.

We can check if we can submit jobs by running the following command:

```console
[andang@gl-login1 ~]$ hostname -s
gl-login1
[andang@gl-login1 ~]$ srun --cpu-bind=none hostname -s
srun: job 29842513 queued and waiting for resources
srun: job 29842513 has been allocated resources
gl3051
[andang@gl-login1 ~]$
```

As you can see, srun is a fully blocking command.

## Loading modules

There is a list of modules available to load in slurm. Here are a list of commands we can use to work with modules:

```bash
module list # list all loaded modules
module avail # list all available modules
module load [module_name] # load a module
module unload [module_name] # unload a module
module purge # unload all modules
module unload [module_name] # unload a specific module
module spider # list all possible modules
module whatis [module_name] # show information about a specific module
module save [module_name] # save the current module state
```

Useful modules on Great Lakes include:

```bash
module load python #python3.13
module load mamba  # smaller version of anaconda... also loads python, can't load python and mamba at the same time
module load matlab
module load gurobi
module load julia
module load cmake
module load gcc
module load git
module load tmux
module load ffmpeg

module load cuda
module load cudnn
module load tensorflow
module load tensorrt

module load code-server
```

These are taken from the documentation [on the man page](https://linux.die.net/man/1/module).

## Inside a node

We can mess around with a node by using `salloc` to allocate resources interactively.
```bash
salloc --account=test
```

Once inside, we can access the `/tmp` directory which is unique for each node. We can check that it is specific to the node by running:

```bash
salloc --account=test
touch /tmp/hello.txt
cat /tmp/hello.txt
exit
cat /tmp/hello.txt
```

And we'll notice that once outside, the file is not there anymore. This is because `/tmp` is a temporary directory that is unique to each node and is cleared when the node is rebooted or when the job ends.

This is a good spot to store temporary files that you don't need to keep after the job ends.

**NOTE**: `/home` directory has a hard limit of 80 GB.

## Creating a batch job

```bash
#!/bin/bash # COMMENT: The interpreter used to execute the script
# COMMENT: #SBATCH directives that convey submission options:
#SBATCH --job-name=example_job
#SBATCH --mail-user=[uniqname]@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000m
#SBATCH --time=10:00
#SBATCH --account=test
#SBATCH --partition=standard
#SBATCH --output=/home/%u/%x-%j.log
# COMMENT: The application(s) to execute along with its input arguments and options: <insert commands here>
```

## One Node, One Processor

```bash
#!/bin/bash #SBATCH --job-name JOBNAME

#SBATCH --nodes=1

#SBATCH --cpus-per-task=1

#SBATCH --mem-per-cpu=1g

#SBATCH --time=00:15:00

#SBATCH --account=test

#SBATCH --partition=standard

#SBATCH --mail-type=NONE

# COMMENT:The application(s) to execute along with its input arguments and options:
echo "Hello World"
srun --cpu-bind=none hostname -s
```

This allocates one node with one processor, 1 GB of memory, and a time limit of 15 minutes. We print "Hello World" to the console.

Output:
```console
Hello World
srun: job 29838616 queued and waiting for resources
srun: job 29838616 has been allocated resources
gl3018
```

## One Node, GPU

```bash
#!/bin/bash 

#SBATCH --job-name test_gpu
#SBATCH --account=test
#SBATCH --nodes=1
#SBATCH --time=00:15:00

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 --gpus=1

#SBATCH --mail-type=NONE
#SBATCH --output=/home/%u/workspace/%x-%j.out

# COMMENT: The application(s) to execute along with its input arguments and options:
module load mamba cuda cudnn
python hello.py
nvidia-smi
```

Using this, we can allocate a gpu node and run nvidia-smi to check the gpu status. This all works. 

## Interactive jobs

We can run salloc to do interafctive job. 
```shell-session
[user@gl-login1 ~]$ salloc --account=test
salloc: Pending job allocation 10081688
salloc: job 10081688 queued and waiting for resources
salloc: job 10081688 has been allocated resources
salloc: Granted job allocation 10081688
salloc: Waiting for resource configuration
salloc: Nodes gl3052 are ready for job
[user@gl3052 ~]$ hostname
gl3052.arc-ts.umich.edu
[user@gl3052 ~]$
```

## Going on interactive mode

[](https://documentation.its.umich.edu/node/4994) Check out this link for more information on how to do things interactively on Great Lakes.

## Scratch directory

Scratch directory is maintained by the account you are under (not the user, but the SLURM account which is typically from the PI).

It is located at `/scratch/[account_name]/[subaccount_name]/[uniqname]`.

We can check the usage of the scratch directory by running:

```bash
scratch-quota [scratch_directory]
```

We can also check user usage of `/home` directory by running:

```bash
home-quota
```

## Turbo directory

For great lakes, we can access this through `/nfs/turbo/[turbo_account_name]/`. Use this to store large files for a long time (doesn't automatically scrap like in `/scratch`).

We can also mount turbo directories.

This is how you do it with sshfs:

```bash
sudo apt-get install sshfs
sudo modprobe fuse
mkdir ~/remote_dir
sshfs username@greatlakes.arc-ts.umich.edu:/nfs/turbo/[account_name]/[uniqname]/something ~/remote_dir
```

To unmount you can use:

```bash
fusermount -u ~/remote_dir # hasn't worked for me
sudo umount -l ~/remote_dir # this works consistently
```