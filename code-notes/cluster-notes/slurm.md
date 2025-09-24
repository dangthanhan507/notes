---
layout: default
title: SLURM
parent: Cluster Notes
nav_order: 8
---

# Some 

## Some commands

```bash
sbatch [sbatch_script].sh
squeue -u [uniqname] # List all jobs of the user submitted
scancel [job_id] # Cancel a job

scontrol show job -dd [job_id] # detailed squeue
scontrol show nodes
scontrol show [node]
scontrol hold [job_id] # Hold a job
scontrol release [job_id] # Release a job
scontrol write batch_script [job_id] # view job batch script

sinfo # cluster status
salloc [args] # start interactive jobsa
salloc [args] --x11 # X11 forwarding

sacct -j [job_num] --format jobid,jobname,NTasks,nodelist,CPUTime,RegMem,Elapsed # monitor or review a job's resource usage\
sacctmgr show assoc user=$USER # view job batch script
sacctmgr show assoc account=[account] # view users with access to account
sacctmgr show assoc user [uniqname] # view default submission account and wckey

my_account_usage -A [account]
my_job_statistics -j [job_id]
```

## Partition options

Partition options:
- Debug: run jobs quickly for debugging
    - max jobs per user: 1
    - max walltime: 4 hours
    - max processor per job: 8
    - max memory per job: 40gb
    - higher scheduling priority
- standard:
    - max walltime: 14 days
    - default partition
- standard-oc: get additional software you can only use on-campus
    - max walltime: 14 days
- gpu: Allows use of Tesla V100 GPUs
    - max walltime: 14 days
- spgpu: Allows use of A40 GPUs
    - max walltime: 14 days
- largemem: allows use of compute node with 1.5TB of RAM
    - max walltime: 14 days


## Batch job: Options for requesting resources

| Option | SLURM Command (#SBATCH) | Example |
| ------ | ----------------------- | ------- |
| Job name | `--job-name=<name>` | `--job-name=asdf` |
| Account | `--account=<account>` | `--account=test` This account is from the provider account  |
| Queue | `--partition=<queue>` | `--partition=standard` We can choose "standard", "gpu" (GPU jobs only), largemem (large memory), viz, debug, standard-oc (oc=on-campus software) |
| Wall time limit | `--time=<dd-hh:mm:ss>` | `--time=10:00` (10 minutes) If you want if you got the write wall time, you can try on debug partition |
| Node count | `--nodes=<count>` | `--nodes=3` |
| Process count per node | `--ntasks-per-node=<count>` | `--ntasks-per-node=1` invoke ntasks on each node |
| Core count (per process) | `--cpus-per-task=<cores>` | `--cpus-per-task=1` Without specifying, it defaults to 1 processor per task |
| Memory limit | `--mem=<limit>` (Memory per node in GiB, MiB, KiB) | `--mem=12000m` (12 GiB roughly) |
| Minimum memory per processor | `--mem-per-cpu=<memory>` | `--mem-per-cpu=1000m` (1 GiB per CPU) |
| Request GPUs | `--gres=gpu:<count>` `--gpus=[type:]<number>` | `--gres=gpu:2` `gpus=2` |
| Process count per GPU | `--ntasks-per-gpu=<count>` Must be used with `--ntasks` or `--gres=gpu:` | `--ntasks-per-gpu=1` `--gres=gpu:4` 2 tasks per GPU times 4 GPUS = 8 tasks total |

## Batch job: Environment variables and logs

| Option | SLURM Command (#SBATCH) | Example |
| ------ | ----------------------- | ------- |
| Copy environment | `--export=ALL` | `--export=ALL` This copies all environment variables to the job |
| Set Env variable | `--export=<variable=value,var2=val2>` | `--export=EDITOR=/bin/vim` |
| Standard output file | `--output=<file path>` (path must exist) | `--output=/home/%u/%x-%j.log` NOTE: %u is the user, %x is the job name, %j is the job ID |
| Standard error file | `--error=<file path>` (path must exist) | `--error=/home/%u/%x-%j.err` |


## Batch job: job control

| Option | SLURM Command (#SBATCH) | Example |
| ------ | ----------------------- | ------- |
| Job array | `--array=<array indices>` | `--array=0-15` |
| Job dependency | `--dependency=after:jobID[:jobID...]` | `--dependency=after:1234[:1233]` |
| Email address | `--mail-user=<email>` | `--mail-user=uniqname@umich.edu` |
| Defer job til time | `--begin=<date/time>` | `--begin=2020-12-25T12:30:00` |