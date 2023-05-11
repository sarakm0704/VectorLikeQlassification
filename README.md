# VectorLikeQlassification
event classification using nn for VLQ example

NN workflow (WIP):

0. setup
```
# in lyoserv start with: 
# wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
# source Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
# (log out from server and log in again)

pip install pandas
pip install deepdish
pip install matplotlib
pip install -U scikit-learn
conda install -c conda-forge root
conda install -c conda-forge tensorflow
cd work
git clone https://github.com/sarakm0704/VectorLikeQlassification.git
cd VectorLikeQlassification
```

1. preprocessing
ntuple will be provided. Convert root file into hdf format
```
python tree2hdf.py -d (for converting files for deep learning)
python tree2hdf.py -m (for merging files from input files (signal + background purpose))
python tree2hdf.py -r (for shuffling indices from input file))
```
in lyoserv you can submit job via slurm batch system, by using ```sbatch sjob.sh```, ```sjob.sh``` would look like:
```
#!/bin/bash

#SBATCH --job-name=m1_VLQlassification
#SBATCH --output=log/std.out
#SBATCH --error=log/std.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --mail-type=ALL
#SBATCH --mail-user={YOUREMAIL}

python3 {YOURCODE}
```
You can check your job status with: ```squeue```. For more detail: [SLURM](https://slurm.schedmd.com/overview.html)

2. training
Train NN : you can even try different network: (shallow) DNN / RNN / CNN
```
python training_dnn.py
```

3. evaluation
Evaluation using different dataset that are not used for training
- Using ```afterTraining.py```<br>
draw input **V**ariables, extract **I**mportance, **C**orrelations, and do **E**valuation after training

```
python afterTraining -v -i -c -e 
```
- ```-v```: Draw Input variable Distributions
- ```-i```: Extract Feature Importance
- ```-c```: Extract Correlations
- ```-e```: do Evaluation of model

4. performance
Compare the performance by drawing ROC curve
```
python drawROC.py
```

5. recurrent feedback for optimization
which structure / input feature / parameter is imporving performance?
