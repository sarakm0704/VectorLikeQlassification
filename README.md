# VectorLikeQlassification
event classification using nn for VLQ example

NN workflow (WIP):

0. setup
```
(in your laptop)
cd work
git clone https://github.com/sarakm0704/VectorLikeQlassification.git
cd VectorLikeQlassification
mkdir ev
python -m venv ev
source ev/bin/activate
(then you are in virtualenv. Maybe you can even put it as alias)
pip install numpy
conda install -c conda-forge root
```

```
(in lyoserv)
source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev4/latest/x86_64-centos7-gcc11-opt/setup.sh
cd work
git clone https://github.com/sarakm0704/VectorLikeQlassification.git
cd VectorLikeQlassification
mkdir ev
python -m venv ev
source ev/bin/activate
pip install deepdish
```

1. preprocessing
ntuple will be provided. Convert root file into hdf format
```
python tree2hdf.py -d (for converting files for deep learning)
python tree2hdf.py -m (for merging files from input files (signal + background purpose))
python tree2hdf.py -r (for shuffling indices from input file))
```

2. training
Train NN : you can even try different network: (shallow) DNN / RNN / CNN
```
python training_dnn.py
```
or
```
python training_cnn.py
```

3. evaluation
Evaluation using different dataset that are not used for training
- Using ```ecid_afterTraining.py```<br>
do **E**valuation and extract **C**orrelations, **I**mportance, **D**istributions of features after training

```
python ecid_afterTraining -e -i -c -d
```
- ```-e```: do Evaluation of model
- ```-i```: Extract Feature Importance
- ```-c```: Extract Correlations
- ```-d```: Draw Input variable Distributions

4. performance
Compare the performance by drawing ROC curve
```
python drawROC.py
```

5. recurrent feedback for optimization
which structure / input feature / parameter is imporving performance?
