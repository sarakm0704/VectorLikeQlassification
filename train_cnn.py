from __future__ import division
import sys, os, re, shutil

import array
import numpy as np
import pandas as pd

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, LSTM, Concatenate
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv1D

from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

from time import time, localtime, strftime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--e', required=False, default=1000, help='epochs')

args = parser.parse_args()
epochs = int(args.e) #Epoch
njets = 5

def split_train_test():
    pd_out = data.filter(items = ['signal'])
    pd_input = data.filter(items = event_var+jet_var)
    np_out = np.array( pd_out )
    np_input = np.array( pd_input )
    # split
    test_size = 0.3
    train_input, valid_input, train_out, valid_out = train_test_split( np_input, np_out, test_size=test_size )

    # train set
    pd_train_input = pd.DataFrame( train_input, columns= event_var+jet_var )
    pd_train_out = pd.DataFrame( train_out, columns=['signal'] )
    # event info
    train_event_input = pd_train_input.filter( items=event_var )
    train_event_input = np.array( train_event_input )
    train_event_out = pd_train_out.filter( items=['signal'] )
    # jet info
    train_jet_input = pd_train_input.filter( items=jet_var )
    train_jet_input = np.array( train_jet_input )
    train_jet_input = train_jet_input.reshape( train_jet_input.shape[0], njets, -1 )
    
    # valid set
    pd_valid_input = pd.DataFrame( valid_input,columns= event_var+jet_var )
    pd_valid_out = pd.DataFrame( valid_out, columns=['signal'] )
    # event info
    valid_event_input = pd_valid_input.filter( items=event_var )
    valid_event_input = np.array( valid_event_input )
    valid_event_out = pd_valid_out.filter( items=['signal'] )
    # jet info
    valid_jet_input = pd_valid_input.filter( items=jet_var )
    valid_jet_input = np.array( valid_jet_input )
    valid_jet_input = valid_jet_input.reshape( valid_jet_input.shape[0], njets, -1 )

    return train_event_input, train_event_out, train_jet_input, valid_event_input, valid_event_out, valid_jet_input


def build_model(model_path):
    Inputs = [ Input( shape=(train_event_input.shape[1],) ), Input( shape=(train_jet_input.shape[1], train_jet_input.shape[2]), ) ]
    
    dropout = 0.1
    nodes = 100
    # BatchNormalization
    event_info = BatchNormalization( name = 'event_input_batchnorm' )(Inputs[0])
    jets = BatchNormalization( name = 'jet_input_batchnorm' )(Inputs[1])
    
    # Dense for event
    event_info = Dense(nodes, activation='relu', name='event_layer1')(event_info)
    event_info = Dropout(dropout)(event_info)
    event_info = Dense(nodes, activation='relu', name='event_layer2')(event_info)
    event_info = Dropout(dropout)(event_info)
    event_info = Dense(nodes, activation='relu', name='event_layer3')(event_info)
    event_info = Dropout(dropout)(event_info)
    
    # CNN for jet
    jets = Conv1D( 128, 1, kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv0')(jets)
    jets = Dropout(dropout)(jets)
    jets = Conv1D( 64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv1')(jets)
    jets = Dropout(dropout)(jets)
    jets = Conv1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv2')(jets)
    jets = Dropout(dropout)(jets)
    jets = Conv1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv3')(jets)
    jets = Dropout(dropout)(jets)
    jets = Conv1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv4')(jets)
    jets = Dropout(dropout)(jets)
    jets = LSTM(20, go_backwards=True, implementation=2, name='jets_lstm')(jets)
    
    # Concatenate
    x = Concatenate()( [event_info, jets] )
    x = Dense(10, activation='relu',kernel_initializer='lecun_uniform', name='concat_layer')(x)
    
    pred_dense = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='event_prediction' )(x)
    model = Model(inputs=Inputs, outputs= pred_dense)
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy','binary_accuracy'])
    model.save(model_path)
    print ('>>> This model is saved in ', model_path)
    
    return model

# input files

# variables
event_var=['nseljets', 'nselbjets', 'goodht', 'relht', 'mindR_dRbb', 'mindR_mbb', 'Chi2_max', 'Chi2_min', 'Chi2_min_H', 'Chi2_min_W', 'Chi2_min_Top', 'mass_h', 'mass_w', 'mass_top', 'mass_wh', 'mass_secondtop', 'mass_leadjets', 'dR_hbb', 'dR_wjj', 'dR_bw', 'dR_tprimeoj', 'dPhi_htop', 'ratio_mass_topH', 'ratio_mass_secondtopW', 'ratio_pt_topsecondtop', 'ratio_pt_htoptprime', 'ratio_pt_tprimehtprimetop']
jet_var=['jet1_eta', 'jet1_btag', 'jet1_e_massnom', 'jet2_eta', 'jet2_btag', 'jet2_e_massnom', 'jet3_eta', 'jet3_btag', 'jet3_e_massnom', 'jet4_eta', 'jet4_btag', 'jet4_e_massnom', 'jet5_eta', 'jet5_btag', 'jet5_e_massnom']

nvar = len(event_var+jet_var)
model_name = 'model_cnn'

# read input
data = pd.read_hdf("./array/array_trainInput_shuffled.h5")
weights = compute_class_weight( class_weight='balanced', classes=np.unique(data['signal']), y=data['signal'])
dic_weights = dict(enumerate(weights))
model_path = 'models/'+model_name

ver = "model1"
newDir = model_path+ver
if os.path.exists( newDir ):
    string = re.split(r'(\d+)', ver)[0]
    num = int( re.split(r'(\d+)', ver)[1] )
    while os.path.exists( newDir ):
        num = num+1
        newDir = model_path+string+str(num)
print ('>>> Results directory: ', newDir)
os.makedirs( newDir )

data = data.reset_index()
train_event_input, train_event_out, train_jet_input, valid_event_input, valid_event_out, valid_jet_input = split_train_test()

train_out = train_event_out
valid_out = valid_event_out

model = build_model(model_path)

earlystop = EarlyStopping(monitor='val_loss', patience=20)
filename = os.path.join(newDir, 'best_model.h5')

checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
hist = model.fit([train_event_input,train_jet_input], train_out, batch_size=1024, epochs=epochs, validation_data=([valid_event_input, valid_jet_input],  valid_out), callbacks=[earlystop, checkpoint], class_weight=dic_weights)
model.summary()

# copy this code for saving structure - TODO to make it simpler
shutil.copy('train_cnn.py',newDir)

# find best epoch
check_loss = model.history.history['val_loss']
bestepoch = np.argmin( check_loss )+1 

model.load_weights( filename )
valid_prediction = model.predict( [valid_event_input, valid_jet_input] )
train_prediction = model.predict( [train_event_input, train_jet_input] )

# pred jet & real jet
pred = np.argmax( valid_prediction, axis=1 )
real = np.argmax( valid_out, axis=1 )

confusion = confusion_matrix(real, pred)
correct = confusion.trace()
sum_row = confusion.sum(axis=1)[:, np.newaxis]
accuracy = correct/len(valid_out)*100

#######################################################################
#                          Plot loss curve                            # 
#######################################################################
plotDir = newDir+'/'
print("Plotting scores")

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train','Valid'], loc='lower right')
plt.savefig(os.path.join(plotDir, 'accuracy.pdf'), bbox_inches='tight')
plt.gcf().clear()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Binary crossentropy')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train','Valid'],loc='upper right')
plt.savefig(os.path.join(plotDir, 'loss.pdf'), bbox_inches='tight')
plt.gcf().clear()

#######################################################################
#                           Plot ROC curve                            # 
#######################################################################
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[1], tpr[1], thresholds1 = roc_curve(valid_out.values.tolist(), valid_prediction.tolist(), pos_label=1)#w.r.t sig is truth in val set
fpr[2], tpr[2], thresholds2 = roc_curve(train_out.values.tolist(), train_prediction.tolist(), pos_label=1)#w.r.t sig is truth in training set, for overtraining check
roc_auc_val = roc_auc_score(valid_out.values.tolist(),valid_prediction.tolist())
roc_auc_train = roc_auc_score(train_out.values.tolist(),train_prediction.tolist())

print("AUC on valid: "+str(roc_auc_val))
print("AUC on train: "+str(roc_auc_train))
plt.plot(tpr[1], 1-fpr[1])#HEP style ROC
plt.plot(tpr[2], 1-fpr[2])#training ROC

pd.DataFrame(tpr[1]).to_csv('roc_curve/tpr_validation_cnn.csv')
pd.DataFrame(fpr[1]).to_csv('roc_curve/fpr_validation_cnn.csv')

plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.title('ROC Curve')
plt.legend(['Valid', 'Train'], loc='lower left')
plt.savefig(os.path.join(plotDir,'fig_score_roc.pdf'))
plt.gcf().clear()
print('ROC curve is saved!')

#######################################################################
#       Overtraining Check, as well as bkg & sig discrimination       # 
#######################################################################
bins = 40

scores = [tpr[1], fpr[1], tpr[2], fpr[2]]
low = min(np.min(d) for d in scores)
high = max(np.max(d) for d in scores)
low_high = (low,high)

# train set is filled
plt.hist(tpr[2],
    color='b', alpha=0.5, range=low_high, bins=bins,
    histtype='stepfilled', density=True, label='S (train)')
plt.hist(fpr[2],
    color='r', alpha=0.5, range=low_high, bins=bins,
    histtype='stepfilled', density=True, label='B (train)')

# valid set is dotted
hist, bins = np.histogram(tpr[1], bins=bins, range=low_high, density=True)
scale = len(tpr[1]) / sum(hist)
err = np.sqrt(hist * scale) / scale
width = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='S (valid)')
hist, bins = np.histogram(fpr[1], bins=bins, range=low_high, density=True)
scale = len(tpr[1]) / sum(hist)
err = np.sqrt(hist * scale) / scale
plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='B (valid)')

plt.xlabel("Deep Learning Score")
plt.ylabel("Arbitrary units")
plt.legend(loc='best')
plt.savefig(os.path.join(plotDir,'fig_score_overtraining.pdf'))
plt.gcf().clear()
print('Overtraining check plot is saved!')

# time
tm = localtime(time())
strtm = strftime('%Y-%m-%d %I:%M:%S %p', tm) 

print ('writing results...')
with open('results.txt', "a") as f_log:
   f_log.write('###  '+strtm+'\n')
   f_log.write(newDir+'\n')
   f_log.write('nvar: '+str(nvar)+'\n')
   f_log.write('training samples: '+str(len(train_out))+'  validation samples: '+str(len(valid_out))+'\n')
   f_log.write('best epoch: '+str(bestepoch)+'  accuracy: '+str(correct)+'/'+str(len(valid_out))+'='+str(accuracy)+'\n\n')


print("Training complete!")
