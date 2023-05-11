#tmux error
import matplotlib
matplotlib.use('Agg')

import os, sys, shutil
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

trainInput = './arrayOut/array_trainInput_shuffled.h5'

# no mass dependents
#name_inputvar=['nseljets', 'nselbjets', 'goodht', 'relht', 'mindR_dRbb', 'mindR_mbb', 'jet1_eta', 'jet2_eta', 'jet3_eta', 'jet4_eta', 'jet5_eta', 'jet1_btag', 'jet2_btag', 'jet3_btag', 'jet4_btag', 'jet5_btag', 'bjet1_eta', 'bjet2_eta', 'bjet1_btag', 'bjet2_btag', 'jet1_e_massnom', 'jet2_e_massnom', 'jet3_e_massnom', 'jet4_e_massnom', 'jet5_e_massnom', 'bjet1_e_massnom', 'bjet2_e_massnom', 'Chi2_max', 'Chi2_min', 'Chi2_min_H', 'Chi2_min_W', 'Chi2_min_Top', 'mass_h', 'mass_w', 'mass_top', 'mass_wh', 'mass_secondtop', 'mass_leadjets', 'dR_hbb', 'dR_wjj', 'dR_bw', 'dR_tprimeoj', 'dPhi_htop', 'ratio_mass_topH', 'ratio_mass_secondtopW', 'ratio_pt_topsecondtop', 'ratio_pt_htoptprime', 'ratio_pt_tprimehtprimetop']
# for test
name_inputvar=['nseljets', 'nselbjets', 'jet1_pt', 'jet2_pt', 'jet3_pt', 'jet4_pt', 'jet5_pt', 'jet1_eta', 'jet2_eta', 'jet3_eta', 'jet4_eta', 'jet5_eta', 'jet1_e', 'jet2_e', 'jet3_e', 'jet4_e', 'jet5_e', 'jet1_btag', 'jet2_btag', 'jet3_btag', 'jet4_btag', 'jet5_btag']

print ("number of variables: "+str(len(list(name_inputvar))))

model_name = 'model_test'
df_data = pd.read_hdf(trainInput)
data = df_data

weights = data.filter(['evWeight'], axis=1)

labels = data.filter(['signal'], axis=1)
all_event = data.filter(['event','signal'], axis=1)

data = data.filter(['signal']+name_inputvar)
data.astype('float32')

data = data.drop('signal', axis=1) #then drop label

print("Checking nan / inf in input dataset...")

data = pd.DataFrame(data).fillna(-999)
data.replace([np.inf, -np.inf], 999, inplace=True)
if not data.isnull().values.any() and not np.isinf(data).values.any(): print("Cleaned data!")

################## split datasets ##################

print(str(len(all_event)))
groupped_event = all_event.drop_duplicates(subset=['event'])

nevt = len(groupped_event)
print("number of total event = "+str(len(groupped_event)))

split_nevt = groupped_event[:int(nevt*0.7)].iloc[-1]
split_point = -1
for idx, row in all_event.iterrows():
  if (row['event'] == split_nevt['event']):
    if split_point < 0:
        split_point = idx
        break

train_event = all_event[:split_point]
valid_event = all_event[split_point:]

train_sig = train_event.loc[labels['signal'] == 1]
train_bkg = train_event.loc[labels['signal'] == 0]

valid_sig = valid_event.loc[labels['signal'] == 1]
valid_bkg = valid_event.loc[labels['signal'] == 0]

train_idx = pd.concat([train_sig, train_bkg]).sort_index().index
valid_idx = pd.concat([valid_sig, valid_bkg]).sort_index().index

data_train = data.loc[train_idx,:].copy()
data_valid = data.loc[valid_idx,:].copy()

labels_train = labels.loc[train_idx,:].copy()
labels_valid = labels.loc[valid_idx,:].copy()

evWeight_train = weights.loc[train_idx,:].copy()

print('## NUMBER OF COMBINAITONS ##')
print('Training signal: '+str(len(train_sig))+' / validing signal: '+str(len(valid_sig))+' / training background: '+str(len(train_bkg))+' / validing background: '+str(len(valid_bkg)))
print('#'*10+'\n')

labels_train = labels_train.values
train_label = labels_train
labels_valid = labels_valid.values
valid_label = labels_valid

################## Standardization and PCA ##################

scaler = StandardScaler()
data_train_sc = scaler.fit_transform(data_train)
data_valid_sc = scaler.fit_transform(data_valid)
train_data = data_train_sc
valid_data = data_valid_sc

all_data = data
all_data = scaler.fit_transform(all_data)

################## Keras model compile and training ################## 
nvar = len(name_inputvar)
a = 300
b = 0.1
init = 'glorot_uniform'

with tf.device("/cpu:0") :
    inputs = Input(shape=(nvar,))
    x = Dense(a, kernel_regularizer=l2(1E-2))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(b)(x)
    x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
    x = Dropout(b)(x)
    x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
    x = Dropout(b)(x)
    x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
    x = Dropout(b)(x)
    x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
    x = Dropout(b)(x)
    x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
    x = Dropout(b)(x)
    outputs = Dense(1, activation='sigmoid')(x)
#    outputs = Dense(2, activation='softmax')(x) # easy way to have confusion matrix
    model = Model(inputs=inputs, outputs=outputs)

if not os.path.exists("models"): os.mkdir("models")

if os.path.exists("models/"+model_name+'/model.h5'): 
    print("\nModel exists already!\n")
    model = load_model("models/"+model_name+'/model.h5')
else:
    # TF < 2.3 style
    #adam=tf.keras.optimizers.Adam(learning_rate=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1E-3)
    # TF >= 2.3 style
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9)
    adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','binary_accuracy'])
    checkpoint = ModelCheckpoint(model_name, monitor='val_binary_accuracy', verbose=1, save_best_only=False)
    history = model.fit(train_data, train_label,
                        epochs=10, batch_size=1024,
                        validation_data=(valid_data,valid_label),
                        sample_weight=evWeight_train
                        )

    os.mkdir("models/"+model_name)
    model.save('models/'+model_name+'/model.h5')

    ##################  prediction on train & valid set ################## 

    pred_data = pd.DataFrame(model.predict(train_data, batch_size=2048), columns=['pred']).set_index(train_event.index)
    pred = pd.concat([pred_data,train_event], axis=1)
    idx = pred.groupby(['event'])['pred'].transform(max) == pred['pred']
    pred = pred[idx]

    pred_val_data = pd.DataFrame(model.predict(valid_data, batch_size=2048), columns=['pred']).set_index(valid_event.index)
    pred_val = pd.concat([pred_val_data,valid_event], axis=1)
    idx = pred_val.groupby(['event'])['pred'].transform(max) == pred_val['pred']
    pred_val = pred_val[idx]

    import matplotlib.pyplot as plt

    ################## Plot loss curve ################## 

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Binary crossentropy')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Valid'],loc='upper right')
    plt.savefig(os.path.join("models/"+model_name+'/','fig_score_loss.pdf'))
    plt.gcf().clear()
    print('Loss curve is saved!')

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Binary accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train','Valid'],loc='best')
    plt.savefig(os.path.join("models/"+model_name+'/','fig_score_acc.pdf'))
    plt.gcf().clear()
    print('Accuracy curve is saved!')

    ################## Plot ROC curve ################## 
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[1], tpr[1], thresholds1 = roc_curve(valid_label.tolist(), pred_val_data['pred'].values.tolist(), pos_label=1)#w.r.t sig is truth in val set
    fpr[2], tpr[2], thresholds2 = roc_curve(train_label.tolist(), pred_data['pred'].values.tolist(), pos_label=1)#w.r.t sig is truth in training set, for overtraining check
    roc_auc_val = roc_auc_score(valid_label.tolist(),pred_val_data['pred'].values.tolist())
    roc_auc_train = roc_auc_score(train_label.tolist(),pred_data['pred'].values.tolist())

    print("AUC on valid: "+str(roc_auc_val))
    print("AUC on train: "+str(roc_auc_train))
    plt.plot(tpr[1], 1-fpr[1])#HEP style ROC
    plt.plot(tpr[2], 1-fpr[2])#training ROC

    pd.DataFrame(tpr[1]).to_csv(f'models/{model_name}/tpr_validation_dnn.csv')
    pd.DataFrame(fpr[1]).to_csv(f'models/{model_name}/fpr_validation_dnn.csv')

    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Rejection')
    plt.title('ROC Curve')
    plt.legend(['Valid', 'Train'], loc='lower left')
    plt.savefig(os.path.join("models/"+model_name+'/','fig_score_roc.pdf'))
    plt.gcf().clear()
    print('ROC curve is saved!')
    
    ################## Overtraining Check, as well as bkg & sig discrimination ################## 
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
    #plt.ylabel("Arbitrary units")
    plt.ylabel("Number of Events")
    plt.legend(loc='best')
    plt.savefig(os.path.join("models/"+model_name+'/','fig_score_overtraining.pdf'))
    plt.gcf().clear()
    print('Overtraining check plot is saved!')

    # copy this code for tracking down structure
    shutil.copy('train_dnn.py',"models/"+model_name+'/')

    print("Training complete!")
