import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import ROOT
from ROOT import *
import sys,os

rocDir1 = "./models/model_test1"
rocDir2 = "./models/model_test2"

tpr1 = pd.read_csv(rocDir1+"tpr_evaluation.csv").set_axis(['Entry','Value'], axis=1)
fpr1 = pd.read_csv(rocDir1+"fpr_evaluation.csv").set_axis(['Entry','Value'], axis=1)
tpr2 = pd.read_csv(rocDir2+"tpr_evaluation.csv").set_axis(['Entry','Value'], axis=1)
fpr2 = pd.read_csv(rocDir2+"fpr_evaluation.csv").set_axis(['Entry','Value'], axis=1)

roc_auc1 = auc(tpr1['Value'], 1-fpr2['Value'])
roc_auc2 = auc(tpr1['Value'], 1-fpr2['Value'])

print("Area under the ROC curve 1: %f" % roc_auc1)
print("Area under the ROC curve 2: %f" % roc_auc2)

plt.xlabel(r'Signal Efficiency ($\epsilon_{s}$)')
plt.ylabel(r'Background Rejection ($1-\epsilon_{b}$)')
plt.title('ROC Curve')
plt.plot(tpr1['Value'], 1-fpr1['Value'],c='#17C2AE')
plt.plot(tpr2['Value'], 1-fpr2['Value'],c='#1f77b4')
plt.legend(['model1','model2'],loc='lower left')
plt.savefig('pdf/fig_score_roc.pdf')
plt.gcf().clear()
print('ROC curve is recoverd!')
