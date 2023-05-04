import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

inputDir = "./arrayOut/"
outputDir = "./pdf/"
inArr = "array_trainInput.h5"

#####################
#Plot input variables
#####################
def drawFeatures(sigdata, bkgdata, signame, bkgname, **kwds):

    print('Plotting input variables')
    bins = 20
    for colname in sigdata:
      dataset = [sigdata, bkgdata]
      low = min(np.min(d[colname].values) for d in dataset)
      high = max(np.max(d[colname].values) for d in dataset)
      #if high > 500: low_high = (low,500)
      #else: low_high = (low,high)
      low_high = (low,high)

      plt.figure()
      sigdata[colname].plot.hist(color='b', density=True, range=low_high, bins=bins, histtype='step', label='signal')
      bkgdata[colname].plot.hist(color='r', density=True, range=low_high, bins=bins, histtype='step', label='background')
      plt.xlabel(colname)
      plt.ylabel('A.U.')
      plt.title('Intput variables')
      plt.legend(loc='best')
      plt.savefig(outputDir+'fig_input_'+colname+'.pdf')
      plt.gcf().clear()
      plt.close()

readData = pd.read_hdf(inputDir+"/"+inArr)

print("cleaning nan / inf ...")
readData = pd.DataFrame(readData).fillna(-999)
readData.replace([np.inf, -np.inf], 999, inplace=True)

##all
name_inputvar=list(readData)
datain = readData.filter(name_inputvar)

drawFeatures(datain.loc[datain['signal'] == 1].drop('signal', axis=1), datain.loc[datain['signal'] == 0].drop('signal', axis=1), 'sig','bkg')
