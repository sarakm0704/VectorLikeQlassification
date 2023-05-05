import pandas as pd
import deepdish.io as io
import numpy as np
import matplotlib.pyplot as plt

inputDir = "./arrayOut"
outputDir = "./pdf"
inArr = "array_trainInput.h5"

def correlations(data, name, **kwds):
    """Calculate pairwise correlation between features.

    Extra arguments are passed on to DataFrame.corr()
    """
    # simply call df.corr() to get a table of
    # correlation values if you do not need
    # the fancy plotting
    corrmat = data.corr(**kwds)

    fig, ax1 = plt.subplots(ncols=1, figsize=(6,5))

    opts = {'cmap': plt.get_cmap("RdBu"),
            'vmin': -1, 'vmax': +1}
    heatmap1 = ax1.pcolor(corrmat, **opts)
    plt.colorbar(heatmap1, ax=ax1)

    labels = corrmat.columns.values
    for ax in (ax1,):
        ax.tick_params(labelsize=6)
        # shift location of ticks to center of the bins
        ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_xticklabels(labels, minor=False, ha='right', rotation=90)
        ax.set_yticklabels(labels, minor=False)

    plt.tight_layout()
    #plt.show()
    if name == 'sig' :
        ax1.set_title("Correlations in Signal")
        plt.savefig(outputDir+'/fig_corr_s.pdf')
        print('Correlation matrix for signal is saved!')
        plt.gcf().clear()
    elif name == 'bkg' :
        ax1.set_title("Correlations in Background")
        plt.savefig(outputDir+'/fig_corr_b.pdf')
        plt.gcf().clear()
        print('Correlation matrix for background is saved!')
    else : print('Wrong class name!')

readData = pd.read_hdf(inputDir+"/"+inArr)

##all
name_inputvar=list(readData)

datain = readData.filter(name_inputvar)

correlations(datain.loc[datain['signal'] == 0].drop('signal', axis=1), 'bkg')
correlations(datain.loc[datain['signal'] == 1].drop('signal', axis=1), 'sig')
