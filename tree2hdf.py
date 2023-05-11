#! /usr/bin/env python3
import sys, os
import numpy as np
from ROOT import *
import pandas as pd
import deepdish.io as io

def random(arrayDir,target):

    print("shuffle input datasets..")

    df = pd.read_hdf(arrayDir+target)
    print("from:\n" + str(df['signal']))
    df = df.sample(frac=1).reset_index(drop=True)
    print("To:\n" + str(df['signal']))
    io.save(arrayDir+"array_trainInput_shuffled.h5",df)
    print("Done")

def merge(arrayDir):
    print("hi")
    selEvent = pd.DataFrame([])

    for process in os.listdir(arrayDir+'/split/'):
        inFile = arrayDir + '/split/' + process
        print("merge process " + process[:-3] + " for training")
        df = pd.read_hdf(inFile)
        max_nevt_num = 0
        if df.size != 0: last = int(df.tail(1)['event'])+1
        df['event'] = df['event'] + max_nevt_num
        selEvent = pd.concat([selEvent,df], axis=0)
        max_nevt_num += last

    selEvent.reset_index(drop=True, inplace=True)
    print(selEvent)
    io.save(arrayDir+"array_trainInput.h5",selEvent)

def makeCombi(inputDir, inputFile, outputDir, makeTrainingInput=False):
    print(str(inputDir+"/"+inputFile)+" start")
    chain = TChain("outputTree")
    chain.Add(inputDir+"/"+inputFile)

    data = False
    if 'Data' in inputDir: data = True
    tprime = False
    if 'tprime' in inputFile: tprime = True

    jetCombi = []

    for i in range(chain.GetEntries()) :
        chain.GetEntry(i)
        if tprime: signal = 1
        if not tprime: signal = 0

        event = i

        if (event % 2) == 0: continue  # select odd for training
        #if (event % 2) != 0: continue  # select even for evaluation
        if event > 242080: continue

        evWeight = chain.evWeight

        nseljets = chain.nselJets
        nselbjets = chain.nselbJets

        jet1_pt = chain.selJet1_pt
        jet2_pt = chain.selJet2_pt
        jet3_pt = chain.selJet3_pt
        jet4_pt = chain.selJet4_pt
        jet5_pt = chain.selJet5_pt

        jet1_pt_massnom = chain.selJet1_pt_massnom
        jet2_pt_massnom = chain.selJet2_pt_massnom
        jet3_pt_massnom = chain.selJet3_pt_massnom
        jet4_pt_massnom = chain.selJet4_pt_massnom
        jet5_pt_massnom = chain.selJet5_pt_massnom

        jet1_eta = chain.selJet1_eta
        jet2_eta = chain.selJet2_eta
        jet3_eta = chain.selJet3_eta
        jet4_eta = chain.selJet4_eta
        jet5_eta = chain.selJet5_eta

        jet1_e = chain.selJet1_e
        jet2_e = chain.selJet2_e
        jet3_e = chain.selJet3_e
        jet4_e = chain.selJet4_e
        jet5_e = chain.selJet5_e

        jet1_e_massnom = chain.selJet1_e_massnom
        jet2_e_massnom = chain.selJet2_e_massnom
        jet3_e_massnom = chain.selJet3_e_massnom
        jet4_e_massnom = chain.selJet4_e_massnom
        jet5_e_massnom = chain.selJet5_e_massnom

        jet1_btag = chain.selJet1_btag
        jet2_btag = chain.selJet2_btag
        jet3_btag = chain.selJet3_btag
        jet4_btag = chain.selJet4_btag
        jet5_btag = chain.selJet5_btag

        jetCombi.append([signal, event, nseljets, nselbjets, jet1_pt, jet2_pt, jet3_pt, jet4_pt, jet5_pt, jet1_eta, jet2_eta, jet3_eta, jet4_eta, jet5_eta, jet1_e, jet2_e, jet3_e, jet4_e, jet5_e, jet1_btag, jet2_btag, jet3_btag, jet4_btag, jet5_btag, evWeight])

    tmp = inputFile[:-5]

    combi = pd.DataFrame(jetCombi,columns=['signal','event','nseljets','nselbjets','jet1_pt','jet2_pt','jet3_pt','jet4_pt','jet5_pt','jet1_eta','jet2_eta','jet3_eta','jet4_eta','jet5_eta','jet1_e','jet2_e','jet3_e','jet4_e','jet5_e','jet1_btag','jet2_btag','jet3_btag','jet4_btag','jet5_btag','evWeight'])

    if makeTrainingInput: combi = combi
    else: combi = combi.drop(['signal'], axis=1)

    print(combi['signal'])
    combi = pd.DataFrame(combi).fillna(-999)
    combi.replace(np.inf,999,inplace=True)

    io.save(outputDir+"split/array_"+tmp+".h5",combi)
    print(str(inputDir+"/"+inputFile)+" end")

if __name__ == '__main__':
    #Options
    from optparse import OptionParser

    parser = OptionParser()
    parser.usage = """
    %prog [options] option
    convert root ntuple to array 
    """

    parser.add_option("-m", "--merge", dest="merge",
                      action = 'store_true',
                      default=False,
                      help='Merge array files for each process')

    parser.add_option("-d", "--deep", dest="deep",
            		      action = 'store_true',
            		      default=False,
            		      help='Run on signal sample for deep learning train')
            
    parser.add_option("-a", "--all", dest="all",
            		      action = 'store_true',
            		      default=False,
            		      help='all root fiels in input directory')

    parser.add_option("-r", "--random", dest="random",
            		      action = 'store_true',
            		      default=False,
            		      help='shuffle row randomly')
            
    (options,args) = parser.parse_args()

    ntupleDir = './dnnTree/hadronic/'
    arrayDir = './arrayOut/'

    processes = []
    if len(args) == 1:
        f = open(args[0], "r") 
        processes = f.read().splitlines()
    else:
        processes = os.listdir(ntupleDir) 

    if options.merge:
        merge(arrayDir)

    if options.deep:
        makeCombi(ntupleDir, 'tprimeAll_2018.root', arrayDir, True)
        makeCombi(ntupleDir, 'tthad2018.root', arrayDir, True)

    if options.random:
        random(arrayDir,"array_trainInput.h5")

    if options.all:
        for process in processes:
                os.makedirs(arrayDir+process)

                proc = process.split(".")[0]
                makeCombi(ntupleDir, process, arrayDir)

