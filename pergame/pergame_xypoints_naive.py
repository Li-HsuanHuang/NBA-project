"""
This code builds transition-rate matrix for each 48-minute game per team. Then we save the lineup model time and lineup real time .mat file,
which is readable in Python and MATLAB.
""" 

# We are using the NBA data for the season 2015-16



import numpy as np
import pandas as pd
from mco1 import *
import time 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as io
# Read team csv file

teams = ['Atl','Bkn','Bos','Cha','Chi','Cle','Dal','Den','Det','GS','Hou',
'Ind','LAC','LAL','Mem','Mia','Mil','Min','NO','NY','OKC','Orl','Phi','Pho',
'Por','SA','Sac','Tor','Uta','Was']

xvals = []
yvals = []
for r in teams:
    dub = pd.read_csv(str(r)+'.csv')
    dub = dub.drop('Unnamed: 0', axis = 1)
    dates = np.unique(dub.rawdate)
    # First figure out dates we don't want.
    toskip = []
    for i in dates:
        tt = dub[dub.rawdate == i].timePlayed
        if np.max(np.cumsum(tt)) > 2880:
            toskip.append(i)
    # These are the dates to use
    tokeep = [i for i in dates if i not in toskip]

    def homeoraway(date):
        test = np.array(dub[dub.rawdate == date].Hteam)
        if (test[0] == r):
            x = 1
        else:
            x = 0
        return x
    

    k = 0
    onenorm = np.zeros(len(tokeep))
    for i in tokeep:
        t = homeoraway(i)
        tg = dub[dub.rawdate == i]
        tt = np.array(tg.timePlayed)
        new = np.cumsum(tt)
        if (t == 1):
            lineup = np.array(tg.HLu,dtype = 'int')-1
        else:    
            lineup = np.array(tg.VLu,dtype = 'int')-1
        ind = np.where(np.diff(lineup)!=0)[0]
        bigstate = np.max(lineup)+1
        localcounts = np.zeros((bigstate,bigstate))
        localtimes = np.zeros(bigstate)
        fixedLU = np.append(lineup[ind],lineup[-1])
        fixedtime = np.append(new[ind],new[-1])
        localcounts += counttrans(fixedLU,bigstate)
        localtimes += statetimetally(fixedLU,fixedtime,bigstate)
        localtimes[fixedLU[0]] += fixedtime[0]       
        nrs = np.where(localtimes!=0.)[0]
        ns = len(nrs)
        newcounts = localcounts[nrs,:]
        newcounts = newcounts[:,nrs] 
        newtimes = localtimes[nrs]
        statefrac = newtimes/np.sum(newtimes)
        
        np.random.seed(7)
        transmat = createCTMC(newcounts,newtimes,ns)
        pivec = equilib(transmat,'CTMC')
        plt.scatter(pivec*2880,newtimes)
        xvals.append(pivec*2880)
        yvals.append(newtimes) 
             
        k += 1


#plt.savefig('allgames_naive.pdf')
dat = {'model':xvals,'real':yvals}
io.savemat('pergame_xypoints_naive',dat)    
