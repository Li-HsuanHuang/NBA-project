"""
This code tries to find the best delta factor in ctmcH,
allowing the optimization solver to find a solution
such that the 1-norm error between
model and true playing time is 0.

For each game, we find the delta factor via a for 
loop on possible factor values. We test values
0.01,0.1,10,100,300,and 500.
If the 1-norm error is less than 1e-07, 
the delta factor is found.

We then plot the model and true playing time and
save the x-y points. 
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

# Create empyt lists for storing model and real lineup times.
xvals = []
yvals = []

# Obtain model and real lineup times per game for games that finished in 48 minutes (2880 seconds). 
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
        f = np.array([0.01,0.1,10,100,300,500])
        for m in f:
            eps, constrviol = fixCTMC(transmat,statefrac,m, forcePos = True)
            l = ns*(ns-1)
            phat = addPert(transmat, eps[0:l],ns, 'CTMC')
            # Obtain equilibirum distribution.
            pivec = equilib(phat,'CTMC')
            onenorm[k] = np.sum(np.abs(pivec*2880 - newtimes))
            # Check one norm. If it is greater than 1e-7, then 
            # continue the loop to find the appropriate delta factor. 
            if onenorm[k] > 1e-7: 
                continue
            else:
                plt.scatter(pivec*2880,newtimes)
                xvals.append(pivec*2880)
                yvals.append(newtimes) 
                break 
        k += 1

# Save results to .mat file, which is readable in Python and MATLAB.
#plt.savefig('allgames.pdf')
dat = {'model':xvals,'real':yvals}
io.savemat('pergame_xypoints',dat)    
