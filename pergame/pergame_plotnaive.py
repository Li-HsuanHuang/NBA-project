# This code plots per game model and true lineup playing time.
# We modeled each game and fixed each CTMC matrix.
# We expect to see 0 training error.

# Correlation between the model and true playing time is computed
# as rho below.

import scipy.io as io
import numpy as np
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt

r = io.loadmat('pergame_xypoints_naive')
model = r['model']
model = np.concatenate(model)
model = np.hstack(model)
model = np.ravel(model)

real = r['real']
real = np.concatenate(real)
real = np.hstack(real)
real = np.ravel(real)

rho,_ = pearsonr(model,real)
print(rho)
plt.scatter(model,real,c='k',s=40)
plt.plot([0.1,3000],[0.1,3000],'r')
plt.axis([0.1,3000,0.1,3000])
plt.xscale('log',basex=10)
plt.yscale('log',basey=10)
plt.xlabel('simulated playing time',fontsize=14)
plt.ylabel('true playing time',fontsize=14)
#plt.savefig('pergame_allteams_naive.pdf')
plt.savefig('pergame_allteams_naive.eps',format='eps',dpi=300)
plt.show()
