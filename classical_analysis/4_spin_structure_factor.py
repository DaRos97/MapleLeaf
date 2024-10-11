"""
Here we compute the ssf of the classical orders we care about.
We do it by brute forse by just defining a big lattice (maple leaf ofc),
and computing the ssf on it for each momentum point Q following the definition (see eq.29 in D.Rossi et al., Phys. Rev. B 108, 144406 (2023))
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import functions_ssf as fs
import functions_cpd as fs_cpd
from pathlib import Path
from tqdm import tqdm
import os,sys
from time import time
from datetime import timedelta

"""
Chose the point to compute in the phase diagram
"""
ind_order = 0 if len(sys.argv)<2 else int(sys.argv[1])
Jh = 1
nn = 51     #largest phase diagram computed so far
bound = 4
ng = 0
Jds = np.linspace(-bound,bound*ng,nn)
Jts = np.linspace(-bound,bound*ng,nn)

nC = '3'
I = 0    #5#12#29      #over 51
J = 25    #40#25#26     #over 51
order = fs_cpd.name_list[nC][ind_order]
print("Computing order ",order," at position Jd=","{:.4f}".format(Jds[I])," and Jt=","{:.4f}".format(Jts[J]))
alpha = 5*np.pi/6
print("alpha: ",alpha)
UC = 30  #number of unit cells (6 sites) in each lattice direction (better to have it commensurate with the order we are computing)
lattice = fs.lattice_functions[order](UC,alpha)

"""
We start by defining the lattice
"""

high_symm = 'b'
factors = {'B':(7,10),'b':(6,4)}
nkx, nky = fs.get_kpoints(factors[high_symm],150)
print("Using ",UC,"x",UC," unit cells and ",nkx,"x",nky," momentum points")

"""
Need to adapt the grid in momentum space to get the high symmetry points.
"""
vecx = fs.B_[1,0] if high_symm=='B' else fs.b_[1,0]
vecy = fs.B_[1,1]/3 if high_symm=='B' else fs.b_[1,1]/3
fx,fy = factors[high_symm]
kxs = np.linspace(-vecx*fx,vecx*fx,nkx)
kys = np.linspace(-vecy*fy,vecy*fy,nky)

"""
Here we compute the ssf
"""
UC_positions = np.array([   #in terms of t1,t2
    [2,0],
    [1,0],
    [0,0],
    [0,1],
    [1,1],
    [0,2]])
SSFzz_fn = fs.get_ssf_fn('zz',order,Jds[I],Jts[J],UC,nkx)
SSFxy_fn = fs.get_ssf_fn('xy',order,Jds[I],Jts[J],UC,nkx)
if not Path(SSFzz_fn).is_file():
    time_initial = time()
    SSFzz = np.zeros((nkx,nky))
    SSFxy = np.zeros((nkx,nky))
    """
    To parallelize this we need, for each k, a UC**2*6 vector of distances (we fix one position to be (UC//2,UC//2,0))
    """
    fUCx = UC//2
    fUCy = UC//2
    #
    distances = np.zeros((UC**2*6,2))
    prod_zz = np.zeros(UC**2*6)
    prod_xy = np.zeros(UC**2*6)
    for fUC in range(6):
        #Probably can be done faster
        for i in range(UC**2*6):
            iUCx, iUCy, iUC = (i//6//UC, i//6%UC, i%6)
            distances[i] = fs.T_[0]*(iUCx-fUCx) + fs.T_[1]*(iUCy-fUCy) + np.dot(fs.t_.T,(UC_positions[iUC]-UC_positions[fUC]))
            prod_zz[i] = lattice[iUCx,iUCy,iUC,2]*lattice[fUCx,fUCy,fUC,2]
            prod_xy[i] = (lattice[iUCx,iUCy,iUC,0]*lattice[fUCx,fUCy,fUC,0]+lattice[iUCx,iUCy,iUC,1]*lattice[fUCx,fUCy,fUC,1])
        """
        We can create SSFzz in one shot by creating a matrix of kx,ky values and doing the dot product at once
        """
        X,Y = np.meshgrid(kxs,kys)
        cos_kd = np.cos(np.einsum('ijm,km->ijk',np.dstack([X.T,Y.T]),distances))/2
        SSFzz += np.sum(cos_kd*prod_zz, axis=2)
        SSFxy += np.sum(cos_kd*prod_xy, axis=2)
    SSFzz /= UC**2*6
    SSFxy /= UC**2*6
    time_final = timedelta(seconds=time()-time_initial)
    print("SSF compute time: ",str(time_final))
    if 0:
        np.save(SSFzz_fn,SSFzz)
        np.save(SSFxy_fn,SSFxy)
else:
    SSFzz = np.load(SSFzz_fn)
    SSFxy = np.load(SSFxy_fn)

"""
Finally we plot the result
"""
fig = plt.figure(figsize=(20,7))
tt = ['zz','xy']
X,Y = np.meshgrid(kxs,kys)
for i in range(2):
    data = SSFzz if i == 0 else SSFxy
    ax = fig.add_subplot(1,2,i+1)
    sc = ax.scatter(X.T,Y.T,c=data,
                marker='s',
                cmap=cm.plasma_r,
                s=30,
                norm=None #if i==0 else norm
              )
    norm = sc.norm
    fs.plot_BZs(ax)
    ax.set_xlim(kxs[0],kxs[-1])
    ax.set_ylim(kys[0],kys[-1])
    title = "SSF "+tt[i]
    title += ", alpha="+"{:.2f}".format(alpha*180/np.pi)+"°" if order=='Coplanar' else ''
    ax.set_title(title)
    plt.colorbar(sc)
    ax.set_aspect('equal')

fig.tight_layout()
plt.show()


























