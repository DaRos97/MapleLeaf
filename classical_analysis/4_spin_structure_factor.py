"""
Here we compute the ssf of the classical orders we care about.
We do it by brute forse by just defining a big lattice (maple leaf ofc),
and computing the ssf on it for each momentum point Q following the definition (see eq.29 in D.Rossi et al., Phys. Rev. B 108, 144406 (2023))
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import functions_ssf as fs_ssf
import functions_cpd as fs_cpd
from pathlib import Path
from tqdm import tqdm
import os,sys
from time import time
from datetime import timedelta

save = True

"""
Chose the point to compute in the phase diagram
"""
ind_choice = 0 if len(sys.argv)<2 else int(sys.argv[1])
order, args, ind_discrete, Jd, Jt = fs_ssf.get_pars(ind_choice)
UC = 21 if len(sys.argv)<3 else int(sys.argv[2]) #number of unit cells (6 sites) in each lattice direction (better to have it commensurate with the order we are computing)
lattice = fs_ssf.fruit_lattice[order](UC,args,ind_discrete)
dic_UC_average = {'kiwi':1,'banana':2,'mango':3}
"""
We start by defining the lattice
"""
high_symm = 'b'     #decide which high symmetry points to include in k-space, thise of the inner BZ (B) or of the external one (b)
factors = (6,10)    #Just how many units to consider 
nkx, nky = fs_ssf.get_kpoints(factors,150)
print("Using ",UC,"x",UC," unit cells and ",nkx,"x",nky," momentum points")
"""
Need to adapt the grid in momentum space to get the high symmetry points.
"""
vecx = np.pi*2/np.sqrt(21)#fs.B_[1,0] if high_symm=='B' else fs.b_[1,0]
vecy = np.pi*2/3/np.sqrt(7)#fs.B_[1,1]/3 if high_symm=='B' else fs.b_[1,1]/3
fx,fy = factors
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
SSFzz_fn = fs_ssf.get_ssf_fn('zz',order,ind_discrete,Jd,Jt,UC,nkx,nky)
SSFxy_fn = fs_ssf.get_ssf_fn('xy',order,ind_discrete,Jd,Jt,UC,nkx,nky)
if not Path(SSFzz_fn).is_file():
    time_initial = time()
    SSFzz = np.zeros((nkx,nky))
    SSFxy = np.zeros((nkx,nky))
    """
    We fix one unit cell and sum over the 6 sites inside
    """
    fUCx = UC//2    #fixed UC
    fUCy = UC//2
    #
    for fff in range(dic_UC_average[order]**2):
        fUCx = UC//2+fff%dic_UC_average[order]
        fUCy = UC//2+fff//dic_UC_average[order]
        for fUC in range(6):
            distances = np.zeros((UC**2*6,2))       #distances of all sites wrt fixed site
            prod_zz = np.zeros(UC**2*6)
            prod_xy = np.zeros(UC**2*6)
            #Probably can be done faster
            for i in range(UC**2*6):
                iUCx, iUCy, iUC = (i//6//UC, i//6%UC, i%6)
                distances[i] = fs_ssf.T_[0]*(iUCx-fUCx) + fs_ssf.T_[1]*(iUCy-fUCy) + np.dot(fs_ssf.t_.T,(UC_positions[iUC]-UC_positions[fUC]))
                prod_zz[i] = lattice[iUCx,iUCy,iUC,2]*lattice[fUCx,fUCy,fUC,2]
                prod_xy[i] = lattice[iUCx,iUCy,iUC,0]*lattice[fUCx,fUCy,fUC,0]+lattice[iUCx,iUCy,iUC,1]*lattice[fUCx,fUCy,fUC,1]
            """
            We can create SSFzz in one shot by creating a matrix of kx,ky values and doing the dot product at once.
            We use the cosine since the SSF is real.
            """
            X,Y = np.meshgrid(kxs,kys)
            cos_kd = np.cos(np.einsum('ijm,km->ijk',np.dstack([X.T,Y.T]),distances))/2
            SSFzz += np.sum(cos_kd*prod_zz, axis=2)
            SSFxy += np.sum(cos_kd*prod_xy, axis=2)
    SSFzz /= UC**2*6*dic_UC_average[order]**2
    SSFxy /= UC**2*6*dic_UC_average[order]**2
    time_final = timedelta(seconds=time()-time_initial)
    print("SSF compute time: ",str(time_final))
    if save:
        print("Saving..")
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
    fs_ssf.plot_BZs(ax)
    ax.set_xlim(kxs[0],kxs[-1])
    ax.set_ylim(kys[0],kys[-1])
#    title = "SSF "+tt[i]
#    ax.set_title(title)
    plt.colorbar(sc)
    ax.set_aspect('equal')

#plt.suptitle(fs_ssf.get_suptitle(order,args,Jd,Jt))

fig.tight_layout()
plt.show()


























