"""
Here we compute the ssf of the classical orders we care about.
We do it by brute forse by just defining a big lattice (maple leaf ofc),
and computing the ssf on it for each momentum point Q following the definition (see eq.29 in D.Rossi et al., Phys. Rev. B 108, 144406 (2023))
"""
import numpy as np
import matplotlib.pyplot as plt
import functions_ssf as fs
import functions_cpd as fs_cpd
from pathlib import Path
from tqdm import tqdm
import os,sys

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

"""
We start by defining the lattice
"""
UC = 9  #number of unit cells (6 sites) in each lattice direction (better to have it commensurate with the order we are computing)
nkx = 61   #number of k points in x,y
nky = 61

"""
Need to adapt the grid in momentum space to get the high symmetry points.
"""
kxs = np.linspace(-np.linalg.norm(fs.b[0]),np.linalg.norm(fs.b[0]),nkx)
kys = np.linspace(-np.linalg.norm(fs.b[0]),np.linalg.norm(fs.b[0]),nky)

lattice = fs.lattice_functions[order](UC)

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
    SSFzz = np.zeros((nkx,nky),dtype=complex)
    SSFxy = np.zeros((nkx,nky),dtype=complex)
    """
    to parallelize this we need, for each k, a UC**2*6xUC**2*6 matrix of distances
    """
    distances = np.zeros((UC**2*6,UC**2*6,2))
    prod_zz = np.zeros((UC**2*6,UC**2*6))
    prod_xy = np.zeros((UC**2*6,UC**2*6))
    for i in range(UC**2*6):
        iUCx, iUCy, iUC = (i//6//UC, i//6%UC, i%6)
        for j in range(UC**2*6):
            jUCx, jUCy, jUC = (j//6//UC, j//6%UC, j%6)
            distances[i,j] = fs.T[0]*(iUCx-jUCx) + fs.T[1]*(iUCy-jUCy) + np.dot(fs.t.T,(UC_positions[iUC] - UC_positions[jUC]))
            prod_zz[i,j] = lattice[iUCx,iUCy,iUC,2]*lattice[jUCx,jUCy,jUC,2]
            prod_xy[i,j] = (lattice[iUCx,iUCy,iUC,0]*lattice[jUCx,jUCy,jUC,0]+lattice[iUCx,iUCy,iUC,1]*lattice[jUCx,jUCy,jUC,1])
    for ik in tqdm(range(nkx*nky)):
        ikx = ik//nky
        iky = ik%nky
        k_ = np.array([kxs[ikx],kys[iky]])
        SSFzz[ikx,iky] = np.sum(np.ravel(np.exp(-1j*np.dot(distances,k_))*prod_zz))
        SSFxy[ikx,iky] = np.sum(np.ravel(np.exp(-1j*np.dot(distances,k_))*prod_xy))
    if 1:
        np.save(SSFzz_fn,SSFzz)
        np.save(SSFxy_fn,SSFxy)
else:
    SSFzz = np.load(SSFzz_fn)
    SSFxy = np.load(SSFxy_fn)

"""
Finally we plot the result
"""
fig = plt.figure(figsize=(20,14))
ax1 = fig.add_subplot(1,2,1)
fs.plot_ssf(SSFzz,kxs,kys,ax1,'SSF zz')
ax2 = fig.add_subplot(1,2,2)
fs.plot_ssf(SSFxy,kxs,kys,ax2,'SSF xy')

fig.tight_layout()
plt.show()


























