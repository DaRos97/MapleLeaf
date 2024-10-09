import numpy as np

def FM(UC):
    lattice = np.zeros((UC,UC,6,3))
    lattice[:,:,:,2] = 1/2
    return lattice

def Neel(UC):
    lattice = np.zeros((UC,UC,6,3))
    lattice[:,:,::2,2] = 1/2
    lattice[:,:,1::2,2] = -1/2
    return lattice

lattice_functions = {'FM':FM,'Neel':Neel}

def get_spec_txt(order,Jd,Jt,UC,nkx):
    return order+'_'+"{:.4f}".format(Jd)+"{:.4f}".format(Jt)+'_'+str(UC)+'_'+str(nkx)

def get_ssf_fn(direction,order,Jd,Jt,UC,nkx):
    return 'results/data_ssf/'+direction+'_'+get_spec_txt(order,Jd,Jt,UC,nkx)+'.npy'

