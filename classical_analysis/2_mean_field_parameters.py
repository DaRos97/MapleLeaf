import numpy as np
import sys, os
import functions_MF as fs_mf
import functions_ssf as fs_ssf

ind_choice = 0 if len(sys.argv)<2 else int(sys.argv[1])
ans, args, ind_discrete, Jd, Jt = fs_ssf.get_pars(ind_choice)

#args = [-np.pi/2,]
lattice = fs_ssf.fruit_lattice[ans](3,args,ind_discrete)/2

list_op_loop = [
        ['L1','alpha'],['L1','beta'],['L2','alpha'],['L2','beta'],['L3','alpha'],['L3','beta']
        ]

res = {}
for loop,type_op in list_op_loop:
    if not type_op in res.keys():   #initiate new dic
        res[type_op] = {}
    disp = False
    res[type_op][loop] = fs_mf.compute_loop(loop,type_op,ans,lattice,disp)
    #



print('Ah: ',fs_mf.Ah(res))
print('At: ',fs_mf.At(res))
print('Ad: ',fs_mf.Ad(res))

print('-----')

print('Bh: ',fs_mf.Bh(res))
print('Bt: ',fs_mf.Bt(res))
print('Bd: ',fs_mf.Bd(res))
exit()

print('--------------------------------------')
print('phi_h: ',fs_mf.phi_h(res,ans))
print('phi_t: ',fs_mf.phi_t(res,ans))
print('phi_d: ',fs_mf.phi_d(res,ans))

print('-----')

print('psi_h: ',fs_mf.psi_h(res,ans))
print('psi_t: ',fs_mf.psi_t(res,ans))
print('psi_d: ',fs_mf.psi_d(res,ans))
