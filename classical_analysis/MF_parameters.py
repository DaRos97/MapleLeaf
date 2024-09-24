import numpy as np
import sys, os
import functions_MF as fs

order = 'NonCoplanar2'        #one of the clasical orders

print("Computing LRO ",order)

if order[:-1] == 'Coplanar':
    print("alpha: ",fs.alpha_,'\n')
if order[:-1] == 'NonCoplanar':
    print("theta: ",fs.theta_)
    print("phi: ",fs.phi_,'\n')


list_op_loop = [
        ['a','alpha'],['b','alpha'],['c','alpha'],['d','alpha'],['e','alpha'],
        ['a','beta'],['b','beta'],['c','beta1'],['c','beta2'],['d','beta1'],['d','beta2'],['e','beta'],
        ['i','beta'],['j','beta'],['e','gamma'],['f','gamma'],
        ]

res = {}
for loop,type_op in list_op_loop:
    if not type_op in res.keys():
        res[type_op] = {}
    pr = True if (type_op == 'beta1' and loop in ['c','d'] and 0) else False
    res[type_op][loop] = fs.compute_loop(loop,type_op,order,pr)
    #
    if pr:
        print(loop,': ',type_op)
        print(res[type_op][loop])
        input()



print('Ah: ',fs.Ah(res,order))
print('Ahp: ',fs.Ahp(res,order))
print('At: ',fs.At(res,order))
print('Atp: ',fs.Atp(res,order))
print('Ad: ',fs.Ad(res,order))

print('-----')

print('Bh: ',fs.Bh(res,order))
print('Bhp: ',fs.Bhp(res,order))
print('Bt: ',fs.Bt(res,order))
print('Btp: ',fs.Btp(res,order))
print('Bd: ',fs.Bd(res,order))

print('--------------------------------------')

print('phi_h: ',fs.phi_h(res,order))
print('phi_hp: ',fs.phi_hp(res,order))
print('phi_t: ',fs.phi_t(res,order))
print('phi_tp: ',fs.phi_tp(res,order))
print('phi_d: ',fs.phi_d(res,order))

print('-----')

print('psi_h: ',fs.psi_h(res,order))
print('psi_hp: ',fs.psi_hp(res,order))
print('psi_t: ',fs.psi_t(res,order))
print('psi_tp: ',fs.psi_tp(res,order))
print('psi_d: ',fs.psi_d(res,order))
