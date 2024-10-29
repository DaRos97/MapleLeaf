import numpy as np
import functions as fs
from time import time
import sys,os
import random

machine = 'loc'

save_to_file = True
"""Choice of ansatz"""
header = ['Jh','Jd','Jt','Energy','Gap','L','Ah','Ahp','argAhp','At','argAt','Atp','argAtp','Ad','argAd','Bh','argBh','Bhp','argBhp','Bt','argBt','Btp','argBtp','Bd','argBd']
"""Parameters of minimization"""
MaxIter = 3000
prec_L = 1e-10       #precision required in L maximization
L_method = 'Brent'
L_bounds = (0,50)       #bound of Lagrange multiplier
cutoff_L = 1e-4
pars_L = (prec_L,L_method,L_bounds)
cutoff_O = 1e-4
"""Parameters of phase diagram"""
index = 0 if len(sys.argv)<2 else int(sys.argv[1])
J_h, J_d, J_t, Spin, K_points = fs.get_pars(index)
print("Using parameters: (Jh,Jd,Jt)=(","{:.3f}".fomrat(J_h),",","{:.3f}".fomrat(J_d),",","{:.3f}".fomrat(J_t),"), S=","{:.3f}".format(Spin),", points in BZ=",str(K_points))
"""Define the BZ and the lattice vectors"""
Kx = K_points;     Ky = K_points
Kx_reference=13;   Ky_reference=13
kxg = np.linspace(0,4*np.pi/np.sqrt(21),Kx)
kyg = np.linspace(0,2*np.pi/np.sqrt(7),Ky)
k_grid = np.zeros((2,Kx,Ky))
for i in range(Kx):     #better way
    for j in range(Ky):
        k_grid[0,i,j] = kxg[i]
        k_grid[1,i,j] = kyg[j]
T1 = np.sqrt(7)*np.array([np.sqrt(3)/2,-1/2])
T2 = np.sqrt(7)*np.array([0,1])
KM = fs.compute_KM(k_grid,T1,T2)
"""Filenames"""
#ReferenceDir = fs.get_res_final_dn(Kx_reference,Ky_reference,txt_S,machine)
filename = fs.get_res_finalfn(pars,Kx,Ky,machine)
"""Initiate self consistent routine"""
pars = (J_h,J_d,J_t,Spin,KM)
Args_O = (k_grid,pars)
Args_L = (pars,pars_L)
P_initial = fs.get_P_initial(header)
"""Can actually use values of previous h point in phase diagram"""
new_O = P_initial;      old_O_1 = new_O;      old_O_2 = new_O
new_L = (L_bounds[0]+L_bounds[1])/2;       old_L_1 = 0;    old_L_2 = 0
#
initial_time = time()
step = 0
continue_loop = True
while continue_loop:
    step += 1
    converged_L = 0
    converged_O = 0
    #Update old L variables
    old_L_2 = float(old_L_1)
    old_L_1 = float(new_L)
    #Compute L with newO
    new_L = fs.compute_L(new_O,Args_L)
    #Update old O variables
    old_O_2 = np.array(old_O_1)
    old_O_1 = np.array(new_O)
    #Compute O with new_L
    temp_O = fs.compute_O_all(new_O,new_L,Args_O)
    #Mix with previous result
    mix_factor = 0.5
    for i in range(len(P_initial)):
        new_O[i] = old_O_1[i]*mix_factor + temp_O[i]*(1-mix_factor)
    #Check if L steady solution
    if np.abs(old_L_2-new_L)/Spin > cutoff_L:
        converged_L = True
    #Check if O steady solution
    for i in range(len(P_initial)):
        if np.abs(old_O_1[i]-new_O[i])/Spin > cutoff_O or np.abs(old_O_2[i]-new_O[i])/Spin > cutoff_O:
            converged_O = 0
            break
        if i == len(P_initial)-1:
            converged_O = True
    if converged_O and converged_L:
        continue_loop = False
        new_L = fs.compute_L(new_O,Args_L)
    #
    if disp:
        print("Step ",step,": ",new_L,*new_O,end='\n')
    #Margin in number of steps
    if step > MaxIter:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Exceeded number of steps!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        break
######################################################################################################
######################################################################################################
print("\nNumber of iterations: ",step,'\n')
conv = True if (converged_O and converged_L) else False
if not conv:
    print("\n\nFound final parameters NOT converged: ",new_L,new_O,"\n")
    exit()
if new_L < inp.L_bounds[0] + 0.01 or new_L > inp.L_bounds[1] - 0.01:
    print("Suspicious L value: ",new_L," NOT saving")
    exit()
################################################### Save solution
E,gap = fs.total_energy(new_O,new_L,Args_O)
if E == 0:
    print("Something wrong with the energy=",E)
    exit()
data = [J_nn,h,E,gap,new_L]
for i in range(len(P_initial)):
    data.append(new_O[i])
DataDic = {}
header = inp.header
for ind in range(len(data)):
    DataDic[header[ind]] = data[ind]
if save_to_file:
    sf.SaveToCsv(DataDic,csvfile)

print(DataDic)
print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################













































































