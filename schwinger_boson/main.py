import numpy as np
import functions as fs
from time import time
from pathlib import Path
import sys,os
import matplotlib.pyplot as plt

machine = fs.get_machine(os.getcwd())

save_to_file = True
disp = 1#True if machine=='loc' else False        #display
compute_ssf = True

"""Parameters of phase diagram"""
index = 0 if len(sys.argv)<2 else int(sys.argv[1])
J_h, J_d, J_t, Spin, K_points = fs.get_pars(index)
Js = (J_h,J_d,J_t)
print("Using parameters: (Jh,Jd,Jt)=(","{:.3f}".format(J_h),",","{:.3f}".format(J_d),",","{:.3f}".format(J_t),"), S=","{:.3f}".format(Spin),", points in BZ=",str(K_points))
"""Choice of ansatz"""
ansatz = 'C3a'        #see dic_mf_ans for defined anstaze
header_g,header_mf = fs.get_header(ansatz,Js)
number_random_ics = 1
initial_conditions = fs.get_initial_conditions(ansatz,header_mf,number_random_ics)
print("Using ansatz: ",ansatz," with classical and ",str(number_random_ics)," random initial conditions")
print("Header: ",header_g+header_mf)
"""Parameters of minimization"""
MaxIter = 3000
prec_L = 1e-10       #precision required in L maximization
L_method = 'Brent'
L_bounds = (0,50)       #bound of Lagrange multiplier
cutoff_L = 1e-4
pars_L = (prec_L,L_method,L_bounds)
cutoff_O = 1e-4
"""Define the BZ and the lattice vectors"""
Kx,Ky = K_points
k_grid = fs.get_k_grid(Kx,Ky)
if 0:   #Plot BZ points
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    for i in range(Kx):
        for j in range(Ky):
            ax.scatter(k_grid[0,i,j],k_grid[1,i,j],color='k',marker='s')
    plt.show()
    exit()
KM = fs.compute_KM(k_grid,fs.T1_,fs.T2_)
"""Filenames"""
#ReferenceDir = fs.get_res_final_dn(Kx_reference,Ky_reference,txt_S,machine)    #implement initial condition from previous result with less ks
pars_general = (Js,Spin,KM,ansatz)
filename_mf = fs.get_res_final_fn(pars_general,machine)
"""Initiate self consistent routine"""
initial_time = time()
if not Path(filename_mf).is_file():
    best_E = 1e10
    for ic in range(len(initial_conditions)):
        P_initial = initial_conditions[ic]
        print("Using initial condition:",ic)
        for i in range(len(header_mf)):
            print(header_mf[i],': ',P_initial[i])
        print('-----------------------------------------------------------')
        #
        new_O = P_initial;      old_O_1 = new_O;      old_O_2 = new_O
        new_L = (L_bounds[0]+L_bounds[1])/2;       old_L_1 = 0;    old_L_2 = 0
        #
        ansatz_initial_time = time()
        step = 0
        while True:
            step += 1
            converged_L = True
            converged_O = True
            #Update old L variables
            old_L_2 = float(old_L_1)
            old_L_1 = float(new_L)
            #Compute L with newO and tempO with newL
            mf_parameters = fs.get_mf_pars(new_O,header_mf,ansatz)
            if 0:   #check mf_parameters
                for i in range(10):
                    print(np.absolute(mf_parameters[i]),np.angle(mf_parameters[i])%(2*np.pi))
                input()
            new_L = fs.compute_L(mf_parameters,pars_general,pars_L)
            temp_O = fs.compute_O_all(mf_parameters,new_L,pars_general)
            #Update old O variables
            old_O_2 = np.array(old_O_1)
            old_O_1 = np.array(new_O)
            #Mix with previous result
            mix_factor = 0.8        #oscillates...
            mix_factor_p = 0
            for i in range(len(P_initial)):
                if header_mf[i][:3]=='arg':
                    new_O[i] = temp_O[i]%(2*np.pi)*(1-mix_factor_p)+old_O_1[i]*mix_factor_p
                else:
                    new_O[i] = old_O_1[i]*mix_factor + temp_O[i]*(1-mix_factor)
            #Check if L steady solution
            if np.abs(old_L_2-new_L)/Spin > cutoff_L:
                if disp:
                    print("L not converged by ","{:.8f}".format(np.abs(old_L_2-new_L)/Spin))
                converged_L = False
            #Check if O steady solution -> no check on phases, they oscillate too much
            for i in range(len(P_initial)):
                if header_mf[i][:3]=='arg':
                    continue
                if np.abs(old_O_1[i]-new_O[i])/Spin > cutoff_O or np.abs(old_O_2[i]-new_O[i])/Spin > cutoff_O:
                    converged_O = False
                    if disp:
                        print("O not converged in ",header_mf[i]," by ","{:.8f}".format(np.abs(old_O_1[i]-new_O[i])/Spin))
                    break
                if i == len(P_initial)-1:
                    converged_O = True
            #
            if disp:
                print("Step ",step)
                print("New L: ","{:.6f}".format(new_L))
                for i in range(len(header_mf)):
                    print(header_mf[i],': ',"{:.6f}".format(new_O[i]))
    #            print("energy: ",energy)
                print('\n')
    #            input()
            if converged_O and converged_L:
                print("Achieved convergence")
                final_mf_parameters = fs.get_mf_pars(new_O,header_mf,ansatz)
                final_L = fs.compute_L(mf_parameters,pars_general,pars_L)
                final_O = new_O
                go_to_next = False
                break
            #Margin in number of steps
            if step > MaxIter:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Exceeded number of steps!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                go_to_next = True
                break
            if new_L>10 and step>3:
                print("very bad, going to next ic")
                input()
                go_to_next = True
                break
        if go_to_next:
            continue
        ######################################################################################################
        ######################################################################################################
        print("\nNumber of iterations: ",step,'\n')
        if new_L < L_bounds[0] + 0.01 or new_L > L_bounds[1] - 0.01:
            print("Suspicious L value: ",new_L," NOT saving")
            exit()
        ################################################### Save solution
        Energy,Gap = fs.mf_energy(final_mf_parameters,final_L,pars_general)
        if Energy == 0:
            print("Something wrong with the energy=",Energy)
            exit()
        print("Final values of minimization:")
        print("Final L: ","{:.6f}".format(final_L))
        for i in range(len(header_mf)):
            print(header_mf[i],': ',"{:.6f}".format(final_O[i]))
        print("Energy: ","{:.6f}".format(Energy))
        print("Gap: ","{:.6f}".format(Gap))
        Dt = time()-ansatz_initial_time
        print("Time: ",str(Dt//60),"mins and ",Dt%60," seconds")
        print('-------------------------------------------------------\n')
        if Energy < best_E:
            best_E = Energy
            best_Gap = Gap
            best_mf = final_mf_parameters
            best_L = final_L
    if save_to_file:
        data_mf = np.zeros(len(best_mf)+3,dtype=complex)
        data_mf[:-3] = best_mf
        data_mf[-3] = best_L
        data_mf[-2] = best_E
        data_mf[-1] = best_Gap
        np.save(filename_mf,data_mf)
else:
    data_mf = np.load(filename_mf)
    best_mf = data_mf[:-3]
    best_L = data_mf[-3]
    best_E = data_mf[-2]
    best_Gap = data_mf[-1]
    print("Energy: ","{:.6f}".format(best_E))
    print("Gap: ","{:.6f}".format(best_Gap))

if compute_ssf:
    """Compute ssf"""
    print("Computing SSF")
    if best_Gap < 1e-3:
        """LRO structure factor. Use method of spins from shape of condensate."""
        print("using gap closing points")
        #Get condensate
        gap_closing_K,columns_V,degenerate = fs.find_gap_closing_points(best_mf,best_L,pars_general)
        UC = 30
        spin_lattice = fs.compute_spin_lattice(gap_closing_K,columns_V,degenerate,UC)
        print("computing ssf")
        SSFzz,SSFxy = fs.spin_structure_factor(spin_lattice)
        #Plot
        fs.plot_ssf(SSFzz,SSFxy)
    else:
        """Structure factor from Bogoliubov form."""
        print("from Bogoliubov bosons")
exit()

"""Save result"""
if save_to_file:
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












































































