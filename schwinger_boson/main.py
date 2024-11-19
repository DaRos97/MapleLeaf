import numpy as np
import scipy.linalg as LA
import functions as fs
from time import time
from pathlib import Path
import sys,os
import matplotlib.pyplot as plt

machine = fs.get_machine(os.getcwd())

save_solution = False if machine=='loc' else True
save_fig = False if machine=='loc' else True
disp = True if machine=='loc' else False        #display
compute_ssf = True

"""Parameters of phase diagram"""
index = 0 if len(sys.argv)<3 else int(sys.argv[2])
J_h, J_d, J_t, Spin, K_points = fs.get_pars(index)
Js = (J_h,J_d,J_t)
print("Using parameters: (Jh,Jd,Jt)=(","{:.3f}".format(J_h),",","{:.3f}".format(J_d),",","{:.3f}".format(J_t),"), S=","{:.3f}".format(Spin),", points in BZ=",str(K_points))
"""Choice of ansatz"""
ansatz = 'C6_5' if len(sys.argv)<2 else sys.argv[1]
header_g,header_mf = fs.get_header(ansatz,Js)
number_random_ics = 1
initial_conditions = fs.get_initial_conditions(ansatz,header_mf,number_random_ics)
print("Using ansatz: ",ansatz," with classical and ",str(number_random_ics)," random initial conditions")
print("Header: ",header_g+header_mf)
print('-------------------------------------------------------------\n')
"""Parameters of minimization"""
MaxIter = 3000
prec_L = 1e-10       #precision required in L maximization
L_method = 'Brent'
L_bounds = (0,100)       #bound of Lagrange multiplier
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
    n_conv = 0  #number of converged results
    best_E = 1e10
    for ic in range(len(initial_conditions)):
        P_initial = initial_conditions[ic]
        print("Initial condition:",ic)
        if disp:
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
                    print(fs.pars_mf_names[i],': ',np.absolute(mf_parameters[i]),np.angle(mf_parameters[i])%(2*np.pi))
                input()
            new_L = fs.compute_L(mf_parameters,pars_general,pars_L)
            temp_O = fs.compute_O_all(mf_parameters,new_L,pars_general)
            if len(temp_O) == 1:
                go_to_next = True
                break
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
                n_conv += 1
                print("Achieved convergence ",n_conv,"/",ic+1)
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
            if new_L>90:
                print("very bad, going to next ic")
                input()
                go_to_next = True
                break
        if go_to_next:
            continue
        ######################################################################################################
        ######################################################################################################
        print("\nNumber of iterations: ",step,'\n')
        Dt = time()-ansatz_initial_time
        print("Time: ",str(Dt//60),"mins and ",Dt%60," seconds")
        print('-------------------------------------------------------\n')
        Energy,Gap = fs.mf_energy(final_mf_parameters,final_L,pars_general)
        if Energy < best_E:
            best_E = Energy
            best_Gap = Gap
            best_mf = final_mf_parameters
            best_L = final_L
        if disp:
            if new_L < L_bounds[0] + 0.01 or new_L > L_bounds[1] - 0.01:
                print("Suspicious L value: ",new_L," NOT saving")
                continue
            ################################################### Save solution
            if Energy == 0:
                print("Something wrong with the energy=",Energy)
                continue
            print("Final values of minimization ",ic,":")
            print("Final L: ","{:.6f}".format(final_L))
            for i in range(len(header_mf)):
                print(header_mf[i],': ',"{:.6f}".format(final_O[i]))
            print("Energy: ","{:.6f}".format(Energy))
            print("Gap: ","{:.6f}".format(Gap))
    if best_E==1e10:
        print("Not a single initial condition converged, NOT saving")
    elif save_solution:
        data_mf = np.zeros(len(best_mf)+3,dtype=complex)
        data_mf[:-3] = best_mf
        data_mf[-3] = best_L
        data_mf[-2] = best_E
        data_mf[-1] = best_Gap
        np.save(filename_mf,data_mf)
    print("Finished minimizations")
    print("-----------------------------------------------------------\n")
else:
    data_mf = np.load(filename_mf)
    best_mf = data_mf[:-3]
    best_L = data_mf[-3]
    best_E = data_mf[-2]
    best_Gap = data_mf[-1]
    print("Loaded best solution which has")
    print("Lagrange multiplier L: ","{:.6f}".format(best_L))
    for i in range(len(best_mf)):
        print(fs.pars_mf_names[i],': ',"{:.6f}".format(best_mf[i]))
    print("Energy: ","{:.6f}".format(best_E))
    print("Gap: ","{:.6f}".format(best_Gap))

if compute_ssf:
    """Compute ssf"""
    print("Computing SSF")
    if 1 and best_Gap < 1e-3:
        """LRO structure factor. Use method of spins from shape of condensate."""
        print("using gap closing points")
        #Get condensate
        gap_closing_K,columns_V,degenerate = fs.find_gap_closing_points(best_mf,best_L,pars_general)
        UC = 30
        spin_lattice = fs.compute_spin_lattice(gap_closing_K,columns_V,degenerate,UC)
        print("computing ssf")
        SSFzz,SSFxy = fs.spin_structure_factor(spin_lattice)
        #Plot
        figure_fn = '' if machine=='loc' else fs.get_figure_fn(pars_general,machine)
        fs.plot_ssf(SSFzz,SSFxy,figure_fn)
    else:
        """Structure factor from Bogoliubov form."""
        Kx = 16     #points for summation over BZ
        Ky = 12
        k_grid = fs.get_k_grid(Kx,Ky)
        Nx = 16    #points of SSF to compute in BZ (Q)
        Ny = 12
        Q_grid = fs.get_k_grid(Nx,Ny)
        D = np.array([          #matrix of positions in UC
                      [np.sqrt(3)/2, 1/2],
                      [0, 1],
                      [0, -1],
                      [np.sqrt(3)/2, -1/2],
                      [-np.sqrt(3)/2, -1/2],
                      [-np.sqrt(3)/2, 1/2],
        ])
        #Result store
        SSFxx = np.zeros((Nx,Ny))
        #M and N at K, -K, K+Q, -K-Q
        k_grid_list = [k_grid, -k_grid, k_grid+Q_grid, -k_grid-Q_grid]
        MMs = []
        m = 6
        J_ = np.identity(2*m)
        for i in range(m):
            J_[i,i] = -1
        for k_g in k_grid_list:
            KMg = fs.compute_KM(k_g,fs.T1_,fs.T2_)
            pars_general_g = (Js,Spin,KMg,ansatz)
            matrix_N = fs.big_Nk(best_mf,best_L,pars_general_g)
            matrix_M = np.zeros(matrix_N.shape,dtype=complex)
            for i in range(Kx):
                for j in range(Ky):
                    N_k = matrix_N[:,:,i,j]
                    try:
                        Ch = LA.cholesky(N_k,check_finite=False)
                        w0,U = LA.eigh(Ch@J_@Ch.T.conj())
                        w = np.diag(np.sqrt(J_@w0))
                        matrix_M[:,:,i,j] = LA.inv(Ch)@U@w
                    except:
                        print("Error in Cholesky....")
                        exit()
            MMs.append(matrix_M)
        #Compute Xi(Q) for Q in BZ
        for xx in range(Nx*Ny):
            ii = xx//Nx
            ij = xx%Ny
            #
            delta = np.zeros((6,6),dtype=complex)
            for u in range(m):
                for g in range(m):
                    delta[u,g] = np.exp(1j*np.dot(Q_grid[:,ii,ij],D[g]-D[u]))
            #
            resxy = 0
            #summation over BZ
            for x in range(Kx*Ky):
                i = x//Kx
                j = x%Ky
                #
                U1,X1,V1,Y1 = fs.split(MMs[0][:,:,i,j],6,6)
                U2,X2,V2,Y2 = fs.split(MMs[1][:,:,i,j],6,6)
                U3,X3,V3,Y3 = fs.split(MMs[2][:,:,i,j],6,6)
                U4,X4,V4,Y4 = fs.split(MMs[3][:,:,i,j],6,6)
                ##############################################
                temp1 = np.einsum('ua,ga->ug',np.conjugate(X1),X1) * np.einsum('ua,ga->ug',np.conjugate(Y4),Y4)
                temp2 = np.einsum('ua,ga->ug',np.conjugate(X1),Y1) * np.einsum('ua,ga->ug',np.conjugate(Y4),X4)
                temp3 = np.einsum('ua,ga->ug',V2,np.conjugate(V2)) * np.einsum('ua,ga->ug',U3,np.conjugate(U3))
                temp4 = np.einsum('ua,ga->ug',V2,np.conjugate(U2)) * np.einsum('ua,ga->ug',U3,np.conjugate(V3))
                temp = (temp1 + temp2 + temp3 + temp4) * delta
                resxy += temp.ravel().sum()
            #
            SSFxx[ii,ij] = np.real(resxy)/(Kx*Ky)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        X,Y = np.meshgrid(np.linspace(0,1,Ny),np.linspace(0,1,Nx))
        ax.plot_surface(X,Y,SSFxx)
        plt.show()
        print("from Bogoliubov bosons -> gapped solution")

Tt = time()-initial_time
print("Time: ",str(Tt//60),"mins and ",Tt%60," seconds")












































































