import numpy as np
from scipy import linalg as LA
from scipy.optimize import minimize_scalar
from scipy.interpolate import RectBivariateSpline as RBS
from pathlib import Path
import csv
import os
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm

def get_pars(index):
    """Get parameters of Hamiltonian."""
    lJh = [1,]
    lJd = [1,]#0,-1]#np.linspace(0,-4,10)
    lJt = [0,]#np.linspace(0,-4,10)
    lSpin = [0.5,]#(np.sqrt(3)+1)/2,0.3,0.2]
    lKpoints = [(16,12),(32,24)]
    ll = [lJh,lJd,lJt,lSpin,lKpoints]
    combs = list(itertools.product(*ll))
    return combs[index]

"""Ansatze mean field parameters. All parameters in the list are subject to minimization."""
dic_mf_ans = {
    'C6a':      ['Ah','At','argAt','Ad','argAd','Bh','argBh','Bt','argBt','Bd','argBd'],
    'C6ar':     ['Ah','At','Ad','Bh','Bt','Bd'],
    'C3a':      ['Ah','Ahp','argAhp','At','argAt','Atp','argAtp','Ad','argAd','Bh','argBh','Bhp','argBhp','Bt','argBt','Btp','argBtp','Bd','argBd'],
    'C3ar':     ['Ah','Ahp','At','Atp','Ad','Bh','Bhp','Bt','Btp','Bd'],
}
"""Classical orders compatible with each ansatz."""
dic_compatible_orders = {
    'C6a':  ['Neel'],
    'C6ar': ['Neel'],
    'C3a':  ['Neel'],
    'C3ar': ['Neel']
}
"""Mean field parameters of classical orders."""
classical_orders_MF = {
    'FM':
    {'Ah':0,'Ahp':0,'argAhp':0,'At':0,'argAt':0,'Atp':0,'argAtp':0,'Ad':0,'argAd':0,
     'Bh':1/2,'argBh':0,'Bhp':1/2,'argBhp':0,'Bt':1/2,'argBt':0,'Btp':1/2,'argBtp':0,'Bd':1/2,'argBd':0,},
    'Neel':
    {'Ah':1/2,'Ahp':1/2,'argAhp':np.pi,'At':0,'argAt':0,'Atp':0,'argAtp':0,'Ad':1/2,'argAd':0,
     'Bh':0,'argBh':0,'Bhp':0,'argBhp':0,'Bt':1/2,'argBt':0,'Btp':1/2,'argBtp':0,'Bd':0,'argBd':0,},
    'Coplanar1':
    {'Ah':1/2,'Ahp':1/2,'argAhp':np.pi,'At':np.sqrt(5)/4,'argAt':0,'Atp':np.sqrt(5)/4,'argAtp':np.pi,'Ad':1/2,'argAd':np.pi,
     'Bh':1/2,'argBh':0,'Bhp':1/2,'argBhp':0,'Bt':1/4,'argBt':np.pi,'Btp':1/4,'argBtp':np.pi,'Bd':1/2,'argBd':0,},
    'Coplanar2':
    {'Ah':1/2,'Ahp':1/2,'argAhp':np.pi,'At':0,'argAt':0,'Atp':np.sqrt(5)/4,'argAtp':0,'Ad':1/2,'argAd':0,
     'Bh':1/2,'argBh':0,'Bhp':1/2,'argBhp':np.pi,'Bt':1/2,'argBt':0,'Btp':1/2,'argBtp':np.pi,'Bd':1/2,'argBd':np.pi,},
}

def compute_L(mf_parameters,pars_general,pars_L):
    """Evaluate best L by maximizing energy wrt L."""
    res = minimize_scalar(lambda l: mf_energy(mf_parameters,l,pars_general,optimize_L=True),  #maximize energy wrt L with fixed P
            method = pars_L[1],          #can be 'bounded' or 'Brent'
            bracket = pars_L[2],
            options={'xtol':pars_L[0]}
            )
    return res.x        #Lambda

def mf_energy(mf_parameters,L,pars_general,optimize_L=False):
    """Computes the energy  using a Bogoliubov transformation.
    If optimize_L is True, only terms containing L are included and is returned -Energy -> maximization.
    """
    Js,Spin,KM,ansatz = pars_general
    Kx,Ky = KM[0].shape
    m = 6   #UC size
    J_ = np.identity(2*m)
    for i in range(m):
        J_[i,i] = -1
    #Check on what value of L is used -> sometimes is a bit stupid
    VL = 100
    if L < 0:
        Res = -VL-(0-L)
        return -Res
    elif L > 50:
        Res = -VL-(L-50)
        return -Res
    #L energy
    Res = -L*(2*Spin+1)
    if not optimize_L:
        Res += O_energy(mf_parameters,pars_general)
    #Compute now the part of the energy coming from the Hamiltonian matrix by the use of a Bogoliubov transformation
    matrix_N = big_Nk(mf_parameters,L,pars_general)
    res = np.zeros((m,Kx,Ky))
    for i in range(Kx):
        for j in range(Ky):
            Nk = matrix_N[:,:,i,j]
            try:
                Ch = LA.cholesky(Nk,check_finite=False)
            except LA.LinAlgError:
                if optimize_L:
                    return -Res+VL-L
                else:
                    print("Matrix is not pos def for some K at energy evaluation.")
                    exit()
            res[:,i,j] = np.linalg.eigvalsh(Ch.T.conj()@J_@Ch)[m:]      #BOTTLE NECK -> compute the eigevalues
    gap = np.amin(res[0].ravel())
    #Now fit the energy values found with a spline curve in order to have a better solution
    if 0:
        r2 = 0
        for i in range(m):
            func = RBS(np.linspace(0,1,Kx),np.linspace(0,1,Ky),res[i])
            r2 += func.integral(0,1,0,1)        #integrate the fitting curves to get the energy of each band
    #        input()
            if 0:
                print(func.integral(0,1,0,1),res[i].ravel().sum()/Kx/Ky)
                kx = np.linspace(0,1,Kx)
                ky = np.linspace(0,1,Ky)
                X,Y = np.meshgrid(ky,kx)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.plot_surface(X,Y,res[i])
                ax.set_title("L:"+"{:.3f}".format(L)+", ind: "+str(i))
                plt.show()
        r2 /= m                             #normalize
    else:
        r2 = np.sum(res)/m/Kx/Ky
    #
    result = Res+r2
    if 0 and not optimize_L:
        print("Energy vals: E_l=",-L*(2*Spin+1),", E_O=",O_energy(mf_parameters,pars_general),"E_w=",r2)
        print("Total=",result)
    return -result if optimize_L else (result,gap)

def big_Nk(mf_parameters,L,pars_general):
    """Hamiltonian matrix for Bogoliubov transformation"""
    Js,Spin,KM,ansatz = pars_general
    J_h,J_d,J_t = Js
    m = 6
    KT1,KT1_,KT2,KT2_,KT12,KT12_ = KM
    Kx,Ky = KT1.shape
    Ah,Ahp,At,Atp,Ad,Bh,Bhp,Bt,Btp,Bd = mf_parameters
    ################
    alpha = np.zeros((m,m,Kx,Ky), dtype=complex)
    beta = np.zeros((m,m,Kx,Ky), dtype=complex)
    delta = np.zeros((m,m,Kx,Ky), dtype=complex)
    #
    alpha[0,1] = J_h/2*Bhp.conj()
    alpha[0,2] = J_t/2*Bt*KT1
    alpha[0,3] = J_h/2*Bh*KT2_
    alpha[0,4] = J_d/2*Bd
    alpha[0,5] = J_t/2*Bt.conj()*KT2_
    alpha[1,2] = J_d/2*Bd.conj()
    alpha[1,3] = J_t/2*Btp
    alpha[1,4] = J_t/2*Btp.conj()
    alpha[1,5] = J_h/2*Bh.conj()*KT12_
    alpha[2,3] = J_h/2*Bhp.conj()
    alpha[2,4] = J_h/2*Bh*KT1_
    alpha[2,5] = J_t/2*Bt*KT12_
    alpha[3,4] = J_t/2*Btp
    alpha[3,5] = J_d/2*Bd.conj()
    alpha[4,5] = J_h/2*Bhp
    alpha += np.conjugate(np.transpose(alpha,axes=(1,0,2,3)))
    #just change B*->B
    delta[0,1] = J_h/2*Bhp
    delta[0,2] = J_t/2*Bt.conj()*KT1
    delta[0,3] = J_h/2*Bh.conj()*KT2_
    delta[0,4] = J_d/2*Bd.conj()
    delta[0,5] = J_t/2*Bt*KT2_
    delta[1,2] = J_d/2*Bd
    delta[1,3] = J_t/2*Btp.conj()
    delta[1,4] = J_t/2*Btp
    delta[1,5] = J_h/2*Bh*KT12_
    delta[2,3] = J_h/2*Bhp
    delta[2,4] = J_h/2*Bh.conj()*KT1_
    delta[2,5] = J_t/2*Bt.conj()*KT12_
    delta[3,4] = J_t/2*Btp.conj()
    delta[3,5] = J_d/2*Bd
    delta[4,5] = J_h/2*Bhp.conj()
    delta += np.conjugate(np.transpose(delta,axes=(1,0,2,3)))
    #need to do both sides here
    beta[0,1] = -J_h/2*Ahp
    beta[0,2] =  J_t/2*At*KT1
    beta[0,3] =  J_h/2*Ah*KT2_
    beta[0,4] =  J_d/2*Ad
    beta[0,5] = -J_t/2*At*KT2_
    beta[1,2] = -J_d/2*Ad
    beta[1,3] =  J_t/2*Atp
    beta[1,4] = -J_t/2*Atp
    beta[1,5] = -J_h/2*Ah*KT12_
    beta[2,3] = -J_h/2*Ahp
    beta[2,4] =  J_h/2*Ah*KT1_
    beta[2,5] =  J_t/2*At*KT12_
    beta[3,4] =  J_t/2*Atp
    beta[3,5] = -J_d/2*Ad
    beta[4,5] =  J_h/2*Ahp
    #Other side has overall - and k->-k
    beta[1,0] =  J_h/2*Ahp
    beta[2,0] = -J_t/2*At*KT1_
    beta[3,0] = -J_h/2*Ah*KT2
    beta[4,0] = -J_d/2*Ad
    beta[5,0] =  J_t/2*At*KT2
    beta[2,1] =  J_d/2*Ad
    beta[3,1] = -J_t/2*Atp
    beta[4,1] =  J_t/2*Atp
    beta[5,1] =  J_h/2*Ah*KT12
    beta[3,2] =  J_h/2*Ahp
    beta[4,2] = -J_h/2*Ah*KT1
    beta[5,2] = -J_t/2*At*KT12
    beta[4,3] = -J_t/2*Atp
    beta[5,3] =  J_d/2*Ad
    beta[5,4] = -J_h/2*Ahp
    #
    final_N = np.zeros((2*m,2*m,Kx,Ky), dtype=complex)
    final_N[:m,:m] = alpha
    final_N[m:,m:] = delta
    final_N[:m,m:] = beta
    final_N[m:,:m] = np.conjugate(np.transpose(beta,axes=(1,0,2,3)))
    #################################### L
    for i in range(2*m):
        final_N[i,i] += L
    return final_N

def compute_O_all(mf_parameters,L,pars_general):
    """Computes expectation value of bond operators."""
    Js,Spin,KM,ansatz = pars_general
    Kx,Ky = KM[0].shape
    m = 6
    J_ = np.identity(2*m)
    for i in range(m):
        J_[i,i] = -1
    #Compute first the transformation matrix M at each needed K -> pt iv of appendix A of PRB 87, 125127 (2013)
    matrix_N = big_Nk(mf_parameters,L,pars_general)
    matrix_M = np.zeros(matrix_N.shape,dtype=complex)
    for i in range(Kx):
        for j in range(Ky):
            N_k = matrix_N[:,:,i,j]
            Ch = LA.cholesky(N_k,check_finite=False)
            w0,U = LA.eigh(Ch@J_@Ch.T.conj())
            w = np.diag(np.sqrt(J_@w0))
            matrix_M[:,:,i,j] = LA.inv(Ch)@U@w
    #For each parameter need to know what it is
    header_mf = get_header(ansatz,Js)[1]
    new_O = np.zeros(len(header_mf))
    k_grid = get_k_grid(Kx,Ky)
    pA = 0  #Phase of Ah -> use to correct others if it is nonzero
    for p in range(len(header_mf)):
        par_name = header_mf[p]
        func_O = compute_A if par_name[0]=='A' else compute_B
        if par_name[:3]=='arg':
            continue    #skip phases, get them from amplitudes
        li_,lj_ = dic_indexes[str(m)][par_name[1:]]         #indices of sites over which to evaluate the bond
        #Create a matrix on the BZ. We evaluate the MF par as in eq.20 of SBMFT paper: integrate over BZ.
        mf_k = np.zeros((Kx,Ky),dtype=complex)
        for i in range(Kx):
            for j in range(Ky):
                U,X,V,Y = split(matrix_M[:,:,i,j],m,m)
                mf_k[i,j] = func_O(U,X,V,Y,li_,lj_,k_grid[:,i,j])
        #
        inds = np.argwhere(np.absolute(mf_k)>1e2)
        if len(inds)>0:
            mf_k[inds[0,0],inds[0,1]] = 0
        #
        #Interpolate and integrate real and imaginary part
        interI = RBS(np.linspace(0,1,Kx),np.linspace(0,1,Ky),np.imag(mf_k))
        res2I = interI.integral(0,1,0,1)
#        res2I = np.imag(mf_k[~np.isnan(mf_k)]).sum()/Kx/Ky
        interR = RBS(np.linspace(0,1,Kx),np.linspace(0,1,Ky),np.real(mf_k))
        res2R = interR.integral(0,1,0,1)
#        res2R = np.real(mf_k[~np.isnan(mf_k)]).sum()/Kx/Ky
        res = (res2R+1j*res2I)/2        #/2 comes from formula eq.20
        #
        new_O[p] = np.absolute(res)         #if the phase of this amplitude is fixed by the ansatz need to check that it is correct
        if 0 and par_name=='Ah':
            pA = np.angle(res)
        if 'arg'+par_name in header_mf:
            new_O[p+1] = np.angle(res)+pA if new_O[p]>1e-3 else 0        #normalize maybe between 0 and 2*pi
        if 0 and par_name=='Ah':
            print("Calculation raw of ",par_name,": ",res)
            kx = np.linspace(0,1,Kx)
            ky = np.linspace(0,1,Ky)
            X,Y = np.meshgrid(ky,kx)
            fig = plt.figure(figsize = (20,20))
            ax = fig.add_subplot(projection='3d')
#            mf_k[5,4] = mf_k[11,8] = np.nan
            ax.plot_surface(X,Y,np.absolute(mf_k),cmap=cm.coolwarm)
#            ax2 = fig.add_subplot(122,projection='3d')
#            ax2.plot_surface(X,Y,np.imag(mf_k),cmap=cm.plasma)
            ax.set_title(par_name)
            plt.show()
    return new_O

"""For each unit cell size (6 or 12), gives the indexes of the sites in the UC
over which to evaluate the bond expectation value"""
dic_indexes =   {'6': {'h': (3,0), 'hp': (0,1),
                       't': (2,0), 'tp': (1,4),
                       'd': (1,2)},
                 '12':{'1': (1,2), '1p': (2,0), #not yet done
                       '2': (1,0), '2p': (5,1),
                       '3': (4,1)}
                 }
def compute_A(U,X,V,Y,li_,lj_,K__):
    """Compute expectation value of A_ij as from eq.20 of paper SBMFT, for a given k."""
    if (li_,lj_) in [(3,0),]:
        dist = -np.sqrt(7)*np.array([0,1])
    elif (li_,lj_) in [(2,0),]:
        dist = np.sqrt(7)*np.array([np.sqrt(3)/2,-1/2])
    else:
        dist = np.zeros(2)
    V_ = V.T.conj()
    Y_ = Y.T.conj()
    return (np.einsum('ln,nm->lm',U,V_)[li_,lj_]   *np.exp( 1j*np.dot(K__,dist))
          - np.einsum('nl,mn->lm',Y_,X)[li_,lj_]   *np.exp(-1j*np.dot(K__,dist)))
########
def compute_B(U,X,V,Y,li_,lj_,K__):
    """Compute expectation value of B_ij as from eq.20 of paper SBMFT, for a given k."""
    if (li_,lj_) in [(3,0),]:
        dist = -np.sqrt(7)*np.array([0,1])
    elif (li_,lj_) in [(2,0),]:
        dist = np.sqrt(7)*np.array([np.sqrt(3)/2,-1/2])
    else:
        dist = np.zeros(2)
    V_ = V.T.conj()
    X_ = X.T.conj()
    return (np.einsum('nl,mn->lm',X_,X)[li_,lj_]   *np.exp(-1j*np.dot(K__,dist))
          + np.einsum('ln,nm->lm',V,V_)[li_,lj_]   *np.exp( 1j*np.dot(K__,dist)))

def split(array, nrows, ncols):
    """Split 2m*2m matrix in 4 matrices of m*m."""
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

def O_energy(mf_parameters,pars_general):
    """Compute k-independent part of GS energy coming from mod square of mf parameters."""
    Js,Spin,KM,ansatz = pars_general
    J_h,J_d,J_t = Js
    Ah,Ahp,At,Atp,Ad,Bh,Bhp,Bt,Btp,Bd = np.absolute(mf_parameters)**2
    #Check
    result = J_h*(Ah+Ahp-Bh-Bhp) + J_d/2*(Ad-Bd) + J_t*(At+Atp-Bt-Btp)
    return result

def get_header(ansatz,Js):
    """Here we give the header of the solution, which changes with the ansatz since there are different MF parameters.
    All the mean field parameters that enter here are subject to minimization.
    --------------------------------------
    Possible ansatze are:
        - C6a: 6-fold symmetric, so i.e. Ah = Ahp. 'a' is for p1=0 -> 6 sites unit cell
        - C6arN: same as C6a but taylored for Neel: At,Bh,Bd=0 and phases of Ah,Ad,Bt fixed
        - C3a: 3-fold symmetric, no constraints at all -> most general
        - C3ar: real C3a
    """
    header_g = ['Jh','Jd','Jt','Energy','Gap','L']
    header_mf = []
    for temp in dic_mf_ans[ansatz]:
        if consider_bond(temp,Js):
            header_mf.append(temp)
    return header_g,header_mf

def consider_bond(temp,Js):
    """
    Condition to consider the mean field parameters of a bond is that its amplitude does not vanish.
    """
    J_h,J_d,J_t = Js
    if ('h' in temp and abs(J_h)>1e-3) or ('d' in temp and abs(J_d)>1e-3) or ('t' in temp and abs(J_t)>1e-3):
        return True
    return False

def get_initial_conditions(ansatz,header_mf,number_random_ic):
    """Gets the list of initial conditions to use depending on the ansatz.
    It takes all the compatible classical orders plus some random initial conditions."""
    ics = []
    for co in dic_compatible_orders[ansatz]:
        ics.append(format_initial_condition(classical_orders_MF[co],header_mf))
    for n in range(number_random_ic):
        ics.append(random_ic(header_mf))
    return ics

def random_ic(header_mf):
    """Here we give some random initial condition.
    0 to 1 for amplitudes and 0 to 1 for phases."""
    res = []
    for i in range(len(header_mf)):
        par = np.random.random()*2*np.pi if header_mf[i][:3]=='arg' else np.random.random()*1
        res.append(par)
    return res

def format_initial_condition(mf,header_mf):
    """Takes dictionary of mf parameters and formats it depending on the ansatz.
    We need it because we need the exact number of mf parameters given by the ansatz."""
    res = []
    for i in range(len(header_mf)):
        res.append(mf[header_mf[i]])
    return res

def get_mf_pars(P,header_mf,ansatz):
    """Computes the list of A,Ap,.. ecc given the list 'P' of free parameters and the ansatz(header).
    This is more complicated than it should.
    """
    full_list_mf = list(classical_orders_MF['FM'].keys())
    res = np.zeros(10,dtype=complex)    #there are 10 complex mf parameters in total
    if ansatz == 'C6a': #Need to compare header_mf to account for possibly Js=0
        res[0] = P[0]*np.exp(1j*0)
        ind_res = 2
        ind_p = 1
        for mf in full_list_mf[3:]:
            if not mf[:3] == 'arg':
                if mf in header_mf:
                    res[ind_res] = P[ind_p]*np.exp(1j*P[ind_p+1])
                    ind_p += 2
                ind_res += 1
        res[1] = -res[0] #Ahp = Ah
        res[3] = res[2] #Atp = At
        res[6] = res[5] #Bhp = Bh
        res[8] = res[7] #Btp = Bt
    elif ansatz == 'C6ar':
        res[0] = P[0]
        ind_res = 1
        ind_p = 1
        for mf in full_list_mf[1:]:
            if not mf[:3] == 'arg':
                if mf in header_mf:
                    res[ind_res] = P[ind_p]
                    ind_p += 1
                ind_res += 1
        res[1] = -res[0] #Ahp = Ah
        res[3] = res[2] #Atp = At
        res[6] = res[5] #Bhp = Bh
        res[8] = res[7] #Btp = Bt
    elif ansatz == 'C3a':
        res[0] = P[0]
        ind_res = 1
        ind_p = 1
        for mf in full_list_mf[1:]:
            if not mf[:3] == 'arg':
                if mf in header_mf:
                    res[ind_res] = P[ind_p]*np.exp(1j*P[ind_p+1])
                    ind_p += 2
                ind_res += 1
    elif ansatz == 'C3ar':
        res[0] = P[0]
        ind_res = 1
        ind_p = 1
        for mf in full_list_mf[1:]:
            if not mf[:3] == 'arg':
                if mf in header_mf:
                    res[ind_res] = P[ind_p]
                    ind_p += 1
                ind_res += 1
        res[1] *= -1
    return res

def get_k_grid(Kx,Ky):
    """Compute grid of BZ points. Here I'm taking the real BZ (rectangular is more practical but wth)."""
    kxg = np.linspace(0,1,Kx,endpoint=False)
    kyg = np.linspace(0,1,Ky,endpoint=False)
    k_grid = np.zeros((2,Kx,Ky))
    for i in range(Kx):     #there is a better way
        for j in range(Ky):
            k_grid[0,i,j] = (kyg[j]+2*kxg[i])*2*np.pi/np.sqrt(3)/np.sqrt(7)
            k_grid[1,i,j] = kyg[j]*2*np.pi/np.sqrt(7)
    return k_grid

def compute_KM(k_grid,T1,T2):
    """Compute the product exp(i k . T) of momenta in BZ and lattice directions."""
    kT1 = np.exp(1j*np.tensordot(T1,k_grid,axes=1));   kT1_ = np.conjugate(kT1);
    kT2 = np.exp(1j*np.tensordot(T2,k_grid,axes=1));   kT2_ = np.conjugate(kT2);
    kT12 = np.exp(1j*np.tensordot(T1+T2,k_grid,axes=1));   kT12_ = np.conjugate(kT12);
    KM_ = (kT1,kT1_,kT2,kT2_,kT12,kT12_)
    return KM_

def get_P_initial(header):
    res = []
    for i in range(len(header)-6):
        if header[i+6][:3]=='arg':
            res.append(0)
        else:
            res.append(1/2)
    return np.array(res)

#################################################################
#################################################################
#################################################################
#################################################################
"""
Spin structure factor functions.
"""
def find_gap_closing_points(mf_parameters,L,pars_general):
    """Returns the indexes of k_grid where the gap closes.
    Still need to implement check on close points and points related by reciprocal lattice vectors.
    """
    Js,Spin,KM,ansatz = pars_general
    Kx,Ky = KM[0].shape
    m = 6   #UC size
    J_ = np.identity(2*m)
    for i in range(m):
        J_[i,i] = -1
    en = np.zeros((Kx,Ky))
    matrix_N = big_Nk(mf_parameters,L,pars_general)
    for i in range(Kx):
        for j in range(Ky):
            Nk = matrix_N[:,:,i,j]
            Ch = LA.cholesky(Nk,check_finite=False)
            en[i,j] = np.linalg.eigvalsh(Ch.T.conj()@J_@Ch)[m]
    #
    k_grid = get_k_grid(Kx,Ky)
    X,Y = np.meshgrid(np.linspace(0,1,Ky),np.linspace(0,1,Kx))
    g = np.argsort(en.ravel())
    k_list = []
    for i in range(2):
        k_list.append([g[i]//Ky,g[i]%Ky])
    #Check that the two minima are not the same point 
    dist = np.linalg.norm(k_grid[:,k_list[0][0],k_list[0][1]]-k_grid[:,k_list[1][0],k_list[1][1]])
    if (dist <= min([np.linalg.norm(k_grid[:,0,0]-k_grid[:,0,1]),np.linalg.norm(k_grid[:,0,0]-k_grid[:,1,0])]) or
        abs(dist-np.linalg.norm(B_[0]))):
        k_list = [k_list[0],]
    if 0:#plot points for additional checking
        fig = plt.figure(figsize=(20,15))
        ax = fig.add_subplot(121,projection='3d')
        ax.plot_surface(X,Y,en,cmap=cm.plasma)
        #
        print("new gap found ",np.amin(en.ravel()))
        ax = fig.add_subplot(122)
        sc = ax.scatter(k_grid[0],k_grid[1],c=en,cmap = cm.plasma)
        plt.colorbar(sc)
        for k in k_list:
            kk = k_grid[:,k[0],k[1]]
            ax.scatter(kk[0],kk[1],c='g',marker='*')
            print('new: ',k)
        plt.show()
    #Now extract the eienvectors relatie to these k-pints
    columns_V = []
    degenerate = False
    for k in k_list:
        Nk = matrix_N[:,:,k[0],k[1]]
        Ch = LA.cholesky(Nk,check_finite=False)
        w0,U = LA.eigh(Ch@J_@Ch.T.conj())
        w = np.diag(np.sqrt(J_@w0))
        Mk = LA.inv(Ch)@U@w
        columns_V.append(Mk[:,m-1])
        if np.abs(w0[m]-w0[m+1]) < 1e-3:          #degeneracy case -> same K
            print("degenerate K point")
            columns_V.append(Mk[:,m+1])
            degenerate = True
    return k_list,columns_V,degenerate

def compute_spin_lattice(gap_closing_K,columns_V,degenerate,UC):
    #Construct lattice
    spin_lattice = np.zeros((UC,UC,6,3))
    #Pauli matrices
    s_ = np.zeros((3,2,2),dtype = complex)
    s_[0] = np.array([[0,1],[1,0]])     #x
    s_[1] = np.array([[0,-1j],[1j,0]])  #y
    s_[2] = np.array([[1,0],[0,-1]])    #z
    if degenerate:                   #FIX degeneracy with multi gap-closing points!!!!!!!!!!!!!!!!!
        if len(gap_closing_K) > 1:
            print("Not supported multi-gap closing points with degeneracy")
            exit()
        gap_closing_K.append(gap_closing_K[0])
    k1 = gap_closing_K[0]
    k2 = gap_closing_K[-1]
    v1 = columns_V[0]/np.linalg.norm(columns_V[0])
    v2 = columns_V[-1]/np.linalg.norm(columns_V[-1])
    #constants of modulo 1 (?) whic give the orientation of the condesate
    c1 = (1)/np.sqrt(2)
    c1_ = np.conjugate(c1)
    c2 = (1)/np.sqrt(2)
    c2_ = np.conjugate(c2)
    c = [1j,1,1,1]
    #
    for iUCx in range(UC):
        for iUCy in range(UC):
            R = iUCx*T1_ + iUCy*T2_
            for iUC in range(6):
                cond = np.zeros(2,dtype=complex)
                for xx in range(len(gap_closing_K)):
                    cond[0] += c[xx]*columns_V[xx][iUC]/np.linalg.norm(columns_V[xx])*np.exp(1j*np.dot(gap_closing_K[xx],R))
                    cond[1] += np.conjugate(c[xx])*np.conjugate(columns_V[xx][iUC+6])/np.linalg.norm(columns_V[xx])*np.exp(-1j*np.dot(gap_closing_K[xx],R))
                for x in range(3):
                    spin_lattice[iUCx,iUCy,iUC,x] = np.real(1/2*np.dot(cond.T.conj(),np.einsum('ij,j->i',s_[x],cond)))
                spin_lattice[iUCx,iUCy,iUC] /= np.linalg.norm(spin_lattice[iUCx,iUCy,iUC])
    return spin_lattice

def spin_structure_factor(spin_lattice):
    UC = spin_lattice.shape[2]
    nkx, nky, kxs, kys = get_kpoints_ssf((6,10),150)
    SSFzz = np.zeros((nkx,nky))
    SSFxy = np.zeros((nkx,nky))
    """
    We fix one unit cell and sum over the 6 sites inside
    """
    fUCx = UC//2    #fixed UC
    fUCy = UC//2
    UC_positions = np.array([   #in terms of t1,t2
        [2,0],
        [1,0],
        [0,0],
        [0,1],
        [1,1],
        [0,2]])
    #
    for fff in range(1):        #can be more
        fUCx = UC//2+fff
        fUCy = UC//2+fff
        for fUC in range(6):
            distances = np.zeros((UC**2*6,2))       #distances of all sites wrt fixed site
            prod_zz = np.zeros(UC**2*6)
            prod_xy = np.zeros(UC**2*6)
            #Probably can be done faster
            for i in range(UC**2*6):
                iUCx, iUCy, iUC = (i//6//UC, i//6%UC, i%6)
                distances[i] = T1_*(iUCx-fUCx) + T2_*(iUCy-fUCy) + np.dot(t_.T,(UC_positions[iUC]-UC_positions[fUC]))
                prod_zz[i] = spin_lattice[iUCx,iUCy,iUC,2]*spin_lattice[fUCx,fUCy,fUC,2]
                prod_xy[i] = spin_lattice[iUCx,iUCy,iUC,0]*spin_lattice[fUCx,fUCy,fUC,0]+spin_lattice[iUCx,iUCy,iUC,1]*spin_lattice[fUCx,fUCy,fUC,1]
            """
            We can create SSFzz in one shot by creating a matrix of kx,ky values and doing the dot product at once.
            We use the cosine since the SSF is real.
            """
            X,Y = np.meshgrid(kxs,kys)
            cos_kd = np.cos(np.einsum('ijm,km->ijk',np.dstack([X.T,Y.T]),distances))/2
            SSFzz += np.sum(cos_kd*prod_zz, axis=2)
            SSFxy += np.sum(cos_kd*prod_xy, axis=2)
    SSFzz /= UC**2*6
    SSFxy /= UC**2*6
    return SSFzz,SSFxy

def get_kpoints_ssf(factors,max_k=100):
    """Get right number of kpoints for calculation of ssf that includes high symmetry points."""
    res = []
    for i in range(2):
        list_ = [2*factors[i]+1,]
        while True:
            val = list_[-1]*2-1
            if val < max_k:
                list_.append(list_[-1]*2-1)
            else:
                res.append(list_[-1])
                break
    vecx = np.pi*2/np.sqrt(21)
    vecy = np.pi*2/3/np.sqrt(7)
    fx,fy = factors
    kxs = np.linspace(-vecx*fx,vecx*fx,res[0])
    kys = np.linspace(-vecy*fy,vecy*fy,res[1])
    return res[0],res[1],kxs,kys

def plot_ssf(SSFzz,SSFxy):
    """Plot of the resulting ssf.
    """
    nkx, nky, kxs,kys = get_kpoints_ssf((6,10),150)
    #
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
        plot_BZs(ax)
        ax.set_xlim(kxs[0],kxs[-1])
        ax.set_ylim(kys[0],kys[-1])
        plt.colorbar(sc)
        ax.set_aspect('equal')

    fig.tight_layout()
    plt.show()

def plot_BZs(ax):
    """Plot the two rotated BZs for the SSF plots.
    """
    lw = 0.5
    for i in range(6):
        ax.plot([vertices_BZ[i,0],vertices_BZ[(i+1)%6, 0]], [vertices_BZ[i,1],vertices_BZ[(i+1)%6, 1]],
                color='k',
                lw=lw,
                ls='--',
                zorder=2,
               )
        ax.plot([vertices_EBZ[i,0],vertices_EBZ[(i+1)%6, 0]], [vertices_EBZ[i,1],vertices_EBZ[(i+1)%6, 1]],
                color='k',
                lw=lw,
                ls='--',
                zorder=2,
               )

##################################################################
################################################################## 
##################################################################
##################################################################
"""
Filenames and I/O functions
"""
def get_res_final_fn(pars_general,machine):
    Js,Spin,KM,ansatz = pars_general
    J_h,J_d,J_t = Js
    Kx,Ky = KM[0].shape
    return get_res_final_dn(Kx,Ky,Spin,machine) +"(Jh,Jd,Jt)=("+"{:.3f}".format(J_h)+","+"{:.3f}".format(J_d)+","+"{:.3f}".format(J_t)+')_ansatz'+ansatz+'.npy'

def get_res_final_dn(Kx,Ky,Spin,machine):
    res_dir = get_res_S_dn(Spin,machine)+'Kx_'+str(Kx)+'Ky_'+str(Ky)+'/'
    if not Path(res_dir).is_dir():
        print("Creating directory ",res_dir)
        os.system("mkdir "+res_dir)
    return res_dir

def get_res_S_dn(Spin,machine):
    res_dir = get_res_dn(machine)+"S_"+"{:.4f}".format(Spin)+'/'
    if not Path(res_dir).is_dir():
        print("Creating directory ",res_dir)
        os.system("mkdir "+res_dir)
    return res_dir

def get_res_dn(machine):
    res_dir = get_home_dn(machine)+'results/'
    if not Path(res_dir).is_dir():
        print("Creating directory ",res_dir)
        os.system("mkdir "+res_dir)
    return res_dir

def get_home_dn(machine):
    if machine == 'loc':
        return '/home/dario/Desktop/git/MapleLeaf/schwinger_boson/'
    elif machine == 'hpc':
        return '/home/users/r/rossid/schwinger_boson/'
    elif machine == 'maf':
        return '/users/rossid/schwinger_boson/'

def SaveToCsv(Data,csvfile):
    header = Data.keys()
    with open(csvfile,'a') as f:
        writer = csv.DictWriter(f, fieldnames = header)
        writer.writeheader()
        writer.writerow(Data)

def get_machine(cwd):
    """Selects the machine the code is running on by looking at the working directory. Supports local, hpc (baobab or yggdrasil) and mafalda.

    Parameters
    ----------
    pwd : string
        Result of os.pwd(), the working directory.

    Returns
    -------
    string
        An acronim for the computing machine.
    """
    if cwd[6:11] == 'dario':
        return 'loc'
    elif cwd[:20] == '/home/users/r/rossid':
        return 'hpc'
    elif cwd[:13] == '/users/rossid':
        return 'maf'

def get_reciprocal_vectors(a):
    """
    Get reciprocal lattice vectors of a1=a[0], a2=a[1]
    """
    a_r = np.zeros((2,2))
    a_r[0] = 2*np.pi*np.array([a[1,1],-a[1,0]])/np.linalg.det(np.array([a[0],a[1]]))
    a_r[1] = 2*np.pi*np.array([-a[0,1],a[0,0]])/np.linalg.det(np.array([a[0],a[1]]))
    return a_r
def angle_between_vectors(A, B):
    angle_rad = np.arccos(np.clip(np.dot(A,B)/np.linalg.norm(A)/np.linalg.norm(B),-1.0,1.0))
    return angle_rad
def R_z(theta):
    """
    Just a rotaion around z.
    """
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
"""
Latice vectors and reciprocal vectors
"""
T1_ = np.sqrt(7)*np.array([np.sqrt(3)/2,-1/2])
T2_ = np.sqrt(7)*np.array([0,1])
T_ = np.array([T1_,T2_])
B_ = get_reciprocal_vectors(T_)
#
t1_ = np.array([np.sqrt(3),-2])/np.sqrt(7)
t2_ = np.array([3/2*np.sqrt(3),1/2])/np.sqrt(7)
t_ = np.array([t1_,t2_])
b_ = get_reciprocal_vectors(t_)
"""
Vertices of BZ and extended BZ
"""
vertices_BZ = np.zeros((6,2))
if abs(angle_between_vectors(B_[0],B_[1])-np.pi/3)<1e-3:
    vertices_BZ[0] = (B_[0] + B_[1])/3
elif abs(angle_between_vectors(B_[0],B_[1])-2*np.pi/3)<1e-3:
    vertices_BZ[0] = (B_[0] - B_[1])/3
for i in range(1,6):
    vertices_BZ[i] = np.matmul(R_z(np.pi/3*i),vertices_BZ[0])
vertices_EBZ = np.zeros((6,2))
if abs(angle_between_vectors(b_[0],b_[1])-np.pi/3)<1e-3:
    vertices_EBZ[0] = (b_[0] + b_[1])/3
elif abs(angle_between_vectors(b_[0],b_[1])-2*np.pi/3)<1e-3:
    vertices_EBZ[0] = (b_[0] - b_[1])/3
for i in range(1,6):
    vertices_EBZ[i] = np.matmul(R_z(np.pi/3*i),vertices_EBZ[0])











