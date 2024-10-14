import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import differential_evolution as d_e
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from matplotlib.lines import Line2D
import functions_cpd as fs
import sys,os

machine = fs.get_machine(os.getcwd())

nC = '6' if len(sys.argv)==1 else sys.argv[1]

plot_PD = 1
plot_val = 0 if plot_PD else 1
n_ord = 6
#
print_txt = 0
II = 43
JJ = 0
#
Jh = 1
nn = 101
bound = 4
ng = 0
Jts = np.linspace(-bound,bound*ng,nn)
Jds = np.linspace(-bound,bound*ng,nn)

res_fn = fs.get_res_cpd_fn(nC,Jh,nn,Jts,Jds)

if not Path(res_fn).is_file():
    Es = np.zeros((nn,nn,len(fs.Eall[nC]),5))
    for i in range(nn):
        print(i)
        for j in range(nn):
            print(j)
            args = (Jh, Jts[j], Jds[i])
            for c in range(len(fs.Eall[nC])):
                if fs.Eall[nC][c] in fs.E0s:
                    Es[i,j,c,0] = fs.Eall[nC][c](*args)
                    Es[i,j,c,1] = Es[i,j,c,2] = np.nan
                    nans = 1        #start index of nans
                elif fs.Eall[nC][c] in fs.E1s:
                    Em = minimize_scalar(
                            fun=fs.Eall[nC][c],
                            bounds=[0,np.pi/2],
                            args=args,
                            method='bounded',
                            )
                    Es[i,j,c,0] = Em.fun
                    Es[i,j,c,1] = Em.x
                    nans = 2
                else:
                    if fs.Eall[nC][c] in fs.E2s:
                        nans = 3
                        bounds = ((0,np.pi),(0,np.pi))
                    if fs.Eall[nC][c] in fs.E3s:
                        nans = 4
                        bounds = ((0,np.pi),(0,np.pi),(0,2*np.pi))
                    if fs.Eall[nC][c] in fs.E4s:
                        nans = 5
                        bounds = ((0,np.pi),(0,np.pi),(0,2*np.pi),(0,2*np.pi))
                    #
                    Em = d_e(
                        fs.Eall[nC][c],
                        args=args,
                        tol=1e-5,
                        popsize=15,
                        bounds=bounds,
                        polish=True,
                        strategy='rand2bin'
                        )
                    Es[i,j,c,0] = Em.fun
                    Es[i,j,c,1:nans] = Em.x
                for NN in range(nans,5):
                    Es[i,j,c,NN] = np.nan
    #
    np.save(res_fn,Es)
else:
    Es = np.load(res_fn)
print("Finished computing")

if not machine=='loc':
    exit()

if plot_PD:#Plot PD
    cmap = plt.get_cmap('tab20')
    n_colors = cmap.N
    colors = [cmap(i/(n_colors-1)) for i in range(n_colors-1)]
    
    cmap2 = plt.get_cmap('tab20b')
    n_colors = cmap2.N
    for i in range(n_colors):
        colors.append(cmap2(i/(n_colors-1)))

    fig,ax = plt.subplots()
    fig.set_size_inches(20,15)
    
    marker = 's'
    for i in range(nn):
        for j in range(nn):
            ind = np.argmin(Es[i,j,:,0])
            lind = np.argwhere(abs(Es[i,j,:,0]-Es[i,j,ind,0])<1e-4)
            if len(lind)>1:
                ind = min(lind)[0]
            if ind >= 6:
                ind = 6 + fs.get_which_NonCoplanar(Es[i,j,6,1:])
            ax.scatter(Jds[i],Jts[j],color=colors[ind],marker=marker,s=50)

    ax.set_xlabel(r"$J_d$",size=30)
    ax.set_ylabel(r"$J_t$",size=30)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)

    legend_entries = []
    for i in [0,1,4,6,7,8]:
        legend_entries.append( Line2D([0], [0], marker='o', color='w', label=fs.name_list[nC][i],
                              markerfacecolor=colors[i], markersize=15)
                              )

    ax.legend(handles=legend_entries,loc='lower right',fontsize=20)

    ax.set_title("Classical phase diagraf of $J_h-J_t-J_d$ maple leaf lattice",size=30)

    ax.autoscale(enable=True, axis='x', tight=True)
    ax.autoscale(enable=True, axis='y', tight=True)

    plt.show()
    exit()

if print_txt:   #print values
    i = II
    j = JJ
    print("Jd,Jt = ",Jds[i],Jts[j])
    print("Energies: ",Es[i,j,:,0])
    print("Order ",n_ord,":")
    print(Es[i,j,n_ord,:])

    args = (Jh,Jts[j],Jds[i])
    exit()

if plot_val: #Plot energies 
    from matplotlib import cm
    
    fig,ax = plt.subplots(subplot_kw={"projection":"3d"},nrows=1,ncols=4)
    fig.set_size_inches(25,10)

    nc = n_ord
    nc = 6

    X,Y = np.meshgrid(Jds,Jts)
    label = ['E',r'$\theta$',r'$\theta_p$',r'$\phi$',r'$\phi_p$']

    ind = np.zeros((nn,nn),dtype=int)
    for i in range(nn):
        for j in range(nn):
            ind[i,j] = np.argmin(Es[i,j,:,0])
            lind = np.argwhere(abs(Es[i,j,:,0]-Es[i,j,ind[i,j],0])<1e-4)
            if len(lind)>1:
                ind[i,j] = min(lind)[0]
    #Select values for which the nc order is GS
    aa = np.argmin(Es[:,:,:,0],axis=2)
    bb = np.where(ind!=nc)
    if 1:
        Es[bb[0],bb[1],nc,1] = np.nan   #Remove values where is not the GS
        Es[bb[0],bb[1],nc,3] = np.nan   #Remove values where is not the GS
        th = Es[:,:,nc,1]
        ph = Es[:,:,nc,3]
        R3 = np.array([[0,0,1],[1,0,0],[0,1,0]])
        psi = np.zeros((nn,nn))
        for i in range(nn):
            for j in range(nn):
                S1 = np.array([np.sin(th[i,j])*np.cos(ph[i,j]),np.sin(th[i,j])*np.sin(ph[i,j]),np.cos(th[i,j])])
                psi[i,j] = np.degrees(np.arccos(np.clip(np.dot(S1,np.matmul(R3,S1)), -1.0, 1.0)))
    for i in range(3):
        Es[bb[0],bb[1],nc+i,0] = np.nan   #Remove values where is not the GS
        Es[bb[0],bb[1],nc+i+1,0] = np.nan   #Remove values where is not the GS
        #
        ax[i].plot_surface(X,Y,abs(Es[:,:,nc+i,0].T-Es[:,:,nc+i+1,0].T),cmap=cm.coolwarm,linewidth=0,antialiased=False)
        ax[i].set_title(label[i])
        ax[i].set_xlabel(r'$J_d$')
        ax[i].set_ylabel(r'$J_t$')

    
    fig.tight_layout()
    plt.show()



































