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

disp = 0#True if machine=='loc' else False

#
plot_PD = 1
plot_val = 0
#what is this
II = 43
JJ = 0
#

Jh = 1
nn = 9
bound = 4
ng = 1  #0 or 1, gives the upper bound
Jds = np.linspace(-bound,bound*ng,nn)
Jts = np.linspace(-bound,bound*ng,nn)

res_fn = fs.get_res_cpd_fn(Jh,nn,Jds,Jts)

txt_ansatz = ['kiwi','banana','mango']
dic_ind_discrete = {'kiwi':8, 'banana':8, 'mango':2}
if not Path(res_fn).is_file():
    Energies = np.zeros((nn,nn,3,8,4))  #Jd, Jt, one of the three ansatz, 8 max number of discrete pars, 4 results max for each solution (energy+3max continuum pars)
    for ind_d in range(nn):
        print("J_d = ",Jds[ind_d])
        for ind_t in range(nn):
            if disp:
                print("J_t = ",Jts[ind_t])
            Js = (Jh, Jds[ind_d], Jts[ind_t])
            for ind_ans in range(3):
                txt = txt_ansatz[ind_ans]
                if disp:
                    print("Ansatz ",txt)
                for ind_discrete in range(dic_ind_discrete[txt]):
                    args=(fs.get_discrete_index(ind_discrete,txt),Js)
                    if disp:
                        print("Discrete pars: ",args[0])
                    if txt == 'kiwi':   #only one continuous parameter
                        Em = minimize_scalar(
                            fun=fs.E_kiwi,
                            bounds=[0,np.pi],
                            args=args,
                            method='bounded'
                        )
                        npars = 1
                    else:
                        bounds = ((0,np.pi),(0,2*np.pi)) if txt=='banana' else ((0,np.pi),(0,np.pi),(0,np.pi))
                        Em = d_e(
                            fs.functions_fruit[txt],
                            args=args,
                            tol=1e-5,
                            popsize=15,
                            bounds=bounds,
                            polish=True,
                            strategy='rand2bin'
                        )
                        npars = len(Em.x)
                    Energies[ind_d,ind_t,ind_ans,ind_discrete,0] = Em.fun
                    Energies[ind_d,ind_t,ind_ans,ind_discrete,1:1+npars] = Em.x
                    for i in range(1+npars,4):
                        Energies[ind_d,ind_t,ind_ans,ind_discrete,i] = np.nan
                    if disp:
                        print("Finished minimization with E=",Em.fun," and pars ",Em.x)
                        print(Energies[ind_d,ind_t,ind_ans,ind_discrete])
                        input()
    #
    np.save(res_fn,Energies)
else:
    Energies = np.load(res_fn)
print("Finished computing")

if not machine=='loc':
    exit()

if plot_PD:#Plot PD
    cmap = plt.get_cmap('tab20')
    n_colors = cmap.N
    colors = [cmap(i/(n_colors-1)) for i in range(n_colors-1)]
    #
    cmap2 = plt.get_cmap('tab20b')
    n_colors = cmap2.N
    for i in range(n_colors):
        colors.append(cmap2(i/(n_colors-1)))
    colors = np.array(colors[:24]).reshape(3,8,4)

    fig,ax = plt.subplots()
    fig.set_size_inches(20,15)
    #
    list_ind = []
    marker = 's'
    for ind_d in range(nn):
        for ind_t in range(nn):
            ind = np.unravel_index(np.argmin(Energies[ind_d,ind_t,:,:,0]),Energies[ind_d,ind_t,:,:,0].shape)
            if ind not in list_ind:
                list_ind.append(ind)
            if 0:   #check degeneracies
                lind = np.argwhere(abs(Es[i,j,:,0]-Es[i,j,ind,0])<1e-4)
                if len(lind)>1:
                    ind = min(lind)[0]
                if ind >= 6:
                    ind = 6 + fs.get_which_NonCoplanar(Es[i,j,6,1:])
            ax.scatter(Jds[ind_d],Jts[ind_t],color=colors[ind[0],ind[1]],marker=marker,s=50)

    ax.set_xlabel(r"$J_d$",size=30)
    ax.set_ylabel(r"$J_t$",size=30)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)

    if 1:   #legend
        legend_entries = []
        list_ind = np.array(list_ind)[np.argsort(np.array(list_ind)[:,0])]  #sort
        for i in range(len(list_ind)):
            ind = list_ind[i]
            txt = txt_ansatz[ind[0]]
            inds = fs.get_discrete_index(ind[1],txt)
            txt += ', ('
            for n in range(len(inds)):
                txt += str(inds[n])
                if not n==len(inds)-1:
                    txt += ','
            txt += ')'
            legend_entries.append( Line2D([0], [0], marker='o', color='w', label=txt,
                                  markerfacecolor=colors[ind[0],ind[1]], markersize=15)
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
    #
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

    #
    fig.tight_layout()
    plt.show()



































