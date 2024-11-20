import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import differential_evolution as d_e
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from matplotlib.lines import Line2D
from matplotlib import cm
import functions_cpd as fs
import sys,os

machine = fs.get_machine(os.getcwd())

disp = 0#True if machine=='loc' else False
plot_PD = 0
plot_values = 0
#
Jh = 1
nn = 65
bound = 4
ng = 1  #0 or 1, gives the upper bound
Jds = np.linspace(-bound,bound*ng,nn)
Jts = np.linspace(-bound,bound*ng,nn)

res_fn = fs.get_res_cpd_fn(Jh,nn,Jds,Jts)

if not Path(res_fn).is_file():
    Energies = np.zeros((nn,nn,3,8,4))  #Jd, Jt, one of the three ansatz, 8 max number of discrete pars, 4 results max for each solution (energy+3max continuum pars)
    for ind_d in range(nn):
        print("J_d = ",Jds[ind_d])
        for ind_t in range(nn):
            if disp:
                print("J_t = ",Jts[ind_t])
            Js = (Jh, Jds[ind_d], Jts[ind_t])
            for ind_ans in range(3):
                txt = fs.txt_ansatz[ind_ans]
                if disp:
                    print("Ansatz ",txt)
                for ind_discrete in range(fs.dic_ind_discrete[txt]):
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
#            ind = np.unravel_index(np.argmin(Energies[ind_d,ind_t,:,:,0]),Energies[ind_d,ind_t,:,:,0].shape)
            minE = np.min(Energies[ind_d,ind_t,:,:,0])
            ind = tuple(np.argwhere(abs(Energies[ind_d,ind_t,:,:,0]-minE)<1e-4)[0])
            if ind not in list_ind:
                list_ind.append(ind)
            ax.scatter(Jds[ind_d],Jts[ind_t],color=colors[ind[0],ind[1]],marker=marker,s=50)

    ax.set_xlabel(r"$J_d$",size=30)
    ax.set_ylabel(r"$J_t$",size=30)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)

    #Legend
    legend_entries = []
    list_ind = np.array(list_ind)[np.argsort(np.array(list_ind)[:,0])]  #sort
    for i in range(len(list_ind)):
        ind = list_ind[i]
        txt = fs.txt_ansatz[ind[0]]
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


# Further checks
if plot_values: #Plot energies and other stuff
    X,Y = np.meshgrid(Jds,Jts)
    ind_ans = 2
    ind_discrete = 0
    label = ['E',r'$\theta$']
    if ind_ans ==1:
        label.append(r'$\phi$')
    if ind_ans == 2:
        label.append(r'$\phi-\eta_R$')

    fig,ax = plt.subplots(subplot_kw={"projection":"3d"},nrows=1,ncols=len(label))
    fig.set_size_inches(25,10)
    #Select values for which the nc order is GS
    for i in range(len(label)):
        if ind_ans == 2 and i==2:
            ax[i].plot_surface(X,Y,(Energies[:,:,ind_ans,ind_discrete,2].T-Energies[:,:,ind_ans,ind_discrete,3].T)%np.pi,cmap=cm.coolwarm)
        else:
            ax[i].plot_surface(X,Y,Energies[:,:,ind_ans,ind_discrete,i].T,cmap=cm.coolwarm)
        ax[i].set_title(label[i])
        ax[i].set_xlabel(r'$J_d$')
        ax[i].set_ylabel(r'$J_t$')

    #
    fig.tight_layout()
    plt.show()



































