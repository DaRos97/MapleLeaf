import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def E_FM(*args):
    h,t,d = args
    return 6*h+6*t+3*d

def E_Neel(*args):
    h,t,d = args
    return -6*h+6*t-3*d

def E_aC6(a,*args):
    h,t,d = args
    return np.sin(a)**2*(6*h+6*t+3*d)+3*np.cos(a)**2*(h-t-d)

def E_aK6(a,*args): #K=\bar(C)
    h,t,d = args
    return np.sin(a)**2*(6*t-6*h-3*d)+3*np.cos(a)**2*(d-h-t)

def E_aC3(a,*args):
    h,t,d = args
    return 6*np.sin(a)**2*(h+t)-3*np.cos(a)**2*(h+t)+3*d

def E_aK3(a,*args):
    h,t,d = args
    return 6*np.sin(a)**2*(t-h)-3*np.cos(a)**2*(t-h)-3*d

def E_aC2(a,*args):
    h,t,d = args
    return (6*h+3*d)*(np.sin(a)**2-np.cos(a)**2)+6*t

def E_aK2(a,*args):
    h,t,d = args
    return (6*h+3*d)*(np.cos(a)**2-np.sin(a)**2)+6*t

ECs = [E_aC6,E_aK6,E_aC3,E_aK3,E_aC2,E_aK2]

def E_NC2(*args):
    h,t,d = args
    return -3*d

def E_NK2(*args):
    h,t,d = args
    return 3*d

def E_NC3(pars,*args):
    be,ga = pars
    h,t,d = args
    return 3*np.sin(ga)**2*(2*h*np.cos(2*be)-t+d*np.cos(2*be-2*np.pi/3))-3*np.cos(ga)**2*(2*h-2*t+d)

name_list = ['FM','Neel','aC6','aK6','aC3','aK3','aC2','aK2','NC2','NK2','NC3','NK3']

Jh = 1
nn = 25
Jts = np.linspace(-4,4,nn)
Jds = np.linspace(-4,4,nn)

res_dn = 'results/'
res_fn = res_dn + 'cpd_'+str(Jh)+'_'+str(nn)+'_'+str(Jts[0])+','+str(Jts[-1])+'_'+str(Jds[0])+','+str(Jds[-1])+'.npy'

if not Path(res_fn).is_file():
    pd = np.zeros((nn,nn,3))

    for i in range(nn):
        for j in range(nn):
            args = (Jh, Jts[j], Jds[i])
            E = []
            alphas = []
            #
            E.append(E_FM(*args))
            E.append(E_Neel(*args))
            #Optimize energies
            for c in range(6):
                Em = minimize_scalar(
                        fun=ECs[c],
                        bounds=[0,np.pi/2],
                        args=args,
                        method='bounded',
                        )
                E.append(Em.fun)
                alphas.append(Em.x)
            #
            E.append(E_NC2(*args))
            E.append(E_NK2(*args))
            #
            EM = minimize(
                fun=E_NC3,
                x0=(np.pi/2,np.pi/3),
                args=args,
                method='Nelder-Mead',
                bounds=((0,np.pi),(0,np.pi)),
                options={
                        'disp':False,
                        'fatol':1e-4,
                        'adaptive':True
                        }
                )
            E.append(EM.fun)
            #
            pd[i,j,0] = np.argmin(E)
            if int(pd[i,j,0]) in [2,3,4,5,6,7]:
                pd[i,j,1] = alphas[int(pd[i,j,0])-2]
            elif int(pd[i,j,0])==10:
                pd[i,j,1:] = EM.x
    #
    np.save(res_fn,pd)
else:
    pd = np.load(res_fn)

#Plot
cmap = plt.get_cmap('Paired')
n_colors = cmap.N
colors = [cmap(i/(n_colors-1)) for i in range(n_colors)]

fig,ax = plt.subplots()

for i in range(nn):
    for j in range(nn):
        ax.scatter(Jds[i],Jts[j],color=colors[int(pd[i,j,0])],marker='o')


legend_entries = []
for i in range(len(name_list)):
    legend_entries.append( Line2D([0], [0], marker='o', color='w', label=name_list[i],
                          markerfacecolor=colors[i], markersize=15)
                          )

ax.legend(handles=legend_entries)
plt.show()

        




































