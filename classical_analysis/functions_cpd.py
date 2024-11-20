import numpy as np
from numpy import cos as C
from numpy import sin as S

def E_kiwi(continuum_pars,*args):
    discrete_pars, Js = args
    h,d,t = Js
    th = continuum_pars
    ep, n = discrete_pars
    return 3*(S(th)**2*(2*h*ep*C(n*np.pi/3)+2*t*C(2*n*np.pi/3)+(-1)**n*d*ep)+C(th)**2*(2*h*ep+2*t+d*ep))

def E_kiwi2(continuum_pars,*args):
    discrete_pars, Js = args
    h,d,t = Js
    th = continuum_pars
    ep, n = discrete_pars
    S = np.array([np.sin(th),0,np.cos(th)])
    R = ep*np.array([[np.cos(n*np.pi/3),-np.sin(n*np.pi/3),0],[np.sin(n*np.pi/3),np.cos(n*np.pi/3),0],[0,0,1]])
    T1 = T2 = np.identity(3)
    Eh = 6*h*S@R@S
    Et = 6*t*S@T1@T2@R@R@S
    Ed = 3*d*S@T2@R@R@R@S
    return Eh+Et+Ed

def E_banana(continuum_pars,*args):
    discrete_pars, Js = args
    h,d,t = Js
    th, ph = continuum_pars
    ep, e1, e2 = discrete_pars
    return -3*d*(C(th)**2+S(th)**2*C(2*ph)) + 6*S(th)*(e2*S(ph)*C(th)*(h*ep+t)+(e1*e2*C(ph)*C(th)+e1*S(th)*C(ph)*S(ph))*(h*ep-t))

def E_banana2(continuum_pars,*args):
    discrete_pars, Js = args
    h,d,t = Js
    th, ph = continuum_pars
    ep, e1, e2 = discrete_pars
    S = np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
    R = ep*np.array([[0,e1,0],[0,0,e2],[e1*e2,0,0]])
    T1 = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    T2 = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
    Eh = 6*h*S@R@S
    Et = 6*t*S@T1@T2@R@R@S
    Ed = 3*d*S@T2@R@R@R@S
    return Eh+Et+Ed

def E_mango(continuum_pars,*args):
    discrete_pars, Js = args
    h,d,t = Js
    th, ph, et = continuum_pars
    ep = discrete_pars[0]
    return 3*t*C(2*th) + 6*h*ep*(S(th)**2*(C(2*et)*C(2*ph)+S(2*et)*S(2*ph))-C(th)**2) - 3/2*d*ep*(C(th)**2+S(th)**2*(C(2*ph)*(C(2*et)+np.sqrt(3)*S(2*et))+S(2*ph)*(S(2*et)-np.sqrt(3)*C(2*et))))

def E_mango2(continuum_pars,*args):
    discrete_pars, Js = args
    h,d,t = Js
    th, ph, et = continuum_pars
    ep = discrete_pars[0]
    S = np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
    R = ep*np.array([[np.cos(2*et),np.sin(2*et),0],[np.sin(2*et),-np.cos(2*et),0],[0,0,-1]])
    T1 = T2 = 1/2*np.array([[-1,-np.sqrt(3),0],[np.sqrt(3),-1,0],[0,0,1]])
    Eh = 6*h*S@R@S
    Et = 6*t*S@T1@T2@R@R@S
    Ed = 3*d*S@T2@R@R@R@S
    return Eh+Et+Ed

def get_discrete_index(ind_discrete,ans):
    if ans == 'kiwi':
        return ((-1)**(ind_discrete//4),ind_discrete%4)
    elif ans == 'banana':
        return ((-1)**(ind_discrete//4),(-1)**(ind_discrete//2),(-1)**(ind_discrete%2))
    elif ans == 'mango':
        return ((-1)**(ind_discrete),)

functions_fruit = {'kiwi':E_kiwi2, 'banana':E_banana2, 'mango':E_mango2}
txt_ansatz = ['kiwi','banana','mango']
dic_ind_discrete = {'kiwi':8, 'banana':8, 'mango':2}

def get_res_cpd_fn(Jh,nn,Jds,Jts):
    res_dn = 'results/data_cpd/'
    return res_dn + 'cpd_Jh'+str(Jh)+'_npts'+str(nn)+'_bounds_'+str(Jds[0])+','+str(Jds[-1])+'_'+str(Jts[0])+','+str(Jts[-1])+'.npy'

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



