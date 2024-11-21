import numpy as np
import functions_cpd as fs_cpd

def get_pars(ind):
    """Here we give the parameters in the phase diagram for the different orders.
    For FM and Neel not really needed.
    We give: FM, Neel, 2 coplanars (close and far from (0,0)) and N non-coplanars (one for each region of the classical phase diagram)
    """
    pars_orders = (
           [0,0,0,0],  #FM in (-4,-4)
           [0,,32,24], #Neel in (0,-1)
           [2,0,40,40], #Coplanar in (1,1)
           [1,0,16,20], #Non-Coplanar a in (-2,-1.5)
           [1,4,10,16], #Non-Coplanar b in (-2.75,-2)
           [1,4,8,25],  #Non-Coplanar b in (-3,-0.875)
           [1,4,56,39], #Non-Coplanar b in (3,8.75)
           )
    ind_order,ind_discrete,ind_d,ind_t = pars_orders[ind]
    #
    Jh = 1
    nn = 65     #largest phase diagram computed so far
    bound = 4
    ng = 1
    Jds = np.linspace(-bound,bound*ng,nn)
    Jts = np.linspace(-bound,bound*ng,nn)
    #
    ans = fs_cpd.txt_ansatz[ind_order]
    print("Computing spin structure factor of",ans)
    print("Using parameters at position Jd=","{:.4f}".format(Jds[ind_d])," and Jt=","{:.4f}".format(Jts[ind_t]))
    pars_fn = fs_cpd.get_res_cpd_fn(Jh,nn,Jds,Jts)
    Energies = np.load(pars_fn)
    args = Energies[ind_d,ind_t,ind_order,ind_discrete,1:]   #don't take the energy
    args = args[~np.isnan(args)]    #remove nans
    print("angle(s): ",args)
    return ans, args, ind_discrete, Jds[ind_d], Jts[ind_t]

def R_z(theta):
    """
    Just a rotaion around z.
    """
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
def R_z3(theta):
    """
    Just a rotaion around z.
    """
    return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
def R_x3(theta):
    return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
def R_y3(theta):
    return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])

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

"""
Latice vectors and reciprocal vectors
"""
T_ = np.array([np.sqrt(7)*np.array([np.sqrt(3)/2,-1/2]),np.sqrt(7)*np.array([0,1])])
B_ = get_reciprocal_vectors(T_)
#
t_ = np.array([1/np.sqrt(7)*np.array([np.sqrt(3),-2]),1/np.sqrt(7)*np.array([np.sqrt(3)*3/2,1/2])])
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
"""
Lattices
"""

"""
2 - 3 - 5
 \ / \ /
  1 - 4
   \ /
    0
hexagonal links: 0-1,2-3,4-5
triangular links: 1-3,3-4,4-1
dimer links: 1-2,3-5,4-0
"""
def kiwi_lattice(UC,args,ind_discrete):
    th = args[0]
    ep, n = fs_cpd.get_discrete_index(ind_discrete,'kiwi')
    S = np.array([np.sin(th),0,np.cos(th)])
    R = ep*np.array([[np.cos(n*np.pi/3),-np.sin(n*np.pi/3),0],[np.sin(n*np.pi/3),np.cos(n*np.pi/3),0],[0,0,1]])
    T1 = T2 = np.identity(3)
    return get_lattice(UC,S,R,T1,T2)

def banana_lattice(UC,args,ind_discrete):
    th,ph = args
    ep, e1, e2 = fs_cpd.get_discrete_index(ind_discrete,'banana')
    S = np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
    R = ep*np.array([[0,e1,0],[0,0,e2],[e1*e2,0,0]])
    T1 = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    T2 = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
    return get_lattice(UC,S,R,T1,T2)

def mango_lattice(UC,args,ind_discrete):
    th,ph,et = args
    ep = fs_cpd.get_discrete_index(ind_discrete,'mango')[0]
    S = np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
    R = ep*np.array([[np.cos(2*et),np.sin(2*et),0],[np.sin(2*et),-np.cos(2*et),0],[0,0,-1]])
    T1 = T2 = 1/2*np.array([[-1,-np.sqrt(3),0],[np.sqrt(3),-1,0],[0,0,1]])
    return get_lattice(UC,S,R,T1,T2)

def get_lattice(UC,S,R,T1,T2):
    lattice = np.zeros((UC,UC,6,3))
    GR_ = []
    for iUC in range(6):
        GR_.append(np.linalg.matrix_power(R,iUC))
    for ix in range(UC):
        tr1 = np.linalg.matrix_power(T1,ix)
        for iy in range(UC):
            tr2 = np.linalg.matrix_power(T2,iy)
            for iUC in range(6):
                lattice[ix,iy,iUC] = tr1@tr2@GR_[iUC]@S
    return lattice

fruit_lattice = {'kiwi':kiwi_lattice,'banana':banana_lattice,'mango':mango_lattice}

def FM(UC,args):
    lattice = np.zeros((UC,UC,6,3))
    lattice[:,:,:,2] = 1/2
    return lattice

def Neel(UC,args):
    lattice = np.zeros((UC,UC,6,3))
    lattice[:,:,[1,3,4],2] = 1/2
    lattice[:,:,[0,2,5],2] = -1/2
    return lattice

def coplanar(UC,args):
    th,tp,ph = args
    lattice = np.zeros((UC,UC,6,3))
    s1 = np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
    s2 = np.array([np.sin(tp),0,np.cos(tp)])
#    s2 = np.matmul(R_z3(alpha),s1)
    for ix in range(UC):
        for iy in range(UC):
            lattice[ix,iy,0] = R_z3(np.pi*2/3*(ix+iy))@s1
            lattice[ix,iy,2] = R_z3(np.pi*2/3*(ix+iy+1))@s1
            lattice[ix,iy,5] = R_z3(np.pi*2/3*(ix+iy+2))@s1
            #
            lattice[ix,iy,1] = R_z3(np.pi*2/3*(ix+iy))@s2
            lattice[ix,iy,3] = R_z3(np.pi*2/3*(ix+iy+1))@s2
            lattice[ix,iy,4] = R_z3(np.pi*2/3*(ix+iy+2))@s2
    return lattice

def noncoplanar_1(UC,args):
    th,tp,ph,pp = args
    lattice = np.zeros((UC,UC,6,3))
    s1 = np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
    s2 = np.array([np.sin(tp)*np.cos(pp),np.sin(tp)*np.sin(pp),np.cos(tp)])
    GR1 = np.array([[0,0,1],[1,0,0],[0,1,0]])
    GR2 = GR1@GR1
    for ix in range(UC):
        for iy in range(UC):
            lattice[ix,iy,0] = R_x3(np.pi*ix)@R_y3(np.pi*iy)@s1
            lattice[ix,iy,2] = GR2@R_x3(np.pi*ix)@R_y3(np.pi*(iy+1))@s1
            lattice[ix,iy,5] = GR1@R_x3(np.pi*(ix+1))@R_y3(np.pi*(iy+1))@s1
            #
            lattice[ix,iy,1] = R_x3(np.pi*ix)@R_y3(np.pi*iy)@s2
            lattice[ix,iy,3] = GR2@R_x3(np.pi*ix)@R_y3(np.pi*(iy+1))@s2
            lattice[ix,iy,4] = GR1@R_x3(np.pi*(ix+1))@R_y3(np.pi*(iy+1))@s2
    return lattice

lattice_functions = {'FM':FM,'Neel':Neel,'Coplanar':coplanar, 'Non-Coplanar Ico':noncoplanar_1}

def get_spec_txt(order,ind_discrete,Jd,Jt,UC,nkx,nky):
    return order+str(ind_discrete)+'_'+"{:.4f}".format(Jd)+"{:.4f}".format(Jt)+'_'+str(UC)+'_'+str(nkx)+'_'+str(nky)

def get_ssf_fn(direction,order,ind_discrete,Jd,Jt,UC,nkx,nky):
    return 'results/data_ssf/'+direction+'_'+get_spec_txt(order,ind_discrete,Jd,Jt,UC,nkx,nky)+'.npy'

def plot_BZs(ax):
    #hexagons
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

def get_kpoints(factors,max_k=100):
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
    return res

def get_suptitle(order,args,Jd,Jt):
    tt = ''
    txt_ = [r'$\theta$',r'$\theta_p $',r'$\phi$',r'$\phi_p$']
    for i in range(len(args)):
        tt += txt_[i]+'='
        tt += "{:.1f}".format(args[i]*180/np.pi) + "°"
        if not i==len(args)-1:
            tt += ', '
    return order + ' at '+r'$J_d=$'+"{:.2f}".format(Jd)+', '+r'$J_t=$'+"{:.2f}".format(Jt)+': ' + tt



















































