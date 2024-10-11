import numpy as np

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
    return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0][-np.sin(theta),0,np.cos(theta)]])

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
t_ = np.array([1/np.sqrt(7)*np.array([np.sqrt(3)*3/2,-1/2]),1/np.sqrt(7)*np.array([np.sqrt(3),2])])
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
def FM(UC,args):
    lattice = np.zeros((UC,UC,6,3))
    lattice[:,:,:,2] = 1/2
    return lattice

def Neel(UC,args):
    lattice = np.zeros((UC,UC,6,3))
    lattice[:,:,[1,3,4],2] = 1/2
    lattice[:,:,[0,2,5],2] = -1/2
    return lattice

def coplanar(UC,alpha=np.pi/3):
    lattice = np.zeros((UC,UC,6,3))
    s1 = np.array([1,0,0])
    s2 = np.matmul(R_z3(alpha),s1)
    for ix in range(UC):
        for iy in range(UC):
            lattice[ix,iy,0] = np.matmul(R_z3(np.pi*2/3*(ix+iy)),s1)
            lattice[ix,iy,2] = np.matmul(R_z3(np.pi*2/3*(ix+iy+1)),s1)
            lattice[ix,iy,5] = np.matmul(R_z3(np.pi*2/3*(ix+iy+2)),s1)
            #
            lattice[ix,iy,1] = np.matmul(R_z3(np.pi*2/3*(ix+iy)),s2)
            lattice[ix,iy,3] = np.matmul(R_z3(np.pi*2/3*(ix+iy+1)),s2)
            lattice[ix,iy,4] = np.matmul(R_z3(np.pi*2/3*(ix+iy+2)),s2)
    return lattice

def noncoplanar_1(UC,theta=np.pi/3,phi=0):
    lattice = np.zeros((UC,UC,6,3))
    s1 = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
    GR = np.array([[0,0,1],[1,0,0],[0,1,0]])
    for ix in range(UC):
        for iy in range(UC):
            lattice[ix,iy,0] = np.matmul(R_y3(np.pi*iy),np.matmul(R_x3(np.pi*ix),s1))
            lattice[ix,iy,1] = np.matmul(np.linalg.matrix_power(GR,2),lattice[ix,iy,0])
            lattice[ix,iy,2] = np.matmul(R_y3(np.pi*iy+1),lattice[ix,iy,1])
            lattice[ix,iy,3] = np.matmul(R_y3(np.pi*iy),np.matmul(,lattice[ix,iy,0]))
            lattice[ix,iy,0] = np.matmul(R_x3(np.pi*ix),s1)
            lattice[ix,iy,2] = np.matmul(R_z3(np.pi*2/3*(ix+iy+1)),s1)
            lattice[ix,iy,5] = np.matmul(R_z3(np.pi*2/3*(ix+iy+2)),s1)
            #
            lattice[ix,iy,1] = np.matmul(R_z3(np.pi*2/3*(ix+iy)),s2)
            lattice[ix,iy,3] = np.matmul(R_z3(np.pi*2/3*(ix+iy+1)),s2)
            lattice[ix,iy,4] = np.matmul(R_z3(np.pi*2/3*(ix+iy+2)),s2)
    return lattice

lattice_functions = {'FM':FM,'Neel':Neel,'Coplanar':coplanar}

def get_spec_txt(order,Jd,Jt,UC,nkx):
    return order+'_'+"{:.4f}".format(Jd)+"{:.4f}".format(Jt)+'_'+str(UC)+'_'+str(nkx)

def get_ssf_fn(direction,order,Jd,Jt,UC,nkx):
    return 'results/data_ssf/'+direction+'_'+get_spec_txt(order,Jd,Jt,UC,nkx)+'.npy'

def plot_BZs(ax):
    #hexagons
    for i in range(6):
        ax.plot([vertices_BZ[i,0],vertices_BZ[(i+1)%6, 0]], [vertices_BZ[i,1],vertices_BZ[(i+1)%6, 1]],
                color='k',
                lw=2,
                ls='--',
                zorder=2,
               )
        ax.plot([vertices_EBZ[i,0],vertices_EBZ[(i+1)%6, 0]], [vertices_EBZ[i,1],vertices_EBZ[(i+1)%6, 1]],
                color='k',
                lw=2,
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




















































