import numpy as np

def R_z(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

"""
Latice vectors and reciprocal vectors
"""
T = np.array([np.sqrt(7)*np.array([np.sqrt(3)/2,-1/2]),np.sqrt(7)*np.array([0,1])])
B = np.zeros((2,2))
B[0] = 2*np.pi*np.array([T[1,1],-T[1,0]])/np.linalg.det(np.array([T[0],T[1]]))
B[1] = 2*np.pi*np.array([-T[0,1],T[0,0]])/np.linalg.det(np.array([T[0],T[1]]))
t = np.array([1/np.sqrt(7)*np.array([np.sqrt(3),-2]),1/np.sqrt(7)*np.array([np.sqrt(3)*3/2,1/2])])
b = np.zeros((2,2))
b[0] = 2*np.pi*np.array([t[1,1],-t[1,0]])/np.linalg.det(np.array([t[0],t[1]]))
b[1] = 2*np.pi*np.array([-t[0,1],t[0,0]])/np.linalg.det(np.array([t[0],t[1]]))
"""
Vertices of BZ and extended BZ
"""
vertices_BZ = np.zeros((6,2))
vertices_BZ[0] = (B[0] + B[1])/3
for i in range(1,6):
    vertices_BZ[i] = np.matmul(R_z(np.pi/3*i),vertices_BZ[0])
vertices_EBZ = np.zeros((6,2))
vertices_EBZ[0] = (b[0] - b[1])/3
for i in range(1,6):
    vertices_EBZ[i] = np.matmul(R_z(np.pi/3*i),vertices_EBZ[0])
"""
Lattices
"""
def FM(UC):
    lattice = np.zeros((UC,UC,6,3))
    lattice[:,:,:,2] = 1/2
    return lattice

def Neel(UC):
    lattice = np.zeros((UC,UC,6,3))
    lattice[:,:,::2,2] = 1/2
    lattice[:,:,1::2,2] = -1/2
    return lattice

lattice_functions = {'FM':FM,'Neel':Neel}

def get_spec_txt(order,Jd,Jt,UC,nkx):
    return order+'_'+"{:.4f}".format(Jd)+"{:.4f}".format(Jt)+'_'+str(UC)+'_'+str(nkx)

def get_ssf_fn(direction,order,Jd,Jt,UC,nkx):
    return 'results/data_ssf/'+direction+'_'+get_spec_txt(order,Jd,Jt,UC,nkx)+'.npy'

def plot_ssf(data,kx,ky,ax,title):
    X,Y = np.meshgrid(kx,ky)
    ax.scatter(X,Y,c=np.real(data),marker='s')
    #hexagons
    for i in range(6):
        ax.plot([vertices_BZ[i,0],vertices_BZ[(i+1)%6, 0]], [vertices_BZ[i,1],vertices_BZ[(i+1)%6, 1]], 'b-')
        ax.plot([vertices_EBZ[i,0],vertices_EBZ[(i+1)%6, 0]], [vertices_EBZ[i,1],vertices_EBZ[(i+1)%6, 1]], 'b-')

    ax.set_aspect('equal')
    ax.set_title(title)

