import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import functions_cpd as fs
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import RegularPolygon

#chose order
Jh = 1
nn = 51
bound = 4
ng = 0
Jts = np.linspace(-bound,bound*ng,nn)
Jds = np.linspace(-bound,bound*ng,nn)

nC = '3'
I = -1
J = -1
n_ord = int(sys.argv[1])   #index of order
name = fs.name_list[nC][n_ord]

res_dn = 'results/'
res_fn = res_dn + 'cp'+nC+'d_'+str(Jh)+'_'+str(nn)+'_'+str(Jts[0])+','+str(Jts[-1])+'_'+str(Jds[0])+','+str(Jds[-1])+'.npy'
if Path(res_fn).is_file():
    Es = np.load(res_fn)
else:
    print("Order not computed")
    print(res_fn)

fig = plt.figure(figsize=(15,10))


#
ax = fig.add_subplot(121)
ax.set_aspect('equal')
#lattice
lw = 0.5
c = 'k'
ls = 'solid'
x = np.sqrt(3)/2
#vertical lines
for i in range(3):
    ax.plot([i*3*x,i*3*x],[-i/2,5-i/2],color=c,ls=ls,lw=lw,zorder=0)
    ax.plot([i*3*x,i*3*x],[7-i/2,8-i/2],color=c,ls=ls,lw=lw,zorder=0)
    ax.plot([x*(1+i*3),x*(1+i*3)],[-1/2-i/2,3-1/2-i/2],color=c,ls=ls,lw=lw,zorder=0)
    ax.plot([x*(1+i*3),x*(1+i*3)],[5-1/2-i/2,8-1/2-i/2],color=c,ls=ls,lw=lw,zorder=0)
    ax.plot([x*(2+i*3),x*(2+i*3)],[-1-i/2,-i/2],color=c,ls=ls,lw=lw,zorder=0)
    ax.plot([x*(2+i*3),x*(2+i*3)],[2-i/2,7-i/2],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([-x,-x],[2+1/2,7+1/2],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([9*x,9*x],[-1-1/2,4-1/2],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([-2*x,-2*x],[5,7],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([10*x,10*x],[-1,1],color=c,ls=ls,lw=lw,zorder=0)
#diagonal negative lines
ax.plot([0,2*x],[0,-1],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([0,5*x],[1,-1-1/2],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([-x,x],[2+1/2,1+1/2],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([3*x,8*x],[1/2,-2],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([-x,4*x],[3+1/2,1],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([6*x,9*x],[0,-1-1/2],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([-2*x,0],[5,4],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([2*x,7*x],[3,1/2],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([9*x,10*x],[-1/2,-1],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([-2*x,3*x],[6,3+1/2],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([5*x,10*x],[2+1/2,0],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([-2*x,-x],[7,6+1/2],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([x,6*x],[5+1/2,3],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([8*x,10*x],[2,1],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([-x,2*x],[7+1/2,6],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([4*x,9*x],[5,2+1/2],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([0,5*x],[8,5+1/2],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([7*x,9*x],[4+1/2,3+1/2],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([3*x,8*x],[7+1/2,5],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([6*x,8*x],[7,6],color=c,ls=ls,lw=lw,zorder=0)

#diagonal positive lines
ax.plot([8*x,10*x],[-2,-1],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([7*x,10*x],[-1-1/2,0],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([5*x,7*x],[-1-1/2,-1/2],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([9*x,10*x],[1/2,1],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([4*x,9*x],[-1,1+1/2],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([2*x,4*x],[-1,0],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([6*x,9*x],[1,2+1/2],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([x,6*x],[-1/2,2],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([8*x,9*x],[3,3+1/2],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([0,x],[0,1/2],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([3*x,8*x],[1+1/2,4],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([0,3*x],[1,2+1/2],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([5*x,8*x],[3+1/2,5],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([0,5*x],[2,4+1/2],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([7*x,8*x],[5+1/2,6],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([-x,0],[2+1/2,3],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([2*x,7*x],[4,6+1/2],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([-x,2*x],[3+1/2,5],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([4*x,6*x],[6,7],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([-x,4*x],[4+1/2,7],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([-2*x,-x],[5,5+1/2],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([x,3*x],[6+1/2,7+1/2],color=c,ls=ls,lw=lw,zorder=0)

ax.plot([-2*x,x],[6,7+1/2],color=c,ls=ls,lw=lw,zorder=0)
ax.plot([-2*x,0],[7,8],color=c,ls=ls,lw=lw,zorder=0)

ax.axis('off')

#Plot unit cell spins
T1 = 1/2*np.array([np.sqrt(3)*3,-1])
T2 = 1/2*np.array([-np.sqrt(3),5])
i_UC = [
        np.array([1,3])/7,
        np.array([-2,1])/7,
        np.array([-3,-2])/7,
        np.array([-1,-3])/7,
        np.array([2,-1])/7,
        np.array([3,2])/7,
        ]

offset = np.array([2*x,1])
marker = 'o'
size = 100

if name in ['FM','Neel','aA3','bA3']:
    UC = 1
    center_x,center_y = offset+T1+T2
    hexagon = RegularPolygon((center_x, center_y), numVertices=6, radius=1.5, orientation=0, color='gray',alpha=0.5,zorder=0) 
    ax.add_patch(hexagon)
elif name in ['3NC2a','3NC2b','3NC2c','3NC2d']:
    UC = 4
    for i in range(2):
        for j in range(2):
            center_x,center_y = offset+i*T1+j*T2
            hexagon = RegularPolygon((center_x, center_y), numVertices=6, radius=1.5, orientation=0, color='gray',alpha=0.5,zorder=0) 
            ax.add_patch(hexagon)
elif name in ['3NC9a','3NC9b']:
    UC = 9
    for i in range(3):
        for j in range(3):
            center_x,center_y = offset+i*T1+j*T2
            hexagon = RegularPolygon((center_x, center_y), numVertices=6, radius=1.5, orientation=0, color='gray',alpha=0.5,zorder=0) 
            ax.add_patch(hexagon)

#Sphere
ax2 = fig.add_subplot(122,projection='3d')
ax2.axis('off')
ax2.set_aspect('equal')
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi, 100)
theta, phi = np.meshgrid(theta, phi)
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)
ax2.plot_surface(x, y, z, cmap='viridis', edgecolor='none',alpha=0.2)
ax2.scatter(0,0,c='k',s=20,lw=0)
#equator
theta_equator = np.linspace(0, 2 * np.pi, 100)
x_equator = np.cos(theta_equator)
y_equator = np.sin(theta_equator)
z_equator = np.zeros_like(theta_equator)
ax2.plot(x_equator, y_equator, z_equator, color='k', linewidth=1)
ax2.view_init(30,200)
if name == 'FM':
    c = 'b'
    for x in range(3):
        for y in range(3):
            off = offset+x*T1+y*T2
            for i in range(6):
                X = T1*i_UC[i][0] + T2*i_UC[i][1] + off
                ax.scatter(X[0],X[1],color=c, lw=0, marker=marker, s=size) 
    #Sphere arrows
    theta_arrow = 0
    phi_arrow = 0
    x_arrow = np.sin(theta_arrow) * np.cos(phi_arrow)
    y_arrow = np.sin(theta_arrow) * np.sin(phi_arrow)
    z_arrow = np.cos(theta_arrow)
    ax2.quiver(0, 0, 0, x_arrow, y_arrow, z_arrow, color=c, arrow_length_ratio=0.1, linewidth=3)
    #
    title = 'Ferromagnetic'
elif name == 'Neel':
    c = ['b','r']
    for x in range(3):
        for y in range(3):
            off = offset+x*T1+y*T2
            for i in range(6):
                X = T1*i_UC[i][0] + T2*i_UC[i][1] + off
                ax.scatter(X[0],X[1],color=c[i%2], lw=0, marker=marker, s=size)
    #Sphere arrows
    for i in range(2):
        theta_arrow = np.pi*i
        phi_arrow = 0
        x_arrow = np.sin(theta_arrow) * np.cos(phi_arrow)
        y_arrow = np.sin(theta_arrow) * np.sin(phi_arrow)
        z_arrow = np.cos(theta_arrow)
        ax2.quiver(0, 0, 0, x_arrow, y_arrow, z_arrow, color=c[i], arrow_length_ratio=0.1, linewidth=3)
    #
    title = 'Neel'
elif name == '3NC9a':
    c1 = ['b','r','g']
    c2 = ['y','m','pink']
    for x in range(3):
        for y in range(3):
            off = offset+x*T1+y*T2
            for i in range(6):
                color = c1[(x+y)%3] if i%2==0 else c2[(x+y)%3]
                X = T1*i_UC[i][0] + T2*i_UC[i][1] + off
                ax.scatter(X[0],X[1],color=color, lw=0, marker=marker, s=size)
    #Sphere arrows
    phi = np.pi/6
    for i in range(3):
        theta_arrow = np.pi/2
        phi_arrow = np.pi/3*2*i
        x_arrow = np.sin(theta_arrow) * np.cos(phi_arrow)
        y_arrow = np.sin(theta_arrow) * np.sin(phi_arrow)
        z_arrow = np.cos(theta_arrow)
        ax2.quiver(0, 0, 0, x_arrow, y_arrow, z_arrow, color=c1[i], arrow_length_ratio=0.1, linewidth=3)
        phi_arrow = np.pi/3*2*i + phi
        x_arrow = np.sin(theta_arrow) * np.cos(phi_arrow)
        y_arrow = np.sin(theta_arrow) * np.sin(phi_arrow)
        z_arrow = np.cos(theta_arrow)
        ax2.quiver(0, 0, 0, x_arrow, y_arrow, z_arrow, color=c2[i], arrow_length_ratio=0.1, linewidth=3)
    #
    theta_al = np.linspace(0, phi, 100)
    x_al = np.cos(theta_al)*0.5
    y_al = np.sin(theta_al)*0.5
    z_al = np.zeros_like(theta_al)
    ax2.plot(x_al, y_al, z_al, color='k', linewidth=1)
    ax2.text(x_al[50], y_al[50], 0.1,r'$\alpha$', color='k')
    
    #
    title = 'Planar'

fig.suptitle(title+' order',size=20)











fig.tight_layout()
plt.show()



