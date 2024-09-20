import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import functions_cpd as fs
import functions_visual as fsv
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import RegularPolygon,Circle


#chose order
Jh = 1
nn = 51
bound = 4
ng = 0
Jts = np.linspace(-bound,bound*ng,nn)
Jds = np.linspace(-bound,bound*ng,nn)

nC = '3'
I = 29    #5#12#29      #over 51
J = 26    #40#25#26     #over 51
n_ord = int(sys.argv[1])   #index of order
name = fs.name_list[nC][n_ord]

print("Order ",n_ord," at Jd=",Jds[I]," and Jt=",Jts[J])

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
args = (x,lw,c,ls)
fsv.plot_lattice(ax,*args)
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
elif name in ['Non-Coplanar Ico']:
    UC = 4
    for i in range(2):
        for j in range(2):
            center_x,center_y = offset+i*T1+j*T2
            hexagon = RegularPolygon((center_x, center_y), numVertices=6, radius=1.5, orientation=0, color='gray',alpha=0.5,zorder=0) 
            ax.add_patch(hexagon)
elif name in ['Coplanar']:
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
#Equator
theta_equator = np.linspace(0, 2 * np.pi, 100)
x_equator = np.cos(theta_equator)
y_equator = np.sin(theta_equator)
z_equator = np.zeros_like(theta_equator)
ax2.plot(x_equator, y_equator, z_equator, color='k', linewidth=1)
#Axis
ax2.quiver(0,0,0,1,0,0, color='k',arrow_length_ratio=0.1,lw=1)
ax2.quiver(0,0,0,0,1,0, color='k',arrow_length_ratio=0.1,lw=1)
ax2.quiver(0,0,0,0,0,1, color='k',arrow_length_ratio=0.1,lw=1)
ax2.text(1, 0, 0.1, r'$x$', color='k')
ax2.text(0, 1, 0.1, r'$y$', color='k')
ax2.text(0, 0, 1.1, r'$z$', color='k')
#Colors
cmap = plt.get_cmap('tab20')
n_colors = cmap.N
colors = [cmap(i/(n_colors-1)) for i in range(n_colors-1)]

cmap2 = plt.get_cmap('tab20b')
n_colors = cmap2.N
for i in range(n_colors):
    colors.append(cmap2(i/(n_colors-1)))
#
if name == 'FM':
    c = colors[0]
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
    #View
    ax2.view_init(30,50)
elif name == 'Neel':
    c = colors[0:4:2]
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
    #View
    ax2.view_init(30,50)
elif name == 'Coplanar':
    c1 = colors[:3]
    c2 = colors[3:6]
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
    ax2.text(x_al[50]+0.1, y_al[50]+0.1, 0,r'$\alpha$', color='k')
    
    #
    title = name
    #View
    ax2.view_init(30,50)
elif name == 'Non-Coplanar Ico':
    R3 = np.array([[0,0,1],[1,0,0],[0,1,0]])
    Gx = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    Gy = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
    th,tp,ph,pp = Es[I,J,n_ord,1:]
    #Sphere arrows
    S1 = np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
    S2 = np.array([np.sin(tp)*np.cos(pp),np.sin(tp)*np.sin(pp),np.cos(tp)])
    diff = [np.matmul(np.linalg.matrix_power(R3,i),S1)-S2 for i in range(3)]
    summ = [np.matmul(np.linalg.matrix_power(R3,i),S1)+S2 for i in range(3)]
    i0 = -1
    title2 = ''
    LW = 0
    angle = np.degrees(np.arccos(np.clip(np.dot(S1,np.matmul(R3,S1)), -1.0, 1.0)))
    for i in range(3):
        if (abs(diff[i])<1e-3).all():
            i0 = i
            print("Spins related by a rotation on the 2 sublattices")
            title2 = 'I'
            LW = 0
        elif (abs(summ[i])<1e-3).all():
            i0 = i
            print("Spins related by a rotation AND flip on the 2 sublattices")
            title2 = 'II small' if angle < 90 else 'II big'
            LW = 2
    #
    c1 = colors[:12]
    if i0!=-1:
        c2 = c1
    else:
        i0 = 0
        c2 = colors[12:]
    for x in range(2):
        for y in range(2):
            off = offset+x*T1+y*T2
            for i in range(6):
                color = c1[i//2+x*3+y*6] if i%2==0 else c2[((i-1)//2+i0)%3+x*3+y*6]
                X = T1*i_UC[i][0] + T2*i_UC[i][1] + off
                lw = LW if i%2==1 else 0
                ax.scatter(X[0],X[1],color=color, linewidths=lw, edgecolors='k', marker=marker, s=size)
            #
            for i in range(3):
                S = np.matmul(np.linalg.matrix_power(Gy,y),np.matmul(np.linalg.matrix_power(Gx,x),np.matmul(np.linalg.matrix_power(R3,i),S1)))
                Sp = np.matmul(np.linalg.matrix_power(Gy,y),np.matmul(np.linalg.matrix_power(Gx,x),np.matmul(np.linalg.matrix_power(R3,i),S2)))
                ax2.quiver(0, 0, 0, S[0], S[1], S[2], color=c1[i+x*3+y*6], arrow_length_ratio=0.1, linewidth=3)
                if c2 != c1:
                    ax2.quiver(0, 0, 0, Sp[0], Sp[1], Sp[2], color=c2[i], arrow_length_ratio=0.1, linewidth=3)
    #
    title = 'Non-Planar '+title2
    #View
    ax2.view_init(10,10)
    if len(title2)>1:
        ax3 = fig.add_subplot(4,6,23)
        ax3.arrow(0,0,0,0.8,color='b',width=0.05,head_starts_at_zero=False)
        ax3.arrow(1,1,0,-0.8,color='b',width=0.05,head_starts_at_zero=False)
        circle = Circle((0,0.5),radius=0.6, fill=False, edgecolor='k',lw=2) 
        ax3.add_patch(circle)
        ax3.set_aspect('equal')
        ax3.text(0.6,0.5,r'$=$',size=20)
        ax3.axis('off')

fig.suptitle(title+' order',size=20)











fig.tight_layout()
plt.show()



