import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import functions_cpd as fs_cpd
import functions_visual as fs_vis
import functions_ssf as fs_ssf
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import RegularPolygon,Circle
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

index = 0 if len(sys.argv)<2 else int(sys.argv[1])  #
ans, args_solution, ind_discrete, Jd, Jt = fs_ssf.get_pars(index)

fig = plt.figure(figsize=(15,10))

#
gs_large = gridspec.GridSpec(2, 2, figure=fig)
ax = fig.add_subplot(gs_large[0,0])
ax.set_aspect('equal')
#lattice
lw = 0.5
c = 'k'
ls = 'solid'
x = np.sqrt(3)/2
args = (x,lw,c,ls)
fs_vis.plot_lattice(ax,*args)
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

if ans=='kiwi':
    UC = 1
    center_x,center_y = offset+T1+T2
    hexagon = RegularPolygon((center_x, center_y), numVertices=6, radius=1.5, orientation=0, color='gray',alpha=0.5,zorder=0)
    ax.add_patch(hexagon)
elif ans=='banana':
    UC = 4
    for i in range(2):
        for j in range(2):
            center_x,center_y = offset+i*T1+j*T2
            hexagon = RegularPolygon((center_x, center_y), numVertices=6, radius=1.5, orientation=0, color='gray',alpha=0.5,zorder=0)
            ax.add_patch(hexagon)
elif ans=='mango':
    UC = 9
    for i in range(2):
        for j in range(2):
            if i==1 and j==0:
                continue
            center_x,center_y = offset+i*T1+j*T2
            hexagon = RegularPolygon((center_x, center_y), numVertices=6, radius=1.5, orientation=0, color='gray',alpha=0.5,zorder=0)
            ax.add_patch(hexagon)

#Sphere
ax2 = fig.add_subplot(gs_large[0,1],projection='3d')
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
#   Colored dots in grid
if index==0:    #FM
    c = colors[0]
    for x in range(3):
        for y in range(3):
            off = offset+x*T1+y*T2
            for i in range(6):
                X = T1*i_UC[i][0] + T2*i_UC[i][1] + off
                ax.scatter(X[0],X[1],color=c, lw=0, marker=marker, s=size) 
    #Sphere arrows
    theta_arrow = args_solution
#    theta_arrow = 0
    phi_arrow = 0
    x_arrow = np.sin(theta_arrow) * np.cos(phi_arrow)
    y_arrow = np.sin(theta_arrow) * np.sin(phi_arrow)
    z_arrow = np.cos(theta_arrow)
    ax2.quiver(0, 0, 0, x_arrow, y_arrow, z_arrow, color=c, arrow_length_ratio=0.1, linewidth=3)
    #
    suptitle = 'Collinear'
    #View
    ax2.view_init(30,50)
elif index==1:  #Neel
    c = colors[0:4:2]
    for x in range(3):
        for y in range(3):
            off = offset+x*T1+y*T2
            for i in range(6):
                X = T1*i_UC[i][0] + T2*i_UC[i][1] + off
                ax.scatter(X[0],X[1],color=c[i%2], lw=0, marker=marker, s=size)
    #Sphere arrows
    for i in range(2):
        theta_arrow = args_solution + np.pi*i
        phi_arrow = 0
        x_arrow = np.sin(theta_arrow) * np.cos(phi_arrow)
        y_arrow = np.sin(theta_arrow) * np.sin(phi_arrow)
        z_arrow = np.cos(theta_arrow)
        ax2.quiver(0, 0, 0, x_arrow, y_arrow, z_arrow, color=c[i], arrow_length_ratio=0.1, linewidth=3)
    #
    suptitle = 'Neel'
    #View
    ax2.view_init(30,50)
elif ans == 'mango':   #coplanar
    c1 = colors[:6]
    ind_c = [1,0,3,2,4,5]   #dark magic, don't touch it
    for x in range(3):
        for y in range(3):
            off = offset+x*T1+y*T2
            for i in range(6):
                color = c1[ind_c[2*((x+y)%3)+(i%2)]] #if i%2==0 else c2[(x+y)%3]
                X = T1*i_UC[i][0] + T2*i_UC[i][1] + off
                ax.scatter(X[0],X[1],color=color, lw=0, marker=marker, s=size)
    #Sphere arrows
    th,ph,et = args_solution
    ep = fs_cpd.get_discrete_index(ind_discrete,ans)
    S1 = np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
    #######################################################################################
    #######################################################################################
    #######################################################################################
    GR6 = ep*np.array([[np.cos(2*et),np.sin(2*et),0],[np.sin(2*et),-np.cos(2*et),0],[0,0,-1]])
    GT = 1/2*np.array([[-1,-np.sqrt(3),0],[np.sqrt(3),-1,0],[0,0,1]])
    exp_t = [0,0,1,1,2,2]
    exp_r = [1,0,1,0,0,1]
    for i in range(6):  #SB unit cell
        x_arrow,y_arrow,z_arrow = np.linalg.matrix_power(GT,exp_t[i])@np.linalg.matrix_power(GR6,exp_r[i])@S1
        ax2.quiver(0, 0, 0, x_arrow, y_arrow, z_arrow, color=c1[i], arrow_length_ratio=0.1, linewidth=3)
    #angle between sublattices
    delta_ph = fs_ssf.angle_between_vectors(S1,GR6@S1)
    theta_al = np.linspace(ph, ph+delta_ph, 100)
    x_al = np.cos(theta_al)*0.5
    y_al = np.sin(theta_al)*0.5
    z_al = np.zeros_like(theta_al)
    ax2.plot(x_al, y_al, z_al, color='k', linewidth=1)
    ax2.text(x_al[50]+0., y_al[50]+0.1, 0.1,r'$\Delta\phi$', color='k')
    #
    title = ans
    #View
    ax2.view_init(30,50)

    suptitle = title+' order at '+r'$J_d=$'+"{:.2f}".format(Jd)+', '+r'$J_t=$'+"{:.2f}".format(Jt)
elif ans == 'banana':
    th,ph = args_solution
    ep,e1,e2 = fs_cpd.get_discrete_index(ind_discrete,ans)
    S1 = np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
    GR6 = ep*np.array([[0,e1,0],[0,0,e2],[e1*e2,0,0]])
    GT1 = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    GT2 = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
    #Sphere arrows
    S1 = np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
    #
    LW = 0 if ep==1 else 1
    c1 = colors[:12]
    for x in range(3):
        for y in range(3):
            off = offset+x*T1+y*T2
            for i in range(6):
                lw = LW if i%2==1 else 0
                color = c1[i%3+3*(x%2)+6*(y%2)]
                X = T1*i_UC[i][0] + T2*i_UC[i][1] + off
                ax.scatter(X[0],X[1],color=color, linewidths=lw, edgecolors='k', marker=marker, s=size)
            #
            continue
    exp_t1 = [0,0,0,0,1,1]
    exp_t2 = [0,0,1,1,1,1]
    exp_r = [5,0,3,4,2,1]
    for i in range(12):
        x = i//3
        y = i//6
        r = i%3
        S = np.linalg.matrix_power(GT1,x)@np.linalg.matrix_power(GT2,y)@np.linalg.matrix_power(GR6,r)@S1
        ax2.quiver(0, 0, 0, S[0], S[1], S[2], color=c1[i], arrow_length_ratio=0.1, linewidth=3)
    #
    title = 'Non-Coplanar'
    #View
    ax2.view_init(10,10)
    if 0 and len(title2)>1:   #inverted spin figure
        gs = gridspec.GridSpec(8, 2, figure=fig)
        ax3 = fig.add_subplot(gs[3,1])
        ax3.arrow(0,0,0,0.8,color='b',width=0.05,head_starts_at_zero=False)
        ax3.arrow(1,1,0,-0.8,color='b',width=0.05,head_starts_at_zero=False)
        circle = Circle((0,0.5),radius=0.6, fill=False, edgecolor='k',lw=2) 
        ax3.add_patch(circle)
        ax3.set_aspect('equal')
        ax3.text(0.6,0.5,r'$=$',size=20)
        ax3.axis('off')

    suptitle = title +' order at '+r'$J_d=$'+"{:.2f}".format(Jd)+', '+r'$J_t=$'+"{:.2f}".format(Jt)

if 1:#SSF
    UC = 71
    fx,fy = (6,10)    #Just how many units to consider 
    nkx, nky = fs_ssf.get_kpoints((fx,fy),150)
    vecx = np.pi*2/np.sqrt(21)#fs.B_[1,0] if high_symm=='B' else fs.b_[1,0]
    vecy = np.pi*2/3/np.sqrt(7)#fs.B_[1,1]/3 if high_symm=='B' else fs.b_[1,1]/3
    kxs = np.linspace(-vecx*fx,vecx*fx,nkx)
    kys = np.linspace(-vecy*fy,vecy*fy,nky)
    SSFzz_fn = fs_ssf.get_ssf_fn('zz',ans,ind_discrete,Jd,Jt,UC,nkx,nky)
    SSFxy_fn = fs_ssf.get_ssf_fn('xy',ans,ind_discrete,Jd,Jt,UC,nkx,nky)
    print(SSFzz_fn)
    if Path(SSFzz_fn).is_file():
        SSFzz = np.load(SSFzz_fn)
        SSFxy = np.load(SSFxy_fn)
        #
        tt = ['zz','xy']
        X,Y = np.meshgrid(kxs,kys)
        for i in range(2):
            data = SSFzz if i == 0 else SSFxy
            ax = fig.add_subplot(gs_large[1,i])
            sc = ax.scatter(X.T,Y.T,c=data,
                        marker='o',
                        cmap=cm.plasma_r,
                        s=5,
                        norm=None #if i==0 else norm
                      )
            fs_ssf.plot_BZs(ax)
            ax.set_xlim(kxs[0],kxs[-1])
            ax.set_ylim(kys[0],kys[-1])
            title = "SSF "+tt[i]
            ax.set_title(title)
            plt.colorbar(sc)
            ax.set_aspect('equal')
    else:
        print("SSF not computed")
#


#fig.suptitle(suptitle,size=20)

fig.tight_layout()
fig.show()
if input("Save?[y/N]")=='y':
    fig.savefig('results/fig_'+ans+str(ind_discrete)+'_'+str(index)+'.png')
#plt.show()



