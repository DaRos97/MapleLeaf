import numpy as np
import matplotlib.pyplot as plt

def plot_lattice(ax,*args):
    x,lw,c,ls = args
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
