# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 15:44:35 2018

@author: rundaji
"""

import numpy as np
import matplotlib.pyplot as plt

def projecting(nb,s,u):
    c=np.zeros((100,nb));
    L2_error=np.zeros((10,10));
    #projection for specific nb
    proj=[0]*100;
    #compute the coef infront of each basis
    for k in range(0,100):
        vec=s[:,k];
        for i in range(0,nb):
            basis=u[:,i];
            magnitude=np.sqrt(basis.T*basis);
            c[k][i]=(vec.T*basis)/magnitude;
            proj[k] += c[k][i]*basis;
        res=vec-proj[k];
        I=int(k/10);
        J=np.remainder(k,10);
        L2_error[I][J]=np.sqrt((res.T*res)/len(res));
        
    z=L2_error;
    M=np.arange(0.5,0.8001,0.3/9);
    alpha=np.arange(0,4.001,4.0/9);
    x,y=np.meshgrid(M,alpha);
    plot_L2_error(x,y,z,nb,'..\\figure\\question_3\\Fig_L2_error_%d_basis.png' %nb);
    return proj;

def plot_L2_error(x,y,z,nb,fname):
    f=plt.figure(figsize=(8,6));
    plt.contourf(x,y,z,20,cmap=plt.cm.jet);
    plt.xlabel(r'Mach number, $M$',fontsize =16);
    plt.ylabel(r'angle of attack, $\alpha$ [degrees]',fontsize =16);
    plt.xlim((0.45,0.85));
    plt.ylim((0,4));    
    plt.colorbar();
    plt.title('$L_2$ error, with %d basis' %nb);
    f.tight_layout();
    plt.savefig(fname,dpi=150);
    plt.show();
    plt.close(f);
    return 0;

def plotsnapshot(x,y,tri,M,title,fname):
    f=plt.figure(figsize=(8,6));
    plt.tricontourf(x,y,tri,M,20,cmap=plt.cm.jet);
    plt.xlabel(r'$x$',fontsize =16);
    plt.ylabel(r'$y$',fontsize =16);
    plt.axis('equal');
    plt.xlim((-0.3,1.5));
    plt.ylim((-0.5,0.5));    
    plt.colorbar();
    plt.title(title);
    f.tight_layout();
    plt.savefig(fname,dpi=150);
    plt.show();
    plt.close(f);
    return 0;

def findmach(N_snapshot,state,title,fname):
    global x,y,tri;
    gamma=1.4;
    M=[None]*N_snapshot;
    for i in range(0,N_snapshot): #i th snapshot
        for j in range(0,len(state[i])):
            #j th node
            rho=state[i][j][0];
            rhou=state[i][j][1];
            rhov=state[i][j][2];
            rhoE=state[i][j][3];
            u=rhou/rho;
            v=rhov/rho;
            speed=np.sqrt(u**2+v**2);
            p=(gamma-1)*(rhoE-0.5*rho*(speed**2));
            a=np.sqrt(gamma*p/rho);
            if M[i]==None:
                M[i]=[speed/a];
            else:
                M[i].append(speed/a);
        plotsnapshot(x,y,tri,M[i],title %2**(i+1),fname %2**(i+1));
    return 0;

def main():
    global x,y,tri;
    #load coordinates
    V = np.loadtxt('..\\data\\V.txt',dtype='float');
    #L_V=len(V);
    #load info of triangles
    E = np.loadtxt('..\\data\\E.txt',dtype='int');
    #L_E=len(E);
    x=V[:,0];
    y=V[:,1];
    tri = np.asarray(E)-1;
    #load test states s
    i=0
    s=np.loadtxt('..\\data\\test_%d_state.txt' %(i+1),dtype='float');
    s=np.reshape(s,(-1,1)); 
    for i in range(1,100): #i th snapshot
        col=np.loadtxt('..\\data\\test_%d_state.txt' %(i+1),dtype='float');
        col=np.reshape(col,(-1,1));
        s=np.hstack((s,col));
    s=np.asmatrix(s);
    #load basis u
    u=np.load('..\\output\\truncated_svd_u.npy');
    u=np.asmatrix(u);
    
    snapshot_55=[None]*4;
    snapshot_90=[None]*4;
    #PROJECTION for different nb
    for i in range(0,4):
        nb=2**(i+1);
        PROJ_ALL=projecting(nb,s,u);
        state=PROJ_ALL[55-1];
        state=np.reshape(state,(-1,5));
        snapshot_55[i]=state;
        state=PROJ_ALL[90-1];
        state=np.reshape(state,(-1,5));
        snapshot_90[i]=state;
    
    snapshot_55=np.asarray(snapshot_55);
    snapshot_90=np.asarray(snapshot_90);
    findmach(4,snapshot_55,'Test snapshot 55 projected on %d basis','..\\figure\\question_3\\Fig_test_55_projected_on_%d_basis');
    findmach(4,snapshot_90,'Test snapshot 90 projected on %d basis','..\\figure\\question_3\\Fig_test_90_projected_on_%d_basis');
    
    return 0;

if __name__=="__main__":
    main()