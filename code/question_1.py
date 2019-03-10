# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 15:25:55 2018

@author: rundaji
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def plotsnapshot(x,y,tri,M,fname):
    f=plt.figure(figsize=(8,6));
    plt.tricontourf(x,y,tri,M,20,cmap=plt.cm.jet);
    plt.xlabel(r'$x$',fontsize =16);
    plt.ylabel(r'$y$',fontsize =16);
    plt.title('Mach Number',fontsize=16);
    plt.axis('equal');
    plt.xlim((-0.3,1.5));
    plt.ylim((-0.5,0.5));    
    plt.colorbar();
    f.tight_layout();
    plt.savefig(fname,dpi=150);
    plt.close(f);
    return 0;
    
def findmach(N_snapshot,state,fname):
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
            speed=math.sqrt(u**2+v**2);
            p=(gamma-1)*(rhoE-0.5*rho*(speed**2));
            a=math.sqrt(gamma*p/rho);
            if M[i]==None:
                M[i]=[speed/a];
            else:
                M[i].append(speed/a);
        plotsnapshot(x,y,tri,M[i],fname %(i+1));
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
    state_train=[None]*20;
    state_test=[None]*100;
    
    for i in range(0,20):
        state_train[i] = np.loadtxt('..\\data\\train_%d_state.txt' %(i+1),dtype='float');
    for i in range(0,100):
        state_test[i] = np.loadtxt('..\\data\\test_%d_state.txt' %(i+1),dtype='float');

    findmach(20,state_train,'..\\figure\\question_1\\Fig_train_%d.png');
    findmach(100,state_test,'..\\figure\\question_1\\Fig_test_%d.png');

    return 0;

if __name__=="__main__":
    main()