# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:17:30 2018

@author: rundaji
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

def plotsemilog(y,fname):
    f=plt.figure(figsize=(8,6));
    index=np.arange(1,21);
    plt.semilogy(index,y,'.-');
    x = range(1,21);
    plt.xticks(x);
    plt.xlabel('Index of singular values',fontsize =16);
    plt.ylabel('Singular values',fontsize =16);
    plt.grid();
    plt.savefig(fname,dpi=150);
    plt.show();
    plt.close(f);
    return 0;

def plotsol(z,t,fname):
    global x,y,tri;
    f=plt.figure(figsize=(8,6));
    plt.tricontourf(x,y,tri,z,20,cmap=plt.cm.jet);
    plt.xlabel(r'$x$',fontsize =16);
    plt.ylabel(r'$y$',fontsize =16);
    plt.axis('equal');
    plt.xlim((-0.3,1.5));
    plt.ylim((-0.5,0.5));    
    plt.colorbar();
    plt.title(t); 
    f.tight_layout();
    plt.savefig(fname,dpi=150);
    plt.show();
    plt.close(f);
    return 0;

def plot4basis(u):
    for i in range(0,4):
        state=u[:,i];
        state=np.reshape(state,(-1,5));
        rho=state[:,0].T;
        x_momentum=state[:,1].T;
        #get rid of [[]]
        rho=np.asarray(rho);
        rho=rho[0];
        x_momentum=np.asarray(x_momentum);
        x_momentum=x_momentum[0];
        plotsol(rho,'Density, $\\rho$, basis %d' %(i+1),'..\\figure\\question_2\\Fig_eig(ATA)_rho_vector_%d.png' %(i+1));
        plotsol(x_momentum,'X-momentum, $\\rho u$, basis %d' %(i+1),'..\\figure\\question_2\\Fig_eig(ATA)_x_momentum_vector_%d.png' %(i+1));
    return 0;

def main():
    global x,y,tri;
    global u,s,vh;
    #load coordinates
    V = np.loadtxt('..\\data\\V.txt',dtype='float');
    #L_V=len(V);
    #load info of triangles
    E = np.loadtxt('..\\data\\E.txt',dtype='int');
    #L_E=len(E);

    x=V[:,0];
    y=V[:,1];
    tri = np.asarray(E)-1;
    state=[None]*20;
    A_row=[None]*20;
    
    for i in range(0,20): #i th snapshot
        #load states
        state[i] = np.loadtxt('..\\data\\train_%d_state.txt' %(i+1),dtype='float');
        #transform list to arrary
        state[i]=np.asarray(state[i]);
        #reshape s[i] to a row
        A_row[i]=np.reshape(state[i],(1,-1));
        #get rid of extra []
        A_row[i]=A_row[i][0];
    
    #combine the rows together
    #transform arrary to matrix
    #each row is one of the 20 state vectors
    #Use a transpose to make those rows become columns -> column vectors
    A=np.asmatrix(A_row).T;
    Density_Matrix=A.T*A;
    w, v = la.eig(Density_Matrix);
    s = np.sqrt(w);
    plotsemilog(s,'..\\figure\\question_2\\Fig_eig(ATA)_Singular_Value.png');
    
    u=A*v[:,0]/s[0]
    for i in range(1,20):
        col=A*v[:,i]/s[i];
        u=np.hstack((u,col));
    #in this method, u is mostly positive
    np.save('..\\output\\eig(ATA)_u.npy',u);
    plot4basis(u);
    return 0;

if __name__=="__main__":
    main()