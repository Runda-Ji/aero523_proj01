# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 15:14:58 2018

@author: rundaji
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

def projecting(nb,s,u):
    c=np.zeros((20,nb));
    #compute the coef infront of each basis
    for k in range(0,20):
        vec=s[:,k];
        for i in range(0,nb):
            basis=u[:,i];
            magnitude=np.sqrt(basis.T*basis);
            c[k][i]=(vec.T*basis)/magnitude;
    return c;

def func_error_linear(a,X,c):
    M=X[:,0];
    alpha=X[:,1];
    c_fitted=a[0]+a[1]*M+a[2]*alpha;
    error=c-c_fitted;
    return error;

def func_error_quadratic(b,X,c):
    M=X[:,0];
    alpha=X[:,1];
    c_fitted=b[0]+b[1]*M+b[2]*alpha+b[3]*M**2+b[4]*M*alpha+b[5]*alpha**2;
    error=c-c_fitted;
    return error;

def compare(nb,a,b,s,u):
    M=np.arange(0.5,0.8001,0.3/9);
    alpha=np.arange(0,4.001,4.0/9);
    L2_error_linear=np.zeros((10,10));
    L2_error_quadratic=np.zeros((10,10));
    sum_linear=[0]*100;
    sum_quadratic=[0]*100;
    for i in range(0,10):   
        for j in range(0,10):
            k=10*i+j;
            vec=s[:,k];
            for n in range(0,nb):
                basis=u[:,n];
                c_linear=a[n][0]+a[n][1]*M[j]+a[n][2]*alpha[i];
                c_quadratic=b[n][0]+b[n][1]*M[j]+b[n][2]*alpha[i]+b[n][3]*M[j]**2+b[n][4]*M[j]*alpha[i]+b[n][5]*alpha[i]**2;
                sum_linear[k] += c_linear*basis;
                sum_quadratic[k] += c_quadratic*basis;
            
            res_linear=vec-sum_linear[k];
            res_quadratic=vec-sum_quadratic[k];
            L2_error_linear[i][j]=np.sqrt((res_linear.T*res_linear)/len(res_linear));
            L2_error_quadratic[i][j]=np.sqrt((res_quadratic.T*res_quadratic)/len(res_quadratic));
    
    z_1=L2_error_linear;
    z_2=L2_error_quadratic;
    x,y=np.meshgrid(M,alpha);
    plot_L2_error(x,y,z_1,z_2,nb,'..\\figure\\question_4\\Fig_Fitting_error_%d_basis.png' %nb);
    return sum_linear,sum_quadratic;
    
def plot_L2_error(x,y,z_1,z_2,nb,fname):
    f=plt.figure(figsize=(16,6));
    
    plt.subplot(1, 2, 1);
    plt.contourf(x,y,z_1,20,cmap=plt.cm.jet);
    plt.xlabel(r'Mach number, $M$',fontsize =16);
    plt.ylabel(r'angle of attack, $\alpha$ [degrees]',fontsize =16);
    plt.xlim((0.45,0.85));
    plt.ylim((0,4));
    plt.title('$L_2$ error of linear fit, with %d basis' %nb,fontsize=16);
    plt.colorbar();
    
    plt.subplot(1, 2, 2);
    plt.contourf(x,y,z_2,20,cmap=plt.cm.jet);
    plt.xlabel(r'Mach number, $M$',fontsize =16);
    #plt.ylabel(r'angle of attack, $\alpha$ [degrees]',fontsize =16);
    plt.xlim((0.45,0.85));
    plt.ylim((0,4));
    plt.title('$L_2$ error of quadratic fit, with %d basis' %nb,fontsize=16);
    plt.colorbar();
    
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
    
    #load train states s
    i=0
    s=np.loadtxt('..\\data\\train_%d_state.txt' %(i+1),dtype='float');
    s=np.reshape(s,(-1,1)); 
    for i in range(1,20): #i th snapshot
        col=np.loadtxt('..\\data\\train_%d_state.txt' %(i+1),dtype='float');
        col=np.reshape(col,(-1,1));
        s=np.hstack((s,col));
    s=np.asmatrix(s);
    
    #load basis u
    u=np.load('..\\output\\truncated_svd_u.npy');
    u=np.asmatrix(u);
    
    #load Mach number and attack angle
    Malpha = np.loadtxt('..\\data\\Malpha.txt',dtype='float');
    
    COEF_ALL=projecting(16,s,u);
    
    c=[None]*16;
    a=[None]*16;
    b=[None]*16;
    a_initial=[0.0,0.0,0.0];
    b_initial=[0.0,0.0,0.0,0.0,0.0,0.0];
    for i in range(0,16):
        c[i]=COEF_ALL[:,i];
        #coef in front basis 1 thru 16
        a[i],success=leastsq(func_error_linear,a_initial,args=(Malpha, c[i]));
        b[i],success=leastsq(func_error_quadratic,b_initial,args=(Malpha, c[i]));
    a=np.asarray(a);
    b=np.asarray(b);
    
    file = open("..\\output\\a&b.txt", 'w');
    for i in range(0,4):
        file.write('%d %d %d\n' %(a[i][0],a[i][1],a[i][2]));
    for i in range(0,4):
        file.write('%d %d %d %d %d %d\n' %(b[i][0],b[i][1],b[i][2],b[i][3],b[i][4],b[i][5]));
    file.close();
    
    #load test states s
    i=0
    s=np.loadtxt('..\\data\\test_%d_state.txt' %(i+1),dtype='float');
    s=np.reshape(s,(-1,1)); 
    for i in range(1,100): #i th snapshot
        col=np.loadtxt('..\\data\\test_%d_state.txt' %(i+1),dtype='float');
        col=np.reshape(col,(-1,1));
        s=np.hstack((s,col));
    s=np.asmatrix(s);
    
    linear_fitted_55=[None]*4;
    linear_fitted_90=[None]*4;
    quadratic_fitted_55=[None]*4;
    quadratic_fitted_90=[None]*4;
    for i in range(0,4):
        nb=2**(i+1);
        LINEAR_ALL,QUADRATIC_ALL=compare(nb,a,b,s,u);
        linear_fitted_55[i]=np.reshape(LINEAR_ALL[55-1],(-1,5));
        linear_fitted_90[i]=np.reshape(LINEAR_ALL[90-1],(-1,5));
        quadratic_fitted_55[i]=np.reshape(QUADRATIC_ALL[55-1],(-1,5));
        quadratic_fitted_90[i]=np.reshape(QUADRATIC_ALL[90-1],(-1,5));
    
    linear_fitted_55=np.asarray(linear_fitted_55);
    linear_fitted_90=np.asarray(linear_fitted_90);
    quadratic_fitted_55=np.asarray(quadratic_fitted_55);
    quadratic_fitted_90=np.asarray(quadratic_fitted_90);
    
    findmach(4,linear_fitted_55,'Linearly fitted test snapshot 55 using %d basis','..\\figure\\question_4\\Fig_linear_fitted_55_projected_on_%d_basis');
    findmach(4,linear_fitted_90,'Linearly fitted test snapshot 90 using %d basis','..\\figure\\question_4\\Fig_linear_fitted_90_projected_on_%d_basis');
    findmach(4,quadratic_fitted_55,'Quadratically fitted test snapshot 55 using %d basis','..\\figure\\question_4\\Fig_quadratic_fitted_55_projected_on_%d_basis');
    findmach(4,quadratic_fitted_90,'Quadratically fitted test snapshot 90 using %d basis','..\\figure\\question_4\\Fig_quadratic_fitted_90_projected_on_%d_basis');
    
    return 0;

if __name__=="__main__":
    main()