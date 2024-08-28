import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from numba import jit
from scipy.ndimage.filters import gaussian_filter
import os
# os.chdir("Base")
# @jit(nopython=True)
def update_surface(x,y,r,asp,nx,ny,theta,dh,surface):
    for ii in np.arange(max(0,int(y-r*asp)),min(int(y+r*asp),ny)):
        for jj in np.arange(max(0,int(x-r*asp)),min(int(x+r*asp),nx)):
            dx=jj-x
            dy=ii-y
            #rotated coordinate
            dx2=dx*np.cos(theta)-dy*np.sin(theta)
            dy2=dx*np.sin(theta)+dy*np.cos(theta)
            r1=np.sqrt((dx2/asp)**2+dy2**2)

            if r1**2<=r**2:
                dz0=-dh/(r**2)*(r1**2)+dh
                surface[ii,jj]=surface[ii,jj]+dz0    
    return 0

# @jit(nopython=True)
def assign_prop(xchange,ychange,x,y,theta,surface0,surface,facies,r,poro,dz,asp,nz,i):
    for n in range(xchange.size):
        ii=ychange[n]
        jj=xchange[n]    
        dx=jj-x
        dy=ii-y
        #rotated coordinate
        dx2=dx*np.cos(theta)-dy*np.sin(theta)
        dy2=dx*np.sin(theta)+dy*np.cos(theta)
        r1=np.sqrt((dx2/asp)**2+dy2**2)            
        #assign value
        bot=int(np.rint(surface0[ii,jj]))
        top=int(min(np.rint(surface[ii,jj]),nz))
#        facies[max(top-4,0):top,ii,jj][facies[max(top-4,0):top,ii,jj]==0]=i
        if top>bot:
            facies[bot:top,ii,jj][facies[bot:top,ii,jj]==0]=i
#        if top-bot>=1:
            for kk in np.arange(bot,top):
    #            Rz=(1-np.sqrt((kk-bot)/dz[ii,jj]))*r
                poro[kk,ii,jj]=(((top-kk)/(top-bot))**1)*0.3*(1-(r1/r))*((1-kk/nz)/2+0.5)+0.05
#            if facies[kk,ii,jj]==i:
#                poro[kk,ii,jj]=0.05
#            
    return 0

#from scipy.ndimage.filters import gaussian_filter
def lobemodeling(surface,nx=50,ny=50,nz=50,dhmax=4,dhmin=4,rmin=42,rmax=44,asp=1.5,theta0=60,m=2):
#    nx=112;ny=112;nz=56;dhmax=8;dhmin=10;rmin=16;rmax=20;asp=1.5;theta0=60;m=2
    facies=np.zeros((nz,ny,nx))
    poro=facies.copy()-0.1
    allsurface=[]
    # surface=0.000001*np.ones((ny,nx))#for prop
    surface0=surface.copy()#old surface
    lat_size=nx*ny
    loc_idx=np.arange(lat_size)
    theta0=theta0/180*np.pi
    allfacies=[]
    allporo=[]
    allsurface.append(surface.copy())
    start=0 #when to start add lobe
    
#    dv=[]
    i=0
    iiii=2000
    while(i<iiii-1):
        
#        poro[facies==2]=0.05
        theta=theta0+np.random.normal(0,20/180*np.pi)
        dh=np.random.uniform(dhmin,dhmax)
        r=np.random.uniform(rmin,rmax)
        surface0=surface.copy()
        #calculate prob
        zz=surface
        prob=(1/(surface-zz.min()+0.001)**m)/np.sum(1/(surface-zz.min()+0.001))
        
   
        #choose loc
        prob=prob/np.sum(prob)
        prob_flat=prob.flatten()
        loc=np.random.choice(loc_idx,p=prob_flat)
        y=loc//nx
        x=loc-nx*y
    
        #update surface
        update_surface(x,y,r,asp,nx,ny,theta,dh,surface)    
#        print(i)
        if i!=0:            
            #do healing correction
            surface2=surface.copy()#initial volume
            surface=surface0+(surface-surface0)*(1-(surface0/surface0.max())**1.2)# after healing volume
            dsurface=(surface-surface0)
            
            #compensate volume
            surface=surface0+dsurface*(np.sum(surface2-surface0)/np.sum(dsurface+0.000000001))
            
        #assign property
        dz=surface-surface0
        ychange,xchange=np.where(dz>0)
        if i>start-0.1:
            allsurface.append(surface.copy())
            assign_prop(xchange,ychange,x,y,theta,surface0,surface,facies,r,poro,dz,asp,nz,i-start+1) 
        i+=1
        # print(i)
        if i==iiii or surface.max()>=nz+4:
            allfacies.append(facies.copy())
            allporo.append(poro.copy())
            break
    return allfacies,allporo,allsurface



np.random.seed(20)
poro=[]
facies=[]

# generate a simple one 
nx=50;ny=50;nz=50
surface=0.000001*np.ones((ny,nx))
allfacies,allporo,allsurface=lobemodeling(surface,nx=nx,ny=ny,nz=nz,dhmax=4,dhmin=4,rmin=42,rmax=44,asp=1.5,theta0=0,m=100)     
prop=allporo[-1]


allsurface=np.stack(allsurface,axis=0)
np.save('poro0.npy',prop)
np.save('facies0.npy',allfacies[0])
np.save('surface0.npy',allsurface)

np.random.seed(10)
poro=[]
facies=[]
nx=50;ny=50;nz=50
surface=0.000001*np.ones((ny,nx))

allfacies,allporo,allsurface=lobemodeling(surface,nx=nx,ny=ny,nz=nz,dhmax=4,dhmin=4,rmin=42,rmax=44,asp=1.5,theta0=0,m=100)     
prop=allporo[-1]

allsurface=np.stack(allsurface,axis=0)
np.save('poro1.npy',prop)
np.save('facies1.npy',allfacies[0])
np.save('surface1.npy',allsurface)



            
        