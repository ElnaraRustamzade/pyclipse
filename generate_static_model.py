import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
# units field

def gen_static_model(upper_zone_thickness=500, lower_zone_thickness=500, ntg1=0.7, ntg2=0.62, perm1=np.log10(10), perm2=np.log10(20)):
    np.random.seed(680)
    nx=50
    ny=50
    nz=101# 1 layer for shale barrier
    nz1=50
    nz2=50
    nz_sh=1
    nz22=nz1+nz_sh

    top=18000
    well_space=4173

    x_len=15000
    y_len=15000
    x_size=x_len/nx

    shale_thickness = 350 # 1 shale 
    dip=20

    por1=0.17
    por2=0.19
    
    ntg1 = 1-ntg1
    ntg2 = 1-ntg2

    por_std=0.03
    perm_std=0.5

    sand_filt=[1.5,2.5,1.5]
    facies_filt=[2.5,5,2.5]
    sand_nug=0.05
    # important surface calculation
    x=np.linspace(0, x_len,nx+1)
    y=np.linspace(0, y_len,ny+1)
    X,Y=np.meshgrid(x,y,indexing='ij')
    z1=Y*np.tan(dip/180*np.pi)+top
    z2=z1+upper_zone_thickness
    z3=z2+shale_thickness
    z4=z3+lower_zone_thickness

    surf_name=['sand1_top','sand1_bot','sand2_top','sand2_bot']
    zz=[z1,z2,z3,z4]

    # for ii in range(4):
    #     with open('./outputs/'+surf_name[ii], 'w') as f:
    #         for i in range(X.shape[0]):
    #             for j in range(X.shape[1]):             
    #                 f.write('%.6f %.6f %.6f\n' % (X[i,j],Y[i,j],zz[ii][i,j]))

    myfolder='./eclipse_files/'
    with open(myfolder+'COORD', 'w') as f:
        f.write('COORD                                   -- Generated : Petrel\n ')
        val_in_line = 0
        for j in range(X.shape[1]-1,-1,-1):  
            for i in range(X.shape[0]):
                if val_in_line == 2:
                    f.write("\n ")
                    val_in_line = 0
                val_in_line += 1         
                f.write(" {:.7g} {:.7g} {:.7g} {:.7g} {:.7g} {:.7g}".format(X[i,j],-Y[i,j],zz[0][i,j],X[i,j],-Y[i,j],zz[-1][i,j]))
        f.write(" /\n")

    with open(myfolder+'ZCORN', 'w') as f:
        f.write('ZCORN                                   -- Generated : Petrel\n ')
        val_in_line = 0
        k_array = np.arange(nz1+1)
        k_array = np.concatenate(([k_array[0]], np.repeat(k_array[1:-1], 2), [k_array[-1]]))
        for k in k_array:
            for j in range(X.shape[1]-1,-1,-1):
                if val_in_line == 10:
                    f.write("\n ")
                    val_in_line = 0
                val_in_line += 1
                if j==0 or j==X.shape[1]-1:
                    if k==0:
                        f.write(" {}*{:.7g}".format(2*nx,zz[0][0,j]))
                    elif k==nz1:
                        f.write(" {}*{:.7g}".format(2*nx,zz[1][0,j]))
                    else:
                        f.write(" {}*{:.7g}".format(2*nx,zz[0][0,j]+k*(zz[1][0,j]-zz[0][0,j])/nz1))
                else:
                    if k==0:
                        f.write(" {}*{:.7g}".format(4*nx,zz[0][0,j]))
                    elif k==nz1:
                        f.write(" {}*{:.7g}".format(4*nx,zz[1][0,j]))
                    else:
                        f.write(" {}*{:.7g}".format(4*nx,zz[0][0,j]+k*(zz[1][0,j]-zz[0][0,j])/nz1))

        for k in range(2):
            for j in range(X.shape[1]-1,-1,-1):
                if val_in_line == 10:
                    f.write("\n ")
                    val_in_line = 0
                val_in_line += 1
                if j==0 or j==X.shape[1]-1:
                    if k==0:
                        f.write(" {}*{:.7g}".format(2*nx,zz[1][0,j]))
                    elif k==1:
                        f.write(" {}*{:.7g}".format(2*nx,zz[2][0,j]))
                else:
                    if k==0:
                        f.write(" {}*{:.7g}".format(4*nx,zz[1][0,j]))
                    elif k==1:
                        f.write(" {}*{:.7g}".format(4*nx,zz[2][0,j]))

        k_array = np.arange(nz2+1)
        k_array = np.concatenate(([k_array[0]], np.repeat(k_array[1:-1], 2), [k_array[-1]]))
        for k in k_array:
            for j in range(X.shape[1]-1,-1,-1):
                if val_in_line == 10:
                    f.write("\n ")
                    val_in_line = 0
                val_in_line += 1
                if j==0 or j==X.shape[1]-1:
                    if k==0:
                        f.write(" {}*{:.7g}".format(2*nx,zz[2][0,j]))
                    elif k==nz2:
                        f.write(" {}*{:.7g}".format(2*nx,zz[3][0,j]))
                    else:
                        f.write(" {}*{:.7g}".format(2*nx,zz[2][0,j]+k*(zz[3][0,j]-zz[2][0,j])/nz2))
                else:
                    if k==0:
                        f.write(" {}*{:.7g}".format(4*nx,zz[2][0,j]))
                    elif k==nz2:
                        f.write(" {}*{:.7g}".format(4*nx,zz[3][0,j]))
                    else:
                        f.write(" {}*{:.7g}".format(4*nx,zz[2][0,j]+k*(zz[3][0,j]-zz[2][0,j])/nz2))
        f.write(" /\n")

    np.random.seed(250)
    # import lobe modeling results
    lobe_poro0=np.load('poro0.npy')
    lobe_poro1=np.load('poro1.npy')

    # change axis from zyx to xyz
    lobe_poro0=np.swapaxes(lobe_poro0, 0, -1)
    lobe_poro1=np.swapaxes(lobe_poro1, 0, -1)
    # facies modeling
    # these are perturbation
    lambda_perturb=0.1 # degree of local perturbation
    facies=gaussian_filter(np.random.normal(0,1,(3*nx,3*ny,3*nz)),facies_filt,mode='wrap')
    facies=facies[nx:2*nx,ny:2*ny,nz:2*nz]
    # add perturbation
    facies[:,:,:nz1]=lobe_poro0+lambda_perturb*facies[:,:,:nz1]
    facies[:,:,nz1+nz_sh:]=lobe_poro1+lambda_perturb*facies[:,:,nz1+nz_sh:]
    # change to be categorical
    facies[:,:,:nz1]=facies[:,:,:nz1]>np.percentile(facies[:,:,:nz1].flatten(),ntg1*100)
    facies[:,:,nz1:nz1+nz_sh]=False
    facies[:,:,nz1+nz_sh:]=facies[:,:,nz1+nz_sh:]>np.percentile(facies[:,:,nz1+nz_sh:].flatten(),ntg2*100)

    # property modeling
    poro=np.random.normal(0,1,(3*nx,3*ny,3*nz))
    poro_nug=np.random.normal(0,1,(3*nx,3*ny,3*nz))
    # sand poro
    # generate stochastic perturbation
    poro=gaussian_filter(poro,sand_filt,mode='wrap')+sand_nug*poro_nug
    poro=poro[nx:2*nx,ny:2*ny,nz:2*nz]
    # add trend
    poro[:,:,:nz1]=lobe_poro0+lambda_perturb*poro[:,:,:nz1]
    poro[:,:,nz1+nz_sh:]=lobe_poro1+lambda_perturb*poro[:,:,nz1+nz_sh:]

    perm=poro.copy()

    # for uppepr unit
    # percentile transform
    mean=[0,0]
    var=[[1,0.6],[0.6,1]]
    mm= multivariate_normal(mean,var)
    sand1=mm.rvs(int(poro[:,:,:nz1].size*1.2))
    sand1[:,0]=sand1[:,0]*por_std+por1
    sand1[:,1]=sand1[:,1]*perm_std+perm1

    sand1=sand1[sand1[:,0]<0.3]
    sand1=sand1[sand1[:,0]>0.05]
    sand1=sand1[sand1[:,1]<3.5]
    sand1=sand1[:poro[:,:,:nz1].size]

    # quantile transformation to force it to be gaussian
    por_zone1=poro[:,:,:nz1].flatten()
    por_zone1_order=np.argsort(por_zone1)
    sand1=sand1[np.argsort(sand1[:poro[:,:,:nz1].size,0])]
    por_zone1[por_zone1_order]=sand1[:,0]
    perm_zone1=por_zone1.copy()
    perm_zone1[por_zone1_order]=sand1[:,1]
    poro[:,:,:nz1]=por_zone1.reshape(poro[:,:,:nz1].shape)
    perm[:,:,:nz1]=perm_zone1.reshape(poro[:,:,:nz1].shape)

    # for lower unit
    mean=[0,0]
    var=[[1,0.6],[0.6,1]]
    mm= multivariate_normal(mean,var)
    sand2=mm.rvs(int(poro[:,:,nz22:].size*1.2))
    sand2[:,0]=sand2[:,0]*por_std+por2
    sand2[:,1]=sand2[:,1]*perm_std+perm2
    sand2=sand2[sand2[:,0]<0.3]
    sand2=sand2[sand2[:,0]>0.05]
    sand2=sand2[sand2[:,1]<3.5]
    sand2=sand2[:poro[:,:,nz22:].size]

    por_zone2=poro[:,:,nz22:].flatten()
    por_zone2_order=np.argsort(por_zone2)
    sand2=sand2[np.argsort(sand2[:poro[:,:,nz22:].size,0])]
    por_zone2[por_zone2_order]=sand2[:,0]
    perm_zone2=por_zone2.copy()
    perm_zone2[por_zone2_order]=sand2[:,1]
    poro[:,:,nz22:]=por_zone2.reshape(poro[:,:,nz22:].shape)
    perm[:,:,nz22:]=perm_zone2.reshape(poro[:,:,nz22:].shape)
    active=facies.astype(int)
    poro_out=poro*facies
    perm_out=(10**perm)*facies

    with open(myfolder+'PORO', 'w') as f:
        f.write('PORO                                   -- Generated : Petrel\n-- Property name in Petrel : Porosity\n ')
        val_in_line = 0
        for k in np.arange(poro.shape[2]):
            for j in np.arange(poro.shape[1]):
                for i in np.arange(poro.shape[0]):
                    if val_in_line == 14:
                        f.write("\n ")
                        val_in_line = 0
                    val_in_line += 1
                    f.write(' %.6f' % abs(poro_out[i,poro.shape[1]-j-1,poro.shape[2]-k-1]))
        f.write(" /\n")
        
    with open(myfolder+'PERMX', 'w') as f:
        f.write('PERMX                                   -- Generated : Petrel\n-- Property name in Petrel : PERMX\n ')
        val_in_line = 0
        for k in np.arange(poro.shape[2]):
            for j in np.arange(poro.shape[1]):
                for i in np.arange(poro.shape[0]):
                    if val_in_line == 14:
                        f.write("\n ")
                        val_in_line = 0
                    val_in_line += 1
                    if abs(perm_out[i,poro.shape[1]-j-1,poro.shape[2]-k-1]) == 0.0:
                        f.write(" 0.00000")
                    else:
                        f.write(" {:.6g}".format(abs(perm_out[i,poro.shape[1]-j-1,poro.shape[2]-k-1])))
        f.write(" /\n")
        
    with open(myfolder+'PERMY', 'w') as f:
        f.write('PERMY                                   -- Generated : Petrel\n-- Property name in Petrel : PERMY\n ')
        val_in_line = 0
        for k in np.arange(poro.shape[2]):
            for j in np.arange(poro.shape[1]):
                for i in np.arange(poro.shape[0]):
                    if val_in_line == 14:
                        f.write("\n ")
                        val_in_line = 0
                    val_in_line += 1
                    if abs(perm_out[i,poro.shape[1]-j-1,poro.shape[2]-k-1]) == 0.0:
                        f.write(" 0.00000")
                    else:
                        f.write(" {:.6g}".format(abs(perm_out[i,poro.shape[1]-j-1,poro.shape[2]-k-1])))
        f.write(" /\n")
        
    with open(myfolder+'PERMZ', 'w') as f:
        f.write('PERMZ                                   -- Generated : Petrel\n-- Property name in Petrel : PERMZ\n ')
        val_in_line = 0
        for k in np.arange(poro.shape[2]):
            for j in np.arange(poro.shape[1]):
                for i in np.arange(poro.shape[0]):
                    if val_in_line == 14:
                        f.write("\n ")
                        val_in_line = 0
                    val_in_line += 1
                    if abs(perm_out[i,poro.shape[1]-j-1,poro.shape[2]-k-1]) == 0.0:
                        f.write(" 0.00000")
                    else:
                        f.write(" {:.6g}".format(abs(0.1*perm_out[i,poro.shape[1]-j-1,poro.shape[2]-k-1])))
        f.write(" /\n")
                    
    with open(myfolder+'ACTNUM', 'w') as f:
        f.write('ACTNUM                                   -- Generated : Petrel\n-- Property name in Petrel : ACTNUM\n ')
        val_in_line = 0
        for k in np.arange(poro.shape[2]):
            for j in np.arange(poro.shape[1]):
                for i in np.arange(poro.shape[0]):
                    if val_in_line == 64:
                        f.write("\n ")
                        val_in_line = 0
                    val_in_line += 1
                    f.write(' %i' % (int(active[i,poro.shape[1]-j-1,poro.shape[2]-k-1])))
        f.write(" /\n")

    return None