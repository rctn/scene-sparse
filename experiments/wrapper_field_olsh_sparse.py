'''
Load Shapes, Compute basis
These basis be sparse
author: Mayur Mudigonda, March 2, 2015
'''

import numpy as np
import scipy.io as scio
from scipy.misc import imread
import glob
from scipy.optimize import minimize 
import os
import ipdb
import sparse_code_gpu
import time
import theano.tensor as T
import theano
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import importlib
import tables
import random
import utilities

def adjust_LR(LR, iterations):
    T = 200
    scale = 1.0/(1.0 + (iterations/T))
    new_LR = scale* LR
    print('.......................New Learning Rate is........................',new_LR)
    return new_LR

#Plot Reconstruction Error
def plot_residuals(residuals):
    plt.plot(residuals[1:])
    plt.xlabel('iterations')
    plt.ylabel('Residual of I - \phi a')
    plt.savefig('Reconstruction_Error.png')
    plt.close()
    return
#Plot SnR
def plot_SNR(SNR):
    plt.plot(SNR[1:])
    plt.xlabel('iterations')
    plt.ylabel('(SNR) ')
    plt.savefig('SNR.png')
    plt.close()
    return

def plot_Energy(Energy):
    plt.plot(Energy[1:])
    plt.xlabel('iterations')
    plt.ylabel('(Energy) ')
    plt.savefig('Energy.png')
    plt.close()
    return

def visualize_data(data,iteration,patchdim,image_shape=None):
    #Use function we wrote previously
    out_image = utilities.tile_raster_images(data.T,patchdim,image_shape)
    plt.imshow(out_image,cmap=cm.Greys_r)
    savepath_image= 'vis_data'+ '_iterations_' + str(iteration) + '.png'
    plt.savefig(savepath_image)
    plt.close()
    return



if __name__ == "__main__":

    #Environment Variables
    DATA = os.getenv('DATA')
    proj_path = DATA + 'scene-sparse/'
    write_path = proj_path + 'experiments/'
    data_dict = scio.loadmat(proj_path+'IMAGES.mat')
    IMAGES = data_dict['IMAGES']
    (imsize, imsize,num_images) = np.shape(IMAGES)
    print('Could not get file handle. Aborting')
    #Inference Variables
    LR = 1e-2 
    training_iter = 2000 
    lam = 5e-2 
    err_eps = 1e-3
    orig_patchdim = 8 
    patch_dim = orig_patchdim**2
    patchdim = np.asarray([0,0])
    patchdim[0] = orig_patchdim 
    patchdim[1] = orig_patchdim
    sz = np.sqrt(patch_dim)
    print('patchdim is ---',patchdim)
    batch = 200 
    data = np.zeros((orig_patchdim**2,batch))
    basis_no = 4*(orig_patchdim**2)
    border = 4
    matfile_write_path = write_path+'IMAGES_' + str(orig_patchdim) + 'x' + str(orig_patchdim) + '__LR_'+str(LR)+'_batch_'+str(batch)+'_basis_no_'+str(basis_no)+'_lam_'+str(lam)+'_basis'

    #Making and Changing directory
    try:
        print('Trying to see if directory exists already')
        os.stat(matfile_write_path)
    except:
        print('Nope nope nope. Making it now')
        os.mkdir(matfile_write_path)

    try:
        print('Navigating to said directory for data dumps')
        os.chdir(matfile_write_path)
    except:
        print('Unable to navigate to the folder where we want to save data dumps')

    #Create object
    lbfgs_sc = sparse_code_gpu.LBFGS_SC(LR=LR,lam=lam,batch=batch,basis_no=basis_no,patchdim=patchdim,savepath=matfile_write_path)
    residual_list=[]
    sparsity_list=[]
    energy_list = []
    snr_list=[]
    for ii in np.arange(training_iter):
        tm1 = time.time()
        print('Loading new Data')
        for i in range(batch):
          #Moving the image choosing inside the loop, so we get more randomness in image choice
          imi = np.ceil(num_images * random.uniform(0, 1))
          r = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
          c = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
          data[:,i] = np.reshape(IMAGES[r:r+sz, c:c+sz, imi-1], patch_dim, 1)
        SNR_I_2 = lbfgs_sc.load_data(data)
        tm2 = time.time()
        print('*****************Adjusting Learning Rate*******************')
        adj_LR = adjust_LR(LR,ii)
        lbfgs_sc.adjust_LR(adj_LR)
        print('Training iteration -- ',ii)
        #Note this way, each column is a data vector
        tm3 = time.time()
        prev_obj = 1e6 
        jj = 0
        ''' 
        while True: 
            obj,active_infer = lbfgs_sc.infer_coeff_gd()            
            if np.mod(jj,10)==0:
                print('Value of objective function from previous iteration of coeff update',obj)
            if (np.abs(prev_obj - obj) < err_eps) or (jj > 100):
                break
            else:
                prev_obj = obj
            jj = jj + 1
        '''
        residual,active,basis,energy=lbfgs_sc.update_basis()
        lbfgs_sc.infer_fista()
        residual_list.append(residual)
        energy_list.append(energy)
        tm5 = time.time()
        denom = lbfgs_sc.recon.get_value()
        denom_var = np.var(denom)
        snr = SNR_I_2/(denom_var)
        snr = 10*np.log10(snr)
        snr_list.append(snr)
        print('Time to load data in seconds', tm2-tm1)
        print('The value of residual after we do learning ....', residual)
        print('The SNR for the model is .........',snr)
        print('The value of active coefficients after we do learning ....',active)
        print('The mean norm of the basis is .....',np.mean(np.linalg.norm(basis,axis=0)))
        #residual_list.append(residual)
        if np.mod(ii,10)==0:
            print('Saving the basis now, for iteration ',ii)
            scene_basis = {
            'basis': lbfgs_sc.basis.get_value(),
            'residuals':residual_list,
            'sparsity':sparsity_list,
            'snr':snr_list
            }
            scio.savemat('basis',scene_basis)
            print('Saving basis visualizations now')
            lbfgs_sc.visualize_basis(ii,[16,16])
            print('Saving data visualizations now')
            visualize_data(data,ii,patchdim,[16,16])
            print('Saving SNR')
            plot_SNR(snr_list)
            print('Saving Energy')
            plot_Energy(energy_list)
            print('Saving R_error')
            plot_residuals(residual_list)
            print('Average Coefficients')
            lbfgs_sc.plot_mean_firing(ii)
            print('Visualizations done....back to work now')
