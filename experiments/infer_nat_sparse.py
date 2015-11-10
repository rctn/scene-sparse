'''
Load Shapes, Compute basis
These basis be sparse
author: Mayur Mudigonda, March 2, 2015
'''

import numpy as np
import scipy.io as scio
from scipy.misc import imread
import glob
from scipy.io import loadmat
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
import utilities

def adjust_LR(LR, iterations):
    if iterations>2000:
        T = 1000
        scale = 1.0/(1.0 + (iterations/T))
        new_LR = scale* LR
    else:
        new_LR = LR
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

def load_list(filename_no_extension):
    """
    :type filename_no_extension: A filename string without the .py extension
    """
    print('loading file ' + filename_no_extension)
    module = importlib.import_module(filename_no_extension)
    assert isinstance(module, object)
    assert isinstance(module.content, list)
    return module.content

def make_data(file_hndl,file_list,batch):

    data=np.zeros([32*32,batch])
    gen_idx = np.random.randint(0,len(file_list),batch)
    idx = 0
    for ii in np.arange(batch):
        try:
            data[:,idx] = file_hndl.root._f_get_child(file_list[gen_idx[ii]])[:]
        except:
            print('Could not extract indoor image with id ',gen_idx[ii])
        idx = idx + 1

    return data

if __name__ == "__main__":

    #Environment Variables
    DATA = os.getenv('DATA')
    proj_path = DATA + 'scene-sparse/'
    write_path = proj_path + 'experiments/'

    print('Loading File lists')
    indoor_list=load_list('indoor_file_list')
    outdoor_list=load_list('outdoor_file_list')
    try:
        h= tables.open_file(DATA+'scene-sparse/places32_whitened.h5','r') 
    except:
        print('Could not get file handle. Aborting')
    #Inference Variables
    LR = 1e-1 
    training_iter = 10000 
    lam = 1e-2
    err_eps = 1e-3
    orig_patchdim = 32 
    patchdim = np.asarray([0,0])
    patchdim[0] = orig_patchdim 
    patchdim[1] = orig_patchdim
    print('patchdim is ---',patchdim)
    batch = 200 
    basis_no =1*(orig_patchdim**2)
    matfile_write_path = write_path+'outdoor_LR_'+str(LR)+'_batch_'+str(batch)+'_basis_no_'+str(basis_no)+'_lam_'+str(lam)+'_basis'

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
    data_obj = loadmat(matfile_write_path + '/basis.mat')
    basis = data_obj['basis']
    sc = sparse_code_gpu.SparseCode(LR=LR,lam=lam,batch=batch,basis_no=basis_no,patchdim=patchdim,savepath=matfile_write_path,basis=basis)
    residual_list=[]
    sparsity_list=[]
    snr_list=[]
    print('Loading new Data')
    data=make_data(h,outdoor_list,batch)
    sc.load_data(data)
    #print('*****************Adjusting Learning Rate*******************')
    #Note this way, each column is a data vector
    tm3 = time.time()
    prev_obj = 1e6 
    sc.infer_fista()
    #residual, residual_avg, active, E, E_rec, E_sp, t_snr = sc.update_basis()
    #residual_list.append(residual_avg)
    #snr = 10*np.log10(snr)
    recon = sc.recon.get_value()
    #snr = np.var(recon,axis=0).mean()/np.var(residual,axis=0).mean()
    #snr_list.append(snr)
    print('Saving data visualizations now')
    sc.visualize_basis(500000,[orig_patchdim,orig_patchdim])
    print('Saving data visualizations now')
    sc.visualize_data(500000,[20,10])
    print('Saving data reconstruction visualizations now')
    sc.visualize_recon(500000,[20,10])
    print('Visualizations done....back to work now')
