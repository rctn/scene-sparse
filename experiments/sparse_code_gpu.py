import numpy as np
from scipy.optimize import minimize
import theano 
from theano import tensor as T
import ipdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import utilities
from collections import OrderedDict
from scipy.linalg import eigvalsh
#FISTA updates
##Courtesy Alexander G Anderson, 2015


dtype = theano.config.floatX

class SparseCode:

    def __init__(self, savepath=None, LR=.1, lam=.1, batch=100,
                 basis_no=512, patchdim=512, N_g_itr=150, basis=None,
                 seed=2015103):
        self.lam = lam
        self.batch = batch
        self.basis_no = basis_no
        self.patchdim = patchdim
        self.N_g_itr = N_g_itr
        if savepath is None:
            self.savepath = os.getcwd() 
        else:
            self.savepath = savepath
        self.rng = np.random.RandomState(seed)
        if basis is None:
            basis = self.rng.randn(self.patchdim[0]*self.patchdim[1], self.basis_no).astype(dtype)
        basis = basis/np.linalg.norm(basis, axis=0)[np.newaxis,:]

        data = self.rng.randn(self.patchdim[0]*self.patchdim[1], self.batch).astype(dtype)
        coeff = self.rng.randn(self.basis_no, self.batch).astype(dtype) 

        self.coeff = theano.shared(coeff)
        self.data = theano.shared(data)
        self.basis = theano.shared(basis)
        self.recon = theano.shared(0.0*self.data.get_value())
        self.LR = theano.shared(np.array(LR, dtype=dtype))
        self.t_T = theano.shared(np.array(1., dtype=dtype), 'fista_T')
        self.t_X = theano.shared(np.zeros((self.basis_no, self.batch), dtype=dtype), 'fista_X')
        self.L = theano.shared(np.array(1.0, dtype=dtype),'fista_L')

        self.t_E_rec = 0.5*T.sum((self.data - self.basis.dot(self.coeff))**2)
        self.t_E_rec.name = 't_E_rec'
        self.t_E_sp = self.lam * abs(self.coeff).sum()
        self.t_E_sp.name = 't_E_sp'
        self.t_E = self.t_E_rec + self.t_E_sp
        self.t_E.name = 't_E'

        #Calling Fista Initialization
        self.fista_init()
        print('Compiling theano inference function')
        self.f = self.create_coeff_fn()
        print('Compiling theano basis function') 
        self.update_basis = self.create_update_basis()
        return 

    def fista_init(self):
        '''
        This function we can just leave it calling Alex's original fista code. The rest
        of it, we can modify to fit into our code base. This function is to be called once
        at the start of the script. So maybe this goes into init?
        Code Adapted From: Alexander G Anderson
        '''
        #This calls the fista_update function and creates a function handle
        self.fista_step = theano.function(inputs = [],
                                     updates = self.fista_updates())
    def calculate_fista_L(self):
        """
        Calculates the 'L' constant for FISTA for the dictionary in t_D.get_value()
        Code adapted from: Alexander G Anderson
        """
        D = self.basis.get_value()
        #Switching the order of D D^{T} solely based on my understanding of how D is setup in Alex' code
        try:
            L = eigvalsh(D.T.dot(D), eigvals=(self.basis_no-1, self.basis_no-1))[0]
            print L
        except ValueError:
            print 'We encountered a Value Error'
            L = np.std(self.basis.get_value())** 2 * self.basis_no # Upper bound on largest eigenvalue
        self.L.set_value(np.array(L, dtype=dtype))

    def reset_fista_variables(self):
        """
        Resets fista variables
        Code adapted from: Alexander G Anderson
        """
        A0 = np.zeros_like(self.coeff.get_value(), dtype=dtype)
        self.coeff.set_value(A0)
        self.t_X.set_value(A0)
        self.t_T.set_value(np.array(1., dtype=dtype))

    def threshold(self,t_X):
        """
        Threshold function
        """
        return T.switch(t_X > 0., t_X, 0.)

    def fista_updates(self, pos_only=False):
        """
        Code to generate theano variables to minimize
            t_E = t_E_rec + t_Alpha * T.sum(abs(t_A))
        with respect to t_A using FISTA
        t_A - Theano shared variable to minimize with respect to
        t_E_rec - Theano Energy function to minimize, not including abs(t_A) term
        t_L - Theano variable for the Lipschitz constant of d t_E / d t_A
        pos_only - Boolean to say if we should do positive only minimization
        Return - a dictionary of updates to run fista to pass to theano.function
        Note: The auxillary variables needed by the algorithm must be
            reset after each run
        t_A = t_fista_X = A0, t_T = 1
        See the end of the script for a sample usage
        Code adapted from: Alexander G Anderson, 2015
        """

        t_B = self.coeff - (1. / self.L) * T.grad(self.t_E_rec, self.coeff)
        t_C = abs(t_B) - self.lam / self.L
        t_A_ista = T.sgn(t_B) * self.threshold(t_C)

        t_X1 = t_A_ista
        t_T1 = 0.5 * (1 + T.sqrt(1. + 4. * self.t_T ** 2))
        t_T1.name = 'fista_T1'
        #t_A1 = t_X1 + (self.t_T - 1) / t_T1 * (t_X1 - self.t_X)
        t_A1 = t_X1 + (t_T1 - 1) / self.t_T * (t_X1 - self.t_X)
        t_A1.name = 'fista_A1'

        updates = OrderedDict()
        updates[self.coeff] = t_A1
        updates[self.t_X] = t_X1
        updates[self.t_T] = t_T1

        return updates

    def ista_updates(self, pos_only = False):
        """
        Code to generate updates for ISTA, since it is simpler and a better debugging starting point
        Adapted from Alex Anderson's code, 2015
        """
        t_B  = self.coeff - (1. / self.L) * T.grad(self.t_E_rec, self.coeff)
        t_C  = abs(t_B) - self.lam/ self.L
        t_A_ista = T.sgn(t_B) * self.threshold(t_C)
        t_A_ista.name = 'A_ista'

        updates = OrderedDict()
        updates[self.coeff] = t_A_ista

        return updates

    def infer_fista(self, show_costs=False):
        """
        Code to train a sparse coding dictionary
        N_itr - Number of iterations, a new batch of images for each
        Alpha - Sparsity cost parameter... E = E_rec + Alpha + |A|_1
        cost_list - list to which the cost at each iteration will be appended
        N_g_itr - number of gradient steps in FISTA
        show_costs - If true, print out costs every N_itr/10 iterations
        Returns I_idx - indices corresponding to the most recent image batch,
            to be used later in visualizations
        """
        if show_costs:
            print 'Iteration, E, E_rec, E_sp, SNR'
        self.calculate_fista_L()
        self.reset_fista_variables()
        for ii in range(self.N_g_itr):
            self.fista_step()
            if np.mod(ii,99)==0:
                print('L ', self.L.get_value())
                print('Mean abs coeff value', self.coeff.get_value().max())
                #print('Mean abs coeff value', self.coeff.get_value()[:,0])
        return 

    def load_data(self, data):
        print('The norm of the data is ', np.mean(np.linalg.norm(data,axis=0)))
        #Update self.data to have new data
        data_norm = np.linalg.norm(data,axis=0)
        #data = data/data_norm[np.newaxis,:]
        data = data/data_norm.max()
        SNR_data = np.var(data)
        self.data.set_value(data.astype(dtype))
        return SNR_data
   
    def adjust_LR(self, LR):
        self.LR.set_value(LR)


    def create_update_basis(self):
        #Update basis with the right update steps
        #Gradient
        dbasis = T.grad(self.t_E_rec, self.basis)
        norm_grad_basis = dbasis**2
        norm_grad_basis = norm_grad_basis.sum(axis=0)
        norm_grad_basis = T.sqrt(norm_grad_basis)
        dbasis = dbasis/T.maximum(norm_grad_basis.dimshuffle('x', 0), 1e-2)
        
        basis = self.basis - self.LR*dbasis
        #Normalize basis
        norm_basis = basis**2
        norm_basis = norm_basis.sum(axis=0)
        norm_basis = T.sqrt(norm_basis)
        basis = basis/norm_basis.dimshuffle('x',0)
        #Now updating reconstructions
        recon = self.basis.dot(self.coeff)

        updates = OrderedDict()
        updates[self.basis]= basis
        updates[self.recon]= recon
        #Now setting the previous basis to this time around
        #Computing Average Residual
        tmp = (self.data - self.basis.dot(self.coeff))**2
        tmp = 0.5 * tmp.sum(axis=0).mean()
        Residual = tmp
        #Computing how much coefficients are "on"
        #num_on = abs(self.coeff).sum().astype(dtype)
        num_on = T.switch(abs(self.coeff)>0, 1., 0.).sum(axis=0).mean()
        outputs = [Residual, num_on, self.t_E, self.t_E_rec, self.t_E_sp]
        f = theano.function([], outputs, updates=updates)
        return f 

    def visualize_basis(self,iteration,image_shape=None):
        #Use function we wrote previously
        tmp = self.basis.get_value()
        out_image = utilities.tile_raster_images(tmp.T,self.patchdim,image_shape,tile_spacing = (1,1))
        plt.imshow(out_image,cmap=cm.Greys_r)
        savepath_image= 'vis_basis'+ '_iterations_' + str(iteration) + '.png'
        plt.savefig(savepath_image)
        plt.close()
        return

    def plot_mean_firing(self,iteration):
        #Plot mean firing
        mean_coeff = self.coeff.get_value()
        tmp = np.abs(mean_coeff).mean(axis=1)
        plt.plot(tmp)
        plt.xlabel('basis number')
        plt.ylabel('mean activation across 200 samples')
        savepath_image=  '_iterations_' + str(iteration) + '.png'
        plt.savefig('mean_coeff'+savepath_image)
        plt.close()
        #Plotting Histograms
        '''
        plt.hist(mean_coeff.flatten(),50)
        plt.xlabel('Bin Values')
        plt.ylabel('Counts of firing rates')
        plt.savefig('hist'+savepath_image)
        plt.close()
        '''
