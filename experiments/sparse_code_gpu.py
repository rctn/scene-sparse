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
#FISTA updates
##Courtesy Alexander G Anderson, 2015

class LBFGS_SC:

    def __init__(self,savepath=None,LR=None,lam=None,batch=None,basis_no=None,patchdim=None):
        if LR is None:
            self.LR = 0.1
        else:
            self.LR = LR
        if lam is None:
            self.lam = 0.1
        else:
            self.lam = lam
        if batch is None:
            self.batch = 30
        else:
            self.batch = batch
        if basis_no is None:
            self.basis_no = 50
        else:
            self.basis_no = basis_no
        if patchdim is None:
            self.patchdim = 512
        else:
            self.patchdim = patchdim
        if savepath is None:
            self.savepath = os.getcwd() 
        else:
            self.savepath = savepath
        self.data = np.random.randn(self.patchdim[0]*self.patchdim[1],self.batch).astype('float32')
        self.coeff = np.random.randn(self.basis_no,self.batch).astype('float32') 
        self.coeff_prev_grad = np.zeros(self.coeff.shape).astype('float32')
        self.basis = np.random.randn(self.patchdim[0]*self.patchdim[1],self.basis_no).astype('float32')
        #self.basis_prev_grad = np.zeros(self.basis.shape).astype('float32') 
        self.coeff = theano.shared(self.coeff)
        self.data = theano.shared(self.data)
        self.basis = theano.shared(self.basis)
        #self.basis_prev_grad = theano.shared(self.basis_prev_grad)
        self.coeff_prev_grad = theano.shared(self.coeff_prev_grad)
        self.recon = theano.shared(0.0*self.data.get_value())
        self.t_T = theano.shared(np.array([1.]).astype(theano.config.floatX), 'fista_T')
        self.t_X = theano.shared(np.zeros(self.coeff.shape).astype(
        print('Compiling theano inference function')
        #self.infer_coeff_gd = self.create_infer_coeff_gd()
        self.f = self.create_coeff_fn()
        print('Compiling theano basis function') 
        self.update_basis = self.create_update_basis()
        return 

    def create_coeff_fn(self):
        coeff_flat = T.dvector('coeffs')
        coeff = T.cast(coeff_flat,'float32')
        coeff = T.reshape(coeff,[self.basis_no,self.batch])
        tmp = (self.data - self.basis.dot(coeff))**2
        tmp = 0.5*tmp.sum(axis=0)
        tmp = tmp.mean()
        sparsity = self.lam * T.abs_(coeff).sum(axis=0).mean()
        obj = tmp + sparsity
        grads = T.grad(obj,coeff_flat)
        f = theano.function([coeff_flat],[obj.astype('float64'),grads.astype('float64')])
        return f 

    def infer_coeff(self):
        init_coeff = np.random.randn(self.basis_no*self.batch).astype('float32')
        res = minimize(fun=self.f,x0=init_coeff,method='L-BFGS-B',jac=True,options={'maxiter':1000,'disp':False})
        self.coeff.set_value(np.reshape(res.x,[self.basis_no,self.batch]).astype('float32'))
        print('Value of objective fun after doing inference',res.fun)
        active = len(res.x[np.abs(res.x)>1e-2])/float(self.basis_no*self.batch)
        print('Number of Active coefficients is ...',active)
        return active,res.fun 

    def fista_init(self):
        '''
        This function we can just leave it calling Alex's original fista code. The rest
        of it, we can modify to fit into our code base. This function is to be called once
        at the start of the script. So maybe this goes into init?
        Code Adapted From: Alexander G Anderson
        '''
        #This calls the fista_update function and creates a function handle
        #fist_updates = fista_updates()
        #_, t_fista_X, t_T = fist_updates.keys()

        fista_step = theano.function(inputs = [],
                                     #outputs = [t_E, t_E_rec, t_E_sp, t_SNR], #These need to be coded up
                                     outputs = [],
                                     updates = self.fista_updates)
    def calculate_fista_L(self):
        """
        Calculates the 'L' constant for FISTA for the dictionary in t_D.get_value()
        Code adapted from: Alexander G Anderson
        """
        D = self.basis.get_value()
        try:
            L = (2 * eigh(np.dot(D, D.T), eigvals_only=True, eigvals=(N_sp-1,N_sp-1))[0]).astype('float32')
        except ValueError:
            L = (2 * D_std ** 2 * N_sp).astype('float32') # Upper bound on largest eigenvalue
        return L

    def reset_fista_variables(self):
        """
        Resets fista variables
        Code adapted from: Alexander G Anderson
        """
        A0 = np.zeros_like(self.coeff.get_value()).astype(theano.config.floatX)
        self.coeff.set_value(A0)
        self_t_X.set_value(A0)
        self.t_T.set_value(np.array([1.]).astype(theano.config.floatX))
        return 1

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

        def threshold(t_X):
            """
            Threshold function
            """
            if pos_only = True:
                return T.switch(t_X > 0., t_X, 0.)
            else:
                return t_X

        t_E_rec = (self.data - self.basis.dot(self.coeff))**2
        t_E_sp = self.lam * T.abs_(coeff).sum(axis=0).mean()
        t_E = t_E_rec + t_E_sp
        #t_E_rec needs to be coded up here, since we don't actually have that variable anywhere
        t_B = self.coeff - (1. / self.L) * T.grad(t_E_rec, self.coeff)
        t_C = T.abs_(t_B) - self.lam / t_L
        t_A_ista = T.switch(t_B > 0, 1., -1.) * threshold(t_C)

        if pos_only:
            t_A_ista = threshold(t_A_ista)
        t_A_ista.name = 'A_ista'

        t_T = theano.shared(np.array([1.]).astype(theano.config.floatX), 'fista_T')
        shape = self.coeff.get_value().shape
        t_X = theano.shared(np.zeros(shape).astype(
            theano.config.floatX), 'fista_X')

        t_X1 = t_A_ista
        t_T1 = 0.5 * (1 + T.sqrt(1. + 4. * self.t_T ** 2))
        t_T1.name = 'fista_T1'
        t_A1 = t_X1 + (t_T1[0] - 1) / t_T[0] * (t_X1 - self.t_X)
        t_A1.name = 'fista_A1'
        updates = OrderedDict()
        updates[self.coeff] = t_A1
        updates[self.t_X] = t_X1
        updates[self.t_T] = t_T1

        return updates

    def infer_fista(self,show_costs = False):
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
        self.reset_fista_variables()
        self.calculate_fista_L()
        for _ in range(N_g_itr):
            E, E_rec, E_sp, SNR = self.fista_step( )
        #dictionary_learning_step(Alpha, Eta, D_std, I_idx)
        # 
        #E, E_rec, E_sp, SNR = costs(I_idx, Alpha)
        cost_list.append(E)
        if ((i + 1) % (1 + N_itr / 10) == 0) and show_costs:
            print i, E, E_rec, E_sp, SNR
        return I_idx


    def create_infer_coeff_gd(self):
        r_err = (self.data - self.basis.dot(self.coeff))**2
        r_err = 0.5*r_err.sum(axis=0)
        r_err = r_err.mean()
        sparsity = self.lam * T.abs_(self.coeff).sum(axis=0).mean()
        obj = r_err + sparsity
        grads = T.grad(obj, self.coeff)
        inf_LR = self.LR*1e-3
        mom_update = 0.5*inf_LR*self.coeff_prev_grad + inf_LR*grads
        #coeff_update = self.coeff - self.LR*1e-1*grads
        coeff_update = self.coeff + mom_update
        updates = {self.coeff:coeff_update.astype('float32'),
                self.coeff_prev_grad:mom_update.astype('float32'),
                }
        active = T.abs_(coeff_update).sum().astype('float32')/float(self.basis_no*self.batch)
        f = theano.function([],[obj.astype('float64'),active.astype('float64')],updates=updates)
        return f


    def load_data(self,data):
        print('The norm of the data is ', np.mean(np.linalg.norm(data,axis=0)))
        #Update self.data to have new data
        data_norm = np.linalg.norm(data,axis=0)
        data = data/data_norm[np.newaxis,:]
        self.data.set_value(data.astype('float32'))
        return
   
    def adjust_LR(self,LR):
        self.LR = LR
        return


    def create_update_basis(self):
        #Update basis with the right update steps
        Residual = self.data - self.basis.dot(self.coeff)
        #Gradient
        dbasis = Residual.dot(self.coeff.T)
        norm_grad_basis = dbasis**2
        norm_grad_basis = norm_grad_basis.sum(axis=0)
        norm_grad_basis = T.sqrt(norm_grad_basis)
        dbasis = dbasis/norm_grad_basis.dimshuffle('x',0)
        
        basis = self.basis + self.LR*dbasis
        #Normalize basis
        norm_basis = basis**2
        norm_basis = norm_basis.sum(axis=0)
        norm_basis = T.sqrt(norm_basis)
        basis = basis/norm_basis.dimshuffle('x',0)
        #Now updating reconstructions
        recon = self.basis.dot(self.coeff)
        updates = {self.basis: basis.astype('float32'),
                   #self.basis_prev_grad:mom_update.astype('float32'),
                   self.recon:recon.astype('float32'),
                  }
        #Now setting the previous basis to this time around
        #Computing Average Residual
        tmp = Residual**2
        tmp = 0.5*tmp.sum()
        Residual = tmp
        #Computing how much coefficients are "on"
        #num_on = T.abs_(self.coeff).sum().astype('float32')/float(self.basis_no*self.batch)
        #Computing sparisty contributions to energy function
        num_on = T.abs_(self.coeff).sum().astype('float32')
        f = theano.function([],[Residual.astype('float32'),num_on,basis], updates=updates)
        return f 

    def visualize_basis(self,iteration,image_shape=None):
        #Use function we wrote previously
        tmp = self.basis.get_value()
        out_image = utilities.tile_raster_images(tmp.T,self.patchdim,image_shape)
        plt.imshow(out_image,cmap=cm.Greys_r)
        savepath_image= 'vis_basis'+ '_iterations_' + str(iteration) + '.png'
        plt.savefig(savepath_image)
        plt.close()
        return

    def visualize_basis_recon(self,iteration):
        #Use function we wrote previously
        f, ax = plt.subplots(20,20,sharex=True,sharey=True)
        #Compute reconstructions first
        recon = self.basis.dot(self.coeff).eval()
        for ii in np.arange(ax.shape[0]):
            for jj in np.arange(0,ax.shape[1],2):
                data = self.data.get_value()
                tmp = data[:,ii*ax.shape[1]+jj]
                im = tmp.reshape([self.patchdim[0],self.patchdim[1]])
                ax[ii,jj].imshow(im,interpolation='nearest',aspect='equal')
                ax[ii,jj].axis('off')
                tmp = self.recon.get_value() 
                im = tmp.reshape([self.patchdim[0],self.patchdim[1]])
                ax[ii,jj+1].imshow(im,interpolation='nearest',aspect='equal')
                ax[ii,jj+1].axis('off')
        savepath_image= 'vis_reconstr'+ '_iterations_' + str(iteration) + '.png'
        f.savefig(savepath_image)
        f.clf()
        plt.close()
        return

    def plot_mean_firing(self,iteration):
        #Plot mean firing
        mean_coeff = self.coeff.get_value()
        tmp = mean_coeff.mean(axis=1)
        plt.plot(tmp)
        plt.xlabel('basis number')
        plt.ylabel('mean activation across 200 samples')
        savepath_image=  '_iterations_' + str(iteration) + '.png'
        plt.savefig('mean_coeff'+savepath_image)
        plt.close()
        #Plotting Histograms
        plt.hist(mean_coeff.flatten(),50)
        plt.xlabel('Bin Values')
        plt.ylabel('Counts of firing rates')
        plt.savefig('hist'+savepath_image)
        plt.close()
        return


