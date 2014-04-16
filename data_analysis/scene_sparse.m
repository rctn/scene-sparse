%This script will do the scene sparse algorithm
function [err] = scene_sparse(path)
addpath('../fast_sc/*')

%Input paths
if nargin <1
	path='/clusterfs/cortex/scratch/mayur/scene-sparse/man_made';
end
	%Load data

	load(path)

	%Initiatlize Parameters for SC
	X_orig = X_man_made;
	num_bases = size(X_orig,1)*4; %We start off with a 4x over complete basis
	batch_size = 1000;
	num_iters = 2;
	sparsity_func = 'epsL1'; %The other option is 'L1'
	epsilon = .001;
	beta=.01;

	Binit = [] ; %Inferred coefficients, start with empty
	[B S stat] = sparse_coding(X_orig, num_bases, beta, sparsity_func, epsilon, num_iters, batch_size, fname_save, Binit )%, resample_size);
%Save Dictionary

%Make Image

end
