% sparsenet.m - Olshausen & Field sparse coding algorithm
% 

%clear all;
%clc;
%load '../data/scene_categories/MITopencountry/TINY_IMAGES';
%load ../data/SUN_tiny/coast/TINY_IMAGES;
%load ../data/bruno_data/IMAGES;
%load ../data/mars/IMAGES;

[imsize imsize num_images]=size(IMAGES);
BUFF=4;

num_trials=5000;
batch_size=100;

% number of outputs
M=1024;

% number of inputs
N=1024;
sz=sqrt(N); % the size of patches from the input images

% initialize basis functions (comment out these lines if you wish to 
% pick up where you left off)
Phi=randn(N,M);
Phi=Phi*diag(1./sqrt(sum(Phi.*Phi)));

% learning rate (start out large, then lower as solution converges)
eta = 2.0;

% lambda - in the hw between 0.1 and 1.
% In the paper labda/sigma set to 0.14 where sigma is the
% variance of the images. Since we made the images all have
% variance of 1, lets set the lambda to 0.14.
% This is not 100% accurate (see original sparsenet code)
lambda = 0.5;

a_var=ones(M,1);
var_eta=.1;

I=zeros(N,batch_size);

display_every=5;
display_network(Phi,a_var);

for t=1:num_trials

    % we are using the whole image rather than patches
    for i=1:batch_size
        % choose an image at random
        imi=ceil(num_images*rand);
        %imi = (t-1)*batch_size + i;
        % add the whole image to the data array
        I(:,i)=reshape(IMAGES(:,:,imi),N,1);
    end

    % calculate coefficients for these data via LCA

    ahat = sparsify(I,Phi,lambda);

    % calculate residual error

    R=I-Phi*ahat;
    %loss(t) = mean(sqrt(sum(R.*R))); This doesnt seem to be used
    % update bases

    dPhi = eta*(R)*ahat'/size(I,2);  %\eta times the difference between input 
    %and the activations times the dictionary elements. 
    Phi = Phi + dPhi;
    Phi=Phi*diag(1./sqrt(sum(Phi.*Phi))); % normalize bases

    % accumulate activity statistics
    
    a_var=(1-var_eta)*a_var + var_eta*mean(ahat.^2,2);

    % display

    if (mod(t,display_every)==0)
        display_network(Phi,a_var);
    end

end
