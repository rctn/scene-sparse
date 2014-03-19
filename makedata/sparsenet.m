% sparsenet.m - Olshausen & Field sparse coding algorithm
% 
%clear all;
%clc;
%load ../data/SUN_tiny/woodland/IMAGES;
%load ../data/bruno_data/IMAGES;
%load ../data/mars/IMAGES;

num_trials=5000;
batch_size=100;

[imsize imsize num_images]=size(IMAGES);
BUFF=4;

% number of outputs 256
M=256;

% number of inputs 256 for 16 X 16 patches
N=256;
sz=sqrt(N);

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
lambda = 0.14;

a_var=ones(M,1);
var_eta=.1;

I=zeros(N,batch_size);

display_every=5;
display_network(Phi,a_var);

for t=1:num_trials
    
    % choose an image for this batch

    imi=ceil(num_images*rand);

    % extract subimages at random from this image to make data array I

    for i=1:batch_size
        r=BUFF+ceil((imsize-sz-2*BUFF)*rand);
        c=BUFF+ceil((imsize-sz-2*BUFF)*rand);
        I(:,i)=reshape(IMAGES(r:r+sz-1,c:c+sz-1,imi),N,1);
    end

    % calculate coefficients for these data via LCA

    ahat = sparsify(I,Phi,lambda);

    % calculate residual error

    R=I-Phi*ahat;
    %loss(t) = mean(sqrt(sum(R.*R)));
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
