% read image files from the data folders and save each
% category in a mat file that holds a 3D matrix of all
% images in that category
clc;
clear all;

% for each category of images, change:
%IMAGE_PATH = '../data/SUN_tiny/woodland/';
%IMAGE_PATH = '../data/mars/';
IMAGE_PATH = '../data/scene_categories/store/';

% output file
out_file = 'TINY_IMAGES.mat';

% current path
PATH = pwd;

% count the number of images in the image folder
M = length(dir(strcat(IMAGE_PATH,'*.jpg')));
% desired size
%N = 512; % for regular images
N = 32; % for tiny images

% create the data matrix
IMAGES = zeros(N, N, M);

% constants for whitening
[fx fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
rho=sqrt(fx.*fx+fy.*fy);
f_0=0.4*N;
filt=rho.*exp(-(rho/f_0).^4);

% load the images into the data matrix
for i = 1:M
    jpgFilename = strcat('image_',sprintf('%04d',i), '.jpg');
    image = imread(strcat(IMAGE_PATH, jpgFilename));
    % make sure it is big enough
    [size1 size2 ~] = size(image);
    if (size1 < N || size2 < N)
        error('myApp:argChk', strcat('Input image too small: ', jpgFilename))
    end
    % turn to grayscale and resize each image if needed
    if (length(size(image))>2)
        image = rgb2gray(image);
    end
    % in any case crop
    imageR = imresizecrop(image, [N N]);
    % whiten each image
    % (code from Bruno's sparsenet - "make your own images")
    If = fft2(imageR);
    imageW = real(ifft2(If.*fftshift(filt)));
    %IMAGES(:,i) = reshape(imageW,N^2,1);
    IMAGES(:,:,i)=imageW;
end

% normalize each image so that the variance
% of the pixels is the same (i.e., 1).
IMAGES=sqrt(0.1)*IMAGES/sqrt(mean(var(reshape(IMAGES,N^2,M))));

% write out the data matrix as a .mat file
save(strcat(IMAGE_PATH,out_file), 'IMAGES');