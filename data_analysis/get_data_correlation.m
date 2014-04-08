function C = get_data_correlation(img_dir, num_images, img_size, image_format, results_file)
% usage: C = get_data_correlation('/clusterfs/cortex/scratch/shiry/image-net-tiny/man_made/', 12573, [32 32], '.JPEG', '/clusterfs/cortex/scratch/shiry/results/data_correlation/man_made.mat')

% create the data matrix where each image will be a column vector
if (length(img_size) == 2)
    M = img_size(1) * img_size(2);
else
    M = img_size(1) * img_size(2) * img_size(3);
end
X = zeros(M,num_images);
idx = 1;

% list all image directories
image_directories = dir(img_dir);
% for each image directory
for i = 1:length(image_directories)
    current_dir = image_directories(i).name;
    if(isempty(strmatch('.',current_dir)))
        current_path = [img_dir current_dir '/'];
        % list all images in this directory
        images = dir([current_path '/*' image_format]);
        for j = 1:length(images)
            current_image = images(j).name;
            I = imread([current_path current_image]);
            % convert image to a column vector
            X(:,idx) = reshape(I, [M 1]);
            idx = idx + 1;
        end
    end
end
% compute the columnwise covariance between images in this dataset
C = cov(X);
save(results_file,'C');
end
