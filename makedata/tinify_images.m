function [] = tinify_images(img_dir, results_dir, N, image_format)
%Usage: for tiny images N = 32;

% list all image directories
image_directories = dir(img_dir);
% for each image directory
for i = 1:length(image_directories)
    current_dir = image_directories(i).name; 
    if(isempty(strmatch('.',current_dir)))
        if (~exist([results_dir '/' current_dir],'dir'))
            mkdir([results_dir '/' current_dir]);
        end
        current_path = [img_dir current_dir '/'];
        % list all images in this directory
        images = dir([current_path '/*' image_format]);
        for j = 1:length(images)
            current_image = images(j).name;
            I = imread([current_path current_image], image_format);
            I_tiny = imresizecrop(I, [N N]);
            imwrite(I_tiny, [results_dir '/' current_dir '/' current_image]);
        end
    end
end
end

