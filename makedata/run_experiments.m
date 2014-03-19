% run all experiments from start to finish(or not)
% using the patches algorithm from 512X512 images

paths = char('bruno_data', 'mars', 'coast', 'highway', ...
    'livingroom', 'mountain', 'office', ...
    'skyscraper', 'street', 'woodland');

num_categories = size(paths,1);

freqs = zeros(256, num_categories);
orientations = zeros(256, num_categories);
color = colormap(jet(num_categories));

figure(2)

for p=1:num_categories
    filename = paths(p,:);
    if (p > 2)
        filename = strcat('sun_tiny/',filename);
    end
     % when loading images
 %   load(strcat('../data/',filename,'/IMAGES'));
 %   filename %print path
 %   sparsenet;
 %   save(strcat('results/',regexprep(filename,'/','_'),'_Phi.mat'), 'Phi');
 
    % when loading basis functions
    load(strcat('results/',regexprep(filename,'/','_'),'_Phi.mat'));
    
    [Wfs, Wor] = fourier_comparison(Phi);
    freqs(:,p) = Wfs;
    orientations(:,p) = Wor;
    % magic for generating plot legend
    figure(2)
    set(gca, 'Xscale', 'log');
    scatter(Wfs,Wor,10,color(p,:),'userdata',paths(p,:));
    hold on
end

%save('results/Wfs_octaves.mat', 'freqs');
%save('results/Wor_degrees.mat', 'orientations');
% generate legend magic
figure(2)
legend(get(gca,'children'),get(get(gca,'children'),'userdata'));
xlabel('Spatial Frequency Bandwidth [octaves]');
ylabel('Orientation Bandwidth [degrees]');