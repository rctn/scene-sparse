% Only produce the scatter plot assuming all the data
% is already saved. Dont save anything.

paths = char('northwest', 'mars', 'coast', 'highway', ...
    'livingroom', 'mountain', 'office', ...
    'skyscraper', 'street', 'woodland');

num_categories = size(paths,1);

color = colormap(hsv(num_categories));

figure(2)

%for p=1:num_categories
for p=3:3:6
    filename = paths(p,:);
%     load 'results/Wf.mat';
%     load 'results/Wor.mat';    
     load 'results/Wfs_octaves.mat';
     load 'results/Wor_degrees.mat';
    % magic for generating plot legend
    figure(2)
    %set(gca, 'Xscale', 'log');
    scatter(freqs(:,p),orientations(:,p),10,color(p,:),'userdata',filename);
    hold on
end

% generate legend magic
figure(2)
legend(get(gca,'children'),get(get(gca,'children'),'userdata'));
%  xlabel('Spatial Frequency Bandwidth');
%  ylabel('Orientation Bandwidth');
 xlabel('Spatial Frequency Bandwidth [octaves]');
 ylabel('Orientation Bandwidth [degrees]');