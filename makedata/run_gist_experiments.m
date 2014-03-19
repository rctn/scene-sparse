% run all experiments from start to finish(or not)
% using the full image gist algorithm from 32X32 images

paths = char('bedroom','CALsuburb','industrial',...
    'kitchen','livingroom','MITcoast','MITforest',...
    'MIThighway','MITinsidecity','MITmountain',...
    'MITopencountry','MITstreet','MITtallbuilding',...
    'PARoffice','store');

num_categories = size(paths,1);
num_basis_functions = 1024;

% freqs = zeros(num_basis_functions, num_categories);
% orientations = zeros(num_basis_functions, num_categories);
% color = colormap(jet(num_categories));
% 
% figure(2)

for p=1
    filename = paths(p,:);
     % when loading images
    load(strcat('../data/scene_categories/',filename,'/TINY_IMAGES'));
    filename %print path
    sparsenet_tiny;
    save(strcat('results/',regexprep(filename,'/','_'),'_gist_Phi_64.mat'), 'Phi');
 
    % when loading basis functions
    %load(strcat('results/',regexprep(filename,'/','_'),'_gist_Phi.mat'));
    
%     [Wfs, Wor] = fourier_comparison(Phi);
%     freqs(:,p) = Wfs;
%     orientations(:,p) = Wor;
%     % magic for generating plot legend
%     figure(2)
%     scatter(Wfs,Wor,10,color(p,:),'userdata',paths(p,:));
%     hold on
end

%save('results/Wf.mat', 'freqs');
%save('results/Wor.mat', 'orientations');
% generate legend magic
% figure(2)
% legend(get(gca,'children'),get(get(gca,'children'),'userdata'));