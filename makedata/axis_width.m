function  [fa,fb,W] = axis_width( point1, point2, mat, origin )
% returns the width of one of the half width binary
% ellipses in matrix mat. the width is taken along an
% axis defined by point1, point2

    % get subscript indices of the pixels on the line
    % accross this axis of the ellipse
    [subi subj]=bresenham(point1(1),point1(2),point2(1),point2(2));
    % get linear indices rather than subscripts
    indices = sub2ind(size(mat),subi,subj);
    % slice the ellipse along this axis
    slice = mat(indices);
    % find the coordinates of the first and last appearences of "1"
    
    % linear index of fa, fb = frequencies at half height
    % before and after the peak on this axis
    fa_ind = indices(find(slice,1,'first'));
    fb_ind = indices(find(slice,1,'last'));
    [fax,fay] = ind2sub(size(mat),fa_ind);
    [fbx,fby] = ind2sub(size(mat),fb_ind);
    % Pythagoras distance of half height full width
    W = sqrt((fax-fbx)^2 + (fay-fby)^2);
    % Frequencies at half height full width before and
    % after the peak (distances from origin)
    fa = sqrt((fax-origin(1))^2 + (fay-origin(2))^2);
    fb = sqrt((fbx-origin(1))^2 + (fby-origin(2))^2);
end

