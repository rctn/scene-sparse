function [ x, y ] = intercept( slope, point, matsize, top )
% Computes the intercept of a line with the top of the image.
% line equation : y = slope * x + b
% inputs:
% slope of the line
% a point on the line
% the size of the 2D image or matrix where the line is contained
    b = point(2) - slope * point(1);
    if top
        x = 1; % try looking at row 1
        y = round(slope * x + b); 
        if (y < 1)
            y = 1;
            x = round((y-b)/slope);
        elseif(y > matsize(2))
            y = matsize(2);
            x = round((y-b)/slope);
        end
    else
        x = matsize(1); % try looking at bottom row
        y = round(slope * x + b);
        if (y > matsize(2))
            y = matsize(2);
            x = round((y-b)/slope);
        elseif (y < 1)
            y = 1;
            x = round((y-b)/slope);
        end
    end
end

