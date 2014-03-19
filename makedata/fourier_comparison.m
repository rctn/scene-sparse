function [Wfs, Wor] = fourier_comparison(A)
%
%  fourier_comparison -- 
%
%  Usage:
%
%    fourier_comparison(A);
%
%    A = basis function matrix

[L M]=size(A);

sz=sqrt(L); % size of a basis function square

Wfs = zeros(M,1); % frequency half width full heights
Wor = zeros(M,1); % orientation half width full heights

for k=1:M
	basis_function = reshape(A(:,k),sz,sz);
    ft = fft2(basis_function,512,512);
    % get the amplitude of the fourier transform
    % shifted so that the low frequency are at the origin
    sft_amp = fftshift(abs(ft));
    
    % find fs the peak frequency.
    % will get two coordinates since transform is symmetrical
    peakf = max(sft_amp(:));
    [row,col] = find(sft_amp == peakf);
    origin = [mean(row), mean(col)];
    [~,findex] = min(row);
    % coordinates of peak
    fs_ind = [row(findex), col(findex)];
    % value of peak frequency (distance from origin)
    fs = sqrt((fs_ind(2)-origin(2))^2 + (fs_ind(1)-origin(1))^2);
    if (fs ==0)
        % ignore these
        Wfs(k) = Wfs(k-1);
        Wor(k) = Wor(k-1);
        continue;
    end
    
    % leave only the half height full width elipses
    % and make a binary mask of them
    hh = zeros(size(sft_amp));
    hh(sft_amp > peakf/2) = 1;
    
    % find the frequency bandwidth %%%%%%%%%%%%
    
    % slope of major axis
    slope = (fs_ind(2)-origin(2))/(fs_ind(1)-origin(1));
    if (isfinite(slope))
        if (slope == 0)
            % if we are on the vertical line
            if (fs_ind(1) < origin(1))
                x = 1;
                y = fs_ind(2);
            else
                x = size(hh,1);
                y = fs_ind(2);
            end 
        else
            % intercept with the y (j - col) axis - zero indexed
            [x, y] = intercept(slope, origin, size(hh), true);
        end
    else
        % if we are on the horizontal line 
        if (fs_ind(2) < origin(2))
            x = fs_ind(1);
            y = 1;
        else
            x = fs_ind(1);
            y = size(hh,2);
        end
    end
    [fa,fb,major_width] = axis_width(origin, [x y], hh, origin);
    % calculate in octaves
    if(fa>fb)
        Wfs(k) = log2(fa/fb);
    else
        Wfs(k) = log2(fb/fa);
    end
    
    % find the orientation bandwidth %%%%%%%%%%%%%
    
    % from the line above, find the perpendicular line
    % at the point fs (for now)
    perp_slope = -1/slope;
    if (isfinite(perp_slope))
        if (perp_slope == 0)
            xtop = 1;
            ytop = fs_ind(2);
            xbottom = size(hh,1);
            ybottom = fs_ind(2);
        else
            [xtop, ytop] = intercept(perp_slope, fs_ind, size(hh), true);
            [xbottom, ybottom] = intercept(perp_slope, fs_ind, size(hh), false);
        end
    else
        % horizontal line
        xtop = fs_ind(1);
        ytop = 1;
        xbottom = fs_ind(1);
        ybottom = size(hh,2);
    end
    [fa,fb,minor_width] = axis_width([xtop ytop], [xbottom ybottom], hh, origin);
    % calculate in degrees
    Wor(k) = radtodeg(atan(minor_width/(2*fs)));
    if(Wor(k)<0 || Wfs(k)<0)
        Wfs(k)
    end
    
%     imshow(sft_amp,[]); % display frequency
%     drawnow
end

% imagesc(basis_function), axis image off
% title('fourier transform')

% drawnow
% scatter(Wfs,Wor);
