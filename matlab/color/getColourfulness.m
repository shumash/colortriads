function C = getColourfulness( im )
%
% C = getColourfulness( im )
%   
% MATLAB algorithm implementation of the 
% "Measuring colourfulness in natural images" 
% (Hasler and Susstrunk, 2003) 
%
%   Input:
%       im  - image in RGB
%
%   Output:
%       C   - colourfulness
%

    i = 1;
    [W, H, RGB] = size(im);
    
    for x=1:W
        for y=1:H
    
            % rg = R - G
            rg(i) = abs(double(im(x,y,1)) - double(im(x,y,2)));
            
            % yb = 1/2(R + G) - B
            yb(i) = abs(.5 * (double(im(x,y,1)) + double(im(x,y,2))) - double(im(x,y,3)));
            
            i = i+1;
        end
    end
    
    max(rg(:))
    min(rg(:))
    
    % standard deviation and the mean value of the pixel
    % cloud along direction, respectively
    stdRG = std(rg)
    meanRG = mean(rg)
    
    stdYB = std(yb)
    meanYB = mean(yb)
    
    stdRGYB = sqrt((stdRG)^2 + (stdYB)^2)
    meanRGYB = sqrt((meanRG)^2 + (meanYB)^2)
    
    C = stdRGYB + 0.3*meanRGYB; 
end