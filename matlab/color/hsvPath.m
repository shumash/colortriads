function hsvPath(inrgb)
    hsv = rgb2hsv(inrgb)
    
    hsvsamples=hsv;
    for i=2:3
        v=hsv(i);
        vstart=max(0.0, v-0.4)
        vend=min(1.0, v+0.4)
        for v=vstart:0.01:vend
            new_val = hsv;
            new_val(i) = v;
            hsvsamples(size(hsvsamples, 1)+1, :) = new_val;
        end
    end
    
    rgbsamples = hsv2rgb(hsvsamples);
    labsamples = rgb2lab(rgbsamples);   
    fh = figure; 
    scatter3(labsamples(1:end,1), labsamples(1:end,2), labsamples(1:end,3), 40, rgbsamples, 'filled');
    ax.XLim = [ 0, 100.0];
ax.YLim = [ -120, 120.0];
ax.ZLim = [ -120, 120.0];
    
end