function [obj_result, area_result, BW_filled] = color_segmentation(image,parameter_file)

%for color segmentation 
%8/8/2016

load(parameter_file);

imdata = image;

[rows columns numberOfColorBands] = size(imdata);

if numberOfColorBands > 1
    R = imdata(:,:,1);
    G = imdata(:,:,2);
    B = imdata(:,:,3);
else
    R = imdata(:,:,1);
    G = imdata(:,:,1);
    B = imdata(:,:,1);
end

R = double(R);
G = double(G);
B = double(B);

%% choose stain type and nearest neighbor classification
      
distance = ( (R - color_markers(1,1)).^2 + (G - color_markers(1,2)).^2 + (B - color_markers(1,3)).^2).^0.5;

%% steps to generate binary image highlighting  object or region of interest

distance = mat2gray(distance);
[X,~] = gray2ind(distance);
if numberOfColorBands > 1
   BW = roicolor(X,[0:index]);
else
   BW = R.*63 > color_markers(1,1).*63-double(index);
end


% properties = regionprops(BW, {'Area', 'ConvexArea', 'Eccentricity', 'EquivDiameter', 'EulerNumber', 'Extent', 'FilledArea', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter'});
% 
% BW = bwpropfilt(BW, 'Area', [-Inf, stats.Area]);
% BW = bwpropfilt(BW, 'EquivDiameter', [0, stats.EquivDiameter]);

%Step 4: get the number
% area = nnz(BW)/prod(size(BW))*100;
% area = bwarea(BW) * 100/numel(BW);
% count = bwconncomp(BW);
if numberOfColorBands > 1
    BW = bwareaopen(BW, 20);
    area = bwarea(BW)* 100/numel(BW);
else
    test = (find(R>0));
    if isempty(test)
       test = numel(BW);
    else 
       test = length(test);
    end
    area = bwarea(BW)* 100/test;    
    if test < (0.7*6084)
        area = 0;
    end
end

BW_filled=imfill(BW,'holes');
obj_result = bwconncomp(BW_filled);
% count_result = obj_result.NumObjects;
area_result = area;
