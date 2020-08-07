function [] = area_of_segmentation()

% close all;

default_input_mat = 'Image_data.mat';

parameter_file = 'color_marker.mat';

if ~exist(default_input_mat,'file')
            error('Please run Color_Multiple_ROI.m to generate file Image_data.mat');
else
    load(default_input_mat);
end

if ~exist(parameter_file,'file')
            error('Please run Color_Multiple_ROI.m to generate file color_marker.mat');
end

ivalue = ivalue;
jvalue = jvalue;
basename = basename;
count_tile = zeros(ivalue,jvalue);
area_tile = zeros(ivalue,jvalue);    
BW_tile = cell([ivalue jvalue]);
    
%% check MATLAB version to create parallel pool on cluster
if verLessThan('matlab','8.4.0')
   % execute code for R2014a or earlier
   try
       parpool('close');  
   catch
   end
   parpool('local');
else
   % execute code for R2014b or later
   % delete parpool if any existed.
   parobj = gcp('nocreate'); 
   if ~isempty(parobj)
       delete(parobj);
   end
   local_cluster_profile=parcluster('local');
   parobj = parpool(local_cluster_profile.NumWorkers-1);
end    
parfor i = 1 : ivalue
    for j = 1 : jvalue
        if ivalue < 100
            iindex = num2str(100+i);
        else
            iindex = num2str(1000+i);
        end
        iindex = iindex(2:end);
        if jvalue < 100
            jindex = num2str(100+j);
        else
            jindex = num2str(1000+j);
        end
        jindex = jindex(2:end);        
        fullfilename = [basename,'_','i',iindex,'j',jindex,'.tif'];

        image = imread(fullfilename);

        [count,area,BW] = color_segmentation(image, parameter_file);
               
         count_tile(i,j) = count;
         area_tile(i,j) = area;
         BW_tile(i,j) = {BW};
    end%j
end%i

BW_final = cell2mat(BW_tile);
h_bw=figure();
imshow(BW_final)
title('Binary Map')
% MyMontage = get(h_bw, 'CData');
% imwrite(BW_final,'BW_whole.tif');

% myfilter = fspecial('gaussian',[10 10], 1);
% % count_tile_filtered = imfilter(count_tile, myfilter, 'replicate');
% area_tile_filtered = imfilter(area_tile, myfilter, 'replicate');
area_tile_filtered = imgaussfilt(area_tile, 0.5);
nii = make_nii(area_tile_filtered);
save_nii(nii,'area.nii');

figure('name', 'Segmented Area', 'numbertitle', 'off')
% subplot(221)
% imshow(fliplr(count_tile),[]);
% colormap('jet');
% title('Raw Count')
% colorbar
% subplot(222)
% imshow(fliplr(count_tile_filtered),[])
% colormap('jet');
% title('Raw Count Filtered')
% colorbar
subplot(121)
imshow(area_tile,[]);
colormap('jet');
title('Raw Area')
colorbar
F = getframe;
imwrite(F.cdata,jet,'area.tif');
subplot(122)
imshow(area_tile_filtered,[])
colormap('jet');
title('Raw Area Filtered')
colorbar
F = getframe;
imwrite(F.cdata,jet,'area_filt.tif');
