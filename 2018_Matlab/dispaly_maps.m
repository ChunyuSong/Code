%%% Display multiple maps in single panel;
%%% created by YZZ on 9/5/2018;

clear all;
maps = {'dti_fa_map','restricted_ratio_1_map','restricted_ratio_2_map','hindered_ratio_map','water_ratio_map','fiber_ratio_map'};
names = {'DTI FA','highly restricted','restricted','hindered','free','anisotropic'};
% dirDBSI = ['C:\Users\zye01\Desktop\GBM\DBSI_results_0.2_0.2_1.5_1.5_2.5_2.5\'];

for i = 1:numel(maps);
    map = maps{i};
    img = char(map);
    name = names{i};
    map_name = char(name);
    subplottight(2,4,i);
    filename = fullfile([img,'.nii']);
    nii = load_nii(filename);
    img = nii.img;
    img = img(:,:,4);
    imshow(img,[0 1],'colormap',jet),colorbar;
    title(map_name);
    hold on;  
end;

subplot(2,4,1);
nii = load_nii('dti_adc_map.nii');
img = nii.img;
img = img(:,:,4);
imshow(img,[0 2],'colormap',gray),colorbar;
title('DTI ADC');
hold on;

subplot(2,4,2);
nii = load_nii('dti_fa_map.nii');
img = nii.img;
img = img(:,:,4);
imshow(img,[0 1],'colormap',jet),colorbar;
title('DTI FA');
hold on;

subplot(2,4,3);
nii = load_nii('restricted_ratio_1_map.nii');
img = nii.img;
img = img(:,:,4);
imshow(img,[0 1],'colormap',jet),colorbar;
title('Higly Restricted');
hold on;

subplot(2,4,4);
nii = load_nii('restricted_ratio_2_map.nii');
img = nii.img;
img = img(:,:,4);
imshow(img,[0 1],'colormap',jet),colorbar;
title('Restricted');
hold on;

subplot(2,4,5);
nii = load_nii('hindered_ratio_map.nii');
img = nii.img;
img = img(:,:,4);
imshow(img,[0 1],'colormap',jet),colorbar;
title('Hindered');
hold on;

subplot(2,4,6);
nii = load_nii('water_ratio_map.nii');
img = nii.img;
img = img(:,:,4);
imshow(img,[0 1],'colormap',jet),colorbar;
title('Hindered');
hold on;

subplot(2,4,7);
nii = load_nii('fiber_ratio_map.nii');
img = nii.img;
img = img(:,:,4);
imshow(img,[0 1],'colormap',jet),colorbar;
title('Anisotropic');
hold on;
