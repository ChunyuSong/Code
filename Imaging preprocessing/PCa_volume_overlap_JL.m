%% to calculate the overlap area of DTI and DBSI map;
% created by yzz, modified by ns & jl on 06/13/2016;
% calculate total prostate voulme & mean value;
% calculate the overlapping degree of DTI- and DBSI-defined PCa for the whole prostate;

%looping over subfolders
listfolderscharacter = ls();
listoffolders = cellstr(listfolderscharacter);
listoffolders = listoffolders(3:length(listoffolders));
myvalues = zeros(length(listoffolders),12);

for k=1:length(listoffolders);
    thisfolder = listoffolders{k};
    fprintf('Processing folder %s\n', thisfolder);
   
    % read nii files;
    mask = load_nii(fullfile(thisfolder,'\mask.nii'));
    ADC = load_nii(fullfile(thisfolder,'\DTI_ADC.nii'));
    restricted = load_nii(fullfile(thisfolder,'\restricted.nii'));
    mask_1 = mask.img;
    ADC_1 = ADC.img;
    restricted_1 = restricted.img;
    map = double(mask_1).*ADC_1;
    index = find(map>0);
    
    % to calculate whole prostate volumes & values;
    prostate_voxel_number = length(index);
    prostate_volume = prostate_voxel_number*4/1000; %to calculate the actual volume, resoluion of image is 1x1x4 mm3;
    prostate_ADC_mean = mean(map(index));
    prostate_ADC_stdv = std(map(index));
    
    % to calculate PCa volumes & vaules based on ADC;
    index_1 = find(map <= str2num(answer{2}) & map > str2num(answer{1}));
    PCa_ADC = map(index_1);
    PCa_ADC_voxel_number = length(PCa_ADC);
    PCa_ADC_volume = PCa_ADC_voxel_number*4/1000; %to calculate the actual volume, resoluion of image is 1x1x4 mm3;
    PCa_ADC_mean = mean(PCa_ADC);
    PCa_ADC_stdv = std(PCa_ADC);
    PCa_ADC_percentage = PCa_ADC_voxel_number / prostate_voxel_number;
    
    % to calculate PCa volumes & values based on DBSI;
    map_1 = double(mask_1).*restricted_1;
    index_3 = find(map_1 > 0);
    prostate_restricted_mean = mean (map_1(index_3));
    index_2 = find(map_1 > str2num(answer{3}) & map_1 < str2num(answer{4}));
    PCa_DBSI = map_1(index_2);
    PCa_DBSI_voxel_number = length(PCa_DBSI);
    PCa_DBSI_volume = PCa_DBSI_voxel_number*4/1000; %to calculate the actual volume, resoluion of image is 1x1x4 mm3;
    PCa_DBSI_mean = mean(PCa_DBSI);
    PCa_DBSI_stdv = std(PCa_DBSI);
    PCa_DBSI_fraction_volume = PCa_DBSI_volume*PCa_DBSI_mean; % to calculate the fraction volume of PCa; 
    PCa_DBSI_percentage = PCa_DBSI_voxel_number / prostate_voxel_number;
    PCa_DBSI_fraction_percentage = PCa_DBSI_fraction_volume / prostate_volume;
    
    
    
    % to calculate the overlaping degree between DBSI and DTI defined PCa for the whole prosate;
    overlap_ratio = length(intersect(index_1, index_2)) / length(union(index_1, index_2));
    
    myvalues(k,1) = PCa_ADC_percentage; 
    myvalues(k,2) = PCa_DBSI_percentage;
    myvalues(k,3) = PCa_DBSI_fraction_percentage;
    myvalues(k,4) = overlap_ratio; 
    myvalues(k,5) = prostate_volume;
    myvalues(k,6) = PCa_ADC_volume; 
    myvalues(k,7) = PCa_DBSI_volume;
    myvalues(k,8) = PCa_DBSI_fraction_volume;
    myvalues(k,9) = prostate_ADC_mean; 
    myvalues(k,10) = PCa_ADC_mean;
    myvalues(k,11) = PCa_DBSI_mean;
    myvalues(k,12) = prostate_restricted_mean;
end

xlswrite(['Z:\Zezhong_Ye\Prostate_Cancer_Project_Shanghai\Josh_Lin_PCa\PCa_ratio_results\' 'ADC_DBSI_overlap_''.xls'],myvalues);