(* ::Package:: *)

%% to calculate the overlap area of DTI and DBSI map;
% calculate total prostate voulme & mean value;
% calculate the overlapping degree of DTI- and DBSI-defined PCa for the whole prostate;

% looping over subfolders
listfolderscharacter = ls();
listoffolders = cellstr(listfolderscharacter);
listoffolders = listoffolders(3:length(listoffolders));
myvalues = zeros(length(listoffolders),12);

for k=1:length(listoffolders);
    thisfolder = listoffolders{k};
    fprintf('Processing folder % s \n', thisfolder);
   
    % read nii files;
    mask = load_nii(fullfile(thisfolder,'\mask.nii'));
    ADC = load_nii(fullfile(thisfolder,'\DTI_ADC.nii'));
    restricted = load_nii(fullfile(thisfolder,'\r estricted.nii'));
    mask_ 1 = mask.img;
    ADC_ 1 = ADC.img;
    restricted_ 1 = restricted.img;
    map = double (mask_ 1).*ADC_ 1;
    index = find(map>0);
    
    % to calculate whole prostate volumes & values;
    prostate_voxel _number = length(index);
    prostate_volume = prostate_voxel _number*4/1000; % to calculate the actual volume, resoluion of image is 1x1x4 mm3;
    prostate_ADC _mean = mean(map(index));
    prostate_ADC _stdv = std(map(index));
    
    % to calculate PCa volumes & vaules based on ADC;
    index_ 1 = find(map <= str2num(answer{2}) & map > str2num(answer{1}));
    PCa_ADC = map(index_ 1);
    PCa_ADC _voxel _number = length(PCa_ADC);
    PCa_ADC _volume = PCa_ADC _voxel _number*4/1000; % to calculate the actual volume, resoluion of image is 1x1x4 mm3;
    PCa_ADC _mean = mean(PCa_ADC);
    PCa_ADC _stdv = std(PCa_ADC);
    PCa_ADC _percentage = PCa_ADC _voxel _number / prostate_voxel_number;
    
    % to calculate PCa volumes & values based on DBSI;
    map_ 1 = double (mask_ 1).*restricted_ 1;
    index_ 3 = find(map_ 1 > 0);
    prostate_restricted _mean = mean (map_ 1(index_ 3));
    index_ 2 = find(map_ 1 > str2num(answer{3}) & map_ 1 < str2num(answer{4}));
    PCa_DBSI = map_ 1(index_ 2);
    PCa_DBSI _voxel _number = length(PCa_DBSI);
    PCa_DBSI _volume = PCa_DBSI _voxel _number*4/1000; % to calculate the actual volume, resoluion of image is 1x1x4 mm3;
    PCa_DBSI _mean = mean(PCa_DBSI);
    PCa_DBSI _stdv = std(PCa_DBSI);
    PCa_DBSI _fraction _volume = PCa_DBSI _volume*PCa_DBSI _mean; % to calculate the fraction volume of PCa; 
    PCa_DBSI _percentage = PCa_DBSI _voxel _number / prostate_voxel_number;
    PCa_DBSI _fraction _percentage = PCa_DBSI _fraction _volume / prostate_volume;
    
    
    
    % to calculate the overlaping degree between DBSI and DTI defined PCa for the whole prosate;
    overlap_ratio = length (intersect(index_ 1, index_ 2)) / length(union(index_ 1, index_ 2));
    
    myvalues(k,1) = PCa_ADC _percentage; 
    myvalues(k,2) = PCa_DBSI _percentage;
    myvalues(k,3) = PCa_DBSI _fraction _percentage;
    myvalues(k,4) = overlap_ratio; 
    myvalues(k,5) = prostate_volume;
    myvalues(k,6) = PCa_ADC _volume; 
    myvalues(k,7) = PCa_DBSI _volume;
    myvalues(k,8) = PCa_DBSI _fraction _volume;
    myvalues(k,9) = prostate_ADC _mean; 
    myvalues(k,10) = PCa_ADC _mean;
    myvalues(k,11) = PCa_DBSI _mean;
    myvalues(k,12) = prostate_restricted _mean;
end

xlswrite(['Z:\Zezhong_Ye\Prostate_Cancer_Project _Shanghai\Josh_Lin_PCa\PCa_ratio_results\' 'ADC_DBSI _overlap _''.xls'],myvalues);
