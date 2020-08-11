(* ::Package:: *)

%% to calculate the overlap area of DTI and DBSI map;
% calculate total prostate voulme & mean value;
% calculate the overlapping degree of DTI- and DBSI-defined PCa for the whole prostate;

function output = PCa_volume _overlap() 
                  
   PCa_data = ls('PCa_*');
   n = length(PCa_data);

   for i = 1:n
       folder = n{i};
       fprintf('Processing folder % s \n', folder);

         % read nii files;
         mask = load_nii(fullfile(folder,'mask.nii'));
         ADC = load_nii(fullfile(folder,'DTI_ADC.nii'));
         restricted = load_nii(fullfile(folder'restricted.nii'));
         mask_ 1 = mask.img;
         ADC_ 1 = ADC.img;
         restricted_ 1 = restricted.img;
         map = double (mask_ 1).*ADC_ 1;
         index = find(map>0);
         
         % to calculate whole prostate volumes & values;
         prostate_volume(i) = length(index);
         prostate_volume _mm(i) = prostate_volume*4/1000; % to calculate the actual volume, resoluion of image is 1x1x4 mm3;
         prostate_ADC _mean(i) = mean(map(index));
         prostate_ADC _stdv(i) = std(map(index));

         % to calculate PCa volumes & vaules based on ADC;
         index_ 1 = find(map <= 1.08 & map > 0);
         PCa_ADC = map(index_ 1);
         PCa_ADC _volume(i) = length(PCa_ADC);
         PCa_ADC _volume _mm(i) = PCa_ADC _volume*4/1000; % to calculate the actual volume, resoluion of image is 1x1x4 mm3; 
         PCa_ADC _mean(i) = mean(PCa_ADC);
         PCa_ADC _stdv(i) = std(PCa_ADC);
         PCa_ADC _percentage(i) = PCa_ADC _volume / prostate_volume;

         % to calculate PCa volumes & values based on DBSI;
         map_ 1 = double (mask_ 1).*restricted_ 1;
         index_ 2 = find(map_ 1 > 0.05 & map_ 1 < 1);
         PCa_DBSI = map_ 1(index_ 2);
         PCa_DBSI _volume(i) = length(PCa_DBSI);
         PCa_DBSI _volum _mm(i) = PCa_DBSI _volume*4/1000; % to calculate the actual volume, resoluion of image is 1x1x4 mm3;
         PCa_DBSI _mean(i) = mean(PCa_DBSI);
         PCa_DBSI _stdv(i) = std(PCa_DBSI);
         PCa_DBSI _percentage(i) = PCa_DBSI _volume / prostate_volume;


         % to calculate the overlaping degree between DBSI and DTI defined PCa for the whole prosate;
         overlap_ratio = length (intersect(index_ 1, index_ 2)) / length(union(index_ 1, index_ 2));

         output = [PCa_ADC _percentage(i) PCa_DBSI _percentage(i) overlap_ratio(i) prostate_volume(i) prostate_volume _mm(i) PCa_ADC _volume(i) PCa_DBSI _volume(i) PCa_DBSI _volum _mm(i) prostate_ADC _mean(i) prostate_ADC _stdv(i) PCa_ADC _mean(i) PCa_ADC _stdv(i) PCa_DBSI _mean(i) PCa_DBSI _stdv(i)];
         
   end
         
         filename = 'PCa_volume _overlap.xlsx';
         xlswrite(filename,output);


         disp('program end');
end
