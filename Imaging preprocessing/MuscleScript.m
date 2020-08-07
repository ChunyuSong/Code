% folder='s_2014122001_C004';
% T1=fidopen2D(128/96,[folder '/sems_05.fid']);
% % MTC=fidopen2D(1.5,[folder '/sems-MTC_01.fid']);
% % PD=fidopen2D(1.5,[folder '/sems-MTC_02.fid']);
% % MTR=1-MTC./PD;
% T2=fidopen2D(128/96,[folder '/mems-T2map_02.fid']);
% T2map=imresize(T2calc(T2,.015:.015:.180),2);


for i=1:size(T1,3)
%     num2str(i,'%.2u')
%     dtiFA(:,:,i)=imresize(fdfsingle([folder '/DBSI_results_0.3_0.3_3_3_perpendicular/dti_fa_map_slide_' num2str(i,'%.2u') '.fdf']),2);
%     dtiADC(:,:,i)=imresize(fdfsingle([folder '/DBSI_results_0.3_0.3_3_3_perpendicular/dti_adc_map_slide_' num2str(i,'%.2u') '.fdf']),2);
%     restricted_ratio(:,:,i)=imresize(fdfsingle([folder '/DBSI_results_0.3_0.3_3_3_perpendicular/restricted_ratio_map_slide_' num2str(i,'%.2u') '.fdf']),2);
%     water_ratio(:,:,i)=imresize(fdfsingle([folder '/DBSI_results_0.3_0.3_3_3_perpendicular/water_ratio_map_slide_' num2str(i,'%.2u') '.fdf']),2);
%     hindered_ratio(:,:,i)=imresize(fdfsingle([folder '/DBSI_results_0.3_0.3_3_3_perpendicular/hindered_ratio_map_slide_' num2str(i,'%.2u') '.fdf']),2);
%     fiber_ratio(:,:,i)=imresize(fdfsingle([folder '/DBSI_results_0.3_0.3_3_3_perpendicular/fiber_ratio_map_slide_' num2str(i,'%.2u') '.fdf']),2);

    imshow(T1(:,:,i),[])
    muscle(:,:,i)=roipoly;
    lesion(:,:,i)=roipoly;


end

% imshow(T1(:,:,i),[])
% muscle=roipoly;
% hole=roipoly;

na_muscle=logical(muscle-lesion);

T2_muscle=nanmean(T2map(muscle));
% mtr_muscle=meanmask(MTR,muscle,i);
dti_ADC_muscle=mean(dtiADC(muscle));
dti_FA_muscle=mean(dtiFA(muscle));
% dti_axial_muscle=mean(dtiAxial(muscle));
% dti_radial_muscle=mean(dtiRadial(muscle));

% fiber_FA_muscle=mean(fiberFA(muscle));
% fiber_axial_muscle=mean(fiberAxial(muscle));
% fiber_radial_muscle=mean(fiberRadial(muscle));

restricted_ratio_muscle=mean(restricted_ratio(muscle));
water_ratio_muscle=mean(water_ratio(muscle));
hindered_ratio_muscle=mean(hindered_ratio(muscle));
fiber_ratio_muscle=mean(fiber_ratio(muscle));


T2_nam=nanmean(T2map(na_muscle));
dti_ADC_nam=mean(dtiADC(na_muscle));
dti_FA_nam=mean(dtiFA(na_muscle));
restricted_ratio_nam=mean(restricted_ratio(na_muscle));
water_ratio_nam=mean(water_ratio(na_muscle));
hindered_ratio_nam=mean(hindered_ratio(na_muscle));
fiber_ratio_nam=mean(fiber_ratio(na_muscle));


T2_lesion=nanmean(T2map(lesion));
dti_ADC_lesion=mean(dtiADC(lesion));
dti_FA_lesion=mean(dtiFA(lesion));
restricted_ratio_lesion=mean(restricted_ratio(lesion));
water_ratio_lesion=mean(water_ratio(lesion));
hindered_ratio_lesion=mean(hindered_ratio(lesion));
fiber_ratio_lesion=mean(fiber_ratio(lesion));

size_muscle=sum(muscle(:));
size_nam=sum(na_muscle(:));
size_lesion=sum(lesion(:));
