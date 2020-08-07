%% flip the DBSI.nii maps by yzz on 04/13/2016
%% edited on 05/05/2016 interpolate maps by twice
%% edited by JLin on 08/10/2017 for complete dbsi parameters
%% edited by JLin on 09/15/2017 to subsitute bicubic interpolation to nearest neighbor interpolation
function output = prostate_cancer()  


                 
   list = cellstr(ls('Patient_*'));

   for i = 1:length(list)
       folder = list{i};
       if length(folder) > 13
           break;
       end
       fprintf('Processing folder %s\n', folder);
       
       % b0
        nii = load_nii('b0_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'b0.nii');

        % model_v_map
        nii = load_nii('model_v_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'model_v.nii');

        % dti_axial_map
        nii = load_nii('dti_axial_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_axial.nii');

        % dti_radial_map
        nii = load_nii('dti_radial_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_radial.nii');

        % DTI_ADC
        nii=load_nii('dti_adc_map.nii');
        img=nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        DTI_ADC=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(DTI_ADC);
        save_nii(nii,'DTI_ADC.nii');

        % DTI_FA
        nii=load_nii('dti_fa_map.nii');
        img=nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        DTI_FA=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(DTI_FA);
        save_nii(nii,'DTI_FA.nii');


        % dti_dirx_map
        nii = load_nii('dti_dirx_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_dirx.nii');

        % dti_diry_map
        nii = load_nii('dti_diry_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_diry.nii');

        % dti_dirz_map
        nii = load_nii('dti_dirz_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_dirz.nii');

        % dti_rgba_map
        nii = load_nii('dti_rgba_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_rgba.nii');

        % dti_rgba_map_itk
        nii = load_nii('dti_rgba_map_itk.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_rgba_itk.nii');

        % dti_r_map
        nii = load_nii('dti_r_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_r.nii');

        % dti_g_map
        nii = load_nii('dti_g_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_g.nii');

        % dti_b_map
        nii = load_nii('dti_b_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'dti_b.nii');

        % fiber_ratio
        nii=load_nii('fiber_ratio_map.nii');
        img=nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        fiber=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(fiber);
        save_nii(nii,'fiber_ratio.nii');

        % fiber1_axial_map
        nii = load_nii('fiber1_axial_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_axial.nii');

        % fiber1_radial_map
        nii = load_nii('fiber1_radial_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_radial.nii');


        % fiber1_fa_map
        nii = load_nii('fiber1_fa_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_fa.nii');


        % fiber1_fiber_ratio_map
        nii = load_nii('fiber1_fiber_ratio_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_fiber_ratio.nii');


        % fiber1_rgba_map
        nii = load_nii('fiber1_rgba_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_rgba.nii');

        % fiber1_dirx_map
        nii = load_nii('fiber1_dirx_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_dirx.nii');

        % fiber1_diry_map
        nii = load_nii('fiber1_diry_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_diry.nii');

        % fiber1_dirz_map
        nii = load_nii('fiber1_dirz_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_dirz.nii');

        % fiber2_axial_map
        nii = load_nii('fiber2_axial_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_axial.nii');

        % fiber2_radial_map
        nii = load_nii('fiber2_radial_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_radial.nii');


        % fiber2_fa_map
        nii = load_nii('fiber2_fa_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_fa.nii');


        % fiber2_fiber_ratio_map
        nii = load_nii('fiber2_fiber_ratio_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_fiber_ratio.nii');

        % fiber2_dirx_map
        nii = load_nii('fiber2_dirx_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_dirx.nii');

        % fiber2_diry_map
        nii = load_nii('fiber2_diry_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_diry.nii');

        % fiber2_dirz_map
        nii = load_nii('fiber2_dirz_map.nii');
        img = nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        b0 = imresize(img_1,2,'nearest'); % interpolate iamge with nearest neighbor method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_dirz.nii');

        % restricted_ratio_1;
        nii=load_nii('restricted_ratio_1_map.nii');
        img=nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        restricted=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(restricted);
        save_nii(nii,'restricted_ratio_1.nii');

        % restricted_adc_1;
        nii=load_nii('restricted_adc_1_map.nii');
        img=nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        restricted=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(restricted);
        save_nii(nii,'restricted_adc_1.nii');

        % restricted_ratio_2;
        nii=load_nii('restricted_ratio_2_map.nii');
        img=nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        restricted=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(restricted);
        save_nii(nii,'restricted_ratio_2.nii');

        % restricted_adc_2;
        nii=load_nii('restricted_adc_2_map.nii');
        img=nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        restricted=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(restricted);
        save_nii(nii,'restricted_adc_2.nii');

        % hindered_ratio
        nii=load_nii('hindered_ratio_map.nii');
        img=nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        hindered=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(hindered);
        save_nii(nii,'hindered_ratio.nii');

        % hindered_adc
        nii=load_nii('hindered_adc_map.nii');
        img=nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        hindered=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(hindered);
        save_nii(nii,'hindered_adc.nii');

        % water_ratio
        nii=load_nii('water_ratio_map.nii');
        img=nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        water=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(water);
        save_nii(nii,'water_ratio.nii');

        % water_adc
        nii=load_nii('water_adc_map.nii');
        img=nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        water=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(water);
        save_nii(nii,'water_adc.nii');

        % iso_adc
        nii=load_nii('iso_adc_map.nii');
        img=nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        water=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(water);
        save_nii(nii,'iso_adc.nii');

        % fraction_rgba_map
        nii=load_nii('fraction_rgba_map.nii');
        img=nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        water=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(water);
        save_nii(nii,'fraction_rgba.nii');

        % fiber1_rgba_map_itk
        nii=load_nii('fiber1_rgba_map_itk.nii');
        img=nii.img;
        img_1 = fliplr(img); % flip iamges vertically
        water=imresize(img_1,2,'nearest'); % interpolate iamgew with nearest neighbor method
        nii=make_nii(water);
        save_nii(nii,'fiber1_rgba_itk.nii');

        % generate DWI_1500 image
        nii = load_nii('dti_adc_map.nii');
        adc=nii.img;
        b0=load_nii('b0_map.nii');
        b0=b0.img;
        img=b0.*exp(-1.5*adc);
        img_1 = fliplr(img); % flip iamges vertically
        dwi_1500=imresize(img_1,2,'nearest'); % interpolate iamgew with bi cubic method
        nii = make_nii(dwi_1500);
        save_nii(nii,'dwi_1500.nii');

        disp('Generate DBSI Results: Completed!');


















