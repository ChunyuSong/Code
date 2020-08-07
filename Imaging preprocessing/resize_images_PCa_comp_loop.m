%% flip the DBSI.nii maps by yzz on 04/13/2016
%% edited on 05/05/2016 interpolate maps by twice
%% edited by JLin on 08/10/2017 for complete dbsi parameters
%% edited by JLin on 09/15/2017 to subsitute with bicubic interpolation to nearest neighbor interpolation




function output = resize_images_PCa_comp_loop()  

    
   list = walkpath_wrapper('.');
                 

   for i = 1:length(list)
       folder = list{i};
       if length(folder) > 29
           break;
       end
       fprintf('Processing folder %s\n', folder);
       
       % b0
        nii = load_nii('b0_map.nii');
        img = nii.img;
 
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'b0_map.nii');

        % model_v_map
        nii = load_nii('model_v_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'model_v_map.nii');

        % dti_axial_map
        nii = load_nii('dti_axial_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'dti_axial_map.nii');

        % dti_radial_map
        nii = load_nii('dti_radial_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'dti_radial_map.nii');

        % DTI_ADC
        nii=load_nii('dti_adc_map.nii');
        img=nii.img;
         
        DTI_ADC=imresize(img,2); % interpolate iamgew with bicubic  method
        nii=make_nii(DTI_ADC);
        save_nii(nii,'DTI_ADC_map.nii');

        % DTI_FA
        nii=load_nii('dti_fa_map.nii');
        img=nii.img;
         
        DTI_FA=imresize(img,2); % interpolate iamgew with bicubic  method
        nii=make_nii(DTI_FA);
        save_nii(nii,'DTI_FA_map.nii');


        % dti_dirx_map
        nii = load_nii('dti_dirx_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'dti_dirx_map.nii');

        % dti_diry_map
        nii = load_nii('dti_diry_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'dti_diry_map.nii');

        % dti_dirz_map
        nii = load_nii('dti_dirz_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'dti_dirz_map.nii');

        % dti_rgba_map
        nii = load_nii('dti_rgba_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'dti_rgba_map.nii');

        % dti_rgba_map_itk
        nii = load_nii('dti_rgba_map_itk.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'dti_rgba_itk_map.nii');

        % dti_r_map
        nii = load_nii('dti_r_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'dti_r_map.nii');

        % dti_g_map
        nii = load_nii('dti_g_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'dti_g_map.nii');

        % dti_b_map
        nii = load_nii('dti_b_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'dti_b_map.nii');

        % fiber_ratio
        nii=load_nii('fiber_ratio_map.nii');
        img=nii.img;
         
        fiber=imresize(img,2); % interpolate iamgew with bicubic  method
        nii=make_nii(fiber);
        save_nii(nii,'fiber_ratio_map.nii');

        % fiber1_axial_map
        nii = load_nii('fiber1_axial_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_axial_map.nii');

        % fiber1_radial_map
        nii = load_nii('fiber1_radial_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_radial_map.nii');


        % fiber1_fa_map
        nii = load_nii('fiber1_fa_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_fa_map.nii');


        % fiber1_fiber_ratio_map
        nii = load_nii('fiber1_fiber_ratio_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_fiber_ratio_map.nii');


        % fiber1_rgba_map
        nii = load_nii('fiber1_rgba_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_rgba_map.nii');

        % fiber1_dirx_map
        nii = load_nii('fiber1_dirx_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_dirx_map.nii');

        % fiber1_diry_map
        nii = load_nii('fiber1_diry_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_diry_map.nii');

        % fiber1_dirz_map
        nii = load_nii('fiber1_dirz_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'fiber1_dirz_map.nii');

        % fiber2_axial_map
        nii = load_nii('fiber2_axial_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_axial_map.nii');

        % fiber2_radial_map
        nii = load_nii('fiber2_radial_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_radial_map.nii');


        % fiber2_fa_map
        nii = load_nii('fiber2_fa_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_fa_map.nii');


        % fiber2_fiber_ratio_map
        nii = load_nii('fiber2_fiber_ratio_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_fiber_ratio_map.nii');

        % fiber2_dirx_map
        nii = load_nii('fiber2_dirx_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_dirx_map.nii');

        % fiber2_diry_map
        nii = load_nii('fiber2_diry_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_diry_map.nii');

        % fiber2_dirz_map
        nii = load_nii('fiber2_dirz_map.nii');
        img = nii.img;
         
        b0 = imresize(img,2); % interpolate iamge with bicubic  method
        nii = make_nii(b0);
        save_nii(nii,'fiber2_dirz_map.nii');

        % restricted_ratio_1;
        nii=load_nii('restricted_ratio_1_map.nii');
        img=nii.img;
         
        restricted=imresize(img,2); % interpolate iamgew with bicubic  method
        nii=make_nii(restricted);
        save_nii(nii,'restricted_ratio_1_map.nii');

        % restricted_adc_1;
        nii=load_nii('restricted_adc_1_map.nii');
        img=nii.img;
         
        restricted=imresize(img,2); % interpolate iamgew with bicubic  method
        nii=make_nii(restricted);
        save_nii(nii,'restricted_adc_1_map.nii');

        % restricted_ratio_2;
        nii=load_nii('restricted_ratio_2_map.nii');
        img=nii.img;
         
        restricted=imresize(img,2); % interpolate iamgew with bicubic  method
        nii=make_nii(restricted);
        save_nii(nii,'restricted_ratio_2_map.nii');

        % restricted_adc_2;
        nii=load_nii('restricted_adc_2_map.nii');
        img=nii.img;
         
        restricted=imresize(img,2); % interpolate iamgew with bicubic  method
        nii=make_nii(restricted);
        save_nii(nii,'restricted_adc_2_map.nii');

        % hindered_ratio
        nii=load_nii('hindered_ratio_map.nii');
        img=nii.img;
         
        hindered=imresize(img,2); % interpolate iamgew with bicubic  method
        nii=make_nii(hindered);
        save_nii(nii,'hindered_ratio_map.nii');

        % hindered_adc
        nii=load_nii('hindered_adc_map.nii');
        img=nii.img;
         
        hindered=imresize(img,2); % interpolate iamgew with bicubic  method
        nii=make_nii(hindered);
        save_nii(nii,'hindered_adc_map.nii');

        % water_ratio
        nii=load_nii('water_ratio_map.nii');
        img=nii.img;
         
        water=imresize(img,2); % interpolate iamgew with bicubic  method
        nii=make_nii(water);
        save_nii(nii,'water_ratio_map.nii');

        % water_adc
        nii=load_nii('water_adc_map.nii');
        img=nii.img;
         
        water=imresize(img,2); % interpolate iamgew with bicubic  method
        nii=make_nii(water);
        save_nii(nii,'water_adc_map.nii');

        % iso_adc
        nii=load_nii('iso_adc_map.nii');
        img=nii.img;
         
        water=imresize(img,2); % interpolate iamgew with bicubic  method
        nii=make_nii(water);
        save_nii(nii,'iso_adc_map.nii');

        % fraction_rgba_map
        nii=load_nii('fraction_rgba_map.nii');
        img=nii.img;
         
        water=imresize(img,2); % interpolate iamgew with bicubic  method
        nii=make_nii(water);
        save_nii(nii,'fraction_rgba_map.nii');

        % fiber1_rgba_map_itk
        nii=load_nii('fiber1_rgba_map_itk.nii');
        img=nii.img;
         
        water=imresize(img,2); % interpolate iamgew with bicubic  method
        nii=make_nii(water);
        save_nii(nii,'fiber1_rgba_itk_map.nii');

        disp('Generate DBSI Results: Completed!');
        
   end
end
 
function list = walkpath(list, current_path)
    fprintf('%d %s\n', length(list), current_path);
    names = dir(current_path);
    for i = 1:length(names)
        abspath = [current_path filesep names(i).name];
        % if prefix is DBSI_results
        if length(strfind(names(i).name,'DHISTO_results')) > 0
            list{end+1} = abspath;
            continue;
        end
        if exist(abspath) == 7 && strcmp(names(i).name, '.') == 0 && strcmp(names(i).name, '..') == 0            
            list = walkpath(list, abspath);
        end
    end
end


function list = walkpath_wrapper(basedir)
    list = walkpath({}, basedir);
end


















