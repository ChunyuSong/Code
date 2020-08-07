function dbsi_save(ini_file)
% RawOutputFileName is the filename of the output class 'classOutput'
% after load the raw output file, class named as 'cRawData'
% function of output module to generate FDF maps
% 
% 2012-09-04: created by Peng
% 2015-01-20: create nifti from image data, independent from the source 
% file format.
% format option: 'nii' or 'ana'


if nargin < 1
    if exist('DBSIClassData.mat','file')
        dbsi_class_file = 'DBSIClassData.mat';
        data_dir = pwd;
    else
        [dbsi_class_file,data_dir,filter_index] = uigetfile(pwd,'Please select the DBSIClassData mat file');
        dbsi_class_file = fullfile(data_dir, dbsi_class_file);
    end
    load(dbsi_class_file); 
    if ~exist('cRawData')
        disp('this is not a proper DBSI output mat file, program exit!')
        return;
    end
    
    prompt = {'Threshold_1(range from 0 to ):','Threshold_2(Range from 0.1 to ):','Threshold_3(Range from 0.5 to ):','Threshold_4(Range to infinity):'};
    dlg_title = 'Input thresholds to differentiate isotropic components';
    num_lines = 1;
    def = {'0.1','0.5','3.0','3.0'};
    answer = inputdlg(prompt,dlg_title,num_lines,def);
    iso_threshold = [str2num(answer{1}) str2num(answer{2}) str2num(answer{3}) str2num(answer{4})]; 
    iso_threshold = 10^-3* iso_threshold;% Adjust unit
    
    [s,v] = listdlg('PromptString','output option:', 'SelectionMode','single','ListString',{'compact','all'});
    if v>0
        output_option = s;
    else
        output_option = 1;
    end
    
    liststr = {'nii','ana'};
    [s,v] = listdlg('PromptString','output option:', 'SelectionMode','single','ListString',liststr);
    if v==0
        output_format = liststr{1};
    else
        output_format = liststr{s};
    end
    
    button = questdlg('output fib','fib file for dsi studio','yes','no','no') ;
    switch button
        case 'yes'
            output_fib = 1;
        case 'no'
            output_fib = 0;
    end
    if output_fib > 0
        prompt = {'x:','y','z'};
        dlg_title = 'image resolution to fib';
        num_lines = 1;
        def = {'1','1','1'};
        answer = inputdlg(prompt,dlg_title,num_lines,def);
        output_fib_res = [str2num(answer{1}) str2num(answer{2}) str2num(answer{3})]; 
    end
else
    if ~exist(ini_file,'file')
        disp('can''t find input ini file for output, program exist!');
        return; 
    end
    ini = IniConfig();
    ini.ReadFile(ini_file);

    data_dir = ini.GetValues('INPUT','data_dir');
    if isempty(data_dir)
        data_dir = pwd;
    end
    dbsi_class_file = ini.GetValues('DBSI', 'dbsi_class_file');
    output_option = ini.GetValues('OUTPUT', 'output_option');
    output_format = ini.GetValues('OUTPUT', 'output_format');
    iso_threshold = ini.GetValues('OUTPUT', 'iso_threshold');
    if isempty(iso_threshold)
        prompt = {'Threshold_1(range from 0 to ):','Threshold_2(Range from 0.1 to ):','Threshold_3(Range from 0.5 to):','CSF Threshold(Range to infinity):'};
        dlg_title = 'Input thresholds to differentiate isotropic components';
        num_lines = 1;
        def = {'0.1','0.5','3.0','3.0'};
        answer = inputdlg(prompt,dlg_title,num_lines,def);
        iso_threshold = [str2num(answer{1}) str2num(answer{2}) str2num(answer{3}) str2num(answer{4})]; 
        ini.SetValues('OUTPUT','iso_threshold',{iso_threshold});
    end
    iso_threshold = 10^-3* iso_threshold;% Adjust unit
    output_fib = ini.GetValues('OUTPUT', 'output_fib');
    output_fib_res = ini.GetValues('OUTPUT','output_fib_res');
    if isempty(output_fib_res)
        output_fib_res = [1 1 1];
    end
    
    dbsi_class_file_full = fullfile(data_dir,dbsi_class_file);
    if ~exist(dbsi_class_file_full,'file')
        disp('can''t find DBSI output mat file in data_dir, tried search current folder');
        dbsi_class_file_full = fullfile(pwd,dbsi_class_file);
        if ~exist(dbsi_class_file_full,'file')
            disp('can''t not find DBSI output mat file in data_dir and local folder, program exit!')
            return;
        end        
    end
    load(dbsi_class_file_full); 
    if ~exist('cRawData','var')
        disp('this is not a proper DBSI output mat file, program exit!')
        return;
    end
end

copyfile(which('versions.txt'),fullfile(data_dir,'save.version'));

outlist_clean = {'b0_map',...
    'dti_axial_map',...
    'dti_radial_map',...
    'dti_adc_map',...
    'dti_fa_map',...
    'fiber_ratio_map',...
    'fiber1_axial_map',...
    'fiber1_radial_map',...
    'fiber1_fa_map',...
    'restricted_ratio_1_map',...
    'restricted_ratio_2_map',...
    'hindered_ratio_map',...
    'water_ratio_map',...
    };

outlist_full = {'b0_map',...
    'model_v_map',...
    'dti_axial_map',...
    'dti_radial_map',...
    'dti_adc_map',...
    'dti_fa_map',...
    'dti_dirx_map',...
    'dti_diry_map',...
    'dti_dirz_map',...
    'dti_rgba_map',...
    'dti_rgba_map_itk',...
    'dti_r_map',...
    'dti_g_map',...
    'dti_b_map',...
    'fiber_ratio_map',...
    'fiber1_axial_map',...
    'fiber1_radial_map',...
    'fiber1_fa_map',...
    'fiber1_fiber_ratio_map',...
    'fiber1_rgba_map',...
    'fiber1_dirx_map',...
    'fiber1_diry_map',...
    'fiber1_dirz_map',...
    'fiber2_axial_map',...
    'fiber2_radial_map',...
    'fiber2_fa_map',...
    'fiber2_fiber_ratio_map',...
    'fiber2_dirx_map',...
    'fiber2_diry_map',...
    'fiber2_dirz_map',...
    'restricted_ratio_1_map',...
    'restricted_adc_1_map',...
    'restricted_ratio_2_map',...
    'restricted_adc_2_map',...
    'hindered_ratio_map',...
    'hindered_adc_map',...
    'water_ratio_map',...
    'water_adc_map',...
    'fraction_rgba_map',...
    'fiber1_rgba_map_itk',...
    };

if output_option == 1
    outlist = outlist_clean;
else
    outlist = outlist_full;
end

% initialize all the maps as matrix
map_dimensions = [cRawData.sImageHeader.image_size, cRawData.sImageHeader.ns];
roi_index=sub2ind(map_dimensions,cRawData.DBSI_aRoiIndex(1,:),cRawData.DBSI_aRoiIndex(2,:),cRawData.DBSI_aRoiIndex(3,:));

%%
total_ratio = sum([cRawData.DBSI_aFiber1Ratio;cRawData.DBSI_aFiber2Ratio;cRawData.DBSI_aIsoSpecdata],1);
cRawData.DBSI_aFiber1Ratio = cRawData.DBSI_aFiber1Ratio./total_ratio;
cRawData.DBSI_aFiber2Ratio = cRawData.DBSI_aFiber2Ratio./total_ratio;

d1st_isotropic_grid = cRawData.DBSI_iIsoSpecGrid < iso_threshold(1);
d2nd_isotropic_grid = cRawData.DBSI_iIsoSpecGrid < iso_threshold(3) & cRawData.DBSI_iIsoSpecGrid > iso_threshold(2) ;
d3rd_isotropic_grid = cRawData.DBSI_iIsoSpecGrid < iso_threshold(4) & cRawData.DBSI_iIsoSpecGrid > iso_threshold(3) ;
d4th_isotropic_grid = cRawData.DBSI_iIsoSpecGrid > iso_threshold(4);
d1st_isotropic_ratio = sum(cRawData.DBSI_aIsoSpecdata(d1st_isotropic_grid,:),1)./total_ratio;
d2nd_isotropic_ratio = sum(cRawData.DBSI_aIsoSpecdata(d2nd_isotropic_grid,:),1)./total_ratio;
d3rd_isotropic_ratio = sum(cRawData.DBSI_aIsoSpecdata(d3rd_isotropic_grid,:),1)./total_ratio;
d4th_isotropic_ratio = sum(cRawData.DBSI_aIsoSpecdata(d4th_isotropic_grid,:),1)./total_ratio;
d1st_isotropic_diff = cRawData.DBSI_iIsoSpecGrid(d1st_isotropic_grid)*cRawData.DBSI_aIsoSpecdata(d1st_isotropic_grid,:);
d2nd_isotropic_diff = cRawData.DBSI_iIsoSpecGrid(d2nd_isotropic_grid)*cRawData.DBSI_aIsoSpecdata(d2nd_isotropic_grid,:);
d3rd_isotropic_diff = cRawData.DBSI_iIsoSpecGrid(d3rd_isotropic_grid)*cRawData.DBSI_aIsoSpecdata(d3rd_isotropic_grid,:);
d3rd_isotropic_diff = cRawData.DBSI_iIsoSpecGrid(d4th_isotropic_grid)*cRawData.DBSI_aIsoSpecdata(d4th_isotropic_grid,:);

fiber1_axial_map = zeros(map_dimensions);
fiber1_axial_map(roi_index)=cRawData.DBSI_aFiber1Axial;
fiber1_radial_map = zeros(map_dimensions);
fiber1_radial_map(roi_index) = cRawData.DBSI_aFiber1Radial;
fiber1_fa_map = zeros(map_dimensions);
fiber1_fa_map(roi_index) =  cRawData.DBSI_aFiber1FA;
fiber1_fiber_ratio_map = zeros(map_dimensions);
fiber1_fiber_ratio_map(roi_index) =  cRawData.DBSI_aFiber1Ratio;  
fiber1_dirx_map = zeros(map_dimensions);
fiber1_diry_map = zeros(map_dimensions);
fiber1_dirz_map = zeros(map_dimensions);
fiber1_dirx_map(roi_index) =  cRawData.DBSI_aFiber1Dir(1,:); % x direction of the fibler
fiber1_diry_map(roi_index) =  cRawData.DBSI_aFiber1Dir(2,:); % y direction of the fibler
fiber1_dirz_map(roi_index) =  cRawData.DBSI_aFiber1Dir(3,:); % z direction of the fibler
fiber2_axial_map = zeros(map_dimensions);
fiber2_axial_map(roi_index) = cRawData.DBSI_aFiber2Axial;
fiber2_radial_map = zeros(map_dimensions);
fiber2_radial_map(roi_index) = cRawData.DBSI_aFiber2Radial;
fiber2_fa_map = zeros(map_dimensions);
fiber2_fa_map(roi_index) =  cRawData.DBSI_aFiber2FA;
fiber2_fiber_ratio_map = zeros(map_dimensions);
fiber2_fiber_ratio_map(roi_index) =  cRawData.DBSI_aFiber2Ratio;    
fiber2_dirx_map = zeros(map_dimensions);
fiber2_diry_map = zeros(map_dimensions);
fiber2_dirz_map = zeros(map_dimensions);
fiber2_dirx_map(roi_index) =  cRawData.DBSI_aFiber2Dir(1,:); % x direction of the fibler
fiber2_diry_map(roi_index) =  cRawData.DBSI_aFiber2Dir(2,:); % y direction of the fibler
fiber2_dirz_map(roi_index) =  cRawData.DBSI_aFiber2Dir(3,:); % z direction of the fibler
restricted_ratio_1_map = zeros(map_dimensions);
restricted_ratio_1_map(roi_index) =  d1st_isotropic_ratio;
restricted_adc_1_map = zeros(map_dimensions);
restricted_adc_1_map(roi_index) =  d1st_isotropic_diff;
restricted_ratio_2_map = zeros(map_dimensions);
restricted_ratio_2_map(roi_index) =  d2nd_isotropic_ratio;
restricted_adc_2_map = zeros(map_dimensions);
restricted_adc_2_map(roi_index) =  d2nd_isotropic_diff;
hindered_ratio_map = zeros(map_dimensions);
hindered_ratio_map(roi_index) =  d3rd_isotropic_ratio;
hindered_adc_map = zeros(map_dimensions);
hindered_adc_map(roi_index) =  d3rd_isotropic_diff;   
water_ratio_map = zeros(map_dimensions);
water_ratio_map(roi_index) =  d4th_isotropic_ratio;
water_adc_map = zeros(map_dimensions);
water_adc_map(roi_index) =  d4th_isotropic_diff;
model_v_map =  zeros(map_dimensions);
model_v_map(roi_index) =  cRawData.DBSI_iModel;
error_v_map =  zeros(map_dimensions);
error_v_map(roi_index) =  cRawData.DBSI_fError;
dti_axial_map =  zeros(map_dimensions);
dti_axial_map(roi_index) =  cRawData.DTI_aAxial;
dti_radial_map =  zeros(map_dimensions);
dti_radial_map(roi_index) =  cRawData.DTI_aRadial;
dti_adc_map =  zeros(map_dimensions);
dti_adc_map(roi_index) =  cRawData.DTI_aADC;
dti_fa_map =  zeros(map_dimensions);
dti_fa_map(roi_index) =  cRawData.DTI_aFA;
dti_dirx_map =  zeros(map_dimensions);
dti_diry_map =  zeros(map_dimensions);
dti_dirz_map =  zeros(map_dimensions);
dti_dirx_map(roi_index) =  cRawData.DTI_aOrientation(1,:);
dti_diry_map(roi_index) =  cRawData.DTI_aOrientation(2,:);
dti_dirz_map(roi_index) =  cRawData.DTI_aOrientation(3,:);
b0_map =  zeros(map_dimensions);
b0_map(roi_index) =  cRawData.DWI_aB0;

% new parameters
fiber_ratio_map = fiber1_fiber_ratio_map + fiber2_fiber_ratio_map ;

r=uint8(abs(restricted_ratio_map)*255);
g=uint8(abs(hindered_ratio_map)*255);
b=uint8(abs(water_ratio_map)*255);
fraction_rgba_map = permute(cat(4,r,g,b),[1 2 4 3]);

r=uint8(abs(dti_dirx_map.*dti_fa_map)*255);
g=uint8(abs(dti_diry_map.*dti_fa_map)*255);
b=uint8(abs(dti_dirz_map.*dti_fa_map)*255);
dti_r_map = uint8(abs(dti_dirx_map.*dti_fa_map)*255);
dti_g_map = uint8(abs(dti_diry_map.*dti_fa_map)*255);
dti_b_map = uint8(abs(dti_dirz_map.*dti_fa_map)*255);
dti_rgba_map = permute(cat(4,r,g,b),[1 2 4 3]);
dti_rgba_map_itk = permute(dti_rgba_map,[1 2 4 5 3]);

r=uint8(abs(fiber1_dirx_map.*fiber_ratio_map)*255);
g=uint8(abs(fiber1_diry_map.*fiber_ratio_map)*255);
b=uint8(abs(fiber1_dirz_map.*fiber_ratio_map)*255);
fiber1_rgba_map = permute(cat(4,r,g,b),[1 2 4 3]);
fiber1_rgba_map_itk = permute(fiber1_rgba_map,[1 2 4 5 3]);


dbsi_results_dir = ['DBSI_results_',num2str(iso_threshold(1)*1000),'_',num2str(iso_threshold(2)*1000),'_',num2str(iso_threshold(3)*1000),'_',num2str(iso_threshold(4)*1000)];
mkdir([data_dir filesep dbsi_results_dir]);
% save(fullfile(data_dir,'dbsi_maps.mat'),'-v7.3');



for i = 1: size(outlist,2)
    varContent = eval(char(outlist(i))); 
    filename = fullfile(data_dir,dbsi_results_dir,[outlist{i},'.nii']);
    if strcmp(output_format,'ana') && size(varContent,5)<2
        ana = make_ana(varContent);
        save_untouch_nii(ana, filename)
    else
        nii = make_nii(varContent);
        save_nii(nii, filename)
    end
end

%% save fib
if output_fib>0
    dbsi_save_fib;
end

%% end of main function
if nargin == 1
    ini.WriteFile(ini_file);
end
disp('-Generate DBSI Results: Completed!');


%% subfunctions
function dbsi_save_fib
    % don't need to get actual rotation matrix since it's been used for
    % bmatrix and q calculation
%     rotation_matrix = ini.GetValues('INPUT', 'rotation_matrix');
%     if isempty(rotation_matrix)
%         [fn,fdir] = uigetfile(fullfile(data_dir,'*.mat'), 'Pick the rotation matrix');
%         rotation_matrix = str2num(fileread(fullfile(fdir,fn)));
%         rotation_matrix = rotation_matrix(1:3,1:3);
%         ini.SetValues('INPUT','rotation_matrix',fn);
%     elseif strcmp(rotation_matrix,'NA')
%         rotation_matrix = eye(3);
%     else
%         rotation_matrix = fullfile(data_dir,rotation_matrix);
%         rotation_matrix = str2num(fileread(rotation_matrix));
%         rotation_matrix = rotation_matrix(1:3,1:3);
%     end
    
    rotation_matrix = eye(3);
    
    % generate fib results
    fib.dimension = [cRawData.sImageHeader.image_size, cRawData.sImageHeader.ns];
    fib.voxel_size=output_fib_res;

    fib.dir0 = zeros([3 fib.dimension]);
    fib.dir0(:,roi_index) = rotation_matrix*cRawData.DTI_aOrientation; fib.dir0 = reshape(fib.dir0,1,[]);
    fib.b0 = b0_map; fib.b0 = reshape(fib.b0,1,[]);  
    fib.fa0 = dti_fa_map; fib.fa0 = reshape(fib.fa0,1,[]); fib.fa0(isnan(fib.fa0))=0;
    fib.dti_fa = dti_fa_map; fib.dti_fa = reshape(fib.dti_fa,1,[]);  fib.dti_fa(isnan(fib.dti_fa))=0;
    fib.dti_axial = dti_axial_map; fib.dti_axial = reshape(fib.dti_axial,1,[]);
    fib.dti_radial = dti_radial_map; fib.dti_radial = reshape(fib.dti_radial,1,[]);    
    fib.dti_adc = dti_adc_map; fib.dti_adc = reshape(fib.dti_adc,1,[]);  

    % save into FIB file
    save(fullfile(data_dir,dbsi_results_dir,'dti_tracking.fib'),'-struct','fib','-v4');      

    fib.dti_fa = dti_fa_map; fib.dti_fa = reshape(fib.dti_fa,1,[]); fib.dti_fa(isnan(fib.dti_fa))=0;
    fib.dti_axial = dti_axial_map; fib.dti_axial = reshape(fib.dti_axial,1,[]);
    fib.dti_radial = dti_radial_map; fib.dti_radial = reshape(fib.dti_radial,1,[]);   
    fib.dti_adc = dti_adc_map; fib.dti_adc = reshape(fib.dti_adc,1,[]);    

    fib.dir0 = zeros([3 fib.dimension]);
    fib.dir0(:,roi_index) = rotation_matrix*cRawData.DBSI_aFiber1Dir; fib.dir0 = reshape(fib.dir0,1,[]);
    fib.dir1 = zeros([3 fib.dimension]);
    fib.dir1(:,roi_index) = rotation_matrix*cRawData.DBSI_aFiber2Dir; fib.dir1 = reshape(fib.dir1,1,[]);

    fib.fa0 = fiber1_fa_map; fib.fa0 = reshape(fib.fa0,1,[]);
    fib.fa1 = fiber1_fa_map; fib.fa1 = reshape(fib.fa1,1,[]);
    fib.fr0 = fiber1_fiber_ratio_map; fib.fr0 = reshape(fib.fr0,1,[]);
    fib.fr1 = fiber2_fiber_ratio_map; fib.fr1 = reshape(fib.fr1,1,[]);
    fib.fiber1_axial = fiber1_axial_map; fib.fiber1_axial = reshape(fib.fiber1_axial,1,[]);
    fib.fiber1_radial = fiber1_radial_map; fib.fiber1_radial = reshape(fib.fiber1_radial,1,[]);
    fib.restricted_ratio = restricted_ratio_map; fib.restricted_ratio = reshape(fib.restricted_ratio,1,[]);
    fib.hindered_ratio = hindered_ratio_map; fib.hindered_ratio = reshape(fib.hindered_ratio,1,[]);
    fib.water_ratio = water_ratio_map; fib.water_ratio = reshape(fib.water_ratio,1,[]);  
    fib.fiber_ratio = fib.fr0+fib.fr1;

    save(fullfile(data_dir,dbsi_results_dir,'dbsi_tracking.fib'),'-struct','fib','-v4'); 
end % end of subfunction dbsi_save_fib

end % end of main function



