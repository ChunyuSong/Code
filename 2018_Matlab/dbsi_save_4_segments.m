(* ::Package:: *)

function dbsi_save(ini_file)
% RawOutputFileName is the filename of the output class 'classOutput'
% after load the raw output file, class named as 'cRawData'
% function of output module to generate FDF maps
% 
% 2015-01-20: create nifti from image data, independent from the source 
% file format.
% format option: 'nii' or 'ana'


if nargin < 1
    if exist('DBSIClassData.mat','file')
        dbsi_class _file = 'DBSIClassData.mat';
        data_dir = pwd;
    else
        [dbsi_class _file,data_dir,filter_index] = uigetfile(pwd,'Please select the DBSIClassData mat file');
        dbsi_class _file = fullfile(data_dir, dbsi_class _file);
    end
    load(dbsi_class _file); 
    if ~exist('cRawData')
        disp('this is not a proper DBSI output mat file, program exit!')
        return;
    end
    
    prompt = {'restricted_ratio _ 1 threshold(range from 0 to ):','restricted_ratio _ 2 threshold(Range from ):','restricted_ratio _ 2 threshold(Range to ):','hindered_ratio threshold(Range from ):','hindered_ratio threshold(Range to ):','CSF threshold(Range to infinity):'};
    dlg_title = 'Input thresholds to differentiate isotropic components';
    num_lines = 1;
    def = {'0.1','0.1','0.8','0.8','1.5','1.5'};
    answer = inputdlg(prompt,dlg_title,num_lines,def);
    iso_threshold = [str2num(answer{1}) str2num(answer{2}) str2num(answer{3}) str2num(answer{4}) str2num(answer{5}) str2num(answer{6})]; 
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
        output_fib _res = [str2num(answer{1}) str2num(answer{2}) str2num(answer{3})]; 
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
    dbsi_class _file = ini.GetValues('DBSI', 'dbsi_class _file');
    output_option = ini.GetValues('OUTPUT', 'output_option');
    output_format = ini.GetValues('OUTPUT', 'output_format');
    iso_threshold = ini.GetValues('OUTPUT', 'iso_threshold');
    if isempty(iso_threshold)
        prompt = {'restricted_ratio _ 1 threshold(range from 0 to ):','restricted_ratio _ 2 threshold(Range from ):','restricted_ratio _ 2 threshold(Range to ):','hindered_ratio threshold(Range from ):','hindered_ratio threshold(Range to ):','CSF threshold(Range to infinity):'};
        dlg_title = 'Input thresholds to differentiate isotropic components';
        num_lines = 1;
          def = {'0.1','0.1','0.8','0.8','1.5','1.5'};
        answer = inputdlg(prompt,dlg_title,num_lines,def);
        iso_threshold = [str2num(answer{1}) str2num(answer{2}) str2num(answer{3}) str2num(answer{4}) str2num(answer{5}) str2num(answer{6})]; 
        ini.SetValues('OUTPUT','iso_threshold',{iso_threshold});
    end
    iso_threshold = 10^-3* iso_threshold;% Adjust unit
    output_fib = ini.GetValues('OUTPUT', 'output_fib');
    output_fib _res = ini.GetValues('OUTPUT','output_fib _res');
    if isempty(output_fib _res)
        output_fib _res = [1 1 1];
    end
    
    dbsi_class _file _full = fullfile(data_dir,dbsi_class _file);
    if ~exist(dbsi_class _file _full,'file')
        disp('can''t find DBSI output mat file in data_dir, tried search current folder');
        dbsi_class _file _full = fullfile(pwd,dbsi_class _file);
        if ~exist(dbsi_class _file _full,'file')
            disp('can''t not find DBSI output mat file in data_dir and local folder, program exit!')
            return;
        end        
    end
    load(dbsi_class _file _full); 
    if ~exist('cRawData','var')
        disp('this is not a proper DBSI output mat file, program exit!')
        return;
    end
end

copyfile(which('versions.txt'),fullfile(data_dir,'save.version'));

outlist_clean = {'b0_map',...
    'dti_axial _map',...
    'dti_radial _map',...
    'dti_adc _map',...
    'dti_fa _map',...
    'fiber_ratio _map',...
    'fiber1_axial _map',...
    'fiber1_radial _map',...
    'fiber1_fa _map',...
    'restricted_ratio _ 1_map',...
    'restricted_ratio _ 2_map',...
    'hindered_ratio _map',...
    'water_ratio _map',...
    'iso_adc _map',...
    };

outlist_full = {'b0_map',...
    'model_v _map',...
    'dti_axial _map',...
    'dti_radial _map',...
    'dti_adc _map',...
    'dti_fa _map',...
    'dti_dirx _map',...
    'dti_diry _map',...
    'dti_dirz _map',...
    'dti_rgba _map',...
    'dti_rgba _map _itk',...
    'dti_r _map',...
    'dti_g _map',...
    'dti_b _map',...
    'fiber_ratio _map',...
    'fiber1_axial _map',...
    'fiber1_radial _map',...
    'fiber1_fa _map',...
    'fiber1_fiber _ratio _map',...
    'fiber1_rgba _map',...
    'fiber1_dirx _map',...
    'fiber1_diry _map',...
    'fiber1_dirz _map',...
    'fiber2_axial _map',...
    'fiber2_radial _map',...
    'fiber2_fa _map',...
    'fiber2_fiber _ratio _map',...
    'fiber2_dirx _map',...
    'fiber2_diry _map',...
    'fiber2_dirz _map',...
    'restricted_ratio _ 1_map',...
    'restricted_adc _ 1_map',...
    'restricted_ratio _ 2_map',...
    'restricted_adc _ 2_map',...
    'hindered_ratio _map',...
    'hindered_adc _map',...
    'water_ratio _map',...
    'water_adc _map',...
    'iso_adc _map',...
    'fraction_rgba _map',...
    'fiber1_rgba _map _itk',...
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

d1st_isotropic _grid = cRawData.DBSI_iIsoSpecGrid < iso_threshold(1);
d2nd_isotropic _grid = cRawData.DBSI_iIsoSpecGrid < iso_threshold(3) & cRawData.DBSI_iIsoSpecGrid > iso_threshold(2) ;
d3rd_isotropic _grid = cRawData.DBSI_iIsoSpecGrid < iso_threshold(5) & cRawData.DBSI_iIsoSpecGrid > iso_threshold(4) ;
d4th_isotropic _grid = cRawData.DBSI_iIsoSpecGrid > iso_threshold(6);
d1st_isotropic _ratio = sum (cRawData.DBSI_aIsoSpecdata(d1st_isotropic _grid,:),1)./total_ratio;
d2nd_isotropic _ratio = sum (cRawData.DBSI_aIsoSpecdata(d2nd_isotropic _grid,:),1)./total_ratio;
d3rd_isotropic _ratio = sum (cRawData.DBSI_aIsoSpecdata(d3rd_isotropic _grid,:),1)./total_ratio;
d4th_isotropic _ratio = sum (cRawData.DBSI_aIsoSpecdata(d4th_isotropic _grid,:),1)./total_ratio;
d1st_isotropic _diff = cRawData.DBSI_iIsoSpecGrid(d1st_isotropic _grid)*cRawData.DBSI_aIsoSpecdata(d1st_isotropic _grid,:)./sum(cRawData.DBSI_aIsoSpecdata(d1st_isotropic _grid,:),1)*1000.0;
d2nd_isotropic _diff = cRawData.DBSI_iIsoSpecGrid(d2nd_isotropic _grid)*cRawData.DBSI_aIsoSpecdata(d2nd_isotropic _grid,:)./sum(cRawData.DBSI_aIsoSpecdata(d2nd_isotropic _grid,:),1)*1000.0;
d3rd_isotropic _diff = cRawData.DBSI_iIsoSpecGrid(d3rd_isotropic _grid)*cRawData.DBSI_aIsoSpecdata(d3rd_isotropic _grid,:)./sum(cRawData.DBSI_aIsoSpecdata(d3rd_isotropic _grid,:),1)*1000.0;
d4th_isotropic _diff = cRawData.DBSI_iIsoSpecGrid(d4th_isotropic _grid)*cRawData.DBSI_aIsoSpecdata(d4th_isotropic _grid,:)./sum(cRawData.DBSI_aIsoSpecdata(d4th_isotropic _grid,:),1)*1000.0;

isotropic_grid = cRawData.DBSI_iIsoSpecGrid < iso_threshold(6);
isotropic_diff = cRawData.DBSI_iIsoSpecGrid(isotropic_grid)*cRawData.DBSI_aIsoSpecdata(isotropic_grid,:)./sum(cRawData.DBSI_aIsoSpecdata(isotropic_grid,:),1)*1000.0;

fiber1_axial _map = zeros(map_dimensions);
fiber1_axial _map(roi_index)=cRawData.DBSI_aFiber1Axial;
fiber1_radial _map = zeros(map_dimensions);
fiber1_radial _map(roi_index) = cRawData.DBSI_aFiber1Radial;
fiber1_fa _map = zeros(map_dimensions);
fiber1_fa _map(roi_index) =  cRawData.DBSI_aFiber1FA;
fiber1_fiber _ratio _map = zeros(map_dimensions);
fiber1_fiber _ratio _map(roi_index) =  cRawData.DBSI_aFiber1Ratio;  
fiber1_dirx _map = zeros(map_dimensions);
fiber1_diry _map = zeros(map_dimensions);
fiber1_dirz _map = zeros(map_dimensions);
fiber1_dirx _map(roi_index) =  cRawData.DBSI_aFiber1Dir(1,:); % x direction of the fibler
fiber1_diry _map(roi_index) =  cRawData.DBSI_aFiber1Dir(2,:); % y direction of the fibler
fiber1_dirz _map(roi_index) =  cRawData.DBSI_aFiber1Dir(3,:); % z direction of the fibler
fiber2_axial _map = zeros(map_dimensions);
fiber2_axial _map(roi_index) = cRawData.DBSI_aFiber2Axial;
fiber2_radial _map = zeros(map_dimensions);
fiber2_radial _map(roi_index) = cRawData.DBSI_aFiber2Radial;
fiber2_fa _map = zeros(map_dimensions);
fiber2_fa _map(roi_index) =  cRawData.DBSI_aFiber2FA;
fiber2_fiber _ratio _map = zeros(map_dimensions);
fiber2_fiber _ratio _map(roi_index) =  cRawData.DBSI_aFiber2Ratio;    
fiber2_dirx _map = zeros(map_dimensions);
fiber2_diry _map = zeros(map_dimensions);
fiber2_dirz _map = zeros(map_dimensions);
fiber2_dirx _map(roi_index) =  cRawData.DBSI_aFiber2Dir(1,:); % x direction of the fibler
fiber2_diry _map(roi_index) =  cRawData.DBSI_aFiber2Dir(2,:); % y direction of the fibler
fiber2_dirz _map(roi_index) =  cRawData.DBSI_aFiber2Dir(3,:); % z direction of the fibler
restricted_ratio _ 1_map = zeros(map_dimensions);
restricted_ratio _ 1_map(roi_index) =  d1st_isotropic _ratio;
restricted_adc _ 1_map = zeros(map_dimensions);
restricted_adc _ 1_map(roi_index) =  d1st_isotropic _diff;
restricted_ratio _ 2_map = zeros(map_dimensions);
restricted_ratio _ 2_map(roi_index) =  d2nd_isotropic _ratio;
restricted_adc _ 2_map = zeros(map_dimensions);
restricted_adc _ 2_map(roi_index) =  d2nd_isotropic _diff;
hindered_ratio _map = zeros(map_dimensions);
hindered_ratio _map(roi_index) =  d3rd_isotropic _ratio;
hindered_adc _map = zeros(map_dimensions);
hindered_adc _map(roi_index) =  d3rd_isotropic _diff;   
water_ratio _map = zeros(map_dimensions);
water_ratio _map(roi_index) =  d4th_isotropic _ratio;
water_adc _map = zeros(map_dimensions);
water_adc _map(roi_index) =  d4th_isotropic _diff;
iso_adc _map = zeros(map_dimensions);
iso_adc _map(roi_index) = isotropic_diff;
model_v _map =  zeros(map_dimensions);
model_v _map(roi_index) =  cRawData.DBSI_iModel;
error_v _map =  zeros(map_dimensions);
error_v _map(roi_index) =  cRawData.DBSI_fError;
dti_axial _map =  zeros(map_dimensions);
dti_axial _map(roi_index) =  cRawData.DTI_aAxial;
dti_radial _map =  zeros(map_dimensions);
dti_radial _map(roi_index) =  cRawData.DTI_aRadial;
dti_adc _map =  zeros(map_dimensions);
dti_adc _map(roi_index) =  cRawData.DTI_aADC;
dti_fa _map =  zeros(map_dimensions);
dti_fa _map(roi_index) =  cRawData.DTI_aFA;
dti_dirx _map =  zeros(map_dimensions);
dti_diry _map =  zeros(map_dimensions);
dti_dirz _map =  zeros(map_dimensions);
dti_dirx _map(roi_index) =  cRawData.DTI_aOrientation(1,:);
dti_diry _map(roi_index) =  cRawData.DTI_aOrientation(2,:);
dti_dirz _map(roi_index) =  cRawData.DTI_aOrientation(3,:);
b0_map =  zeros(map_dimensions);
b0_map(roi_index) =  cRawData.DWI_aB0;

% new parameters
fiber_ratio _map = fiber1_fiber _ratio _map + fiber2_fiber _ratio _map ;

restricted_ratio _map = restricted_ratio _ 1_map + restricted_ratio _ 2_map ;
r=uint8(abs(restricted_ratio _map)*255);
g=uint8(abs(hindered_ratio _map)*255);
b=uint8(abs(water_ratio _map)*255);
fraction_rgba _map = permute(cat(4,r,g,b),[1 2 4 3]);

r=uint8(abs(dti_dirx _map.*dti_fa _map)*255);
g=uint8(abs(dti_diry _map.*dti_fa _map)*255);
b=uint8(abs(dti_dirz _map.*dti_fa _map)*255);
dti_r _map = uint8(abs(dti_dirx _map.*dti_fa _map)*255);
dti_g _map = uint8(abs(dti_diry _map.*dti_fa _map)*255);
dti_b _map = uint8(abs(dti_dirz _map.*dti_fa _map)*255);
dti_rgba _map = permute(cat(4,r,g,b),[1 2 4 3]);
dti_rgba _map _itk = permute(dti_rgba _map,[1 2 4 5 3]);

r=uint8(abs(fiber1_dirx _map.*fiber_ratio _map)*255);
g=uint8(abs(fiber1_diry _map.*fiber_ratio _map)*255);
b=uint8(abs(fiber1_dirz _map.*fiber_ratio _map)*255);
fiber1_rgba _map = permute(cat(4,r,g,b),[1 2 4 3]);
fiber1_rgba _map _itk = permute(fiber1_rgba _map,[1 2 4 5 3]);


dbsi_results _dir = ['DBSI_results _',num2str(iso_threshold(1)*1000),'_',num2str(iso_threshold(2)*1000),'_',num2str(iso_threshold(3)*1000),'_',num2str(iso_threshold(4)*1000),'_',num2str(iso_threshold(5)*1000),'_',num2str(iso_threshold(6)*1000)];
mkdir([data_dir filesep dbsi_results _dir]);
% save(fullfile(data_dir,'dbsi_maps.mat'),'-v7 .3');



for i = 1: size(outlist,2)
    varContent = eval(char(outlist(i))); 
    filename = fullfile(data_dir,dbsi_results _dir,[outlist{i},'.nii']);
    if strcmp(output_format,'ana') && size(varContent,5)<2
        ana = make_ana(varContent);
        save_untouch _nii(ana, filename)
    else
        nii = make_nii(varContent);
        save_nii(nii, filename)
    end
end

%% save fib
if output_fib>0
    dbsi_save _fib;
end

%% end of main function
if nargin == 1
    ini.WriteFile(ini_file);
end
disp('-Generate DBSI Results: Completed!');


%% subfunctions
function dbsi_save _fib
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
    fib.voxel_size=output_fib _res;

    fib.dir0 = zeros([3 fib.dimension]);
    fib.dir0(:,roi_index) = rotation_matrix*cRawData.DTI_aOrientation; fib.dir0 = reshape(fib.dir0,1,[]);
    fib.b0 = b0_map; fib.b0 = reshape(fib.b0,1,[]);  
    fib.fa0 = dti_fa _map; fib.fa0 = reshape(fib.fa0,1,[]); fib.fa0(isnan(fib.fa0))=0;
    fib.dti_fa = dti_fa _map; fib.dti_fa = reshape(fib.dti_fa,1,[]);  fib.dti_fa(isnan(fib.dti_fa))=0;
    fib.dti_axial = dti_axial _map; fib.dti_axial = reshape(fib.dti_axial,1,[]);
    fib.dti_radial = dti_radial _map; fib.dti_radial = reshape(fib.dti_radial,1,[]);    
    fib.dti_adc = dti_adc _map; fib.dti_adc = reshape(fib.dti_adc,1,[]);  

    % save into FIB file
    save(fullfile(data_dir,dbsi_results _dir,'dti_tracking.fib'),'-struct','fib','-v4');      

    fib.dti_fa = dti_fa _map; fib.dti_fa = reshape(fib.dti_fa,1,[]); fib.dti_fa(isnan(fib.dti_fa))=0;
    fib.dti_axial = dti_axial _map; fib.dti_axial = reshape(fib.dti_axial,1,[]);
    fib.dti_radial = dti_radial _map; fib.dti_radial = reshape(fib.dti_radial,1,[]);   
    fib.dti_adc = dti_adc _map; fib.dti_adc = reshape(fib.dti_adc,1,[]);    

    fib.dir0 = zeros([3 fib.dimension]);
    fib.dir0(:,roi_index) = rotation_matrix*cRawData.DBSI_aFiber1Dir; fib.dir0 = reshape(fib.dir0,1,[]);
    fib.dir1 = zeros([3 fib.dimension]);
    fib.dir1(:,roi_index) = rotation_matrix*cRawData.DBSI_aFiber2Dir; fib.dir1 = reshape(fib.dir1,1,[]);

    fib.fa0 = fiber1_fa _map; fib.fa0 = reshape(fib.fa0,1,[]);
    fib.fa1 = fiber1_fa _map; fib.fa1 = reshape(fib.fa1,1,[]);
    fib.fr0 = fiber1_fiber _ratio _map; fib.fr0 = reshape(fib.fr0,1,[]);
    fib.fr1 = fiber2_fiber _ratio _map; fib.fr1 = reshape(fib.fr1,1,[]);
    fib.fiber1_axial = fiber1_axial _map; fib.fiber1_axial = reshape(fib.fiber1_axial,1,[]);
    fib.fiber1_radial = fiber1_radial _map; fib.fiber1_radial = reshape(fib.fiber1_radial,1,[]);
    fib.restricted_ratio_ 1 = restricted_ratio _ 1_map; fib.restricted_ratio_ 1 = reshape(fib.restricted_ratio_ 1,1,[]);
    fib.restricted_ratio_ 2 = restricted_ratio _ 2_map; fib.restricted_ratio_ 2 = reshape(fib.restricted_ratio_ 2,1,[]);
    fib.hindered_ratio = hindered_ratio _map; fib.hindered_ratio = reshape(fib.hindered_ratio,1,[]);
    fib.water_ratio = water_ratio _map; fib.water_ratio = reshape(fib.water_ratio,1,[]);  
    fib.fiber_ratio = fib.fr0+fib.fr1;

    save(fullfile(data_dir,dbsi_results _dir,'dbsi_tracking.fib'),'-struct','fib','-v4'); 
end % end of subfunction dbsi_save _fib

end % end of main function



