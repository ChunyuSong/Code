% function dbsi_read_dicom
% read DICOM file for DBSI analysis (Philips Scanner)

file = dir ('IM*');
dbsi_data = [];
slice_location = [];
b_value = [];
b_vector = [];
dicom_info = [];
for i = 1:length(file)
    dicom_info = dicominfo(file(i).name);
    slice_location(i)=dicom_info.SliceLocation;
    b_value(i)=dicom_info.DiffusionBValue;
    b_vector(:,i)=dicom_info.DiffusionGradientOrientation;
    dbsi_data(:,:,:,i)= dicomread(file(i).name);
end

slice_location=int32(slice_location);
slice_location=unique(slice_location);

length(file)
length(slice_location)

% generate nii
img = permute(reshape(dbsi_data,dicom_info.Rows,dicom_info.Columns,length(file)/length(slice_location),length(slice_location)),[1,2,4,3]);
nii = make_nii(rot90(img));
save_nii( nii, fullfile(pwd,'data.nii'));

% save bvalue and bvector
dlmwrite( fullfile(pwd,'bval'),b_value(1:length(file)/length(slice_location)),',');

dlmwrite( fullfile(pwd,'bvec'),b_vector(:,1:length(file)/length(slice_location)),',');
% end
