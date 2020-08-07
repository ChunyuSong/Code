% function dbsi_read_IMA_dicom
% read DICOM file for DBSI analysis (SIEMENS Scanner)

file = dir ('ZH*');
dbsi_data = [];
slice_position = [];
temp_b_value = [];
b_value = [];
b_vector = [];
dicom_info = [];
slice_counter = 1;
counter = 1;
for i = 1:length(file)      
        dicom_info = dicominfo(file(i).name);
        temp_b_value(i)=dicom_info.Private_0019_100c;
        if i~=1
            if  temp_b_value(i-1)==temp_b_value(i)
                slice_counter = slice_counter + 1;
            else
                if temp_b_value(i)==0
                    b_vector(:,counter+1)=[0;0;0];
                else
                    b_vector(:,counter+1)=dicom_info.Private_0019_100e;
                end
                b_value(counter+1) = temp_b_value(i);
                slice_position(counter)=slice_counter;
                slice_counter = 1;
                counter = counter + 1;
                
            end  
        else
            b_value(i) = temp_b_value(i);
            if temp_b_value(i)==0
                b_vector(:,i)=[0;0;0];
            else
                b_vector(:,i)=dicom_info.Private_0019_100e;
            end
        end

        dbsi_data(:,:,i)= dicomread(file(i).name);

end

% slice_position=int32(slice_position);
% slice_position=unique(slice_position);
% 
% length(file)
% length(slice_position)

% generate nii
% img = permute(reshape(dbsi_data,dicom_info.Rows,dicom_info.Columns,counter, slice_counter),[1,2,4,3]);
img = reshape(dbsi_data,dicom_info.Rows,dicom_info.Columns,slice_counter,counter);

nii = make_nii(rot90(img));
save_nii( nii, fullfile(pwd,'data.nii'));
% 
% % save bvalue and bvector
dlmwrite( fullfile(pwd,'bval'),b_value(1:counter),',');

dlmwrite( fullfile(pwd,'bvec'),b_vector(:,1:counter),',');
% end
