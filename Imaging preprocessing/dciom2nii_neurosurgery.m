files = ls('IM*');
data = [];
acquisitionnumber=1;
for i=1:size(files,1)
    i
    info = dicominfo(char(files(i,:)));
    slicelocation(i) = info.SliceLocation;
    data(:,:,i) = dicomread(char(files(i,:)));

end
[sorted, index] = sort(slicelocation);
data = data(:,:,index);

nii = make_nii(rot90(data));
save_nii(nii, 'data.nii');
clear all;