function basis_set_new
%% input parameters
axial1area = 0: 0.15: 1.5;
radial1area = 0: 0.15: 1.5;

axial2area = 0:0.05:0.5;
radial2area = 0:0.05:0.5;

axial3area = 0:0.05:0.5;
radial3area = 0:0.05:0.5;

%% generate basis set
axial2 = zeros(1,length(axial1area)*length(radial1area));
radial2 = zeros(1,length(axial1area)*length(radial1area));
for ii = 1 : length(axial1area)
    for jj = 1 : length(radial1area)
        axial2((ii-1)*length(radial1area) + jj) = axial1area(ii);
        radial2((ii-1)*length(radial1area) + jj) = radial1area(jj);
    end
end
index = find((axial2>radial2));
axial3 = axial2(index);
radial3 = radial2(index);

axial4 = zeros(1,length(axial2area)*length(radial2area));
radial4 = zeros(1,length(axial2area)*length(radial2area));
for ii = 1 : length(axial2area)
    for jj = 1 : length(radial2area)
        axial4((ii-1)*length(radial2area) + jj) = axial2area(ii);
        radial4((ii-1)*length(radial2area) + jj) = radial2area(jj);
    end
end
index = find(axial4>radial4);
axial5 = axial4(index);
radial5 = radial4(index);

axial6 = zeros(1,length(axial3area)*length(radial3area));
radial6 = zeros(1,length(axial3area)*length(radial3area));
for ii = 1 : length(axial3area)
    for jj = 1 : length(radial3area)
        axial6((ii-1)*length(radial3area) + jj) = axial3area(ii);
        radial6((ii-1)*length(radial3area) + jj) = radial3area(jj);
    end
end
index = find(axial6>radial6);
axial7 = axial6(index);
radial7 = radial6(index);

fAnisotropicBasisAxial=[axial3 axial5 axial7];
fAnisotropicBasisRadial=[radial3 radial5 radial7];
D = [fAnisotropicBasisAxial;fAnisotropicBasisRadial];
D = D';
D = unique(D,'rows');
D = D';
fAnisotropicBasisAxial = D(1,:);
fAnisotropicBasisRadial = D(2,:);
for xx=1:length(fAnisotropicBasisAxial)
    fa(xx) = sqrt(1/2)*(sqrt((fAnisotropicBasisAxial(xx)- fAnisotropicBasisRadial(xx))^2+(fAnisotropicBasisAxial(xx)- fAnisotropicBasisRadial(xx))^2+(0)^2))/(sqrt( fAnisotropicBasisAxial(xx)^2+ fAnisotropicBasisRadial(xx)^2+ fAnisotropicBasisRadial(xx)^2));
end
index=find(fa>=0.3);
fAnisotropicBasisAxial = fAnisotropicBasisAxial(index);
fAnisotropicBasisRadial = fAnisotropicBasisRadial(index);




%% plot basis set
scatter(fAnisotropicBasisRadial,fAnisotropicBasisAxial,'r*');
axis equal;
xlabel('RD');
ylabel('AD');

%% save to new files
filename = 'diffusivities_basis_set.mat';
save(filename,'fAnisotropicBasisAxial','fAnisotropicBasisRadial');
clear
display('program end')