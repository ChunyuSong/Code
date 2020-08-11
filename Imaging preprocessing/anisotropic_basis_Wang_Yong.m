load dbsi_input %sig b_value q  bmatrix
sig = dbsi_data(1,1,:,1); 
[lambdas2, eigenVectors2,residualImage2] = dtiLeastSquaresW2(sig', bmatrix);
dir = eigenVectors2(:,1)';

ind = find(b_value<15000);
sig = sig(ind);
b_value = b_value(ind);
q = q(ind,:);

sig = sig/max(sig);
axial = 0 : 0.025 : 1.5;
radial = 0 : 0.025 : 1.0;

axial2 = zeros(1,length(axial)*length(radial));
radial2 = zeros(1,length(axial)*length(radial));
for i = 1 : length(axial)
    for j = 1 : length(radial)
        axial2((i-1)*length(radial) + j) = axial(i);
        radial2((i-1)*length(radial) + j) = radial(j);
    end
end
NN = length(axial2);
N  =  length(b_value);
R = zeros(N,NN);
for i =  1 : N,
    trb = b_value(i)/1000;
    for j = 1 : NN,
            cosv = sum(q(i,:).*dir)/norm(q(i,:))/norm(dir);% q and x are unit vectors      
            lamda_axial = axial2(j);
            lamda_perd = radial2(j);
            R(i,j) = exp(-1*trb*lamda_perd) * exp(-3*trb*((lamda_axial+lamda_perd*2)/3-lamda_perd)*cosv^2);
    end%j
end

fFirstStepPanelty = 0.1;
H = fFirstStepPanelty*eye(size(R,2));
d0 = zeros(size(R,2),1);
reso = lsqnonneg([R;H],[sig';d0]);

map = reshape(reso,length(radial),length(axial));
figure;imshow(map',[]);
colormap('jet')
 
 

