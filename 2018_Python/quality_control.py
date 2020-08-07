def QC(proj_dir, roi_files):
    QC_dir = os.path.join(proj_dir,'QC')
    if not os.path.exists(QC_dir):
        os.makedirs(QC_dir)
        
    for i,roi_file in enumerate(roi_files):
        rois = nib.load(roi_file).get_data()
        roi_path,roi_name = os.path.split(roi_file)
        
        current_dir = roi_path
        for roi_id in range(1,100):
            if not any(rois.ravel() == roi_id):
                continue

            idx = np.asarray(np.where(rois==roi_id))      
            
            temp_atlas = copy.deepcopy(rois)
            temp_atlas = temp_atlas.astype(float)
            temp_atlas[temp_atlas[:]==0]=np.nan
            
            fig = plt.figure(figsize=(24,15))
            ax_1 = fig.add_subplot(2,1,1)
            ax_1.axis('off')
            img_dti_adc = nib.load(glob.glob(os.path.join(current_dir,'dti_adc_map.nii'))[0]).get_data()
            ax_1.imshow(img_dti_adc[:,:,idx[2,0]],cmap='gray',vmin=0.2,vmax=2.0,aspect='equal')
            ax_1.imshow(temp_atlas[:,:,idx[2,0]],cmap='rainbow',alpha=0.8,aspect='equal')
            ax_1.set_title('DTI ADC',fontweight='bold')
            ax_2 = fig.add_subplot(2,2,1)
            ax_2.axis('off')
            ax_2.set_title('DTI FA',fontweight='bold')
            img_dti_fa = nib.load(glob.glob(os.path.join(current_dir,'dti_fa_map.nii'))[0]).get_data()
            ax_2.imshow(img_dti_fa[:,:,idx[2,0]],cmap='gray',vmin=0.2,vmax=0.8,aspect='equal')
            ax_2.imshow(temp_atlas[:,:,idx[2,0]],cmap='rainbow',alpha=0.8,aspect='equal')
            plt.savefig(os.path.join(QC_dir,'qc_%s_%s.png'%(roi_name[:-7],roi_id)),format='png')
            plt.close()   