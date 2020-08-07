import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from Tkinter import Tk
from tkFileDialog import askopenfilename
import pickle

print 'PO1 generate ROI reports: start'

po1dir = r"\\bmrw207.rad.wustl.edu\nmr35\Peng\PO1"
print "checking folder %s...... " % po1dir
if not os.path.exists(po1dir):
    print "root folder does not existed ", po1dir
    sys.exit("%s folder does not existed!" % (po1dir))

dbsidir = 'DBSI_results_0.3_0.3_3_3'
datadir = 'Final'
dbsifiles = ['dti_fa_map.hdr','dti_adc_map.hdr','dti_axial_map.hdr','dti_radial_map.hdr',
             'fiber_ratio_map.hdr','fiber1_fa_map.hdr','fiber1_axial_map.hdr','fiber1_radial_map.hdr',
             'restricted_ratio_map.hdr','hindered_ratio_map.hdr','water_ratio_map.hdr']
optionalfiles = ['mprage_mni152.hdr', 'flair_mni152.hdr', 't2w_mni152.hdr','mtc_mni152.hdr','t1wpre_mni152.hdr', 't1wpost_mni152.hdr']

new_roi_files = []
#for roi_file in glob.glob(os.path.join(po1dir,'*','DBSI_*[0-9][0-9]','ROI_PO1_*.hdr')):
for roi_file in glob.glob(os.path.join(po1dir,'*','DBSI_*[0-9][0-9]','*.hdr')):
    if not os.path.exists(roi_file.replace('.hdr','.csv')):  #"%s.csv" % (os.path.splitext(roi_file)[0])
        new_roi_files.append(roi_file)
        print "New ROI file %s found"%(roi_file) 
        
for new_roi_file in new_roi_files:
    roi_dir = os.path.dirname(new_roi_file)
    pid = os.path.basename(roi_dir)

    #if 'Volunteer' in roi_dir:
        #structfiles = ['mtc_mni152.hdr']
    #else:
        #structfiles = ['t1wpre_mni152.hdr'] #['t2w_mni152.hdr','mtc_mni152.hdr','t1wpre_mni152.hdr','t1wpost_mni152.hdr']

    #completesets = []
    #data_sets = glob.glob(os.path.join(roi_dir,'[0-9][0-9]_[0-9][0-9][0-9][0-9]',datadir,dbsidir))
    #for data_set in data_sets:
        #sid = data_set.split('\\')[-3]
        #if all([glob.glob(os.path.join(data_set, "%s_%s_%s"%(pid,sid,map_file))) for map_file in dbsifiles]):
            #if all([glob.glob(os.path.join(os.path.dirname(data_set), "PO1_%s_%s_%s"%(pid,sid,map_file))) for map_file in structfiles]):
                #print data_set+': data is complete'
                #completesets.append(data_set)
            #else:
                #print data_set+': data is not complete'  
    #completesets.sort()
    completesets = []
    data_sets = glob.glob(os.path.join(roi_dir,'[0-9][0-9]_[0-9][0-9][0-9][0-9]',datadir,dbsidir))
    for data_set in data_sets:
        sid = data_set.split('\\')[-3]
        if all([glob.glob(os.path.join(data_set, "%s_%s_%s"%(pid,sid,map_file))) for map_file in dbsifiles]):
            print data_set+': data is complete'
            completesets.append(data_set)
        else:
            print data_set+': data is not complete'  
    completesets.sort()
    
    mask = nib.analyze.load(new_roi_file).get_data()
    mastercsvfile = os.path.join(roi_dir,"%s.csv" % (os.path.splitext(new_roi_file)[0])) 
  
    visit=0
    for data_set in completesets:
        sid = data_set.split('\\')[-3]
        for i in range(1,100):    
            inx = np.where(mask==i) 
            if len(inx[0])>0:
                data = np.zeros( (len(inx[0]),len(dbsifiles)) )
                df = pd.DataFrame()
                for dbsi_file in dbsifiles:
                    dbsimap = nib.analyze.load(os.path.join(data_set, "%s_%s_%s" % (pid,sid,dbsi_file))).get_data()
                    if len(dbsimap.shape)>3:
                        df[dbsi_file] = dbsimap[:,:,:,0][inx]
                    else:
                        df[dbsi_file] = dbsimap[inx]
                
                # mandatory structural files
                #for struct_file in structfiles:
                    #structmap = nib.analyze.load(os.path.join(os.path.dirname(data_set), "PO1_%s_%s_%s" % (pid,sid,struct_file))).get_data()
                    #df.insert(0,struct_file,structmap[inx])
                
                for optional_file in optionalfiles:
                    optional_file_name = os.path.join(os.path.dirname(data_set), "PO1_%s_%s_%s" % (pid,sid,optional_file))
                    if os.path.isfile(optional_file_name):
                        structmap = nib.analyze.load(optional_file_name).get_data()
                        df.insert(0,optional_file,structmap[inx]) 
                    else:
                        df.insert(0,optional_file,'NA') 
                    
                    
                # optional structural files    
                #for struct_file in structfiles:
                    #struct_file_fullname = os.path.join(os.path.dirname(data_set), "PO1_%s_%s_%s" % (pid,sid,struct_file))
                    #if os.path.isfile(struct_file_fullname):
                        #structmap = nib.analyze.load(struct_file_fullname).get_data()
                        #df.insert(0,struct_file,structmap[inx])
                    #else:
                        #df.insert(0,struct_file,np.nan)                
                
                #for fileid,dbsifile in enumerate(dbsifiles):
                    #dbsimap = nib.analyze.load(os.path.join(data_set, "%s_%s_%s" % (pid,sid,dbsifile))).get_data()
                    #data[:,fileid] = dbsimap[:,:,:,0][inx]

                #df = pd.DataFrame(data,columns=dbsifiles)

                #for fileid,structfile in enumerate(structfiles):
                    #filename = glob.glob(os.path.join(data_set, "*_%s"% structfile))[0]
                    #structmap = nib.analyze.load(filename).get_data()
                    #df.insert(0,structfile,structmap[inx])
                    
                #if not 'mprage_mni152.hdr' in df.columns:
                    #df.insert(0,'mprage_mni152.hdr','NA')  
                                    
                #if not 't2w_mni152.hdr' in df.columns:
                    #df.insert(0,'t2w_mni152.hdr','NA')  
                    
                #if not 'mtc_mni152.hdr' in df.columns:
                    #df.insert(0,'mtc_mni152.hdr','NA')
                
                t1wpre_file_name = os.path.join(os.path.dirname(data_set), "PO1_%s_%s_%s" % (pid,sid,'t1wpre_mni152.hdr'))
                t1wpost_file_name = os.path.join(os.path.dirname(data_set), "PO1_%s_%s_%s" % (pid,sid,'t1wpost_mni152.hdr'))                
                if os.path.isfile(t1wpre_file_name) and os.path.isfile(t1wpost_file_name):
                    df.insert(0,'t1w_diffratio',(df['t1wpost_mni152.hdr']-df['t1wpre_mni152.hdr'])/df['t1wpre_mni152.hdr'])
                    df = df.drop('t1wpost_mni152.hdr',1) 
                    df = df.drop('t1wpre_mni152.hdr',1)
                else:
                    df.insert(0,'t1w_diffratio','NA')
                    if 't1wpre_mni152.hdr' in df.columns:
                        df = df.drop('t1wpre_mni152.hdr',1)
                    if 't1wpost_mni152.hdr' in df.columns:
                        df = df.drop('t1wpost_mni152.hdr',1)                    
                                           
                df.insert(0,'Z',inx[2])
                df.insert(0,'Y',inx[1])
                df.insert(0,'X',inx[0])
                df.insert(0,'VOXEL ID',np.arange(len(inx[0]))+1)
                df.insert(0,'ROI ID',i)
                df.insert(0,'SCAN NUM',visit+1)
                scandate = '%s/%s/20%s' %(sid[3:5],sid[5:7],sid[0:2])
                df.insert(0,'SCAN DATE',scandate) 
                df.insert(0,'PATIENT ID',pid.lstrip("DBSI_ELPVelpv")) 
                df.insert(0,'GROUP',pid.strip("DBSI_0123456789"))
                
                # set NA to fiber axial/radial/fa if fiber ratio is 0
                df['fiber1_axial_map.hdr'][df['fiber_ratio_map.hdr'] < 0.01]='NA'
                df['fiber1_radial_map.hdr'][df['fiber_ratio_map.hdr'] < 0.01]='NA'
                df['fiber1_fa_map.hdr'][df['fiber_ratio_map.hdr'] < 0.01]='NA'
                    
                if visit+i<2:
                    df.to_csv(mastercsvfile,mode='a',header=True, index=False)
                else:
                    df.to_csv(mastercsvfile,mode='a',header=False, index=False)
        visit +=1
    
print 'PO1 generate ROI reports: end'