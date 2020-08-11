from subprocess import call
import os
import matlab.engine
import ConfigParser


def main():
    data_path = '/bmrp092temp/Prostate_Cancer_ex_vivo/WU_Class_Data/'
    files=[]
    eng = matlab.engine.start_matlab()
    eng.addpath('/bmrs090temp/dbsi_release/','/bmrs090temp/dbsi_release/Misc/','/bmrs090temp/dbsi_release/Misc/NIfTI_20140122/')
    for dirName, subdirList, fileList in os.walk(data_path):
        for fname in range(len(fileList)):
            if fileList[fname] == 'DBSIClassData.mat':
                files.append(os.path.join(dirName,fileList[fname]))
    new_data_path = data_path.replace('WU_Class_Data','WU_PCa_ex_vivo')
    os.makedirs(new_data_path)
    for file in range(len(files)):
        folder = os.path.dirname(files[file])
        new_folder = folder.replace('WU_Class_Data','WU_PCa_ex_vivo')
        os.makedirs(new_folder)
        cmd = ['cp','-rf',"%s"%files[file],"%s"%new_folder]
        call(cmd)
        current_dir = new_folder
        os.chdir(current_dir)
        Config = ConfigParser.ConfigParser()
        cfgfile = open("config.ini",'w')
        Config.add_section('INPUT')
        Config.set('INPUT','data_dir',current_dir)
        Config.set('INPUT','dwi_file','NA')
        Config.set('INPUT','mask_file','NA')
        Config.set('INPUT','rotation_matrix','NA')
        Config.set('INPUT','bval_file','NA')
        Config.set('INPUT','bvec_file','NA')
        Config.set('INPUT','preprocess','NA')
        Config.set('INPUT','slices_to_compute',0)
        Config.set('INPUT','dbsi_mode','map')
        Config.set('INPUT','norm_by_bvec','no')
        Config.set('INPUT','bmax_dbsi',' ')
        Config.set('INPUT','dbsi_input_file','NA')
        Config.add_section('DBSI')
        Config.set('DBSI','dbsi_input_file','NA')
        Config.set('DBSI','dbsi_config_file','NA')
        Config.set('DBSI','dbsi_class_file','DBSIClassData.mat')
        Config.add_section('OUTPUT')
        Config.set('OUTPUT','output_option',1)
        Config.set('OUTPUT','output_format','nii')
        Config.set('OUTPUT','iso_threshold','0.1,0.1,0.8,0.8,1.5,1.5')
        Config.set('OUTPUT','output_fib',1)
        Config.set('OUTPUT','output_fib_res','1,1,1')
        Config.write(cfgfile)
        cfgfile.close()
        eng.dbsi_save_4_segments(current_dir + '/' + 'config.ini',nargout=0)
if __name__ == '__main__':
    main()
