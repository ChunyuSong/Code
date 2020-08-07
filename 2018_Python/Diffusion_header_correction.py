import tkFileDialog
import Tkinter
import os
import nibabel as nib
import ntpath

def main():
    root = Tkinter.Tk()
    dbsi_path = tkFileDialog.askdirectory(title="Directory of dbsi data")
    dbsi_folder = os.path.basename(dbsi_path)
    files = tkFileDialog.askopenfilenames(parent=root,title="Select all diffusion parameter maps",initialdir=dbsi_path)
    main_folder = os.path.dirname(dbsi_path)
    if not os.path.isdir(os.path.join(main_folder,'%s_header_corrected'%dbsi_folder)):
        os.makedirs(os.path.join(main_folder,'%s_header_corrected'%dbsi_folder))
    save_path = os.path.join(main_folder,'%s_header_corrected'%dbsi_folder)
    aff_file = tkFileDialog.askopenfilenames(parent=root,title="Select Diffusion Weighted Image",initialdir=dbsi_path)
    affine = nib.load(aff_file[0]).get_affine()
    for file in range(len(files)):
        pathname, filename = ntpath.split(files[file])
        img = nib.load(files[file]).get_data()
        img_save = nib.Nifti1Image(img,affine)
        img = img_save.get_data()
        img_save = nib.Nifti1Image(img,affine)
        nib.save(img_save, os.path.join(save_path + '/' + filename))

if __name__ == '__main__':
    main()
