import numpy as np #for checking ROI map for voxels of interest
import nibabel as nib #for extracting nifti image data
import xlsxwriter as xw #for writing to excel
import glob #for retriving file paths
import sys #for reading parameters
import os #for cleaning up directory/file paths
import datetime #for appending date to file name for easier file organization

today = datetime.datetime.now()
#these are the maps we want to get data for
maps_of_interest = ['hindered_fr_map.nii',
					'fiber_fr_map.nii',
					'highly_restricted_fr_map.nii',
					'restricted_fr_map.nii']

#INPUTS: directory of ROIs <PARENT FOLDER>/<PATIENT FOLDER1><PATIENT FOLDER2>...<PATIENT FOLDERN>/<ROI.NII>
#	 	 directory of results <PARENT FOLDER>/<PATIENT FOLDER1><PATIENT FOLDER2>...<PATIENT FOLDERN>/<MAPS OF INTEREST.NII>

roi_directory = sys.argv[1];
results_directory = sys.argv[2];
#dictionary to hold patient name and the coordinates of the patients roi
roi_dict = {}

#loop to get the patients we have rois for
for pr in glob.glob(roi_directory + '/[P|p]atient*/'):

	#strip the patient name from the folder path string and convert to lowercase
	p_name = os.path.basename(os.path.normpath(pr))

	#get the roi file for patient
	roi_file = glob.glob(pr + '/*.nii')

	#get the voxels of our roi
	roi_coords = np.where(nib.load(roi_file[0]).get_data() > 0)

	#array to hold our formatted/organized coordinates
	roi_coords_formatted = []

	#loop through the results of our voxel coordinates and reformat into (x, y, z)
	for coords in range(0, len(roi_coords[0])):
		roi_coords_formatted.append((roi_coords[0][coords], roi_coords[1][coords], roi_coords[2][coords]))
	#end coordinates formatting loop

	#insert our patient name and the roi coordinates for the patient into the dictionary
	roi_dict[p_name] = roi_coords_formatted
#end loop to get patients roi

for patient, coords in roi_dict.iteritems():

	#create text file to store our patient data
	patient_data_file = open(str(today.year) + '-' + str(today.month) + '-' + str(today.day) + '_' + patient + '_data_file', 'w')

	#loop through our maps of interest to get the data from the result maps for this patient
	for maps in maps_of_interest:

		#write map name and directory to help distinguish where data starts and ends
		patient_data_file.write('====' + maps + ': ' + results_directory + '/' + patient + '\n')

		map_data = []
		m = glob.glob(results_directory + '/' + patient + '/' + maps)
		if len(m) == 0:
			map_data.append('no data')
		else:
			m = nib.load(m[0]).get_data()
			for c in coords:
				map_data.append(c)
				map_data.append(m[c[0]][c[1]][c[2]])
		map_data_string = ",".join([str(d) for d in map_data]) #use ast.literal_eval() to read coordinate string back into tuple for easier x,y,z extraction
		patient_data_file.write(map_data_string + '\n')
	patient_data_file.close()
	#end map data extraction loop

#end looping through patients
#end dxgps-read-roi.py
