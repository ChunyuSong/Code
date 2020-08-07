import numpy as np #for checking ROI map for voxels of interest and data array
import pandas as pd #for creating dataframe to easily dump data to excel
import nibabel as nib #for extracting nifti image data
import matplotlib as mpl #for plotting data
mpl.use('TkAgg') #make matplotlib compatible with mac
from matplotlib import pyplot as plt #for plotting data
from sklearn import datasets as ds, metrics as mt, model_selection as ms, svm#for training SVM to fit data
# from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
import xlsxwriter as xw #for writing to excel
import glob #for retriving file paths
import sys #for reading parameters
import os #for cleaning up directory/file paths
import datetime #for appending date to file name for easier file organization
import re #for parsing patient info from filename using regex
import time #for timing tasks


#complete list of result maps
result_maps_list = ['b0_map.nii',#0
			   'dti_adc_map.nii',#1
			   'dti_axial_map.nii',#2
			   'dti_b_map.nii',#3
			   'dti_dirx_map.nii',#4
			   'dti_diry_map.nii',#5
			   'dti_dirz_map.nii',#6
			   'dti_fa_map.nii',#7
			   'dti_g_map.nii',#8
			   'dti_radial_map.nii',#9
			   'dti_rgba_map.nii',#10
			   'dti_rgba_map_itk.nii',#11
			   'dti_r_map.nii',#12
			   'fiber1_axial_map.nii',#13
			   'fiber1_dirx_map.nii',#14
			   'fiber1_diry_map.nii',#15
			   'fiber1_dirz_map.nii',#16
			   'fiber1_fa_map.nii',#17
			   'fiber1_fiber_ratio_map.nii',#18 
			   'fiber1_radial_map.nii',#19
			   'fiber1_rgba_map.nii',#20
			   'fiber1_rgba_map_itk.nii',#21
			   'fiber2_axial_map.nii',#22
			   'fiber2_dirx_map.nii',#23
			   'fiber2_diry_map.nii',#24
			   'fiber2_dirz_map.nii',#25
			   'fiber2_fa_map.nii',#26
			   'fiber2_fiber_ratio_map.nii',#27
			   'fiber2_radial_map.nii',#28
			   'fiber_ratio_map.nii',#29
			   'fraction_rgba_map.nii',#30
			   'hindered_adc_map.nii',#31 DONT USE
			   'hindered_ratio_map.nii',#32
			   'iso_adc_map.nii',#33 DONT USE
			   'model_v_map.nii',#34
			   'restricted_adc_1_map.nii',#35 DONT USE
			   'restricted_adc_2_map.nii',#36 DONT USE
			   'restricted_ratio_1_map.nii',#37
			   'restricted_ratio_2_map.nii',#38
			   'water_adc_map.nii',#39 DONT USE
			   'water_ratio_map.nii'#40
			   ]
#DTI_maps = ['b0_map.nii','dti_axial_map.nii','dti_radial_map.nii','dti_adc_map.nii','dti_fa_map.nii']
#DHisto_1_maps = ['fiber1_axial_map.nii','fiber1_radial_map.nii','fiber1_fa_map.nii','fiber1_fiber_ratio_map.nii',
#				  'fiber2_axial_map.nii','fiber2_radial_map.nii','fiber2_fa_map.nii','fiber2_fiber_ratio_map.nii','restricted_ratio_1_map.nii',
#				  'restricted_ratio_2_map.nii','water_ratio_map.nii','hindered_ratio_map.nii','iso_adc_map.nii']
#DHisto_2_maps = ['fiber1_axial_map.nii','fiber1_radial_map.nii','fiber1_fiber_ratio_map.nii','restricted_ratio_1_map.nii',
#				 'restricted_ratio_2_map.nii','water_ratio_map.nii','hindered_ratio_map.nii','iso_adc_map.nii']
#COMBINATIONS:
#1 DTI_maps
#2 DHisto_1_maps
#3 DHisto_2_maps
#4 DTI_maps + DHisto_1_maps
#TODO add new parameter combination
#set which combination we want for easier referencing below
maps_of_interest = [0,1,2,7,9,13,17,18,19,22,26,27,28,29,32,37,38,40]

#dictionary for easy referencing and labeling based on Gleason score
grade_dict = {'0&0': 0,
			  '3&3': 1,
			  '3&4': 2,
			  '3&5': 5,
			  '4&3': 3,
			  '4&4': 4,
			  '4&5': 6,
			  '5&3': 0,
			  '5&4': 7,
			  '5&5': 8,
			  '6&0': 0}

#tag for if we are including cancer grade in our training
#False : train benign vs pca first then take pca group and train on grading
#True : train benign vs pca grades
direct_grading = False
#if additional arg is passed, we want to train on pca grade as well
if len(sys.argv) == 6:
	#set our tag to signify we are doing pca grading training
	direct_grading = True
#end if statement to check for extra argument

class Patient:
	def __init__(self, n, no, t, g, nccn, r, d):
		self.name = n
		self.number = no
		self.type = t
		self.gleason = g
		self.nccn = nccn
		self.roi = r
		self.result_maps = d
#end Patient class definition

def coordinates_formatter(roi):
	# print('coordinates_formatter')
	"""function to format roi coordsinates into tuple format (x, y, z)
	params:
		roiDir: string representing directory containing our ROIs
	return:
	    roi_dict: dictionary of rois and their respective formatted coordinates [(x1,y1,z1),(x2,y2,z2),...,(xN,yN,zN)]
	"""
	roi_dict = {}
	coords_formatted = []
	coords = np.where(nib.load(roi).get_data() > 0)
	for c in range(0, len(coords[0])):
		coords_formatted.append((coords[0][c], coords[1][c], coords[2][c]))
	roi_dict[roi] = coords_formatted
	return roi_dict
#end coordinates_formatter

def prep_patient_info(roiDir, dataDir, patientType):
	# print('prep_patient_info')
	""" 
		function to create patient objects for each patient
	    that we find an ROI for
	    params:
	    	roiDir: directory string defining location of patient ROIs
	    	dataDir: directory string defining location of patient data
	    	patientType: string defining benign vs pca
	    return:
	    	patients: array of patient objects
	"""
	patients = []
	name_prefix = 'p' if patientType == 'benign' else 'pca_'
	patient_rois = glob.glob(roiDir + '/*.nii')
	patient_names = [name_prefix + str(os.path.basename(os.path.normpath(p)).split('_')[0]) for p in glob.glob(dataDir + '/*/')]

	for roi in patient_rois:
		roi_name = os.path.splitext(os.path.basename(os.path.normpath(roi)))[0]
		roi_name_parts = roi_name.split('_')
		roi_patient_name = roi_name_parts[0] if roi_name_parts[0].lower() != 'pca' else 'pca_' + str(roi_name_parts[1])
		patient_number = re.sub('\D+', '', roi_name_parts[0]) if patientType == 'benign' else roi_name_parts[1]
		gleason = 'N/A' if patientType == 'benign' else str(roi_name_parts[2])
		nccn = -1 if patientType == 'benign' else roi_name_parts[3]
		if roi_patient_name in patient_names:
			roi_dict = coordinates_formatter(roi)
			result_maps = []
			# patient_data = glob.glob(dataDir + '/' + patient_number + '_*' + '/DBSI_results*/*.nii')
			for maps_i in maps_of_interest:
				result_maps.append(glob.glob(dataDir + '/' + patient_number + '_*' + '/DBSI_results*/' + result_maps_list[maps_i])[0])
			# for maps in patient_data:
			# 	if result_maps_list.index(os.path.basename(maps)) in maps_of_interest:
			# 		result_maps.append(maps)
			patient = Patient(roi_patient_name, patient_number, patientType, gleason, nccn, roi_dict, result_maps)
			patients.append(patient)
	return patients
#end prep_patient_info

def get_num_vox(patientsArray):
	# print('get_num_vox')
	num_vox = 0
	patient_vox_dict = {}
	for patient in patientsArray:
		patient_vox_dict[patient] = 0
		for roi, coords in patient.roi.iteritems():
			num_vox += len(coords)
			patient_vox_dict[patient] += len(coords)

	return {'total': num_vox, 'patient_totals': patient_vox_dict}
#end get_num_vox

def prep_svm_data(patientsArray, patientType):
	# print('prep_svm_data')
	"""
		function to create numpy arrays that we will contain voxel data and labels used for svm training
		params:
		    patientsArray: array of Patient objects that contain patient information
		    patientType: string defining benign vs pca patient
		return: 
			numpy_dict: dict of numpy arrays for svm data and labels
	"""
	num_vox_info = get_num_vox(patientsArray)
	raw_data = []
	normalized_data = []
	target_array = np.zeros((num_vox_info.get('total'),), dtype=int)
	if patientType == 'pca':
		target_array.fill(1)
	target_array_2 = np.empty((num_vox_info.get('total'),), dtype=int)
	if patientType == 'benign':
		target_array_2.fill(0)
	coordinates = []
	continue_index = 0
	for patient in patientsArray:
		patient_raw_data = np.empty([num_vox_info.get('patient_totals').get(patient), len(maps_of_interest)])
		for roi, coords in patient.roi.iteritems():
			coordinates += coords
			for m_i, m in enumerate(patient.result_maps):
				map_coord_tracker = 0
				m_data = nib.load(m).get_data()
				for c_i, c in enumerate(coords):
					patient_raw_data[c_i + map_coord_tracker][m_i] = m_data[c[0]][c[1]][c[2]]
					if patientType == 'pca':
						target_array_2[c_i + continue_index] = int(grade_dict.get(patient.gleason))
				map_coord_tracker = c_i
		raw_data.append(patient_raw_data)
		normalized_data.append(amend_data(normalize_data(amend_data(patient_raw_data))))
		continue_index += len(coords)
	data_array = np.vstack(normalized_data)
	raw_array = np.vstack(raw_data)
	# if patientType == 'pca':
	# 	pca_data = clean_up_data(data_array, target_array, target_array_2)
	numpy_dict = {'data': data_array,
				  'raw_data': raw_array,
				  'label1': target_array,
				  'label2': target_array_2,
				  'coords': np.array(coordinates, dtype='int,int,int')}
	return numpy_dict
#end prep_svm_data

def amend_data(numpyArray):
	# print('amend_data')
	"""
		function to ammend data of NaN values
		params:
			numpyArray: numpy array of data
		return:
			numpyArray: numpy array with amended values to remove NaN values
		TODO: test replacing NaN with 0
	"""
	numpyArray_amend = np.nan_to_num(numpyArray)
	return numpyArray_amend
#end amend_data

def normalize_data(numpyArray):
	# print('normalize_data')
	"""
		function to normalize our data before using it to train svm
		params:
			numpyArray: numpy array containing data
		return:
			data_array_norm: normalized numpy array with voxel data
	"""
	numpyArray_norm = numpyArray / np.amax(numpyArray, axis=0)
	return numpyArray_norm
#end normalize_data

def clean_up_data(dataDict):
	# print('clean_up_data')
	"""
		function to remove non-pca data for use during pca training
		params:
			dataset: numpy array containing our pca data
			labels: numpy array containing our labels for pca vs benign
			pcaLabels: pca grading labels
		return:
			dict containing data set with label 0 data removed
	"""
	data = dataDict.get('data')
	# labels = dataDict.get('label1')
	pca_labels = dataDict.get('label2')
	remove_index = np.argwhere(pca_labels == 0)
	# labels = np.delete(labels, remove_index, 0)
	pca_labels = np.delete(pca_labels, remove_index, 0)
	data = np.delete(data, remove_index, 0)

	return {'data': data, 'label2': pca_labels}
#end clean_up_data

def combine_data(dataset1, dataset2):
	# print('combine_data')
	"""
		function to combine numpy data sets into 1 numpy array so they can
		later be split into training and testing sets
		params:
			dataset1: dictionary of numpy arrays
			dataset2: dictionary of numpy arrays
		return:
			combined_data: dictionary of combined numpy arrays
	"""
	combine_data = np.vstack((dataset1.get('data'), dataset2.get('data')))
	combine_target = np.concatenate((dataset1.get('label1'), dataset2.get('label1')))
	combine_target_2 = np.concatenate((dataset1.get('label2'), dataset2.get('label2')))

	combined_data = {'data': combine_data,
					 'label1': combine_target,
					 'label2': combine_target_2}
	return combined_data

def svm_training_1(combined_data):
	print('svm_training_1')
	"""
		function to perform svm training
		1. benign vs pca training
		2. pca grading training on pca group
		params:
			benignData: dictionary of numpy arrays for benign patients
			pcaData: dictionary of numpy arrays for pca patients
		TODO: BALANCE DATA SETS SO EACH GRADE HAS SIMILAR NUMBER OF VOXELS
			  OPTIMIZE KERNELS TO SEE WHICH ONE FITS OUR DATA THE BEST
	"""
	start = time.time()
	training_data, test_data, training_target, test_target = ms.train_test_split(combined_data.get('data'), combined_data.get('label1'), test_size=0.2)
	svm_plain_classifier = OneVsRestClassifier(svm.SVC(C=1000.0, cache_size=200, class_weight='balanced', decision_function_shape=None, gamma=0.1, kernel='rbf',
		max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False))
	svm_plain_classifier.fit(training_data, training_target)
	score = svm_plain_classifier.decision_function(test_data)
	false_pos = dict()
	true_pos = dict()
	roc_auc = dict()
	svm_predict = svm_plain_classifier.predict(test_data)
	accuracy = svm_plain_classifier.score(test_data, test_target)
	end = time.time()
	runtime = end - start
	print('runtime: ' + str(runtime))
	print('score: ')
	print(score)
	print("benign vs pca accuracy: " + str(accuracy))
	print(mt.confusion_matrix(test_target, svm_predict))
	classes = np.unique(combined_data.get('label1'))
	print(mt.classification_report(test_target, svm_predict, target_names=map(str,classes)))
	# benign_v_pca_results = [test_data, test_target, svm_predict]
	print(type(test_target))
	print(test_target.shape)
	print(type(score))
	print(score.shape)
	sys.exit()
	for i in range(len(classes)):
		false_pos[i], true_pos[i], _ = mt.roc_curve(test_target[:, i], score[:, i])
		roc_auc[i] - mt.auc(false_pos[i], true_pos[i])

	false_pos["micro"], true_pos["micro"], _ = roc_curve(test_target.ravel(), score.ravel())
	roc_auc["micro"] = auc(false_pos["micro"], true_pos["micro"])

	plt.figure()
	lw = 2
	plt.plot(false_pos[2], true_pos[2], color='darkorange', lw=lw, label='ROC curve(area = %0.2f)' % roc_auc[2])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characterisitc Example')
	plt.legend(loc='lower right')
	plt.show()
	return [[test_data, test_target, svm_predict]]
#end svm_training_1

def svm_training_2(data):
	print('svm_training_2')
	"""
		function to perform svm training directly with pca grading
		params:
			benignData: dictionary of numpy arrays for benign patients
			pcaData: dictionary of numpy arrays for pca patients
	"""
	start = time.time()
	training_data, test_data, training_target, test_target = ms.train_test_split(data.get('data'), data.get('label2'), test_size=0.2)
	svm_plain_classifier = OneVsRestClassifier(svm.SVC(C=1000.0, cache_size=200, class_weight='balanced', decision_function_shape=None, gamma=0.1, kernel='rbf',
		max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False))
	svm_plain_classifier.fit(training_data, training_target)
	svm_predict = svm_plain_classifier.predict(test_data)
	accuracy = svm_plain_classifier.score(test_data, test_target)
	end = time.time()
	runtime = end - start
	print('runtime: ' + str(runtime))
	print("direct cancer grading accuracy: " + str(accuracy))
	classes = np.unique(data.get('label2'))
	print(mt.confusion_matrix(test_target, svm_predict))
	print(mt.classification_report(test_target, svm_predict, target_names=map(str,classes)))

	return [[test_data, test_target, svm_predict]]
#end svm_training_2

def convert_to_gleason(scoresArray):
	# print('convert_to_gleason')
	"""
		function to backwards convert score to gleason score based on grades_dict
		params:
			scoresArray: numpy array of our scores obtained from grades_dict lookup
		return:
			gleason_scores: array of gleason scores from backwards lookup of grades_dict
	"""
	gleason_scores = []
	for x in scoresArray:
		for g, s in grade_dict.items():
			if x == s:
				gleason_scores.append(str(g))
	return gleason_scores
def plot_data(set1, set2):
	# print('plot_data')
	"""
		function to plot our data points
		params:
			set1, set2: dictionaries of numpy arrays that contain data we want to plot
	"""
	benign_data_points = set1.get('data')
	pca_data_points = set2.get('data')
	
	fig = plt.figure()
	plot = fig.add_subplot(1, 1, 1)

	for i in range(len(maps_of_interest)):
		plot.scatter(benign_data_points.shape[0] * maps_of_interest, benign_data_points[:,i], s=0.5, c='g')
		plot.scatter(pca_data_points.shape[0] * maps_of_interest, pca_data_points[:,i], s=0.5, c='r')
	plt.show()

def data_log(benign, pca, predicted):
	# print('data_log')
	"""
		function to log our patient data into an excel sheet
		params:
			benign: dictionary of numpy arrays for benign patient data
			pca: dictionary of numpy arrays for pca patient data
	"""
	today = datetime.datetime.now()
	today_files = glob.glob(os.path.dirname(__file__) + str(today.year) + str(today.month) + str(today.day) + '_data_log*.xlsx')
	if len(today_files) > 0:
		excel_writer = pd.ExcelWriter(str(today.year) + str(today.month) + str(today.day) + '_data_log(' + str(len(today_files)) + ').xlsx')
	else:
		excel_writer = pd.ExcelWriter(str(today.year) + str(today.month) + str(today.day) + '_data_log.xlsx')

	benign_raw_df = pd.DataFrame(benign.get('raw_data'))
	benign_raw_df.columns = [result_maps_list[i] for i in maps_of_interest]
	benign_raw_df.insert(0, 'Coords', benign.get('coords'))

	benign_df = pd.DataFrame(benign.get('data'))
	benign_df.columns = [result_maps_list[i] for i in maps_of_interest]
	benign_df.insert(0, 'Coords', benign.get('coords'))

	pca_raw_df = pd.DataFrame(pca.get('raw_data'))
	pca_raw_df.columns = [result_maps_list[i] for i in maps_of_interest]
	pca_raw_df.insert(0, 'Coords', pca.get('coords'))

	pca_df = pd.DataFrame(pca.get('data'))
	pca_df.columns = [result_maps_list[i] for i in maps_of_interest]
	pca_df.insert(0, 'Coords', pca.get('coords'))
	pca_grade = pca.get('label2')

	gleason_score = convert_to_gleason(pca_grade)
	pca_raw_df.insert(0, 'Gleason', gleason_score)
	pca_df.insert(0, 'Gleason', gleason_score)

	benign_raw_df.to_excel(excel_writer, sheet_name="Benign RAW")
	pca_raw_df.to_excel(excel_writer, sheet_name="PCa RAW")
	benign_df.to_excel(excel_writer, sheet_name="Benign")
	pca_df.to_excel(excel_writer, sheet_name="PCa")

	grades_predicted = convert_to_gleason(predicted[1][2])
	grades_actual = convert_to_gleason(predicted[1][1])

	training_results_df1 = pd.DataFrame(predicted[0][0])
	training_results_df1.insert(0, 'PCa', predicted[0][1])
	training_results_df1.insert(0, 'Predicted', predicted[0][2])
	training_results_df1.to_excel(excel_writer, sheet_name="Benign v PCa")

	training_results_df2 = pd.DataFrame(predicted[1][0])
	training_results_df2.insert(0, 'Gleason', grades_actual)
	training_results_df2.insert(0, 'Predicted', grades_predicted)
	training_results_df2.to_excel(excel_writer, sheet_name=" PCa Grading")

	grades_predicted = convert_to_gleason(predicted[0][2])
	grades_actual = convert_to_gleason(predicted[0][1])

	# one_step_df = pd.DataFrame(predicted[0][0])
	# one_step_df.insert(0, 'Gleason', grades_actual)
	# one_step_df.insert(0, 'Predicted', grades_predicted)
	# one_step_df.to_excel(excel_writer, sheet_name="1-step Gleason")

	xlsx_book = excel_writer.book
	xlsx_book_format = xlsx_book.add_format({'align':'center', 'valign':'center'})
	xlsx_sheet_raw_b = excel_writer.sheets['Benign RAW']
	xlsx_sheet_raw_b.set_column('B:Z', 22, xlsx_book_format)
	xlsx_sheet_b = excel_writer.sheets['Benign']
	xlsx_sheet_b.set_column('B:Z', 22, xlsx_book_format)
	xlsx_sheet_raw_p = excel_writer.sheets['PCa RAW']
	xlsx_sheet_raw_p.set_column('B:Z', 22, xlsx_book_format)
	xlsx_sheet_p = excel_writer.sheets['PCa']
	xlsx_sheet_p.set_column('B:Z', 22, xlsx_book_format)

	excel_writer.save()

def main():
	# print('main')
	benign_patients = prep_patient_info(sys.argv[1], sys.argv[2], 'benign')
	# print(len(benign_patients))
	benign_numpy = prep_svm_data(benign_patients, 'benign')

	pca_patients = prep_patient_info(sys.argv[3], sys.argv[4], 'pca')
	# print(len(pca_patients))
	pca_numpy = prep_svm_data(pca_patients, 'pca')
	pca_numpy_cleaned = clean_up_data(pca_numpy)

	# plot_data(benign_numpy, pca_numpy)
	# sys.exit()
	combined_data = combine_data(benign_numpy, pca_numpy)

	# if not direct_grading:
	predicted1 = svm_training_1(combined_data)
	# predicted2 = svm_training_2(pca_numpy_cleaned)
	# else:
	# 	predicted = svm_training_2(combined_data)
	# data_log(benign_numpy, pca_numpy, predicted)
#end main

if __name__ == '__main__':
	main()