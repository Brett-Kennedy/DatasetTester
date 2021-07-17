import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from warnings import filterwarnings
import shutil
from time import perf_counter

# todo: put back!!
#from DatasetsEvaluator.DatasetsEvaluator import DatasetsTester

import sys  
sys.path.insert(0, 'C:\python_projects\DatasetsEvaluator_project\DatasetsEvaluator')
import DatasetsEvaluator as de


def run_single_process(datasets_tester, estimators_arr, results_folder):
	print("Running a single process...")
	start_time = perf_counter()
	summary_df, _ = datasets_tester.run_tests(estimators_arr, results_folder=results_folder) 
	end_time = perf_counter()
	print(f"Total Time: {end_time-start_time}")


def run_multiple_processes(datasets_tester, estimators_arr, results_folder):
	print("\n\nRunning multiple processes...")
	start_time = perf_counter()
	summary_df, _ = datasets_tester.run_tests(estimators_arr,  results_folder=results_folder, run_parallel=True) 
	end_time = perf_counter() 
	print(f"Total Time: {end_time-start_time}")


def main():
	filterwarnings('ignore')

	cache_folder = "c:\\dataset_cache"
	results_folder = "c:\\results"

	datasets_tester = de.DatasetsTester()
	matching_datasets = datasets_tester.find_datasets( 
	    problem_type = "classification",
	    min_num_classes = 2,
	    max_num_classes = 20,
	    min_num_minority_class = 5,
	    max_num_minority_class = np.inf,
	    min_num_features = 0,
	    max_num_features = np.inf,
	    min_num_instances = 500,
	    max_num_instances = 5_000,
	    min_num_numeric_features = 2,
	    max_num_numeric_features = 50,
	    min_num_categorical_features=0,
	    max_num_categorical_features=50)

	print(f"Number matching datasets found: {len(matching_datasets)}" )

	datasets_tester.collect_data(max_num_datasets_used=20,
	                             method_pick_sets='pick_first', 
	                             preview_data=False,
	                             save_local_cache=True, 
	                             check_local_cache=True,
	                             path_local_cache=cache_folder)

	dt_1 = tree.DecisionTreeClassifier(min_samples_split=50, max_depth=6, random_state=0)
	dt_2 = tree.DecisionTreeClassifier(min_samples_split=25, max_depth=5, random_state=0)
	knn_1 = KNeighborsClassifier(n_neighbors=5)
	knn_2 = KNeighborsClassifier(n_neighbors=10)

	estimators_arr = [
		        ("Decision Tree", "Original Features", "min_samples_split=50, max_depth=6", dt_1),
		        ("Decision Tree", "Original Features", "min_samples_split=25, max_depth=5", dt_2),
		        ("kNN", "Original Features", "n_neighbors=5", knn_1),
		        ("kNN", "Original Features", "n_neighbors=10", knn_2)]

	run_single_process(datasets_tester, estimators_arr, results_folder)
	run_multiple_processes(datasets_tester, estimators_arr, results_folder)

if __name__ == "__main__":
	main()