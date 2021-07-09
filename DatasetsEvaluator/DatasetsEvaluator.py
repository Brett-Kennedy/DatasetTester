import pandas as pd
import numpy as np
import openml
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from statistics import mean, stdev
from warnings import filterwarnings, resetwarnings
from time import time
from os import mkdir, listdir
from shutil import rmtree

class DatasetsTester():
    """
    Tool to compare predictors (classifiers or regressors) on a set of datasets collected from openml.org.

    This simplifies automatically comparing the performance of predictors on potentially large numbers
    of datasets, thereby supporting more thorough and accurate testing of predictors. 
    """
    
    def __init__(self):
        pass
      
    def find_by_name(self, names_arr, problem_type):
        """
        Identifies, but does not collect, the set of datasets meeting the specified set of names.

        Parameters
        ----------
        names_arr: array of dataset names

        problem_type: str
            Either "classification" or "regression"         
            All estimators will be compared using the same metric, so it is necessary that all
            datasets used are of the same type.

        Returns
        -------
        dataframe with a row for each dataset on openml meeting the specified set of names. 

        """

        self.problem_type = problem_type
        self.openml_df = openml.datasets.list_datasets(output_format="dataframe")
        self.openml_df = self.openml_df[self.openml_df.name.isin(names_arr)]
        return self.openml_df
        
    def find_datasets(self, 
                     problem_type, 
                     min_num_classes=2,
                     max_num_classes=10,
                     min_num_minority_class=5,
                     max_num_minority_class=np.inf, 
                     min_num_features=0,
                     max_num_features=100,
                     min_num_instances=500, 
                     max_num_instances=5000, 
                     min_num_numeric_features=0,
                     max_num_numeric_features=50,
                     min_num_categorical_features=0,
                     max_num_categorical_features=50):
        """
        Identifies, but does not collect, the set of datasets meeting the specified set of names.
        This or find_by_name() must be called to identify the potential set of datasets to be collected.

        Parameters
        ----------
        problem_type: str
            Either "classification" or "regression".        
            All estimators will be compared using the same metric, so it is necessary that all
            datasets used are of the same type.

        All other parameters are direct checks of the statistics about each dataset provided by openml.org.

        Returns
        -------
        dataframe with a row for each dataset on openml meeting the specified set of criteria. 

        """
        
        if problem_type not in ["classification", "regression"]:
            print("problem_type must be either 'classification' or 'regression'.")
            return None
        if problem_type == "classification" and (min_num_classes<=0 or max_num_classes<=0):
            print("For classification datasets, both min_num_classes and max_num_classes must be specified.")
            return None

        self.problem_type = problem_type
        self.min_num_classes = min_num_classes
        self.max_num_classes = max_num_classes
        self.min_num_minority_class = min_num_minority_class
        self.max_num_minority_class = max_num_minority_class
        self.min_num_features = min_num_features
        self.max_num_features = max_num_features        
        self.min_num_instances = min_num_instances
        self.max_num_instances = max_num_instances
        self.min_num_numeric_features = min_num_numeric_features
        self.max_num_numeric_features = max_num_numeric_features
        self.min_num_categorical_features = min_num_categorical_features
        self.max_num_categorical_features = max_num_categorical_features
        
        self.openml_df = openml.datasets.list_datasets(output_format="dataframe")
        
        # Filter out datasets where some key attributes are unspecified
        self.openml_df = self.openml_df[ 
                        (np.isnan(self.openml_df.NumberOfFeatures) == False) &
                        (np.isnan(self.openml_df.NumberOfInstances) == False) &
                        (np.isnan(self.openml_df.NumberOfInstancesWithMissingValues) == False) &
                        (np.isnan(self.openml_df.NumberOfMissingValues) == False) &
                        (np.isnan(self.openml_df.NumberOfNumericFeatures) == False) &
                        (np.isnan(self.openml_df.NumberOfSymbolicFeatures) == False) 
                     ]   

        self.openml_df = self.openml_df[
                    #(self.openml_df.NumberOfClasses == 0) &
                    (self.openml_df.NumberOfFeatures >= min_num_features) & 
                    (self.openml_df.NumberOfFeatures <= max_num_features) &            
                    (self.openml_df.NumberOfInstances >= self.min_num_instances) & 
                    (self.openml_df.NumberOfInstances <= self.max_num_instances) &
                    (self.openml_df.NumberOfNumericFeatures >= min_num_numeric_features) &
                    (self.openml_df.NumberOfNumericFeatures <= max_num_numeric_features) &
                    (self.openml_df.NumberOfSymbolicFeatures >= min_num_categorical_features) &
                    (self.openml_df.NumberOfSymbolicFeatures <= max_num_categorical_features)
                    ]    

        if problem_type == "classification":
            self.openml_df = self.openml_df[ 
                        (np.isnan(self.openml_df.MajorityClassSize) == False) &
                        (np.isnan(self.openml_df.MaxNominalAttDistinctValues) == False) &
                        (np.isnan(self.openml_df.MinorityClassSize) == False) &
                        (np.isnan(self.openml_df.NumberOfClasses) == False) 
                    ]     
            
            self.openml_df = self.openml_df[
                        (self.openml_df.NumberOfClasses >= min_num_classes) & 
                        (self.openml_df.NumberOfClasses <= max_num_classes) &
                        (self.openml_df.MinorityClassSize >= min_num_minority_class) &
                        (self.openml_df.MinorityClassSize <= max_num_minority_class) 
                    ]    

        return self.openml_df
    
    def collect_data(self, 
                     max_num_datasets_used=-1,
                     method_pick_sets="pick_first",
                     exclude_list=None,
                     max_cat_unique_vals = 20,
                     keep_duplicated_names=False,
                     save_local_cache=False, 
                     check_local_cache=False, 
                     path_local_cache="",
                     preview_data=False,
                     one_hot_encode=True,
                     fill_nan_and_inf_zero=True):
        """
        This method collects the data from openml.org, unless check_local_cache is True and the dataset is avaialble 
        in the local folder. This will collec the specifed subset of datasets identified by the most recent call 
        to find_by_name() or find_datasets(). This allows users to call those methods until a suitable 
        collection of datasets have been identified.

        Parameters
        ----------
        max_num_datasets_used: integer 
            The maximum number of datasets to collect.

        method_pick_sets: str
            If only a subset of the full set of matches are to be collected, this identifies if those
            will be selected randomly, or simply using the first matches

        exclude_list: array
            list of names of datasets to exclude

        max_cat_unique_vals: int
            As categorical columns are one-hot encoded, it may not be desirable to one-hot encode categorical
            columns with large numbers of unique values. Columns with a greater number of unique values than
            max_cat_unique_vals will be dropped. 

        keep_duplicated_names: bool
            If False, for each set of datasets with the same name, only the one with the highest 
            version number will be used. 

        save_local_cache: bool
            If True, any collected datasets will be saved locally in path_local_cache

        check_local_cache: bool
            If True, before collecting any datasets from openml.org, each will be checked to determine if
            it is already stored locally in path_local_cache

        path_local_cache: str
            Folder identify the local cache of datasets, stored in .csv format.

        preview_data: bool
            Indicates if the first rows of each collected dataset should be displayed

        one_hot_encode: bool
            If true, categorical columns are one-hot encoded. This is necessary for many types of predictor, but
            may be done elsewhere, for example in a pipeline passed to the run_tests() function.

        fill_nan_and_inf_zero: bool
            If true, all instances of NaN, inf and -inf are replaced with 0.0. Replacing these values with something 
            valid is necessary for many types of predictor, butmay be done elsewhere, for example in a pipeline passed 
            to the run_tests() function.

        Returns
        -------

        drops any categorical columns with more than max_cat_unique_vals unique values. 
        if keep_duplicated_names is False, then only one version of each dataset name is kept. This can reduce
        redundant test. In some cases, though, different versions of a dataset are significantly different. 
        """
        
        assert method_pick_sets in ['pick_first','pick_random']
        
        if (len(self.openml_df)==0):
            print("Error. No datasets specified. Call find_datasets() or find_by_name() before collect_data().")
            return None        
        
        if keep_duplicated_names==False:
            self.openml_df = self.openml_df.drop_duplicates(subset=["name"], keep="last")           
        
        self.dataset_collection = []
        
        if max_num_datasets_used > -1 and max_num_datasets_used < len(self.openml_df) and method_pick_sets == "pick_random":
            openml_subset_df = self.openml_df.sample(frac=1, random_state=0)
        else:
            openml_subset_df = self.openml_df
        
        usable_dataset_idx = 0
        for dataset_idx in range(len(openml_subset_df)):
            if (max_num_datasets_used>-1) and (len(self.dataset_collection) >= max_num_datasets_used):
                break

            dataset_did = int(openml_subset_df.iloc[dataset_idx].did)
            dataset_name = openml_subset_df.iloc[dataset_idx]['name']
            dataset_version = openml_subset_df.iloc[dataset_idx]['version']

            if not exclude_list is None and dataset_name in exclude_list:
                continue

            dataset_df = None
            if check_local_cache: 
                try: 
                    path_to_file = path_local_cache + "/" + dataset_name + '.csv'
                    X_with_y = pd.read_csv(path_to_file)
                    dataset_df = X_with_y.drop("y", axis=1)
                    y = X_with_y["y"]
                    print(f"Reading from local cache: {dataset_idx}, id: {dataset_did}, name: {dataset_name}")
                except Exception as e:
                    if "No such file or directory:" not in str(e):
                        print(f" Error reading file: {e}")
                    else:
                        print(" File not found in cache.")
                    dataset_df = None

            if dataset_df is None:
                print(f"Loading dataset from openml: {dataset_idx}, id: {dataset_did}, name: {dataset_name}")
                dataset = openml.datasets.get_dataset(dataset_did)            
                try: 
                    X, y, categorical_indicator, attribute_names = dataset.get_data(
                        dataset_format="dataframe", 
                        target=dataset.default_target_attribute
                    )
                except Exception as e:
                    print(f" Error collecting file with did: {dataset_did}, name: {dataset_name}. Error: {e}")
                    continue
                if X is None or y is None:
                    print(f" Error collecting file with did: {dataset_did}, name: {dataset_name}. X or y is None")
                    continue
                dataset_df = pd.DataFrame(X, columns=attribute_names)

            if (len(dataset_df)==len(y)):
                if preview_data: display(dataset_df.head())

                if save_local_cache:
                    X_with_y = dataset_df.copy()
                    X_with_y['y'] = y 
                    X_with_y.to_csv(path_local_cache + "/" + dataset_name + '.csv', index=False)

                if (self.problem_type == "regression") and (is_numeric_dtype(y)==False):
                    print(" Dataset is classification")
                    continue

                dataset_df = self.__clean_dataset(dataset_df, max_cat_unique_vals, one_hot_encode, fill_nan_and_inf_zero)
                self.dataset_collection.append((usable_dataset_idx, dataset_name, dataset_version, dataset_df, y))
                usable_dataset_idx+=1
            else:
                print(f" Error collecting file with did: {dataset_did}, name: {dataset_name}. Number rows in X: {len(X)}. Number rows in y: {len(y)}")
        
    def __clean_dataset(self, X, max_cat_unique_vals, one_hot_encode, fill_nan_and_inf_zero):
        
        # One-hot encode the categorical columns
        if one_hot_encode:
            # The categorical_indicator provided by openml isn't 100% reliable, so we also check panda's is_numeric_dtype
            categorical_indicator = [False]*len(X.columns)
            for c in range(len(X.columns)):
                if is_numeric_dtype(X[X.columns[c]]) == False:
                    categorical_indicator[c]=True

            new_df = pd.DataFrame()
            for c in range(len(categorical_indicator)):
                col_name = X.columns[c]
                if categorical_indicator[c] == True:
                    if X[col_name].nunique() > max_cat_unique_vals:
                        pass
                    else:
                        one_hot_cols = pd.get_dummies(X[col_name], prefix=col_name, dummy_na=True, drop_first=False)
                        new_df = pd.concat([new_df, one_hot_cols], axis=1)
                else:
                    new_df[col_name] = X[col_name]
            X = new_df

        # Remove any NaN or inf values
        if fill_nan_and_inf_zero:
            X = X.fillna(0.0)
            X = X.replace([np.inf, -np.inf], 0.0)                        
        
        return X.reset_index(drop=True)
    
    def get_dataset_collection(self):
        return self.dataset_collection

    def run_tests(self, estimators_arr, num_cv_folds=5, scoring_metric='', show_warnings=False, starting_point=0, ending_point=np.inf, partial_result_folder=""):
        """
        Evaluate all estimators on all datasets. 
        
        Parameters
        ----------
        estimators_arr: array of tuples, with each tuple containing: 
            str: estimator name, 
            str: a description of the features used
            str: a description of the hyperparameters used
            estimator: the estimator to be used. This should not be fit yet, just have the hyperparameters set.

        num_cv_folds: int
            the number of folds to be used in the cross validation process used to evaluate the predictor

        scoring_metric: str
            one of the set of scoring metrics supported by sklearn. Set to '' to indicate to use the default.
            The default for classification is f1_macro and for regression is normalized root mean square error.

        show_warnings: bool
            if True, warnings will be presented for calls to cross_validate(). These can get very long in in some
            cases may affect only a minority of the dataset-predictor combinations, so is False by default. Users
            may wish to set to True to determine the causes of any NaNs in the final summary dataframe.   

        starting_point: int
            This may be used to resume long-running tests where previous runs have not completed the full test or
            where previous calls to this method set ending_point

        ending_point: int
            This may be used to divide up the datasets, potentially to 

        partial_result_folder: string
            path to folder where partial results are saved. 

        Returns
        -------
        a dataframe summarizing the performance of the estimators on each dataset. There is one row
        for each combination of dataset and estimator. 
        """

        self.estimators_arr = estimators_arr

        scoring_metric_specified = True
        if self.problem_type == "classification":
            if scoring_metric == '':
                scoring_metric_specified = False
                scoring_metric = 'f1_macro'
        else:
            if scoring_metric == '':
                scoring_metric_specified = False
                scoring_metric = 'neg_root_mean_squared_error'

        # Dataframes used to store the test results
        column_names = ['Dataset Index',
                        'Dataset',
                        'Dataset Version',
                        'Model',                                                          
                        'Feature Engineering Description',
                        'Hyperparameter Description']
        if scoring_metric_specified==False and self.problem_type == "regression":
            column_names.append('Avg NRMSE')
        else:
            column_names.append('Avg ' + scoring_metric)
        column_names +=[
                        'Std dev between folds', 
                        'Train-Test Gap', 
                        '# Columns',
                        'Model Complexity',
                        'Fit Time']
        summary_df = pd.DataFrame(columns=column_names)

        if show_warnings:
            filterwarnings('default')
        else:
            filterwarnings('ignore')

        if starting_point==0 and partial_result_folder != "":
            try:
                mkdir(partial_result_folder)
            except:
                pass

        print(f"\nRunning test on {len(self.dataset_collection)} datastets")
        for dataset_tuple in self.dataset_collection: 
            dataset_index, dataset_name, version, X, y = dataset_tuple
            if (dataset_index < starting_point):
                continue
            if (dataset_index >= ending_point):
                continue
            print(f"Running tests on dataset index: {dataset_index}, dataset: {dataset_name}")
            for estimator_desc in self.estimators_arr:
                model_name, engineering_description, hyperparameters_description, clf = estimator_desc
                print(f"\tRunning tests with model: {model_name} ({engineering_description}), ({hyperparameters_description})")
                scores = cross_validate(clf, X, y, cv=num_cv_folds, scoring=scoring_metric, return_train_score=True, return_estimator=True)
                train_scores = scores['train_score']
                test_scores = scores['test_score']
                if scoring_metric_specified == False and self.problem_type == "regression":
                    # Convert from neg_root_mean_squared_error to NRMSE
                    train_scores =abs(train_scores/(y.mean()))                        
                    test_scores = abs(test_scores/(y.mean()))                        
                avg_test_score = test_scores.mean()
                scores_std_dev = stdev(test_scores)
                avg_train_score = train_scores.mean()
                avg_fit_time = scores['fit_time'].mean()

                # Model Complexity is currently only supported for decision trees, and measures the number of nodes.
                estimators_arr = scores['estimator']                
                if type(estimators_arr[0]) == Pipeline:
                    for p_idx in range(len(estimators_arr[0])):
                        short_estimators_arr = [e[p_idx] for e in estimators_arr]
                        model_complexity = self.check_model_complexity(short_estimators_arr)       
                        if (model_complexity>0):
                            break
                else:
                    model_complexity = self.check_model_complexity(estimators_arr)

                summary_row = [dataset_index,
                               dataset_name, 
                               version,
                               model_name, 
                               engineering_description,
                               hyperparameters_description,
                               avg_test_score, 
                               scores_std_dev, 
                               avg_train_score-avg_test_score,
                               len(X.columns),
                               model_complexity,
                               avg_fit_time]
                summary_df = summary_df.append(pd.DataFrame([summary_row], columns=summary_df.columns))
            
            if (partial_result_folder != ""):
                intermediate_file_name = partial_result_folder + "\\intermediate_" + str(dataset_index) + ".csv"
                summary_df.to_csv(intermediate_file_name, index=False)
        
        resetwarnings()

        if starting_point>0 and partial_result_folder != "":
            summary_df = self.get_previous_results(summary_df, partial_result_folder)

        return summary_df.reset_index(drop=True)

    def run_tests_grid_search(self, estimators_arr, parameters_arr, num_cv_folds=5, scoring_metric='', show_warnings=False, starting_point=0, ending_point=np.inf, partial_result_folder=""):
        """
        Evaluate all estimators on all datasets. 
        
        Parameters
        ----------
        estimators_arr: array of tuples, with each tuple containing: 
            str: estimator name, 
            str: a description of the features used
            str: a description of the hyperparameters used
            estimator: the estimator to be used. This should not be fit yet, just have the hyperparameters set.

        parameters_arr: array of dictionaries
            Each dictionary describes the range of parameters to be tested on the matching estimator

        num_cv_folds: int
            the number of folds to be used in the cross validation process used to evaluate the predictor

        scoring_metric: str
            one of the set of scoring metrics supported by sklearn. Set to '' to indicate to use the default.
            The default for classification is f1_macro and for regression is neg_root_mean_squared_error.

        show_warnings: bool
            if True, warnings will be presented for calls to cross_validate(). These can get very long in in some
            cases may affect only a minority of the dataset-predictor combinations, so is False by default. Users
            may wish to set to True to determine the causes of any NaNs in the final summary dataframe.   

        Returns
        -------
        a dataframe summarizing the performance of the estimators on each dataset. There is one row
        for each combination of dataset and estimator. 
        """

        self.estimators_arr = estimators_arr
        self.parameters_arr = parameters_arr

        scoring_metric_specified = True
        if self.problem_type == "classification":
            if scoring_metric == '':
                scoring_metric_specified = False
                scoring_metric = 'f1_macro'
        else:
            if scoring_metric == '':
                scoring_metric_specified = False
                scoring_metric = 'neg_root_mean_squared_error'

        # Dataframes used to store the test results
        column_names = ['Dataset Index',
                        'Dataset',
                        'Dataset Version',
                        'Model',                                                          
                        'Feature Engineering Description',
                        'Hyperparameter Description']
        if scoring_metric_specified==False and self.problem_type == "regression":
            column_names.append('NRMSE')
        else:
            column_names.append(scoring_metric)
        column_names +=[
                        'Train-Test Gap', 
                        '# Columns',
                        'Model Complexity',
                        'Fit Time']
        summary_df = pd.DataFrame(columns=column_names)

        if show_warnings:
            filterwarnings('default')
        else:
            filterwarnings('ignore')

        if starting_point==0 and partial_result_folder != "":
            try:
                mkdir(partial_result_folder)
            except:
                pass            

        print(f"\nRunning test on {len(self.dataset_collection)} datastets")
        for dataset_tuple in self.dataset_collection: 
            dataset_index, dataset_name, version, X, y = dataset_tuple
            if (dataset_index < starting_point):
                continue
            if (dataset_index >= ending_point):
                continue            
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
            print(f"Running tests on dataset: {dataset_index}: {dataset_name}")
            for estimator_idx, estimator_desc in enumerate(self.estimators_arr):
                model_name, engineering_description, hyperparameters_description, estimator = estimator_desc
                parameters = self.parameters_arr[estimator_idx]
                print(f"\tRunning tests with model: {model_name} ({engineering_description})")
                gs_estimator = GridSearchCV(estimator, parameters, scoring=scoring_metric)
                start_time = time()
                gs_estimator.fit(X_train, y_train)
                end_time = time()
                y_pred_train = gs_estimator.predict(X_train)  
                y_pred_test = gs_estimator.predict(X_test) 
                train_score = f1_score(list(y_pred_train), list(y_train), average="macro") 
                test_score = f1_score(list(y_pred_test), list(y_test), average="macro") 
                if scoring_metric_specified == False and self.problem_type == "regression":
                    # Convert from neg_root_mean_squared_error to NRMSE
                    train_score = abs(train_score/(y.mean()))                        
                    test_score = abs(test_score/(y.mean()))                                        

                print("\ttest_score: ", test_score)  

                if type(gs_estimator.best_estimator_ == Pipeline):
                    for p_idx in range(len(gs_estimator.best_estimator_)):
                        est = gs_estimator.best_estimator_[p_idx]
                        model_complexity = self.check_model_complexity([est])       
                        if (model_complexity>0):
                            break
                else:
                    model_complexity = self.check_model_complexity([gs_estimator.best_estimator_])

                summary_row = [dataset_index,
                               dataset_name, 
                               version,
                               model_name, 
                               engineering_description,
                               hyperparameters_description,
                               test_score, 
                               train_score-test_score,
                               len(X.columns),
                               model_complexity,
                               round(end_time-start_time,2)]
                summary_df = summary_df.append(pd.DataFrame([summary_row], columns=summary_df.columns))
        
            if (partial_result_folder != ""):
                intermediate_file_name = partial_result_folder + "\\intermediate_" + str(dataset_index) + ".csv"
                summary_df.to_csv(intermediate_file_name, index=False)
 
        resetwarnings()

        if starting_point>0 and partial_result_folder != "":
            summary_df = self.get_previous_results(summary_df, partial_result_folder)
        
        return summary_df.reset_index(drop=True)

    def check_model_complexity(self, estimators_arr):
        if hasattr(estimators_arr[0], "tree_"):
            total_num_nodes = 0
            for est in estimators_arr:
                 total_num_nodes += len(est.tree_.feature)
            model_complexity = total_num_nodes / len(estimators_arr)
        elif hasattr(estimators_arr[0], "get_num_nodes"):
            total_num_nodes = 0
            for est in estimators_arr:
                 total_num_nodes += est.get_num_nodes()
            model_complexity = total_num_nodes / len(estimators_arr)                    
        else:
            model_complexity = 0
        return model_complexity

    def get_previous_results(self, summary_df, partial_result_folder):
        # Load in partial the results from previous runs and combine the results
        prev_res = listdir(partial_result_folder)
        for f in prev_res:
            f = partial_result_folder + "\\" + f
            prev_df = pd.read_csv(f)
            summary_df = prev_df.append(summary_df)

        summary_df = summary_df.drop_duplicates(subset=["Dataset Index", 
                                                        "Dataset Version", 
                                                        "Model", 
                                                        "Feature Engineering Description", 
                                                        "Hyperparameter Description"], 
                                                keep="last")  
        try:
            shutil.rmtree(partial_result_folder)
        except:
            pass                                                         
        return summary_df
