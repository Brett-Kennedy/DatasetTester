import pandas as pd
import numpy as np
import openml
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.pipeline import Pipeline
from statistics import stdev
from warnings import filterwarnings, resetwarnings
from time import time
from datetime import datetime
from os import mkdir, listdir
from shutil import rmtree
import concurrent
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Process, Queue


def get_single_dataset(q, dataset_did, dataset_name):
    dataset = openml.datasets.get_dataset(dataset_did)
    print(f" Loading {dataset_name} from openml.org")
    q.put(dataset)


class DatasetsTester:
    """
    Tool to compare predictors (classifiers or regressors) on a set of datasets collected from openml.org.

    This simplifies automatically comparing the performance of predictors on potentially large numbers
    of datasets, thereby supporting more thorough and accurate testing of predictors. 
    """

    # have the directories and problem type set here
    def __init__(self, problem_type, path_local_cache=""):
        """
        problem_type: str
            Either "classification" or "regression"
            All estimators will be compared using the same metric, so it is necessary that all
            datasets used are of the same type.

        path_local_cache: str
            Folder identify the local cache of datasets, stored in .csv format.
        """

        self.problem_type = problem_type
        self.path_local_cache = path_local_cache
        self.openml_df = None
        self.scoring_metric = ""

    def check_problem_type(self):
        problem_type_okay = self.problem_type in ["classification", "regression", "both"]
        if not problem_type_okay:
            print("problem_type must be one of: 'classification', 'regression', 'both'")
        return problem_type_okay

    def find_by_name(self, names_arr, use_cache=False):
        """
        Identifies, but does not collect, the set of datasets meeting the specified set of names.

        Parameters
        ----------
        names_arr: array of dataset names

        use_cache: bool
            If True, the local cache will be checked for the file.

        Returns
        -------
        dataframe with a row for each dataset on openml meeting the specified set of names. 

        """
        if not self.check_problem_type():
            return None
        #self.openml_df = openml.datasets.list_datasets(output_format="dataframe")
        self.__get_datasets_list(use_cache)
        self.openml_df = self.openml_df[self.openml_df.name.isin(names_arr)]
        return self.openml_df

    def find_by_tag(self, my_tag):
        """
        Identifies, but does not collect, the set of datasets attached to the specified tag.

        Parameters
        ----------
        my_tag: the dataset tag

        Returns
        -------
        dataframe with a row for each dataset on openml meeting the specified tag. 

        """
        if not self.check_problem_type():
            return None
        self.openml_df = openml.datasets.list_datasets(tag=my_tag, output_format="dataframe")
        return self.openml_df

    def find_datasets(self,
                      use_cache=True,
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
        This, find_by_name(), or find_by_tag() must be called to identify the potential set of datasets to be collected.

        Parameters
        ----------
        All other parameters are direct checks of the statistics about each dataset provided by openml.org.

        Returns
        -------
        dataframe with a row for each dataset on openml meeting the specified set of criteria.

        """

        if not self.check_problem_type():
            return None
        if self.problem_type == "classification" and (min_num_classes <= 0 or max_num_classes <= 0):
            print("For classification datasets, both min_num_classes and max_num_classes must be specified.")
            return None

        '''
        read_dataset_list = False  # Set True if manage to read from cache. Otherwise read from openml.org.
        if use_cache and self.path_local_cache != "":
            try:
                path_to_file = self.path_local_cache + "/dataset_list.csv"
                self.openml_df = pd.read_csv(path_to_file)
                read_dataset_list = True
            except Exception as e:
                if "No such file or directory:" not in str(e):
                    print(f" Error reading file: {e}")
                else:
                    print(" File not found in cache.")
        if not read_dataset_list:
            self.openml_df = openml.datasets.list_datasets(output_format="dataframe")
            if use_cache and self.path_local_cache != "":
                try:
                    mkdir(self.path_local_cache)
                except FileExistsError:
                    pass
                except Exception as e:
                    print(f"Error creating local cache folder: {e}")
                path_to_file = self.path_local_cache + "/dataset_list.csv"
                self.openml_df.to_csv(path_to_file)
        '''
        self.__get_datasets_list(use_cache)

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
            (self.openml_df.NumberOfFeatures >= min_num_features) &
            (self.openml_df.NumberOfFeatures <= max_num_features) &
            (self.openml_df.NumberOfInstances >= min_num_instances) &
            (self.openml_df.NumberOfInstances <= max_num_instances) &
            (self.openml_df.NumberOfNumericFeatures >= min_num_numeric_features) &
            (self.openml_df.NumberOfNumericFeatures <= max_num_numeric_features) &
            (self.openml_df.NumberOfSymbolicFeatures >= min_num_categorical_features) &
            (self.openml_df.NumberOfSymbolicFeatures <= max_num_categorical_features)
            ]

        if self.problem_type == "classification":
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

        if self.problem_type == "regression":
            self.openml_df = self.openml_df[self.openml_df.NumberOfClasses == 0]

        return self.openml_df

    def __get_datasets_list(self, use_cache):
        read_dataset_list = False  # Set True if manage to read from cache. Otherwise read from openml.org.
        if use_cache and self.path_local_cache != "":
            try:
                path_to_file = self.path_local_cache + "/dataset_list.csv"
                self.openml_df = pd.read_csv(path_to_file)
                read_dataset_list = True
            except Exception as e:
                if "No such file or directory:" not in str(e):
                    print(f" Error reading file: {e}")
                else:
                    print(" File not found in cache.")
        if not read_dataset_list:
            self.openml_df = openml.datasets.list_datasets(output_format="dataframe")
            if use_cache and self.path_local_cache != "":
                try:
                    mkdir(self.path_local_cache)
                except FileExistsError:
                    pass
                except Exception as e:
                    print(f"Error creating local cache folder: {e}")
                path_to_file = self.path_local_cache + "/dataset_list.csv"
                self.openml_df.to_csv(path_to_file)

    def collect_data(self,
                     max_num_datasets_used=-1,
                     method_pick_sets="pick_random",
                     shuffle_random_state=0,
                     exclude_list=None,
                     use_automatic_exclude_list=False,
                     max_cat_unique_vals=20,
                     keep_duplicated_names=False,
                     check_local_cache=False,
                     check_online=True,
                     save_local_cache=False,
                     preview_data=False,
                     one_hot_encode=True,
                     fill_nan_and_inf_zero=True,
                     verbose=False):
        """
        This method collects the data from openml.org, unless check_local_cache is True and the dataset is available
        in the local folder. This will collect the specified subset of datasets identified by the most recent call
        to find_by_name() or find_datasets(). This allows users to call those methods until a suitable 
        collection of datasets have been identified.

        Parameters
        ----------
        max_num_datasets_used: integer 
            The maximum number of datasets to collect.

        method_pick_sets: str
            If only a subset of the full set of matches are to be collected, this identifies if those
            will be selected randomly, or simply using the first matches

        shuffle_random_state: int
            Where method_pick_sets is "pick_random", this is used to shuffle the order of the datasets

        exclude_list: array
            list of names of datasets to exclude

        use_automatic_exclude_list: bool
            If set True, any files that can't be loaded will be appended to a list and subsequent calls will not attempt
            to load them. This may be set to save time. However, if there are errors simply due to internet problems or
            temporary issues, this may erroneously exclude some datasets.

        max_cat_unique_vals: int
            As categorical columns are one-hot encoded, it may not be desirable to one-hot encode categorical
            columns with large numbers of unique values. Columns with a greater number of unique values than
            max_cat_unique_vals will be dropped. 

        keep_duplicated_names: bool
            If False, for each set of datasets with the same name, only the one with the highest 
            version number will be used. In some cases, different versions of a dataset are significantly different.

        save_local_cache: bool
            If True, any collected datasets will be saved locally in path_local_cache

        check_local_cache: bool
            If True, before collecting any datasets from openml.org, each will be checked to determine if
            it is already stored locally in path_local_cache

        check_online: bool
            If True, openml.org may be checked for the dataset, unless check_local_cache is True and the dataset has
            been cached.

        preview_data: bool
            Indicates if the first rows of each collected dataset should be displayed

        one_hot_encode: bool
            If True, categorical columns are one-hot encoded. This is necessary for many types of predictor, but
            may be done elsewhere, for example in a pipeline passed to the run_tests() function.

        fill_nan_and_inf_zero: bool
            If True, all instances of NaN, inf and -inf are replaced with 0.0. Replacing these values with something 
            valid is necessary for many types of predictor, butmay be done elsewhere, for example in a pipeline passed 
            to the run_tests() function.

        verbose: bool
            If True, messages will be displayed indicating errors collecting any datasets. 

        Returns
        -------
        dataset_collection: dictionary containing: index in this collection, dataset_name, version, X, y 
        This method will attempt to collect as many datasets as specified, even where additional datasets must
        be examined. 
        """

        def append_auto_exclude_list(did):
            if not use_automatic_exclude_list:
                return
            auto_exclude_list.append(did)

        def read_auto_exclude_list():
            nonlocal auto_exclude_list
            if not use_automatic_exclude_list or self.path_local_cache == "":
                return
            try:
                path_to_file = self.path_local_cache + "/exclude_list.csv"
                auto_list_df = pd.read_csv(path_to_file)
            except Exception as e:
                print(f" Error reading file: {e}")
                return
            auto_exclude_list = auto_list_df['List'].tolist()

        def save_auto_exclude_list():
            nonlocal auto_exclude_list
            if not use_automatic_exclude_list or self.path_local_cache == "" or len(auto_exclude_list) == 0:
                return
            try:
                mkdir(self.path_local_cache)
            except FileExistsError:
                pass
            except Exception as e:
                print(f"Error creating local cache folder: {e}")
            path_to_file = self.path_local_cache + "/exclude_list.csv"
            pd.DataFrame({'List': auto_exclude_list}).to_csv(path_to_file)

        assert method_pick_sets in ['pick_first', 'pick_random']
        q = Queue()

        if self.openml_df is None or len(self.openml_df) == 0:
            print("Error. No datasets specified. Call find_datasets() or find_by_name() before collect_data().")
            return None

        if not keep_duplicated_names:
            self.openml_df = self.openml_df.drop_duplicates(subset=["name"], keep="last")

        self.dataset_collection = []

        if -1 < max_num_datasets_used < len(self.openml_df) and method_pick_sets == "pick_random":
            openml_subset_df = self.openml_df.sample(frac=1, random_state=shuffle_random_state)
        else:
            openml_subset_df = self.openml_df

        auto_exclude_list = []
        read_auto_exclude_list()
        usable_dataset_idx = 0
        for dataset_idx in range(len(openml_subset_df)):
            if (max_num_datasets_used > -1) and (len(self.dataset_collection) >= max_num_datasets_used):
                break

            dataset_did = int(openml_subset_df.iloc[dataset_idx].did)
            dataset_name = openml_subset_df.iloc[dataset_idx]['name']
            dataset_version = openml_subset_df.iloc[dataset_idx]['version']

            if not exclude_list is None and dataset_name in exclude_list:
                continue
            if dataset_did in auto_exclude_list:
                continue

            print(f"Collecting {usable_dataset_idx}: {dataset_name}")

            dataset_df = None
            dataset_source = ""
            if check_local_cache:
                try:
                    path_to_file = self.path_local_cache + "/" + dataset_name + '.csv'
                    X_with_y = pd.read_csv(path_to_file)
                    dataset_df = X_with_y.drop("y", axis=1)
                    y = X_with_y["y"]
                    dataset_source = "cache"
                except Exception as e:
                    if "No such file or directory:" not in str(e):
                        print(f" Error reading file: {e}")
                    else:
                        print(" File not found in cache.")
                    dataset_df = None

            if not check_online and dataset_df is None:
                continue

            if dataset_df is None:
                p = Process(target=get_single_dataset, name="get_single_dataset", args=(q, dataset_did, dataset_name))
                p.start()
                p.join(timeout=20)
                if q.empty():
                    print(f" Unable to collect {dataset_name} from openml.org")
                    append_auto_exclude_list(dataset_did)
                    continue
                dataset = q.get()

                try:
                    X, y, categorical_indicator, attribute_names = dataset.get_data(
                        dataset_format="dataframe",
                        target=dataset.default_target_attribute
                    )
                except Exception as e:
                    if verbose:
                        print(f" Error collecting file with did: {dataset_did}, name: {dataset_name}. Error: {e}")
                    append_auto_exclude_list(dataset_did)
                    continue
                if X is None or y is None:
                    if verbose:
                        print(f" Error collecting file with did: {dataset_did}, name: {dataset_name}. X or y is None")
                    append_auto_exclude_list(dataset_did)
                    continue
                dataset_df = pd.DataFrame(X, columns=attribute_names)

            if len(dataset_df) != len(y):
                if verbose:
                    print(f" Error collecting file with did: {dataset_did}, name: {dataset_name}. Number rows in X: {len(X)}. Number rows in y: {len(y)}")
                append_auto_exclude_list(dataset_did)
                continue

            if preview_data:
                print(dataset_df.head())

            if save_local_cache:
                try:
                    mkdir(self.path_local_cache)
                except FileExistsError:
                    pass
                except Exception as e:
                    print(f"Error creating local cache folder: {e}")

                X_with_y = dataset_df.copy()
                X_with_y['y'] = y
                X_with_y.to_csv(self.path_local_cache + "/" + dataset_name + '.csv', index=False)

            if (self.problem_type == "regression") and (is_numeric_dtype(y) == False):
                continue

            if dataset_source == "cache":
                print(f" Reading from local cache: {usable_dataset_idx}, id: {dataset_did}, name: {dataset_name}")
            else:
                print(f" Loaded dataset from openml: {usable_dataset_idx}, id: {dataset_did}, name: {dataset_name}")

            dataset_df = self.__clean_dataset(dataset_df, max_cat_unique_vals, one_hot_encode,
                                              fill_nan_and_inf_zero)
            self.dataset_collection.append({'Index': usable_dataset_idx,
                                            'Dataset_name': dataset_name,
                                            'Dataset_version': dataset_version,
                                            'X': dataset_df,
                                            'y': y})
            usable_dataset_idx += 1
        save_auto_exclude_list()

    def __clean_dataset(self, X, max_cat_unique_vals, one_hot_encode, fill_nan_and_inf_zero):

        # The categorical_indicator provided by openml isn't 100% reliable, so we also check panda's is_numeric_dtype
        categorical_indicator = [False] * len(X.columns)
        for c in range(len(X.columns)):
            if not is_numeric_dtype(X[X.columns[c]]):
                categorical_indicator[c] = True

        # Remove any NaN or inf values
        if fill_nan_and_inf_zero:
            for c_idx, col_name in enumerate(X.columns):
                if categorical_indicator[c_idx] == True:
                    if hasattr(X[col_name], "cat"):
                        X[col_name] = X[col_name].cat.add_categories("").fillna("")
                    else:
                        X[col_name] = X[col_name].fillna("")
                else:
                    X[col_name] = X[col_name].fillna(0.0)

        # One-hot encode the categorical columns
        if one_hot_encode:
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

        return X.reset_index(drop=True)

    def get_dataset_collection(self):
        return self.dataset_collection

    def get_dataset(self, dataset_name):
        """
        Returns a single dataset.

        Parameters
        ----------
        dataset_name: str
            The name as it appears in the openml list of datasets

        Returns
        -------
        Returns both the X and y. Returns the first match if multiple versions are present.
        """

        for dataset_dict in self.dataset_collection:
            # if dataset_name == dataset_name_:
            if dataset_dict['Dataset_name'] == dataset_name:
                return dataset_dict['X'], dataset_dict['y']
        return None, None

    def run_tests(self,
                  estimators_arr,
                  feature_selection_func=None,
                  num_cv_folds=5,
                  scoring_metric='',
                  show_warnings=False,
                  starting_point=0,
                  ending_point=np.inf,
                  partial_result_folder="",
                  results_folder="",
                  run_parallel=False):
        """
        Evaluate all estimators on all datasets. 
        
        Parameters
        ----------
        estimators_arr: array of tuples, with each tuple containing: 
            str: estimator name, 
            str: a description of the features used
            str: a description of the hyper-parameters used
            estimator: the estimator to be used. This should not be fit yet, just have the hyper-parameters set.

        feature_selection_func: function
            Optional function, used to select the features used. The specified function must accept the parameters
            X and y, representing the data in the form of a pandas dataframe and series, and must return a
            new version of X with the same or less columns.

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
            This may be used to divide up the datasets, potentially to spread the work over a period of time, or 
            to use some datasets purely for testing.

        partial_result_folder: string
            path to folder where partial results are saved. 

        results_folder: string
            path to folder where results are saved.         

        run_parallel: bool
            If set to True, the datasets will be tested in parallel. This speeds up computation, but is set to 
            False by default as it makes the print output harder to follow and the process of recovering from
            partial runs more complicated. 

        Returns
        -------
        a dataframe summarizing the performance of the estimators on each dataset. There is one row
        for each combination of dataset and estimator. 

        the name of the saved results if any were saved
        """

        self.estimators_arr = estimators_arr

        scoring_metric_specified = True
        if self.problem_type == "classification":
            if scoring_metric == '':
                scoring_metric_specified = False
                scoring_metric = 'f1_macro'
        elif self.problem_type == "regression":
            if scoring_metric == '':
                scoring_metric_specified = False
                scoring_metric = 'neg_root_mean_squared_error'
        else:
            assert False, "problem type must be 'classification' or 'regression' if running tests. "
        self.scoring_metric = scoring_metric

        # Dataframes used to store the test results
        column_names = ['Dataset Index',
                        'Dataset',
                        'Dataset Version',
                        'Model',
                        'Feature Engineering Description',
                        'Hyperparameter Description']
        if scoring_metric_specified == False and self.problem_type == "regression":
            column_names.append('Avg NRMSE')
        else:
            column_names.append('Avg ' + scoring_metric)
        column_names += [
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

        self.__create_folders(starting_point, ending_point, partial_result_folder, results_folder)

        print(f"\nRunning test on {len(self.dataset_collection)} datastets")

        if not run_parallel:
            summary_df = self.run_subset(summary_df,
                                         starting_point,
                                         ending_point,
                                         feature_selection_func,
                                         partial_result_folder,
                                         num_cv_folds,
                                         scoring_metric,
                                         scoring_metric_specified)
        else:
            ending_point = min(ending_point, len(self.dataset_collection) - 1)
            process_arr = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for dataset_idx in range(starting_point, ending_point + 1):
                    print(f"Starting process for dataset: {dataset_idx}")
                    f = executor.submit(self.run_subset,
                                        summary_df,
                                        dataset_idx,
                                        dataset_idx,
                                        feature_selection_func,
                                        partial_result_folder,
                                        num_cv_folds,
                                        scoring_metric,
                                        scoring_metric_specified)
                    process_arr.append(f)
                for f in process_arr:
                    summary_df = summary_df.append(f.result())

        resetwarnings()

        if starting_point > 0 and partial_result_folder != "":
            summary_df = self.__get_previous_results(summary_df, partial_result_folder)

        summary_file_name = ""
        if ending_point >= (len(self.dataset_collection) - 1) and results_folder != "" and len(summary_df) > 0:
            n = datetime.now()
            dt_string = n.strftime("%d_%m_%Y_%H_%M_%S")
            summary_file_name = "results_" + dt_string
            final_file_name = results_folder + "\\" + summary_file_name + ".csv"
            print(f"Writing results to {final_file_name}")
            summary_df.to_csv(final_file_name, index=False)
            self.__remove_partial_results(partial_result_folder)

        return summary_df.reset_index(drop=True), summary_file_name

    def run_subset(self, summary_df, starting_point, ending_point, feature_selection_func, partial_result_folder,
                   num_cv_folds, scoring_metric, scoring_metric_specified):
        for dataset_dict in self.dataset_collection:
            dataset_index = dataset_dict['Index']
            dataset_name = dataset_dict['Dataset_name']
            version = dataset_dict['Dataset_version']
            X = dataset_dict['X']
            y = dataset_dict['y']

            if feature_selection_func:
                X = feature_selection_func(X, y)

            # Normally the dataset_index values are sequential within the dataset_collection, but
            # this handles where they are not. 
            if dataset_index < starting_point:
                continue
            if dataset_index > ending_point:
                continue
            print(f"Running tests on dataset index: {dataset_index}, dataset: {dataset_name}")
            for estimator_desc in self.estimators_arr:
                model_name, engineering_description, hyperparameters_description, clf = estimator_desc
                print(
                    f"\tRunning tests with model: {model_name} ({engineering_description}), ({hyperparameters_description})")
                scores = cross_validate(clf, X, y, cv=num_cv_folds, scoring=scoring_metric, return_train_score=True,
                                        return_estimator=True)
                print(f"\tscores for {model_name}: {scores['test_score']}")
                train_scores = scores['train_score']
                test_scores = scores['test_score']
                if scoring_metric_specified == False and self.problem_type == "regression":
                    # Convert from neg_root_mean_squared_error to NRMSE
                    train_scores = abs(train_scores / (y.mean()))
                    test_scores = abs(test_scores / (y.mean()))
                avg_test_score = test_scores.mean()
                scores_std_dev = stdev(test_scores)
                avg_train_score = train_scores.mean()
                avg_fit_time = scores['fit_time'].mean()

                # Model Complexity is currently only supported for decision trees, and measures the number of nodes.
                estimators_arr = scores['estimator']
                if type(estimators_arr[0]) == Pipeline:
                    for p_idx in range(len(estimators_arr[0])):
                        short_estimators_arr = [e[p_idx] for e in estimators_arr]
                        model_complexity = self.__check_model_complexity(short_estimators_arr)
                        if model_complexity > 0:
                            break
                else:
                    model_complexity = self.__check_model_complexity(estimators_arr)

                summary_row = [dataset_index,
                               dataset_name,
                               version,
                               model_name,
                               engineering_description,
                               hyperparameters_description,
                               avg_test_score,
                               scores_std_dev,
                               avg_train_score - avg_test_score,
                               len(X.columns),
                               model_complexity,
                               avg_fit_time]
                summary_df = summary_df.append(pd.DataFrame([summary_row], columns=summary_df.columns))

            if (partial_result_folder != ""):
                intermediate_file_name = partial_result_folder + "\\intermediate_" + str(dataset_index) + ".csv"
                summary_df.to_csv(intermediate_file_name, index=False)

        return summary_df

    def run_tests_parameter_search(
            self,
            estimators_arr,
            parameters_arr,
            search_method='random',
            num_cv_folds=5,
            scoring_metric='',
            show_warnings=False,
            starting_point=0,
            ending_point=np.inf,
            partial_result_folder="",
            results_folder="",
            run_parallel=False):
        """
        Evaluate all estimators on all datasets. 
        
        Parameters
        ----------
        All parameters are the same as in run_tests() with the addition of:

        estimators_arr: <fill in> #todo: add all parameters to the docstring

        parameters_arr: array of dictionaries
            Each dictionary describes the range of parameters to be tested on the matching estimator

        search_method: str
            Either "grid" or "random"

        Returns
        -------
        a dataframe summarizing the performance of the estimators on each dataset. There is one row
        for each combination of dataset and estimator. 
        """

        assert search_method in ['grid', 'random']

        self.estimators_arr = estimators_arr
        self.parameters_arr = parameters_arr

        scoring_metric_specified = True
        if self.problem_type == "classification":
            if scoring_metric == '':
                scoring_metric_specified = False
                scoring_metric = 'f1_macro'
        elif self.problem_type == "regression":
            if scoring_metric == '':
                scoring_metric_specified = False
                scoring_metric = 'neg_root_mean_squared_error'
        else:
            assert False, "problem_type must be 'classification' or 'regression' to run tests."

        # Dataframes used to store the test results
        column_names = ['Dataset Index',
                        'Dataset',
                        'Dataset Version',
                        'Model',
                        'Feature Engineering Description',
                        'Hyperparameter Description']
        if not scoring_metric_specified and self.problem_type == "regression":
            column_names.append('NRMSE')
        else:
            column_names.append(scoring_metric)
        column_names += [
            'Train-Test Gap',
            '# Columns',
            'Model Complexity',
            'Fit Time',
            'Best Hyperparameters']
        summary_df = pd.DataFrame(columns=column_names)

        if show_warnings:
            filterwarnings('default')
        else:
            filterwarnings('ignore')

        self.__create_folders(starting_point, ending_point, partial_result_folder, results_folder)

        print(f"\nRunning test on {len(self.dataset_collection)} datastets")
        if not run_parallel:
            summary_df = self.run_subset_cv_parameter_search(
                summary_df,
                starting_point,
                ending_point,
                partial_result_folder,
                num_cv_folds,
                search_method,
                scoring_metric,
                scoring_metric_specified)
        else:
            ending_point = min(ending_point, len(self.dataset_collection) - 1)
            process_arr = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for dataset_idx in range(starting_point, ending_point + 1):
                    print(f"Starting process for dataset: {dataset_idx}")
                    f = executor.submit(self.run_subset_cv_parameter_search,
                                        summary_df,
                                        dataset_idx,
                                        dataset_idx,
                                        partial_result_folder,
                                        num_cv_folds,
                                        search_method,
                                        scoring_metric,
                                        scoring_metric_specified)
                    process_arr.append(f)
                for f in process_arr:
                    summary_df = summary_df.append(f.result())

        resetwarnings()

        if starting_point > 0 and partial_result_folder != "":
            summary_df = self.__get_previous_results(summary_df, partial_result_folder)

        summary_file_name = ""
        if ending_point >= (len(self.dataset_collection) - 1) and results_folder != "" and len(summary_df) > 0:
            n = datetime.now()
            dt_string = n.strftime("%d_%m_%Y_%H_%M_%S")
            summary_file_name = "results_" + dt_string
            final_file_name = results_folder + "\\" + summary_file_name + ".csv"
            print(f"Writing results to {final_file_name}")
            summary_df.to_csv(final_file_name, index=False)
            self.__remove_partial_results(partial_result_folder)

        return summary_df.reset_index(drop=True), summary_file_name

    def run_subset_cv_parameter_search(self,
                                     summary_df,
                                     starting_point,
                                     ending_point,
                                     partial_result_folder,
                                     num_cv_folds,
                                     search_method,
                                     scoring_metric,
                                     scoring_metric_specified):
        for dataset_dict in self.dataset_collection:
            dataset_index = dataset_dict['Index']
            dataset_name = dataset_dict['Dataset_name']
            version = dataset_dict['Dataset_version']
            X = dataset_dict['X']
            y = dataset_dict['y']
            if dataset_index < starting_point:
                continue
            if dataset_index > ending_point:
                continue
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
            print(f"Running tests on dataset: {dataset_index}: {dataset_name}")
            for estimator_idx, estimator_desc in enumerate(self.estimators_arr):
                model_name, engineering_description, hyperparameters_description, estimator = estimator_desc
                parameters = self.parameters_arr[estimator_idx]
                print(
                    f"\tRunning tests with model: {model_name} ({engineering_description}), ({hyperparameters_description})")
                if search_method == "grid":
                    gs_estimator = GridSearchCV(estimator, parameters, scoring=scoring_metric)
                elif search_method == "random":
                    gs_estimator = RandomizedSearchCV(estimator, parameters, scoring=scoring_metric, n_iter=10)
                start_time = time()
                gs_estimator.fit(X_train, y_train)
                end_time = time()
                y_pred_train = gs_estimator.predict(X_train)
                y_pred_test = gs_estimator.predict(X_test)

                if self.problem_type == "classification":
                    if scoring_metric == "f1_macro":
                        train_score = f1_score(list(y_pred_train), list(y_train), average="macro")
                        test_score = f1_score(list(y_pred_test), list(y_test), average="macro")
                    else:
                        assert False, "Only f1_macro currently supported."
                else:
                    if self.problem_type == "regression":
                        if scoring_metric_specified == False or scoring_metric == "neg_root_mean_squared_error" or scoring_metric == "NRMSE":
                            train_score = (-1) * mean_squared_error(y_train, y_pred_train)
                            test_score = (-1) * mean_squared_error(y_test, y_pred_test)
                            if not scoring_metric_specified:
                                # Convert from neg_root_mean_squared_error to NRMSE
                                train_score = abs(train_score / (y.mean()))
                                test_score = abs(test_score / (y.mean()))
                    else:
                        assert False, "Only NRMSE and neg_root_mean_squared_error currently supported,"

                print("\ttest_score: ", test_score)

                if type(gs_estimator.best_estimator_) == Pipeline:
                    for p_idx in range(len(gs_estimator.best_estimator_)):
                        est = gs_estimator.best_estimator_[p_idx]
                        model_complexity = self.__check_model_complexity([est])
                        if (model_complexity > 0):
                            break
                else:
                    model_complexity = self.__check_model_complexity([gs_estimator.best_estimator_])

                summary_row = [dataset_index,
                               dataset_name,
                               version,
                               model_name,
                               engineering_description,
                               hyperparameters_description,
                               test_score,
                               train_score - test_score,
                               len(X.columns),
                               model_complexity,
                               round(end_time - start_time, 2),
                               str(gs_estimator.best_params_)]
                summary_df = summary_df.append(pd.DataFrame([summary_row], columns=summary_df.columns))

            if partial_result_folder != "":
                intermediate_file_name = partial_result_folder + "\\intermediate_" + str(dataset_index) + ".csv"
                summary_df.to_csv(intermediate_file_name, index=False)

        return summary_df

    def summarize_results(self, summary_df, accuracy_metric="", saved_file_name="", results_folder="", show_std_dev=False):
        """
        Creates a 2nd results dataframe, summarizing the first data frame by dataset. This is returned and
        optionally written to disk. The returned dataframe has one row per model type, summarizing over all
        datasets for that model type.

        Parameters
        ----------
        summary_df: pandas dataframe
            The results found calling run_tests() or run_tests_parameter_search()

        accuracy_metric: str
            The accuracy metric used in summary_df. This is one of the column headings.

        saved_file_name: str
            The full path to the file saved in run_tests() or run_tests_parameter_search(). This will have been returned
            by the method

        results_folder: str
            path where the dataframe created here will be saved to disk

        show_std_dev: bool
            If True, the standard deviations between folds will be reported

        Returns
        -------
        The pandas dataframe created by this function.
        """

        if len(summary_df) == 0:
            return
        if accuracy_metric == "":
            accuracy_metric = "Avg " + self.scoring_metric
        g = summary_df.groupby(['Model', 'Feature Engineering Description', 'Hyperparameter Description'], sort=False)
        p = pd.DataFrame(g[accuracy_metric].mean())
        if show_std_dev:
            p['Avg. Std dev between folds'] = g['Std dev between folds'].mean()
        p['Avg. Train-Test Gap'] = g['Train-Test Gap'].mean()
        p['Avg. Fit Time'] = g['Fit Time'].mean()
        if g['Model Complexity'].sum().any() > 0.0:
            p['Avg. Complexity'] = g['Model Complexity'].mean()
        if saved_file_name and results_folder:
            results_summary_filename = results_folder + "\\" + saved_file_name + "_summarized" + ".csv"
            p.to_csv(results_summary_filename, index=True)
        return p

    def plot_results(self, summary_df, accuracy_metric="", saved_file_name="", results_folder=""):
        """
        Creates plots summarizing the results. This includes two plots:
        1) a pair of line graphs:
            a) a line graph comparing each detector by accuracy. The x-axis lists the datasets arranged from lowest
            to highest accuracy on the first detector; the y-axis indicates the accuracy in the specified metric.
            A line is drawn for each detector.

            b) a similar line graph comparing model complexity. This applies only to models such as decision trees
            where the complexity can be readily measured.

        2) a heatmap indicating how often each model is the most accurate and most interpretable. Where the complexity
            cannot be measured, this simply compares how often each model had the highest accuracy.

        summary_df: pandas dataframe
            The results found calling run_tests() or run_tests_parameter_search()

        accuracy_metric: str
            The accuracy metric used in summary_df. This is one of the column headings

        saved_file_name: str
            The full path to the file saved in run_tests() or run_tests_parameter_search(). This will have been returned
            by the method

        results_folder: str
            path where the dataframe created here will be saved to disk

        Returns
        -------
        None
        """

        def line_plot(summary_df):
            if len(summary_df) == 0:
                return

            sns.set()

            def set_axis(ax, labels):
                ax.xaxis.set_tick_params(rotation=90)
                ax.xaxis.set_ticks_position('bottom')
                ax.xaxis.set_tick_params(which='major', labelsize=6)
                ax.set_xticks(np.arange(0, len(labels)))
                ax.set_xticklabels(labels)
                ax.set_xlabel('Dataset', fontsize=5)

            # Collect the set of all combinations of model type and feature engineering. In this example, there
            # should just be the two.
            combinations_df = summary_df.groupby(
                ['Model', 'Feature Engineering Description', 'Hyperparameter Description'], sort=False).size().reset_index()

            summary_df = summary_df.dropna(subset=[accuracy_metric])

            fig_width = min(len(summary_df) / 4, 20)
            fig_width = max(fig_width, 5)
            show_complexity_plot = summary_df['Model Complexity'].sum() > 0
            fig, ax = plt.subplots(nrows=1+show_complexity_plot, ncols=1, figsize=(fig_width, 10))

            ax0 = ax
            if show_complexity_plot:
                ax0 = ax[0]

            # Draw a single plot, with a line for each feature engineering description. Along the x-axis we have each
            # dataset ordered by lowest to highest score when using the original features.
            for row_idx in range(len(combinations_df)):
                m = combinations_df.iloc[row_idx]['Model']
                f = combinations_df.iloc[row_idx]['Feature Engineering Description']
                h = combinations_df.iloc[row_idx]['Hyperparameter Description']

                # Get the subset of summary_df for the current feature engineering method. 
                if row_idx == 0:
                    subset_df_1 = summary_df[(summary_df['Model'] == m) &
                                             (summary_df['Feature Engineering Description'] == f) &
                                             (summary_df['Hyperparameter Description'] == h)
                                            ].sort_values(by=accuracy_metric).reset_index()
                    x_coords = subset_df_1.index
                    y_coords = subset_df_1[accuracy_metric]
                else:
                    subset_df_2 = summary_df[(summary_df['Model'] == m) &
                                             (summary_df['Feature Engineering Description'] == f) &
                                             (summary_df['Hyperparameter Description'] == h)]
                    y_coords = []
                    for i in range(len(subset_df_1)):
                        ds = subset_df_1.iloc[i]['Dataset']
                        y_coords.append(subset_df_2[subset_df_2['Dataset'] == ds][accuracy_metric])
                ax0.plot(list(x_coords), list(y_coords), label=m + "(" + f + ") (" + h + ")")
                #data = np.hstack((np.array(x_coords).reshape(-1, 1), np.array(y_coords).reshape(-1, 1)))
                #sns.lineplot(data=data, ax=ax0)
            ax0.set_title(accuracy_metric + " by dataset")

            if show_complexity_plot:
                for row_idx in range(len(combinations_df)):
                    m = combinations_df.iloc[row_idx]['Model']
                    f = combinations_df.iloc[row_idx]['Feature Engineering Description']
                    h = combinations_df.iloc[row_idx]['Hyperparameter Description']

                    subset_df_2 = summary_df[(summary_df['Model'] == m) &
                                             (summary_df['Feature Engineering Description'] == f) &
                                             (summary_df['Hyperparameter Description'] == h)]
                    y_coords = []
                    for i in range(len(subset_df_1)):
                        ds = subset_df_1.iloc[i]['Dataset']
                        y_coords.append(subset_df_2[subset_df_2['Dataset'] == ds]['Model Complexity'])
                    ax[1].plot(x_coords, y_coords, label=m + "(" + h + ")")
                ax[1].set_title("Model Complexity by dataset")

            t = subset_df_1['Dataset']
            t = [x[:15] + "..." if len(x) > 15 else x for x in t]
            ax0.legend(bbox_to_anchor=(1.05, 1))
            #for ax_ in ax:
            #    set_axis(ax_, t)
            set_axis(ax[1], t)

            plt.subplots_adjust(hspace=0.5)
            if saved_file_name and results_folder:
                results_plot_filename = results_folder + "\\" + saved_file_name + "_plot" + ".png"
                fig.savefig(results_plot_filename, bbox_inches='tight', dpi=150)

        def heatmap_plot(summary_df):
            g = summary_df.groupby('Dataset')
            # Matrix containing a count of how often each estimator is the best in terms of accuracy & interpretability
            output_matrix = None
            names_arr = []
            sns.set_style('white')

            for file_name, group in g:
                # Each group represents a set of rows in summary_df for a given dataset. Each group covers the same set
                # of estimators being compared. For the first group, get the list of estimators and initialize the
                # output_matrix
                if output_matrix == None:
                    output_matrix = [[]] * len(group)
                    for i in range(len(group)):
                        output_matrix[i] = [0] * len(group)

                    for i in range(len(group)):
                        row = summary_df.iloc[i]
                        model_name = row['Model']
                        eng_name = row['Feature Engineering Description']
                        param_name = row['Hyperparameter Description']
                        name = model_name
                        if eng_name:
                            name += "(" + eng_name + ")"
                        if param_name:
                            name += "(" + param_name + ")"
                        names_arr.append(name)

                acc_col = summary_df.loc[group.index].loc[:, accuracy_metric].astype(float).reset_index(drop=True)
                max_acc_idx = acc_col.idxmax()

                int_col = summary_df.loc[group.index].loc[:, "Model Complexity"].astype(float).reset_index(drop=True)
                min_compl_idx = int_col.idxmin()

                output_matrix[min_compl_idx][max_acc_idx] += 1

            if not output_matrix:
                return

            fig, ax = plt.subplots()
            ax.imshow(output_matrix, cmap='Blues', interpolation="nearest")

            # We want to show all ticks...
            ax.set_xticks(np.arange(len(names_arr)))
            ax.set_yticks(np.arange(len(names_arr)))
            # ... and label them with the respective list entries
            ax.set_xticklabels(names_arr)
            ax.set_yticklabels(names_arr)

            # Let the horizontal axes labeling appear on top.
            ax.tick_params(top=True, bottom=False,
                           labeltop=True, labelbottom=False)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                     rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            for i in range(len(names_arr)):
                for j in range(len(names_arr)):
                    color = 'w'
                    if output_matrix[i][j] == 0: color = 'b'
                    ax.text(j, i, output_matrix[i][j], ha="center", va="center", color=color)

            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Interpretability")
            ax.set_title("Counts of Best Models with respect to Accuracy and Interpretability")

            # Turn spines off and create white grid.
            ax.spines[:].set_visible(False)

            ax.set_xticks(np.arange(len(output_matrix) + 1) - .5, minor=True)
            ax.set_yticks(np.arange(len(output_matrix) + 1) - .5, minor=True)
            ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
            ax.tick_params(which="minor", bottom=False, left=False)

            if saved_file_name and results_folder:
                results_plot_filename = results_folder + "\\" + saved_file_name + "_heatmap" + ".png"
                fig.savefig(results_plot_filename, bbox_inches='tight', dpi=150)

        if accuracy_metric == "":
            accuracy_metric = "Avg " + self.scoring_metric

        line_plot(summary_df)
        heatmap_plot(summary_df)

    def __check_model_complexity(self, estimators_arr):
        if hasattr(estimators_arr[0], "get_model_complexity"):
            total_model_complexity = 0
            for est in estimators_arr:
                total_model_complexity += est.get_model_complexity()
            model_complexity = total_model_complexity / len(estimators_arr)
        elif hasattr(estimators_arr[0], "get_num_nodes"):
            total_num_nodes = 0
            for est in estimators_arr:
                total_num_nodes += est.get_num_nodes()
            model_complexity = total_num_nodes / len(estimators_arr)
        elif hasattr(estimators_arr[0], "tree_"):
            total_num_nodes = 0
            for est in estimators_arr:
                total_num_nodes += len(est.tree_.feature)
            model_complexity = total_num_nodes / len(estimators_arr)
        elif hasattr(estimators_arr[0], "coef_"):
            model_complexity = 0
            for est in estimators_arr:
                coef_arr = est.coef_
                if est.coef_.ndim > 1:
                    coef_arr = est.coef_[0]
                non_zero_coefs = [x for x in coef_arr if x != 0]
                model_complexity += len(non_zero_coefs)
            model_complexity = model_complexity / len(estimators_arr)
        else:
            model_complexity = 0
        return model_complexity

    def __create_folders(self, starting_point, ending_point, partial_result_folder, results_folder):
        if starting_point == 0 and partial_result_folder != "":
            try:
                mkdir(partial_result_folder)
            except FileExistsError:
                pass
            except Exception as e:
                print(f"Error creating partial results folder: {e}")

        if ending_point >= len(self.dataset_collection) and results_folder != "":
            try:
                mkdir(results_folder)
            except FileExistsError:
                pass
            except Exception as e:
                print(f"Error creating partial results folder: {e}")

    def __get_previous_results(self, summary_df, partial_result_folder):
        # Load in partial the results from previous runs and combine the results
        prev_res = listdir(partial_result_folder)
        for f in prev_res:
            f = partial_result_folder + "\\" + f
            prev_df = pd.read_csv(f)
            prev_df['Feature Engineering Description'] = prev_df['Feature Engineering Description'].fillna("")
            prev_df['Hyperparameter Description'] = prev_df['Hyperparameter Description'].fillna("")
            summary_df = prev_df.append(summary_df)

        summary_df = summary_df.drop_duplicates(subset=["Dataset Index",
                                                        "Dataset Version",
                                                        "Model",
                                                        "Feature Engineering Description",
                                                        "Hyperparameter Description"],
                                                keep="last")
        try:
            rmtree(partial_result_folder)
        except Exception as e:
            print("Error deleting partial results folder: {e}")
        return summary_df

    def __remove_partial_results(self, partial_result_folder):
        if partial_result_folder == "":
            return
        try:
            rmtree(partial_result_folder)
        except Exception as e:
            print(f"Error deleting partial results folder: {partial_result_folder}. Error: {e}")
