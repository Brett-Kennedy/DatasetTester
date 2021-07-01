import pandas as pd
import numpy as np
import openml
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import cross_validate
from statistics import mean, stdev
from warnings import filterwarnings, resetwarnings

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
            Either "classifiction" or "regression"         
            All esimators will be compared using the same metric, so it is necessary that all
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
                     min_num_classes=0,
                     max_num_classes=0,
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
            Either "classifiction" or "regression".        
            All esimators will be compared using the same metric, so it is necessary that all
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
                        (self.openml_df.MinorityClassSize <= max_num_minority_class) &
                        (self.openml_df.NumberOfFeatures >= min_num_features) & 
                        (self.openml_df.NumberOfFeatures <= max_num_features) &            
                        (self.openml_df.NumberOfInstances >= self.min_num_instances) & 
                        (self.openml_df.NumberOfInstances <= self.max_num_instances) &
                        (self.openml_df.NumberOfNumericFeatures >= min_num_numeric_features) &
                        (self.openml_df.NumberOfNumericFeatures <= max_num_numeric_features) &
                        (self.openml_df.NumberOfSymbolicFeatures >= min_num_categorical_features) &
                        (self.openml_df.NumberOfSymbolicFeatures <= max_num_categorical_features)
                        ]    
        else: # regression
            self.openml_df = self.openml_df[
                        (self.openml_df.NumberOfClasses == 0) &
                        (self.openml_df.NumberOfFeatures >= min_num_features) & 
                        (self.openml_df.NumberOfFeatures <= max_num_features) &            
                        (self.openml_df.NumberOfInstances >= self.min_num_instances) & 
                        (self.openml_df.NumberOfInstances <= self.max_num_instances) &
                        (self.openml_df.NumberOfNumericFeatures >= min_num_numeric_features) &
                        (self.openml_df.NumberOfNumericFeatures <= max_num_numeric_features) &
                        (self.openml_df.NumberOfSymbolicFeatures >= min_num_categorical_features) &
                        (self.openml_df.NumberOfSymbolicFeatures <= max_num_categorical_features)
                        ]    

        return self.openml_df
    
    def collect_data(self, 
                     max_num_datasets_used=-1,
                     method_pick_sets="pick_first",
                     max_cat_unique_vals = 20,
                     keep_duplicated_names=False,
                     save_local_cache=False, 
                     check_local_cache=False, 
                     path_local_cache="",
                     preview_data=False):
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
            self.openml_df = self.openml_df.drop_duplicates(subset=["name"],keep="last")           
        
        self.dataset_collection = []
        
        if max_num_datasets_used > 0 and max_num_datasets_used < len(self.openml_df):
            if method_pick_sets == "pick_first":
                openml_subset_df = self.openml_df.head(max_num_datasets_used)
            else:
                openml_subset_df = self.openml_df.sample(max_num_datasets_used, random_state=0)
        else:
            openml_subset_df = self.openml_df
        
        for dataset_idx in range(len(openml_subset_df)):
            dataset_did = int(openml_subset_df.iloc[dataset_idx].did)
            dataset_name = openml_subset_df.iloc[dataset_idx]['name']
            dataset_version = openml_subset_df.iloc[dataset_idx]['version']

            dataset_df = None
            if check_local_cache: 
                try: 
                    path_to_file = path_local_cache + "/" + dataset_name + '.csv'
                    X_with_y = pd.read_csv(path_to_file)
                    dataset_df = X_with_y.drop("y", axis=1)
                    y = X_with_y["y"]
                    #categorical_indicator = ['False']*dataset_df.shape[1] # todo: read the categorical_indicator from disk too
                    path_to_file = path_local_cache + "/" + dataset_name + '_cat_ind.csv'
                    categorical_indicator_df = pd.read_csv(path_to_file)
                    categorical_indicator = categorical_indicator_df['0'].tolist()
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
                dataset_df = self.__clean_dataset(dataset_df, categorical_indicator, max_cat_unique_vals)
                self.dataset_collection.append((dataset_name, dataset_version, dataset_df, y))
                if save_local_cache:
                    X_with_y = dataset_df.copy()
                    X_with_y['y'] = y 
                    X_with_y.to_csv(path_local_cache + "/" + dataset_name + '.csv')
                    categorical_indicator_df = pd.DataFrame(categorical_indicator)
                    categorical_indicator_df.to_csv(path_local_cache + "/" + dataset_name + '_cat_ind.csv')
            else:
                print(f" Error collecting file with did: {dataset_did}, name: {dataset_name}. Number rows in X: {len(X)}. Number rows in y: {len(y)}")
        
    def __clean_dataset(self, X, categorical_indicator, max_cat_unique_vals):
        # categorical_indicator provided by openml isn't 100% reliable, so we also check panda's is_numeric_dtype
        for c in range(len(X.columns)):
            if is_numeric_dtype(X[X.columns[c]]) == False:
                categorical_indicator[c]=True
        
        # One-hot encode the categorical columns
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
        X = X.fillna(0.0)
        X = X.replace([np.inf, -np.inf], 0.0)                        
        
        return X.reset_index()
    
    def run_tests(self, estimators_arr, num_cv_folds=5, scoring_metric='', show_warnings=False):
        """
        
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

        if scoring_metric == '':
            if self.problem_type == "classification": 
                scoring_metric = 'f1_macro'
            else: 
                scoring_metric = 'neg_root_mean_squared_error'

        # Dataframes used to store the test results
        column_names = ['Dataset',
                        'Dataset Version',
                        'Model',                                                          
                        'Feature Engineering Description',
                        'Hyperparameter Description',
                        'Avg ' + scoring_metric,
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

        print(f"\nRunning test on {len(self.dataset_collection)} datastets")
        for dataset_tuple in self.dataset_collection: 
            dataset_name, version, X, y = dataset_tuple
            print(f"Running tests on dataset: {dataset_name}")
            for estimator_desc in self.estimators_arr:
                model_name, engineering_description, hyperparameters_description, clf = estimator_desc
                print(f"\tRunning tests with model: {model_name} ({engineering_description}), ({hyperparameters_description})")
                # use scores['fit_time'] instead. take the average.
                scores = cross_validate(clf, X, y, cv=num_cv_folds, scoring=scoring_metric, return_train_score=True, return_estimator=True)
                test_scores = scores['test_score']
                train_scores = scores['train_score']
                avg_test_score = test_scores.mean()
                scores_std_dev = stdev(test_scores)
                avg_train_score = train_scores.mean()
                avg_fit_time = scores['fit_time'].mean()

                # Model Complexity is currently only supported for decision trees, and measures the number of nodes.
                estimators_arr = scores['estimator']
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

                summary_row = [dataset_name, 
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
        
        resetwarnings()
        return summary_df.reset_index()
