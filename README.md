# DatasetsEvaluator
DatasetTester is a tool to collect datasets from openml.org and make it easier to test predictors (classifiers or regressors) against these files. Our hope is this eases the work required to test predictors and so encourages researchers to test predictors against larger numbers of datasets, taking greater advantage of the collection on openml.org. Ideally, this can lead to greater accuracy and reduced bias in the evaluation of ML tools. 

## Installation

`
pip install DatasetsEvaluator
`

    
## Examples

The tool works by calling a series of methods: First: find_datasets() (or find_by_name()). Second: collect_data(). And finally: run_tests(). For example:

```python
from DatasetsEvaluator import DatasetsEvaluator as de

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
```
This returns a pandas dataframe containing the list of datasets on openml.org matching the provided criteria. In this example, we're specifying datasets with between 500 and 5,000 rows, between 2 and 50 numeric columns, and so on.

The returned list may be examined and the parameters refined if desired. Alternatively, users may call datasets_tester.find_by_name() to specify a specific list of dataset names.

A call is then made such as:

    
```python
datasets_tester.collect_data()
```

This will return all datasets identified by the previous call to find_datasets() or find_by_name(). Alternatively, users may specify to return a subset of the datasets identified, for example:

```python
datasets_tester.collect_data(max_num_datasets_used=5, method_pick_sets='pick_first', keep_duplicated_names=False)
```

This collects the first 5 datasets found above. Note though, as keep_duplicated_names=False is specified, in cases where openml.org has multiple datasets with the same name, but different versions, only the last version will be collected.

A call to run_tests() may then be made to test one or more predictors on the collected datasets. For example:

```python
dt = tree.DecisionTreeRegressor(min_samples_split=50, max_depth=5, random_state=0)
knn = KNeighborsRegressor(n_neighbors=10)

summary_df = datasets_tester.run_tests(estimators_arr = [
                                        ("Decision Tree", "Original Features", "Default", dt),
                                        ("kNN", "Original Features", "Default", knn)],
                                       num_cv_folds=5,
                                       scoring_metric='r2',
                                       show_warnings=True) 

display(summary_df)
```

This compares the accuracy of the created decision tree and kNN classifiers on the collected datasets. 

An example notebook provides further examples. 

## Methods

## find_by_name()

```
find_by_name(names_arr, problem_type)
```
Identifies, but does not collect, the set of datasets meeting the specified set of names.

**Parameters**

**names_arr** : array of dataset names

**problem_type** : str

**Return Type**

A dataframe with a row for each dataset on openml meeting the specified set of names.

**Discussion**

problem_type must be either "classifiction" or "regression". All esimators will be compared using the same metric, so it is necessary that all datasets used are of the same type. 
---
## find_datasets()

```
find_datasets(   problem_type, 
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
                 max_num_categorical_features=50) 
```

This method collects the data from openml.org, unless check_local_cache is True and the dataset is avaialble 
in the local folder. This will collec the specifed subset of datasets identified by the most recent call 
to find_by_name() or find_datasets(). This allows users to call those methods until a suitable 
collection of datasets have been identified.

**Parameters**

**problem_type**: str

Either "classifiction" or "regression". All esimators will be compared using the same metric, so it is necessary that all datasets used are of the same type.

All other parameters are direct checks of the statistics about each dataset provided by openml.org.

**Return Type**

dataframe with a row for each dataset on openml meeting the specified set of criteria. 


---
## collect_datasets()

```
def collect_data(max_num_datasets_used=-1,
                 method_pick_sets="pick_first",
                 max_cat_unique_vals = 20,
                 keep_duplicated_names=False,
                 save_local_cache=False, 
                 check_local_cache=False, 
                 path_local_cache="",
                 preview_data=False)
```

#### Parameters
**max_num_datasets_used**: integer 
    
The maximum number of datasets to collect.

**method_pick_sets**: str
    
If only a subset of the full set of matches are to be collected, this identifies if those will be selected randomly, or simply using the first matches

**max_cat_unique_vals**: int
    
As categorical columns are one-hot encoded, it may not be desirable to one-hot encode categorical    columns with large numbers of unique values. Columns with a greater number of unique values than max_cat_unique_vals will be dropped. 

**keep_duplicated_names**: bool
    
If False, for each set of datasets with the same name, only the one with the highest version number will be used. 

**save_local_cache**: bool
    
If True, any collected datasets will be saved locally in path_local_cache

**check_local_cache**: bool
    
If True, before collecting any datasets from openml.org, each will be checked to determine if it is already stored locally in path_local_cache

**path_local_cache**: str

Folder identify the local cache of datasets, stored in .csv format.

**preview_data**: bool
    
Indicates if the first rows of each collected dataset should be displayed


**Return Type**

Returns reference to self.

**Discussion**

This drops any categorical columns with more than max_cat_unique_vals unique values. 
If keep_duplicated_names is False, then only one version of each dataset name is kept. This can reduce redundant test. In some cases, though, different versions of a dataset are significantly different. 

---

## run_tests()

```
run_tests(estimators_arr, num_cv_folds=5, scoring_metric='', show_warnings=False)
```

**Parameters**

**estimators_arr**: array of tuples, with each tuple containing: 
        
+ str: estimator name, 
+ str: a description of the features used
+ str: a description of the hyperparameters used
+ estimator: the estimator to be used. This should not be fit yet, just have the hyperparameters set.

**num_cv_folds**: (int)
    
The number of folds to be used in the cross validation process used to evaluate the predictor

**scoring_metric**: (str) 
    
One of the set of scoring metrics supported by sklearn. Set to '' to indicate to use the default. The default for classification is f1_macro and for regression is neg_root_mean_squared_error.

**show_warnings**: (bool) 
    
if True, warnings will be presented for calls to cross_validate(). These can get very long in in some     cases may affect only a minority of the dataset-predictor combinations, so is False by default. Users may wish to set to True to determine the causes of any NaNs in the final summary dataframe.   

**Return Type**

A dataframe summarizing the performance of the estimators on each dataset. There is one row for each combination of dataset and estimator. 

