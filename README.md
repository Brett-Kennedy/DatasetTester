# DatasetsEvaluator
DatasetTester is a tool to collect datasets from openml.org and make it easier to test predictors (classifiers or regressors) against these files. Our hope is this eases the work required to test predictors and so encourages researchers to test predictors against larger numbers of datasets, taking greater advantage of the collection on openml.org. Ideally, this can lead to greater accuracy and reduced bias in the evaluation of ML tools. Ideally, this will support making the testing of predictors more consitent and more objective. 

The tool also allows researchers, to work with a large number of datasets, such that separate datasets may be used for training and testing, allowing a higher level of separation than most current methods, which maintain a holdout test set, or use cross validation, with each dataset. For example, a set of datasets may be used to determine good default hyperparameters for a tool, while a completely separate set of datasets may evaluate these. 

## Installation

`
pip install DatasetsEvaluator
`

    
## Examples

The tool works by calling a series of methods: First: find_datasets() (or find_by_name()). Second: collect_data(). And finally: run_tests(). For example:

```python
from DatasetsEvaluator import DatasetsEvaluator as de
import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from IPython.display import display

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

The returned list may be examined and the parameters refined if desired. 

Alternatively, users may call datasets_tester.find_by_name() or datasets_tester.find_by_tag() to request a list of specific dataset names or tags as returned by the OpenML API.

A call is then made, such as:

    
```python
datasets_tester.collect_data()
```

This will return all datasets identified by the previous call to find_datasets(), find_by_tag(), or find_by_name(). Alternatively, users may specify to return a subset of the datasets identified, for example:

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

An example notebook and example .py file (TestMultiprocessing.py) provide further examples. 

## Example Files

Two example files are provided

**DatasetTester** is a notebook that provides basic examples of using the tool. This includes examples collecting datasets, running tests, and plotting the results. 

**TestMultiProcessing** is a python file that evaluates running the tests in parallel, which can provide a quicker evaluation where many datasets are used. Running run_tests_parameter_search() is more expensive than run_tests() and can benefit more from parallel execution. 

## Methods

### find_by_name()

```
find_by_name(names_arr)
```
Identifies, but does not collect, the set of datasets meeting the specified set of names. In many cases, multiple versions of the same file may be returned. 

**Parameters**

**names_arr** : array of dataset names

**Return Type**

A dataframe with a row for each dataset on openml meeting the specified set of names.
##

### find_by_tag()

```
find_by_tag(my_tag)
```
Identifies, but does not collect, the set of datasets attached to the specified tag.

**Parameters**

**my_tag** : a dataset tag

**Return Type**

A dataframe with a row for each dataset on openml meeting the specified set of names.
##

### find_datasets()

```
find_datasets(   use_cache=True,
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
in the local folder. This will collect the specifed subset of datasets identified by the most recent call 
to find_by_name() or find_datasets(). This allows users to call those methods until a suitable 
collection of datasets have been identified.

**Parameters**

**use_cache**: bool

If set True, the local cache will be searched first to find the complete list of datasets available and the returned set will be a subset of this. If a cached file can not be found, openml.org will be queried. If set False, the local cache will not be examined.

**Other Parameters**

All other parameters are direct checks on the properties of the datasets returned by openml.org.

**Return Type**

dataframe with a row for each dataset on openml meeting the specified set of criteria. 


---
### collect_datasets()

```
def collect_data(max_num_datasets_used=-1,
                 method_pick_sets="pick_random",
                 shuffle_random_state=0,
                 max_cat_unique_vals = 20,
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
```

This method reads in the datasets matching the parameters here and in the previous call to find_by_name(), find_by_tag(), or find_datasets(). These are stored by the DatasetTester object and are used for subsequent calls to run_tests() and run_tests_grid_search(). This method provides options for simple preprocessing of the data, such as filling NaN values and removing infinite values. However, for most preprocessing, such as scaling, imputing missing values, feature selection, etc, pipelines should be passed to the run_tests() and run_tests_grid_search() methods as opposed to plain predictors. Examples are provided in the example notebook. 

#### Parameters
**max_num_datasets_used**: integer 
    
The maximum number of datasets to collect.

**method_pick_sets**: str
    
Either 'pick_first' or 'pick_random'. If only a subset of the full set of matches are to be collected, this identifies if those will be selected randomly, or simply using the first matches. Using 'pick_random' is preferred, as it removes any bias towards selecting files towards the beginning of the collection. 

**max_cat_unique_vals**: int
    
As categorical columns are one-hot encoded, it may not be desirable to one-hot encode categorical    columns with large numbers of unique values. Columns with a greater number of unique values than max_cat_unique_vals will be dropped. 

**keep_duplicated_names**: bool
    
If False, for each set of datasets with the same name, only the one with the highest version number will be used. 

**save_local_cache**: bool
    
If True, any collected datasets will be saved locally in path_local_cache.

**check_local_cache**: bool
    
If True, before collecting any datasets from openml.org, each will be checked to determine if it is already stored locally in path_local_cache.

**path_local_cache**: str

Folder identify the local cache of datasets, stored in .csv format.

**preview_data**: bool
    
Indicates if the first rows of each collected dataset should be displayed.


#### Return Type**

Returns reference to self.

**Discussion**

This drops any categorical columns with more than max_cat_unique_vals unique values. 
If keep_duplicated_names is False, then only one version of each dataset name is kept. This can reduce redundant test. In some cases, though, different versions of a dataset are significantly different. 
##

### run_tests()

```
run_tests(
    estimators_arr,
    num_cv_folds=5,
    scoring_metric='',
    show_warnings=False,
    starting_point=0,
    ending_point=np.inf,
    partial_result_folder="",
    results_folder="",
    run_parallel=False)
```
This allows faster evaluation where multiprocessing is enabled, but this does make the screen output more difficult to follow, and makes it more difficult to resume from partial results if an earlier test failed part way through, as the set of tests completed may have gaps. 

#### Parameters

**estimators_arr**: array of tuples, with each tuple containing: 
        
+ str: estimator name, such as "Decision Tree" or "kNN".
+ str: a description of the features used, such as "Original Features".
+ str: a description of the hyperparameters used. For "Decision Tree", "min_samples_split=X, max_depth=X". For "kNN", "n_neighbors=X".
+ estimator: the estimator to be used. This should not be fit yet, just have the hyperparameters set as dt_X or kNN_X.

**num_cv_folds**: int
    
The number of folds to be used in the cross validation process used to evaluate the predictor.

**scoring_metric**: str
    
One of the set of scoring metrics supported by sklearn. Set to '' to indicate to use the default. The default for classification is f1_macro and for regression is neg_root_mean_squared_error.

**show_warnings**: bool
    
if True, warnings will be presented for calls to cross_validate(). These can get very long in in some     cases may affect only a minority of the dataset-predictor combinations, so is False by default. Users may wish to set to True to determine the causes of any NaNs in the final summary dataframe.   

**starting_point**: int

This may be used to resume long-running tests where previous runs have not completed the full test or
            where previous calls to this method set ending_point

**ending_point**: int
            
This may be used to divide up the datasets, potentially to spread the work over a period of time, or 
            to use some datasets purely for testing.

**partial_result_folder**: string

path to folder where partial results are saved. 

**results_folder**: string

path to folder where results are saved.         

**run_parallel**: bool

If set to True, the datasets will be tested in parallel. This speeds up computation, but is set to 
            False by default as it makes the print output harder to follow and the process of recovering from
            partial runs more complicated. The TestMultiProcessing.py file provides an example of its use. 



#### Return Type

A dataframe summarizing the performance of the estimators on each dataset. There is one row for each combination of dataset and estimator. 
##

### run_tests_parameter_search()

```
run_tests_parameter_search(
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
    run_parallel=False)
```

This may be run as an alternative to run_tests(), which assumes a certain set of hyperparameter for each predictor. This instead uses either grid search or random search to attempt to identify the best hyperparamters for each predictor, which may provide a fairer comparison between them. The majority of the parameters are the same as with run_tests() with the exceptions of the following:

**parameters_arr**: dictionary

Each dictionary describes the range of parameters to be tested on the matching estimator

**search_method**: str

Must be either "grid" or "random". This determines how the hyperparameter search is conducted.
##

### summarize_results()

```
summarize_results(
    summary_df, 
    accuracy_metric, 
    saved_file_name="", 
    results_folder="", 
    show_std_dev=False):
```

Returns a pandas dataframe summarizing the results, providing an overview of how each predictor did, averaged over all datasets used. 

##

### plot_results()
```
plot_results(  
    summary_df, 
    accuracy_metric, 
    saved_file_name="", 
    results_folder="")
```

Returns an image of a plot summarizing the results, providing an overview of how each predictor did. The x-axis lists all datasets, ordered by the permance of the first predictor. For this purpose, it is recommended the first specified detector act as a baseline against which to compare the other detectors. A line is drawn for each detector, plotting it's score for each dataset. 

##



