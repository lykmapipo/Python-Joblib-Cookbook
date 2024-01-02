# Python Joblib Cookbook

A step-by-step guide to master various aspects of [Joblib](https://github.com/joblib/joblib), and utilize its functionalities for parallel computing and task handling in Python.


## Requirements

- [Python 3.8+](https://www.python.org/)
- [joblib 1.3+](https://github.com/joblib/joblib)
- [numpy 1.24+](https://github.com/numpy/numpy)
- [scikit-learn 1.3+](https://github.com/scikit-learn/scikit-learn)


## Installing Joblib

**Objective:** Learn how to install Joblib using pip.

```sh
pip install joblib
```

**Tips:** Ensure the appropriate Python environment is activated before running the installation command.


## Basic Usage

**Objective:** Understand the fundamental usage of Joblib for parallelizing functions.

```python
import joblib


def square(x):
    return x**2


results = joblib.Parallel(n_jobs=-1)(joblib.delayed(square)(i) for i in range(10))

print(results)

```

**Tips:** Experiment with different functions to parallelize and observe the behavior with varying inputs.


## Parallelizing a For Loop

**Objective:** Parallelize a for loop using Joblib.

```python
import joblib


def process_item(item):
    return item**2


items = list(range(10))

results = joblib.Parallel(n_jobs=-1)(joblib.delayed(process_item)(item) for item in items)

print(results)
```

**Tips:** Adjust the number of items in the list and observe performance changes when parallelizing.


### Memoizing a Function Results

**Objective:** Use Joblib's `Memory` to cache function results and speed up repeated computations.

```python
import time

import joblib

mem = joblib.Memory("./tmp/cache", verbose=0)


@mem.cache
def process_item(item):
    return item**2


items = list(range(100))

start = time.time()
results = joblib.Parallel(n_jobs=-1)(joblib.delayed(process_item)(item) for item in items)
stop = time.time()

print(results)
print("Elapsed time for the entire processing: {:.2f} s".format(stop - start))

```

**Tips:** Adjust the number of items in the list and observe performance changes when caching.


## Memory Mapping Large Arrays

**Objective:** Use memory mapping with Joblib for handling large arrays efficiently.

```python
import joblib
import numpy as np

data = np.random.rand(1000, 1000)
filename = "./tmp/large_array.dat"

joblib.dump(data, filename, compress=3, protocol=4)
loaded_data = joblib.load(filename)

print(loaded_data)

```

**Tips:** Experiment with different compression levels and pickle protocols for optimization.


## Customizing Joblib Parallel Backend

**Objective:** Customize Joblib's parallel backend for specific requirements.

```python
import joblib


def square(x):
    return x**2


with joblib.parallel_config(backend="threading", n_jobs=2):
    results = joblib.Parallel()(joblib.delayed(square)(i) for i in range(10))

print(results)

```

**Tips:** Explore different parallel backends and adjust the number of jobs for performance comparison.


## Parallelizing Machine Learning Training

**Objective:** Parallelize machine learning model training using Joblib.

```python
import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)

with joblib.parallel_config(backend="loky", n_jobs=-1):
    clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")

```

**Tips:** Experiment with different machine learning models and datasets to observe performance gains.


## What's Next

1. **Explore Advanced Joblib Features:** Delve deeper into Joblib's advanced features such as caching, lazy evaluation, and distributed computing for more complex tasks.

2. **Apply Joblib to Real-world Projects:** Implement Joblib in your own projects involving data processing, machine learning, or any CPU-intensive tasks to experience its benefits firsthand.

3. **Discover Related Libraries:** Explore other Python libraries for parallel computing and optimization, such as Dask or Multiprocessing, to broaden your toolkit.

4. **Stay Updated:** Keep an eye on Joblib's updates and enhancements in future releases to leverage the latest functionalities and optimizations.
