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


## Multi log-files Data Processing

**Objective:** Process multiple log files concurrently.

```python
import re
from datetime import datetime
from pathlib import Path

from joblib import Parallel, delayed, parallel_config


def parse_log_line(log_line):
    log_pattern = r"\[(?P<datetime>.*?)\] (?P<level>\w+): (?P<message>.*)"
    log_match = re.match(log_pattern, log_line)

    log_datetime = datetime.strptime(log_match.group("datetime"), "%Y-%m-%d %H:%M:%S")
    log_level = log_match.group("level")
    log_message = log_match.group("message")
    return log_datetime, log_level, log_message


def process_log_file(log_file=None):
    with open(log_file, "r") as file:
        log_lines = file.readlines()
        with parallel_config():
            logs = Parallel(n_jobs=-1)(delayed(parse_log_line)(log_line) for log_line in log_lines)
        return logs


def glob_log_files(logs_dir=None):
    logs_dir_path = Path(logs_dir).expanduser().resolve()
    yield from logs_dir_path.glob("*.txt")


log_files = glob_log_files(logs_dir="./data/raw/logs")
with parallel_config():
    logs = Parallel(n_jobs=-1)(delayed(process_log_file)(log_file) for log_file in log_files)

print(logs)

```

**Tips:** Experiment with different parallel backends and data formats.


## Distributed Computing with Dask

**Objective:** Utilize `Dask` as a Joblib backend, to enable distributed computing capabilities.

```sh
pip install dask
```

```python
from dask.distributed import Client, LocalCluster
from joblib import Parallel, delayed, parallel_config


def square(x):
    return x**2


# See: https://docs.dask.org/en/stable/deploying.html#distributed-computing
if __name__ == "__main__":
    with LocalCluster() as cluster:
        with Client(cluster) as client:
            with parallel_config(backend="dask"):
                results = Parallel(verbose=10)(delayed(square)(i) for i in range(10))

    print(results)

```

**Tips:** Experiment with many ways to [deploy and run Dask clusters](https://docs.dask.org/en/stable/deploying.html#distributed-computing) and observe performance gains.


## What's Next

1. **Explore Advanced Joblib Features:** Delve deeper into Joblib's advanced features such as caching, lazy evaluation, and distributed computing for more complex tasks.

2. **Apply Joblib to Real-world Projects:** Implement Joblib in your own projects involving data processing, machine learning, or any CPU-intensive tasks to experience its benefits firsthand.

3. **Discover Related Libraries:** Explore other Python libraries for parallel computing and optimization, such as Dask or Multiprocessing, to broaden your toolkit.

4. **Stay Updated:** Keep an eye on Joblib's updates and enhancements in future releases to leverage the latest functionalities and optimizations.


## Gotchas

1. **Choose the Right Backend:** Select the appropriate Joblib backend based on your task and available resources. For CPU-bound tasks, `loky` or `multiprocessing` might be suitable. For I/O-bound tasks, `threading` or specific distributed computing backends like `dask` might be better.

2. **Optimal Number of Workers:** Experiment with the number of workers (`n_jobs`) to find the optimal configuration. Too many workers can lead to resource contention, while too few might underutilize resources.

3. **Data Transfer Overhead:** Minimize data transfer overhead between processes/threads. Large data transfers between parallel workers can become a bottleneck. Avoid unnecessary data sharing or copying if possible.

4. **Memory Consideration:** Be mindful of memory usage, especially when processing large datasets in parallel. Parallelism can increase memory consumption, potentially leading to resource contention or out-of-memory issues.

5. **Cleanup Resources:** Ensure proper cleanup of resources (e.g., closing files, releasing memory) after the parallel tasks complete to avoid resource leaks.

6. **Proper Error Handling:** Implement proper error handling mechanisms, especially when dealing with parallel tasks, to manage exceptions and prevent deadlocks or crashes.

7. **Benchmark and Profile:** Measure the performance of your parallelized code using benchmarking tools (`timeit`, `time`, etc.) to identify bottlenecks and areas for improvement.
