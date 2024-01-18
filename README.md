# Python Joblib Cookbook

A step-by-step guide to master various aspects of [Joblib](https://github.com/joblib/joblib), and utilize its functionalities for parallel computing and task handling in Python.


## Requirements

- [Python 3.8+](https://www.python.org/)
- [pip 23.3+](https://github.com/pypa/pip)
- [joblib 1.3+](https://github.com/joblib/joblib)
- [numpy 1.24+](https://github.com/numpy/numpy)
- [scikit-learn 1.3+](https://github.com/scikit-learn/scikit-learn)
- [dask 2023.5+](https://github.com/dask/dask)
- [ray 2.9+](https://github.com/ray-project/ray)


---

## Installing Joblib

**Objective:** Learn how to install and verify Joblib using `pip`.

```sh
pip install joblib
```

```sh
pip show joblib
```

**Tips:**

- Ensure the appropriate [Python virtual environment](https://docs.python.org/3/library/venv.html) is activated before running the installation command.

- Ensure [pip](https://pip.pypa.io/en/stable/installation/) is installed before before running the installation command.

- If you want use [docker](https://www.docker.com/) run:

```sh
docker build -t python-joblib-cookbook:3.8-slim-bookworm .

docker run -it --rm \
    -v $(pwd)/data:/python-joblib-cookbook/data \
    -v $(pwd)/tmp:/python-joblib-cookbook/tmp \
    -v $(pwd)/scripts:/python-joblib-cookbook/scripts\
    python-joblib-cookbook:3.8-slim-bookworm

```


---

## Basic Usage

**Objective:** Understand the fundamental usage of Joblib for parallelizing functions.

```python
from joblib import Parallel, delayed


def square(x):
    return x**2


results = Parallel(n_jobs=-1, verbose=50)(delayed(square)(i) for i in range(10))

print(results)

```

**Tips:**

- Adjust the `n_jobs` to `0, 1, etc`, to control the number of parallel jobs (`-1` uses all available `cpu cores`)

- Adjust the `vebosity` to `0, 1, 2, 3, 10, 50 etc.`, to control level of progress messages that are printed.


---

## Basic Configuration
**Objective:** Understand how to configure Joblib (i.e to set `backend`, `n_jobs`, `verbose` etc).

```python
from joblib import Parallel, delayed, parallel_config


def square(x):
    return x**2


with parallel_config(backend="loky", n_jobs=-1, verbose=50):
    results = Parallel()(delayed(square)(i) for i in range(10))

print(results)

```

**Tips:**

- It is particularly useful (recommended) to use `parallel_config` when configuring joblib, especially when using libraries (e.g [scikit-learn](https://github.com/scikit-learn/scikit-learn)) that uses joblib internally.

- `backend` specifies the parallelization backend to use. By default, available backends are `loky`, `threading` and `multiprocessing`. Custom backends i.e `Dask`, `Ray` etc., need to be registered before usage.

- `n_jobs` specifies the maximum number of parallel jobs. If `-1` all CPU cores are used.

- `verbose` specifies level of progress messages to be printed, when executiong the jobs.


---

## Parallelizing a For Loop

**Objective:** Parallelize a for loop using Joblib.

```python
from joblib import Parallel, delayed, parallel_config


def process_item(item):
    return item**2


items = list(range(10))

with parallel_config(backend="loky", n_jobs=-1, verbose=50):
    results = Parallel()(delayed(process_item)(item) for item in items)

print(results)

```

**Tips:**

- Adjust the number of items in the list and observe performance changes when parallelizing.


---

## Memoizing a Function Results

**Objective:** Use Joblib's `Memory` to cache function results and speed up repeated computations.

```python
from joblib import Memory, Parallel, delayed, parallel_config

mem = Memory("./tmp/cache", verbose=10)


@mem.cache
def process_item(item):
    return item**2


items = list(range(100))

with parallel_config(backend="loky", n_jobs=-1, verbose=50):
    results = Parallel()(delayed(process_item)(item) for item in items)

print(results)


```

**Tips:**

- Adjust the number of items in the list, re-run the codes and observe performance changes when caching.

- Adjust `Memory` verbose level to `0, 2, 10, 50 etc.` to see if cached results are used.


---

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

**Tips:**

- Experiment with different compression levels and pickle protocols for optimization.


---

## Customizing Joblib Parallel Backend

**Objective:** Customize Joblib's parallel backend for specific requirements.

```python
from joblib import Parallel, delayed, parallel_config


def square(x):
    return x**2


with parallel_config(backend="threading", n_jobs=-1, verbose=50):
    results = Parallel()(delayed(square)(i) for i in range(10))

print(results)

```

**Tips:**

- Explore different parallel backends and adjust the number of jobs for performance comparison.


---


## Exception Handling
**Objective:** Implement proper exception handling for parallelized tasks.

```python
from joblib import Parallel, delayed, parallel_config


def divide(x, y):
    try:
        result = x / y
    except ZeroDivisionError:
        result = float("nan")
    return result


data = [(1, 2), (3, 0), (5, 2)]

with parallel_config(backend="loky", n_jobs=-1, verbose=50):
    results = Parallel()(delayed(divide)(x, y) for x, y in data)

print(results)

```

**Tips:**

- Ensure proper error handling within the parallelized function.


---

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


with joblib.parallel_config(backend="loky", n_jobs=-1, verbose=50):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=50)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")

```

**Tips:**

- Experiment with different machine learning models and datasets to observe performance gains.


---

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
        with parallel_config(backend="threading", n_jobs=-1, verbose=50):
            logs = Parallel()(delayed(parse_log_line)(log_line) for log_line in log_lines)
        return logs


def glob_log_files(logs_dir=None):
    logs_dir_path = Path(logs_dir).expanduser().resolve()
    yield from logs_dir_path.glob("*.txt")


log_files = glob_log_files(logs_dir="./data/raw/logs")
with parallel_config(backend="loky", n_jobs=-1, verbose=50):
    logs = Parallel()(delayed(process_log_file)(log_file) for log_file in log_files)

print(logs)

```

**Tips:**

- Experiment with different parallel backends and data formats.


---

## Distributed Computing with Dask

**Objective:** Utilize `Dask` as a Joblib backend, to enable distributed computing capabilities.

```sh
pip install dask distributed
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
            with parallel_config(backend="dask", n_jobs=-1, verbose=50):
                results = Parallel()(delayed(square)(i) for i in range(10))

    print(results)

```

**Tips:**

- Experiment with many ways to [deploy and run Dask clusters](https://docs.dask.org/en/stable/deploying.html#distributed-computing) and observe performance gains.


---

## Distributed Computing with Ray

**Objective:** Utilize `Ray` as a Joblib backend, to enable distributed computing capabilities.

```sh
pip install ray
```

```python
from joblib import Parallel, delayed, parallel_config
from ray.util.joblib import register_ray


def square(x):
    return x**2


# Register Ray Backend to be called with parallel_config(backend="ray")
register_ray()

# See: https://docs.ray.io/en/latest/ray-core/walkthrough.html
if __name__ == "__main__":
    with parallel_config(backend="ray", n_jobs=-1, verbose=50):
        results = Parallel()(delayed(square)(i) for i in range(10))

    print(results)

```

**Tips:**

- Experiment with many ways to [deploy and run Ray clusters](https://docs.ray.io/en/latest/cluster/getting-started.html) and observe performance gains.


---

## What's Next

1. **Explore Advanced Joblib Features:** Delve deeper into Joblib's advanced features such as caching, lazy evaluation, and distributed computing for more complex tasks.

2. **Apply Joblib to Real-world Projects:** Implement Joblib in your own projects involving data processing, machine learning, or any CPU-intensive tasks to experience its benefits firsthand.

3. **Discover Related Libraries:** Explore other Python libraries for parallel computing and optimization, such as Dask, Ray or Multiprocessing, to broaden your toolkit.

4. **Stay Updated:** Keep an eye on Joblib's updates and enhancements in future releases to leverage the latest functionalities and optimizations.


## Gotchas

1. **Choose the Right Backend:** Select the appropriate Joblib backend based on your task and available resources. For CPU-bound tasks, `loky` or `multiprocessing` might be suitable. For I/O-bound tasks, `threading` or specific distributed computing backends like `dask` might be better.

2. **Optimal Number of Workers:** Experiment with the number of workers (`n_jobs`) to find the optimal configuration. Too many workers can lead to resource contention, while too few might underutilize resources.

3. **Data Transfer Overhead:** Minimize data transfer overhead between processes/threads. Large data transfers between parallel workers can become a bottleneck. Avoid unnecessary data sharing or copying if possible.

4. **Memory Consideration:** Be mindful of memory usage, especially when processing large datasets in parallel. Parallelism can increase memory consumption, potentially leading to resource contention or out-of-memory issues.

5. **Cleanup Resources:** Ensure proper cleanup of resources (e.g., closing files, releasing memory) after the parallel tasks complete to avoid resource leaks.

6. **Proper Error Handling:** Implement proper error handling mechanisms, especially when dealing with parallel tasks, to manage exceptions and prevent deadlocks or crashes.

7. **Benchmark and Profile:** Measure the performance of your parallelized code using benchmarking tools (`timeit`, `time`, etc.) to identify bottlenecks and areas for improvement.
