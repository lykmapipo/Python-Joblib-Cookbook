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
