from joblib import Parallel, delayed, parallel_config


def square(x):
    return x**2


with parallel_config(backend="loky", n_jobs=-1, verbose=50):
    results = Parallel()(delayed(square)(i) for i in range(10))

print(results)
