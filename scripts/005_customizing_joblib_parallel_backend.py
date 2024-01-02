import joblib


def square(x):
    return x**2


with joblib.parallel_config(backend="threading", n_jobs=2):
    results = joblib.Parallel()(joblib.delayed(square)(i) for i in range(10))

print(results)
