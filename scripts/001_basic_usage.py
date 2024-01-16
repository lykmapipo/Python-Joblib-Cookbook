from joblib import Parallel, delayed


def square(x):
    return x**2


results = Parallel(n_jobs=-1, verbose=50)(delayed(square)(i) for i in range(10))

print(results)
