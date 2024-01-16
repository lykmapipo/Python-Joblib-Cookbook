from joblib import Parallel, delayed, parallel_config


def process_item(item):
    return item**2


items = list(range(10))

with parallel_config(backend="loky", n_jobs=-1, verbose=50):
    results = Parallel()(delayed(process_item)(item) for item in items)

print(results)
