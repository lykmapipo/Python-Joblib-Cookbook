import time

from joblib import Memory, Parallel, delayed, parallel_config

mem = Memory("./tmp/cache", verbose=0)


@mem.cache
def process_item(item):
    return item**2


items = list(range(100))

start = time.time()
with parallel_config(backend="loky", n_jobs=-1, verbose=50):
    results = Parallel()(delayed(process_item)(item) for item in items)
stop = time.time()

print(results)
print("Elapsed time for the entire processing: {:.2f} s".format(stop - start))
