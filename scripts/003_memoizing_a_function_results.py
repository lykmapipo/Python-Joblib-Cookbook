import time

from joblib import Memory, Parallel, delayed

mem = Memory("./tmp/cache", verbose=0)


@mem.cache
def process_item(item):
    return item**2


items = list(range(100))

start = time.time()
results = Parallel(n_jobs=-1)(delayed(process_item)(item) for item in items)
stop = time.time()

print(results)
print("Elapsed time for the entire processing: {:.2f} s".format(stop - start))
