import joblib
import numpy as np

data = np.random.rand(1000, 1000)
filename = "./tmp/large_array.dat"

joblib.dump(data, filename, compress=3, protocol=4)
loaded_data = joblib.load(filename)

print(loaded_data)
