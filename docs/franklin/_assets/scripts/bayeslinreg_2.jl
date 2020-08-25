# Data.
data_len = 100
x = [Float64(i) for i in 1 : data_len]
obs = target([(:y => i, ) => 3.0 * x[i] + randn() for i in 1 : data_len])
