# 1
import numpy as np

# 2
print(np.__version__)
np.show_config()

# 3
z = np.zeros((10))
print(z, z.shape)

# 4
f32 = np.array([1, 2, 3], dtype=np.float32)
print("f32.size: {}".format(f32.size))
print("f32.itemsize: {}".format(f32.itemsize))
print("memory size of f32: {}".format(f32.size * f32.itemsize))

f64 = np.array([1, 2, 3], dtype=np.float64)
print("f64.size: {}".format(f64.size))
print("f64.itemsize: {}".format(f64.itemsize))
print("memory size of f64: {}".format(f32.size * f64.itemsize))

z = np.zeros((10, 20))
print("z.size: {}".format(z.size))
print("z.itemsize: {}".format(z.itemsize))
print("memory size of z: {}".format(z.size * z.itemsize))

# 5
print(np.info(np.add))

# 6
z = np.zeros(10)
z[4] = 1
print(z)

# 7
a = np.arange(10, 50)
print(a)

# 8
print(a[::-1])

# 9
print(np.arange(9).reshape(3, 3))

# 10
a = np.array([1, 2, 0, 0, 4, 0], dtype=np.int64)
print(a.nonzero())
print(np.nonzero(a))

# 11
print(np.eye(3))

# 12
print(np.random.random((3, 3, 3)))

# 13
a = np.random.random((10, 10))
print(np.min(a), np.max(a))
print(a.min(), a.max())

# 14
a = np.random.random((30))
print(np.mean(a))
print(a.mean())

# 15, Notice the difference between x[:][3] and x[:, 3]
a = np.ones((4, 4))
a[1:-1, 1:-1] = 0
print(a)

# 16. This might be useful.
a = np.arange(9).reshape(3, 3)
print(np.pad(a, (1, 2), 'constant'))
print(np.pad(a, ((1, 2), (2, 1)), 'constant'))

# 17
# NaN, False, False, NaN, False
print("0*np.nan: {}".format(0*np.nan))
print("np.nan == np.nan: {}".format(np.nan == np.nan))
print("np.inf > np.nan: {}".format(np.inf > np.nan))
print("np.nan - np.nan: {}".format(np.nan - np.nan))
print("0.3 == 3 * 0.1: {}".format(0.3 == 3 * 0.1))

# 18 #
print(np.diag(1+np.arange(4), k=-1))

# 19
cb = np.zeros((8, 8))
cb[::2, ::2] = 1
cb[1::2, 1::2] = 1
print(cb)

# 20
a = np.arange(6*7*8).reshape(6, 7, 8)
print(np.unravel_index(100, (6, 7, 8)))

# 21
print(np.tile(np.eye(2), (4, 4)))

# 22
a = np.random.randint(100)*np.random.random((5, 5))
print((a - np.min(a)) / (np.max(a) - np.min(a)))

# 23

# 24
a = np.arange(15).reshape(5, 3)
b = np.ones((3, 2))
print(np.dot(a, b))
print(a @ b)

# 25
a = np.arange(11)
a[(a > 3) & (a < 8)] *= -1
print(a)

# 26
print(sum(range(5), -1))
from numpy import sum
print(sum(range(5), -1))

# 27
z = np.array([2])
print("z**z: {}".format(z**z))
print("2 << z >> 2: {}".format(2 << z >> 2))
print("z <- z: {}".format(z <- z))
print("1j*z: {}".format(1j*z))
print("z/1/1: {}".format(z/1/1))
print("z<z>z: {}".format(z<z>z))

# 28

# 29
z = np.random.uniform(-10, 10, 10)
print(z)
print(np.copysign(np.ceil(np.abs(z)), z))

# 30
a = np.array([1, 2, 2, 2, 3, 1, 7])
b = np.array([0, 0, 4, 2, 1, 1, 1])
print(np.intersect1d(a, b))

# 33
today = np.datetime64('today', 'D')
tomorrow = today + np.timedelta64(1, 'D')
yesterday = today - np.timedelta64(1, 'D')
print("np.datetime64('today', 'D'): {}".format(np.datetime64('today', 'D')))
print("yesterday: {}".format(yesterday))
print("tomorrow: {}".format(tomorrow))

