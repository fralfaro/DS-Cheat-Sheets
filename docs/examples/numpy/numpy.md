<div style="display: flex; align-items: center;">
    <a href="https://colab.research.google.com/github/fralfaro/DS-Cheat-Sheets/blob/main/docs/examples/numpy/numpy.ipynb" target="_parent">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
</div>

# NumPy 

<img src="https://raw.githubusercontent.com/fralfaro/DS-Cheat-Sheets/main/docs/examples/numpy/numpy2.png" alt="numpy logo" width = "200">

[NumPy](https://numpy.org/) is the core library for scientific computing in
Python. It provides a high-performance multidimensional array
object, and tools for working with these arrays.

## Install and import NumPy

`
$ pip install numpy
`


```python
# Import NumPy convention
import numpy as np
```

## NumPy Arrays

<img src="https://raw.githubusercontent.com/fralfaro/DS-Cheat-Sheets/main/docs/examples/numpy/np_01.png" alt="numpy logo" >



```python
# Create a 1D array
a = np.array([1, 2, 3])

# Create a 2D array with specified dtype
b = np.array([
    (1.5, 2, 3),
    (4, 5, 6)
], dtype=float)

# Create a 3D array with specified dtype
c = np.array([
    [(1.5, 2, 3), (4, 5, 6)],
    [(3, 2, 1), (4, 5, 6)]
], dtype=float)

print("Array a:")
print(a)

print("\nArray b:")
print(b)

print("\nArray c:")
print(c)
```

    Array a:
    [1 2 3]
    
    Array b:
    [[1.5 2.  3. ]
     [4.  5.  6. ]]
    
    Array c:
    [[[1.5 2.  3. ]
      [4.  5.  6. ]]
    
     [[3.  2.  1. ]
      [4.  5.  6. ]]]
    

## Initial Placeholders



```python
# Create an array of zeros
zeros_arr = np.zeros((3, 4))

# Create an array of ones
ones_arr = np.ones((2, 3, 4))

# Create an array of evenly spaced values (step value)
d = np.arange(10, 25, 5)

# Create an array of evenly spaced values (number of samples)
e = np.linspace(0, 2, 9)

# Create a constant array
f = np.full((2, 2), 7)

# Create a 2x2 identity matrix
g = np.eye(2)

# Create an array with random values
random_arr = np.random.random((2, 2))

# Create an empty array
empty_arr = np.empty((3, 2))

print("zeros_arr:")
print(zeros_arr)

print("\nones_arr:")
print(ones_arr)

print("\nd:")
print(d)

print("\ne:")
print(e)

print("\nf:")
print(f)

print("\ng:")
print(g)

print("\nrandom_arr:")
print(random_arr)

print("\nempty_arr:")
print(empty_arr)

```

    zeros_arr:
    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    
    ones_arr:
    [[[1. 1. 1. 1.]
      [1. 1. 1. 1.]
      [1. 1. 1. 1.]]
    
     [[1. 1. 1. 1.]
      [1. 1. 1. 1.]
      [1. 1. 1. 1.]]]
    
    d:
    [10 15 20]
    
    e:
    [0.   0.25 0.5  0.75 1.   1.25 1.5  1.75 2.  ]
    
    f:
    [[7 7]
     [7 7]]
    
    g:
    [[1. 0.]
     [0. 1.]]
    
    random_arr:
    [[0.00439382 0.02702873]
     [0.19578698 0.34798592]]
    
    empty_arr:
    [[1.5 2. ]
     [3.  4. ]
     [5.  6. ]]
    

## NumPy Data Types



```python
# Signed 64-bit integer types
int64_type = np.int64

# Standard double-precision floating point
float32_type = np.float32

# Complex numbers represented by 128 floats
complex_type = np.complex128

# Boolean type storing TRUE and FALSE values
bool_type = np.bool_

# Python object type
object_type = np.object_

# Fixed-length string type
string_type = np.string_

# Fixed-length unicode type
unicode_type = np.unicode_

print("int64_type:", int64_type)
print("float32_type:", float32_type)
print("complex_type:", complex_type)
print("bool_type:", bool_type)
print("object_type:", object_type)
print("string_type:", string_type)
print("unicode_type:", unicode_type)
```

    int64_type: <class 'numpy.int64'>
    float32_type: <class 'numpy.float32'>
    complex_type: <class 'numpy.complex128'>
    bool_type: <class 'numpy.bool_'>
    object_type: <class 'numpy.object_'>
    string_type: <class 'numpy.bytes_'>
    unicode_type: <class 'numpy.str_'>
    

## Inspecting Array Properties


```python
# Array dimensions
a_shape = a.shape

# Length of array
a_length = len(a)

# Number of array dimensions
b_ndim = b.ndim

# Number of array elements
e_size = e.size

# Data type of array elements
b_dtype = b.dtype

# Name of data type
b_dtype_name = b.dtype.name

# Convert an array to a different type
b_as_int = b.astype(int)

print("a_shape:")
print(a_shape)

print("\na_length:")
print(a_length)

print("\nb_ndim:")
print(b_ndim)

print("\ne_size:")
print(e_size)

print("\nb_dtype:")
print(b_dtype)

print("\nb_dtype_name:")
print(b_dtype_name)

print("\nb_as_int:")
print(b_as_int)

```

    a_shape:
    (3,)
    
    a_length:
    3
    
    b_ndim:
    2
    
    e_size:
    9
    
    b_dtype:
    float64
    
    b_dtype_name:
    float64
    
    b_as_int:
    [[1 2 3]
     [4 5 6]]
    

## Arithmetic Operations


```python
# Example values for arrays
a = np.array([1, 2, 3])
b = np.array([
    (1.5, 2, 3),
    (4, 5, 6)
], dtype=float)
e = np.array([2, 3, 4])
d = np.arange(10, 25, 5)

# Subtraction
subtraction_result = a - b
subtraction_np = np.subtract(a, b)

# Addition
addition_result = b + a
addition_np = np.add(b, a)

# Division
division_result = a / b
division_np = np.divide(a, b)

# Multiplication
multiplication_result = a * b
multiplication_np = np.multiply(a, b)

# Exponentiation
exponentiation_result = np.exp(b)

# Square root
sqrt_result = np.sqrt(b)

# Sine of an array
sin_result = np.sin(a)

# Element-wise cosine
cos_result = np.cos(b)

# Element-wise natural logarithm
log_result = np.log(a)

# Dot product
dot_product_result = np.dot(e, d)

print("subtraction_result:")
print(subtraction_result)

print("\nsubtraction_np:")
print(subtraction_np)

print("\naddition_result:")
print(addition_result)

print("\naddition_np:")
print(addition_np)

print("\ndivision_result:")
print(division_result)

print("\ndivision_np:")
print(division_np)

print("\nmultiplication_result:")
print(multiplication_result)

print("\nmultiplication_np:")
print(multiplication_np)

print("\nexponentiation_result:")
print(exponentiation_result)

print("\nsqrt_result:")
print(sqrt_result)

print("\nsin_result:")
print(sin_result)

print("\ncos_result:")
print(cos_result)

print("\nlog_result:")
print(log_result)

print("\ndot_product_result:")
print(dot_product_result)
```

    subtraction_result:
    [[-0.5  0.   0. ]
     [-3.  -3.  -3. ]]
    
    subtraction_np:
    [[-0.5  0.   0. ]
     [-3.  -3.  -3. ]]
    
    addition_result:
    [[2.5 4.  6. ]
     [5.  7.  9. ]]
    
    addition_np:
    [[2.5 4.  6. ]
     [5.  7.  9. ]]
    
    division_result:
    [[0.66666667 1.         1.        ]
     [0.25       0.4        0.5       ]]
    
    division_np:
    [[0.66666667 1.         1.        ]
     [0.25       0.4        0.5       ]]
    
    multiplication_result:
    [[ 1.5  4.   9. ]
     [ 4.  10.  18. ]]
    
    multiplication_np:
    [[ 1.5  4.   9. ]
     [ 4.  10.  18. ]]
    
    exponentiation_result:
    [[  4.48168907   7.3890561   20.08553692]
     [ 54.59815003 148.4131591  403.42879349]]
    
    sqrt_result:
    [[1.22474487 1.41421356 1.73205081]
     [2.         2.23606798 2.44948974]]
    
    sin_result:
    [0.84147098 0.90929743 0.14112001]
    
    cos_result:
    [[ 0.0707372  -0.41614684 -0.9899925 ]
     [-0.65364362  0.28366219  0.96017029]]
    
    log_result:
    [0.         0.69314718 1.09861229]
    
    dot_product_result:
    145
    

## Comparison Operations



```python
# Element-wise comparison for equality
equality_comparison = a == b

# Element-wise comparison for less than
less_than_comparison = a < 2

# Array-wise comparison using np.array_equal
np_equal = np.array_equal(a, b)

print("equality_comparison:")
print(equality_comparison)

print("\nless_than_comparison:")
print(less_than_comparison)

print("\nnp_equal:")
print(np_equal)
```

    equality_comparison:
    [[False  True  True]
     [False False False]]
    
    less_than_comparison:
    [ True False False]
    
    np_equal:
    False
    

## Aggregate Functions



```python
# Array-wise sum
array_sum = a.sum()

# Array-wise minimum value
array_min = a.min()

# Maximum value of an array row
row_max = b.max(axis=0)

# Cumulative sum of the elements
cumulative_sum = b.cumsum(axis=1)

# Mean
array_mean = a.mean()

# Median
array_median = np.median(b)

# Correlation coefficient (not valid for 1D array)
corr_coefficient = np.corrcoef(a, b[0])

# Standard deviation
std_deviation = np.std(b)

print("array_sum:")
print(array_sum)

print("\narray_min:")
print(array_min)

print("\nrow_max:")
print(row_max)

print("\ncumulative_sum:")
print(cumulative_sum)

print("\narray_mean:")
print(array_mean)

print("\narray_median:")
print(array_median)

print("\ncorr_coefficient:")
print(corr_coefficient)

print("\nstd_deviation:")
print(std_deviation)
```

    array_sum:
    6
    
    array_min:
    1
    
    row_max:
    [4. 5. 6.]
    
    cumulative_sum:
    [[ 1.5  3.5  6.5]
     [ 4.   9.  15. ]]
    
    array_mean:
    2.0
    
    array_median:
    3.5
    
    corr_coefficient:
    [[1.         0.98198051]
     [0.98198051 1.        ]]
    
    std_deviation:
    1.5920810978785667
    

## Copying Arrays



```python
# Create a view of the array with the same data
array_view = a.view()

# Create a copy of the array
array_copy = np.copy(a)

# Create a deep copy of the array
array_deep_copy = a.copy()

# Sort an array
a.sort()

# Sort the elements of an array's axis
c.sort(axis=0)

print("array_view:")
print(array_view)

print("\narray_copy:")
print(array_copy)

print("\narray_deep_copy:")
print(array_deep_copy)
```

    array_view:
    [1 2 3]
    
    array_copy:
    [1 2 3]
    
    array_deep_copy:
    [1 2 3]
    

## Sorting Arrays



```python
# Sort an array
a.sort()

# Sort the elements of an array's axis
c.sort(axis=0)

print("Sorted a:")
print(a)

print("\nSorted c (axis=0):")
print(c)
```

    Sorted a:
    [1 2 3]
    
    Sorted c (axis=0):
    [[[1.5 2.  1. ]
      [4.  5.  6. ]]
    
     [[3.  2.  3. ]
      [4.  5.  6. ]]]
    

## Subsetting, Slicing, and Indexing


```python
# Subsetting
element_at_2nd_index = a[2] 

# Select the element at row 1, column 2
element_row_1_col_2 = b[1, 2] 

# Slicing
sliced_a = a[0:2]

# Select items at rows 0 and 1 in column 1
sliced_b = b[0:2, 1]

# Select all items at row 0
sliced_c = b[:1] 

# Reversed array
reversed_a = a[::-1] 

# Boolean Indexing
a_less_than_2 = a[a < 2]

# Fancy Indexing
fancy_indexing_result = b[ 
    [1, 0, 1, 0], 
    [0, 1, 2, 0]
]
fancy_indexing_subset = b[[1, 0, 1, 0]][:, [0, 1, 2, 0]]

print("element_at_2nd_index:")
print(element_at_2nd_index)

print("\nelement_row_1_col_2:")
print(element_row_1_col_2)

print("\nsliced_a:")
print(sliced_a)

print("\nsliced_b:")
print(sliced_b)

print("\nsliced_c:")
print(sliced_c)

print("\nreversed_a:")
print(reversed_a)

print("\na_less_than_2:")
print(a_less_than_2)

print("\nfancy_indexing_result:")
print(fancy_indexing_result)

print("\nfancy_indexing_subset:")
print(fancy_indexing_subset)
```

    element_at_2nd_index:
    3
    
    element_row_1_col_2:
    6.0
    
    sliced_a:
    [1 2]
    
    sliced_b:
    [2. 5.]
    
    sliced_c:
    [[1.5 2.  3. ]]
    
    reversed_a:
    [3 2 1]
    
    a_less_than_2:
    [1]
    
    fancy_indexing_result:
    [4.  2.  6.  1.5]
    
    fancy_indexing_subset:
    [[4.  5.  6.  4. ]
     [1.5 2.  3.  1.5]
     [4.  5.  6.  4. ]
     [1.5 2.  3.  1.5]]
    

## Array Manipulation



```python
# Example values for arrays
a = np.array([3, 1, 2])
b = np.array([
    (1.5, 2, 3),
    (4, 5, 6)
], dtype=float)
h = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
g = np.array([7, 8, 9])
d = np.array([4, 5, 6])
e = np.array([10, 11])
f = np.array([12, 13])
c = np.array([
    (3, 1, 2),
    (6, 4, 5)
], dtype=int)

# Transposing Array
transposed_b = np.transpose(b)
transposed_b_T = transposed_b.T

# Changing Array Shape
flattened_h = h.ravel()
reshaped_g = g.reshape(3, -2)

# Adding/Removing Elements
resized_h = np.resize(h, (2, 6))  # Using np.resize to avoid the error
appended_array = np.append(h, g)
inserted_array = np.insert(a, 1, 5)
deleted_array = np.delete(a, [1])

# Combining Arrays
concatenated_arrays = np.concatenate((a, d), axis=0)
vstacked_arrays = np.vstack((a, b))
hstacked_arrays = np.hstack((e, f))
column_stacked_arrays = np.column_stack((a, d))
c_stacked_arrays = np.c_[a, d]

# Splitting Arrays
hsplit_array = np.hsplit(a, 3)
vsplit_array = np.vsplit(c, 2)

print("transposed_b:")
print(transposed_b)

print("\ntransposed_b_T:")
print(transposed_b_T)

print("\nflattened_h:")
print(flattened_h)

print("\nreshaped_g:")
print(reshaped_g)

print("\nresized_h:")
print(resized_h)

print("\nappended_array:")
print(appended_array)

print("\ninserted_array:")
print(inserted_array)

print("\ndeleted_array:")
print(deleted_array)

print("\nconcatenated_arrays:")
print(concatenated_arrays)

print("\nvstacked_arrays:")
print(vstacked_arrays)

print("\nhstacked_arrays:")
print(hstacked_arrays)

print("\ncolumn_stacked_arrays:")
print(column_stacked_arrays)

print("\nc_stacked_arrays:")
print(c_stacked_arrays)

print("\nhsplit_array:")
print(hsplit_array)

print("\nvsplit_array:")
print(vsplit_array)

```

    transposed_b:
    [[1.5 4. ]
     [2.  5. ]
     [3.  6. ]]
    
    transposed_b_T:
    [[1.5 2.  3. ]
     [4.  5.  6. ]]
    
    flattened_h:
    [1 2 3 4 5 6]
    
    reshaped_g:
    [[7]
     [8]
     [9]]
    
    resized_h:
    [[1 2 3 4 5 6]
     [1 2 3 4 5 6]]
    
    appended_array:
    [1 2 3 4 5 6 7 8 9]
    
    inserted_array:
    [3 5 1 2]
    
    deleted_array:
    [3 2]
    
    concatenated_arrays:
    [3 1 2 4 5 6]
    
    vstacked_arrays:
    [[3.  1.  2. ]
     [1.5 2.  3. ]
     [4.  5.  6. ]]
    
    hstacked_arrays:
    [10 11 12 13]
    
    column_stacked_arrays:
    [[3 4]
     [1 5]
     [2 6]]
    
    c_stacked_arrays:
    [[3 4]
     [1 5]
     [2 6]]
    
    hsplit_array:
    [array([3]), array([1]), array([2])]
    
    vsplit_array:
    [array([[3, 1, 2]]), array([[6, 4, 5]])]
    

## Asking for Help



```python
# Get information about a NumPy function or object
np.info(np.ndarray.dtype)
```

    Data-type of the array's elements.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    d : numpy dtype object
    
    See Also
    --------
    numpy.dtype
    
    Examples
    --------
    >>> x
    array([[0, 1],
           [2, 3]])
    >>> x.dtype
    dtype('int32')
    >>> type(x.dtype)
    <type 'numpy.dtype'>
    

## Saving & Loading 

**On Disk**

``` python
# Save a NumPy array to a file
a = np.array([1, 2, 3])
np.save('my_array', a)

# Save multiple NumPy arrays to a compressed file
b = np.array([
    (1.5, 2, 3), 
    (4, 5, 6)
    ], dtype=float)
np.savez('array.npz', a=a, b=b)

# Load a NumPy array from a file
loaded_array = np.load('my_array.npy')
```

**Text Files**

``` python
# Load data from a text file
loaded_txt = np.loadtxt("myfile.txt")

# Load data from a CSV file with specified delimiter
loaded_csv = np.genfromtxt(
    "my_file.csv",
     delimiter=',')

# Save a NumPy array to a text file
a = np.array([1, 2, 3])
np.savetxt(
    "myarray.txt", 
    a, 
    delimiter=" ")
```
