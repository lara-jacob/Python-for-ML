# ESSENTIAL PYTHON LIBRARIES FOR ML
# `NUMPY`

## Introduction

NumPy, short for Numerical Python, is a fundamental library in Python for numerical computing, widely used in scientific computing, data analysis, and machine learning applications. Its primary data structure, `numpy.ndarray`, is an n-dimensional array that efficiently stores and manipulates numerical data. In this technical overview, we'll delve into the key features and capabilities of NumPy that make it such a powerful tool for numerical computation.

## Key Features

### Arrays and Memory Management

NumPy's `ndarray` is the cornerstone of its functionality. Unlike Python lists, `ndarray` is a contiguous block of memory containing elements of the same data type, allowing for efficient storage and manipulation of large datasets. This layout facilitates fast element-wise operations and eliminates the overhead associated with dynamic typing.

### Vectorized Operations and Performance

NumPy's vectorized operations enable efficient element-wise operations on arrays, leveraging optimized, low-level implementations for improved performance. By avoiding explicit iteration over array elements, vectorized operations take advantage of hardware-level optimizations and parallelism, resulting in significant speedups compared to traditional Python loops.

### Broadcasting

Broadcasting is a powerful mechanism in NumPy that allows arrays with different shapes to be combined in arithmetic operations. NumPy automatically aligns dimensions to perform operations, simplifying code and improving readability. This feature is particularly useful when working with arrays of different shapes or when performing operations between arrays and scalars.

### Mathematical Functions and Numerical Computations

NumPy provides a comprehensive suite of mathematical functions for numerical computations. These functions cover a wide range of mathematical operations, including basic arithmetic, trigonometry, logarithms, exponentials, and more. NumPy's ability to apply these functions element-wise to entire arrays enables efficient numerical computations on large datasets.

### Random Number Generation

The `numpy.random` module offers robust tools for generating random numbers from various probability distributions. Random number generation is essential for tasks such as model initialization, data simulation, and statistical analysis. NumPy's random number generation capabilities provide researchers and practitioners with tools to simulate stochastic processes and analyze probabilistic outcomes effectively.

### Integration with Low-Level Languages and Existing Libraries

NumPy seamlessly integrates with code written in low-level languages such as C and Fortran, allowing Python developers to leverage existing optimized numerical routines. This integration enhances computational efficiency and enables access to a vast ecosystem of pre-existing algorithms and numerical methods. NumPy's interoperability with other libraries, such as SciPy, Matplotlib, and scikit-learn, further extends its capabilities for scientific computing and machine learning applications.


## Importing NumPy
Once you've installed NumPy you can import it as a library:
```python
import numpy as np
import numpy as np
```
## NumPy Arrays
 NumPy arrays essentially come in two flavors: vectors and matrices. Vectors are strictly 1-dimensional (1D) arrays and matrices are 2D (but you should note a matrix can still have only one row or one column).

### Why use Numpy array? Why not just a list?
There are lot's of reasons to use a Numpy array instead of a "standard" python list object. 
Our main reasons are:

> Memory Efficiency of Numpy Array vs list
> Easily expands to N-dimensional objects
> Speed of calculations of numpy array
> Broadcasting operations and functions with numpy
> All the data science and machine learning libraries we use are built with Numpy

### Simple Example of what numpy array can do
```python
my_list = [1,2,3]
my_array = np.array([1,2,3])
type(my_list)
```
Output list
​
Let's begin our introduction by exploring how to create NumPy arrays.

### Creating NumPy Arrays from Objects
From a Python List
We can create an array by directly converting a list or list of lists:
```python
my_list = [1,2,3]
my_list
```
Output [1, 2, 3]
```python
np.array(my_list)
```
Output array([1, 2, 3])
```python
my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
my_matrix
```
Output [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
np.array(my_matrix)
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
### Built-in Methods to create arrays
There are lots of built-in ways to generate arrays.

#### arange
Return evenly spaced values within a given interval. [reference]
```python
np.arange(0,10)
```
Output array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```python
np.arange(0,11,2)
```
Output array([ 0,  2,  4,  6,  8, 10])
#### zeros and ones
Generate arrays of zeros or ones. [reference]
```python
np.zeros(3)
```
Output array([0., 0., 0.])
```python
np.zeros((5,5))
```
output
array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]])
```python
np.ones(3)
```
output array([1., 1., 1.])
```python
np.ones((3,3))
```
output 
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]])
#### linspace
Return evenly spaced numbers over a specified interval. [reference]
```python
np.linspace(0,10,3)
```
Output array([ 0.,  5., 10.])
``` python
np.linspace(0,5,20)
```
Output
array([0.        , 0.26315789, 0.52631579, 0.78947368, 1.05263158,
       1.31578947, 1.57894737, 1.84210526, 2.10526316, 2.36842105,
       2.63157895, 2.89473684, 3.15789474, 3.42105263, 3.68421053,
       3.94736842, 4.21052632, 4.47368421, 4.73684211, 5.        ])
Note that .linspace() includes the stop value. To obtain an array of common fractions, increase the number of items:
```python
np.linspace(0,5,21)
```
Output 
array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  , 2.25, 2.5 ,
       2.75, 3.  , 3.25, 3.5 , 3.75, 4.  , 4.25, 4.5 , 4.75, 5.  ])
#### eye
Creates an identity matrix [reference]
```python
np.eye(4)
```
Output
array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]])
#### Random
Numpy also has lots of ways to create random number arrays:

##### rand
Creates an array of the given shape and populates it with random samples from a uniform distribution over [0, 1). [reference]
```python
np.random.rand(2)
```
Output array([0.37065108, 0.89813878])
``` python
np.random.rand(5,5)
```
Output
array([[0.03932992, 0.80719137, 0.50145497, 0.68816102, 0.1216304 ],
       [0.44966851, 0.92572848, 0.70802042, 0.10461719, 0.53768331],
       [0.12201904, 0.5940684 , 0.89979774, 0.3424078 , 0.77421593],
       [0.53191409, 0.0112285 , 0.3989947 , 0.8946967 , 0.2497392 ],
       [0.5814085 , 0.37563686, 0.15266028, 0.42948309, 0.26434141]])
##### randn
Returns a sample (or samples) from the "standard normal" distribution [σ = 1]. Unlike rand which is uniform, values closer to zero are more likely to appear. [reference]
```python
np.random.randn(2)
```
Output array([-0.36633217, -1.40298731])
``` python
np.random.randn(5,5)
```
Output
array([[-0.45241033,  1.07491082,  1.95698188,  0.40660223, -1.50445807],
       [ 0.31434506, -2.16912609, -0.51237235,  0.78663583, -0.61824678],
       [-0.17569928, -2.39139828,  0.30905559,  0.1616695 ,  0.33783857],
       [-0.2206597 , -0.05768918,  0.74882883, -1.01241629, -1.81729966],
       [-0.74891671,  0.88934796,  1.32275912, -0.71605188,  0.0450718 ]])
##### randint
Returns random integers from low (inclusive) to high (exclusive). [reference]
```python
np.random.randint(1,100)
```
Output 61
``` python
np.random.randint(1,100,10)
```
Output array([39, 50, 72, 18, 27, 59, 15, 97, 11, 14])
##### seed
Can be used to set the random state, so that the same "random" results can be reproduced. [reference]
``` python
np.random.seed(42)
np.random.rand(4)
```
Output array([0.37454012, 0.95071431, 0.73199394, 0.59865848])
``` python
np.random.seed(42)
np.random.rand(4)
```
Output array([0.37454012, 0.95071431, 0.73199394, 0.59865848])

### Array Attributes and Methods
Let's discuss some useful attributes and methods for an array:
```python
arr = np.arange(25)
ranarr = np.random.randint(0,50,10)
arr
```
Output
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24])
ranarr
array([38, 18, 22, 10, 10, 23, 35, 39, 23,  2])

#### Reshape
Returns an array containing the same data with a new shape. [reference]
``` python
arr.reshape(5,5)
```
Output 
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24]])
### max, min, argmax, argmin
These are useful methods for finding max or min values. Or to find their index locations using argmin or argmax

#### ranarr
``` python
array([38, 18, 22, 10, 10, 23, 35, 39, 23,  2])
ranarr.max()
```
39
``` python
ranarr.argmax()
```
7
``` python
ranarr.min()
```
2
``` python
ranarr.argmin()
```
9
#### Shape
Shape is an attribute that arrays have (not a method): [reference]
``` python
# Vector
arr.shape
```
(25,)
Notice the two sets of brackets
``` python
arr.reshape(1,25)
```
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24]])
 ```python
arr.reshape(1,25).shape
```
(1, 25)
``` python
arr.reshape(25,1)
```
array([[ 0],
       [ 1],
       [ 2],
       [ 3],
       [ 4],
       [ 5],
       [ 6],
       [ 7],
       [ 8],
       [ 9],
       [10],
       [11],
       [12],
       [13],
       [14],
       [15],
       [16],
       [17],
       [18],
       [19],
       [20],
       [21],
       [22],
       [23],
       [24]])
 ```python
arr.reshape(25,1).shape
```
(25, 1)
#### dtype
You can also grab the data type of the object in the array: [reference]
``` python
arr.dtype
```
dtype('int32')
```python
arr2 = np.array([1.2, 3.4, 5.6])
arr2.dtype
```
dtype('float64')
​
## NumPy Indexing and Selection
```python
import numpy as np
#Creating sample array
arr = np.arange(0,11)
#Show
arr
```
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
### Bracket Indexing and Selection
The simplest way to pick one or some elements of an array looks very similar to python lists:
```python
#Get a value at an index
arr[8]
```
8
```python
#Get values in a range
arr[1:5]
```
array([1, 2, 3, 4])
```python
#Get values in a range
arr[0:5]
```
array([0, 1, 2, 3, 4])
### Broadcasting
NumPy arrays differ from normal Python lists because of their ability to broadcast. With lists, you can only reassign parts of a list with new parts of the same size and shape. That is, if you wanted to replace the first 5 elements in a list with a new value, you would have to pass in a new 5 element list. With NumPy arrays, you can broadcast a single value across a larger set of values:
```python
#Setting a value with index range (Broadcasting)
arr[0:5]=100​
#Show
arr
```
array([100, 100, 100, 100, 100,   5,   6,   7,   8,   9,  10])
```python
#Reset array, we'll see why I had to reset in  a moment
arr = np.arange(0,11)
​#Show
arr
```
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
```python
#Important notes on Slices
slice_of_arr = arr[0:6]
​#Show slice
slice_of_arr
```
array([0, 1, 2, 3, 4, 5])
```python
#Change Slice
slice_of_arr[:]=99
​#Show Slice again
slice_of_arr
```
array([99, 99, 99, 99, 99, 99])
Now note the changes also occur in our original array!
```python
arr
```
array([99, 99, 99, 99, 99, 99,  6,  7,  8,  9, 10])

Data is not copied, it's a view of the original array! This avoids memory problems!
#To get a copy, need to be explicit
```python
arr_copy = arr.copy()
arr_copy
```
array([99, 99, 99, 99, 99, 99,  6,  7,  8,  9, 10])

### Indexing a 2D array (matrices)
The general format is arr_2d[row][col] or arr_2d[row,col]. I recommend using the comma notation for clarity.
``` python
arr_2d = np.array(([5,10,15],[20,25,30],[35,40,45]))
​#Show
arr_2d
```
array([[ 5, 10, 15],
       [20, 25, 30],
       [35, 40, 45]])
``` python
#Indexing row
arr_2d[1]
```
array([20, 25, 30])
``` python
#Format is arr_2d[row][col] or arr_2d[row,col]
​#Getting individual element value
arr_2d[1][0]
```
20
``` python
#Getting individual element value
arr_2d[1,0]
```
20
``` python
#2D array slicing
​#Shape (2,2) from top right corner
arr_2d[:2,+]
```
array([[10, 15],
       [25, 30]])
``` python
#Shape bottom row
arr_2d[2]
```
array([35, 40, 45])
``` python
#Shape bottom row
arr_2d[2,:]
```
array([35, 40, 45])

### More Indexing Help
Indexing a 2D matrix can be a bit confusing at first, especially when you start to add in step size. Try google image searching NumPy indexing to find useful images, like this one:

Image Image source: http://www.scipy-lectures.org/intro/numpy/numpy.html

### Conditional Selection
This is a very fundamental concept that will directly translate to pandas later on, make sure you understand this part!

Let's briefly go over how to use brackets for selection based off of comparison operators.
``` python
arr = np.arange(1,11)
arr
```
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
``` python
arr > 4
```
array([False, False, False, False,  True,  True,  True,  True,  True,
        True])
``` python
bool_arr = arr>4
bool_arr
```
array([False, False, False, False,  True,  True,  True,  True,  True,
        True])
``` python
arr[bool_arr]
```
array([ 5,  6,  7,  8,  9, 10])
``` python
arr[arr>2]
```
array([ 3,  4,  5,  6,  7,  8,  9, 10])
``` python
x = 2
arr[arr>x]
```
array([ 3,  4,  5,  6,  7,  8,  9, 10])

## NumPy Operations
### Arithmetic
You can easily perform array with array arithmetic, or scalar with array arithmetic. Let's see some examples:
``` python
import numpy as np
arr = np.arange(0,10)
arr
```
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```python
arr + arr
```
array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])
``` python
arr * arr
```
array([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])
``` python
arr - arr
```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
``` python
#This will raise a Warning on division by zero, but not an error!
#It just fills the spot with nan
arr/arr
```
``` python
C:\Anaconda3\envs\tsa_course\lib\site-packages\ipykernel_launcher.py:3: RuntimeWarning: invalid value encountered in true_divide
  This is separate from the ipykernel package so we can avoid doing imports until
```
array([nan,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
``` python
# Also a warning (but not an error) relating to infinity
1/arr
C:\Anaconda3\envs\tsa_course\lib\site-packages\ipykernel_launcher.py:2: RuntimeWarning: divide by zero encountered in true_divide
  ```
array([       inf, 1.        , 0.5       , 0.33333333, 0.25      ,
       0.2       , 0.16666667, 0.14285714, 0.125     , 0.11111111])
 ```python
arr**3
```
array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729], dtype=int32)
### Universal Array Functions
NumPy comes with many universal array functions, or ufuncs, which are essentially just mathematical operations that can be applied across the array.
Let's show some common ones:
``` python
#Taking Square Roots
np.sqrt(arr)
```
array([0.        , 1.        , 1.41421356, 1.73205081, 2.        ,
       2.23606798, 2.44948974, 2.64575131, 2.82842712, 3.        ])
       ``` python
#Calculating exponential (e^)
np.exp(arr)
```
array([1.00000000e+00, 2.71828183e+00, 7.38905610e+00, 2.00855369e+01,
       5.45981500e+01, 1.48413159e+02, 4.03428793e+02, 1.09663316e+03,
       2.98095799e+03, 8.10308393e+03])
``` python
#Trigonometric Functions like sine
np.sin(arr)
```
array([ 0.        ,  0.84147098,  0.90929743,  0.14112001, -0.7568025 ,
       -0.95892427, -0.2794155 ,  0.6569866 ,  0.98935825,  0.41211849])
       ``` python
#Taking the Natural Logarithm
np.log(arr)
```
C:\Anaconda3\envs\tsa_course\lib\site-packages\ipykernel_launcher.py:2: RuntimeWarning: divide by zero encountered in log
  
array([      -inf, 0.        , 0.69314718, 1.09861229, 1.38629436,
       1.60943791, 1.79175947, 1.94591015, 2.07944154, 2.19722458])

### Summary Statistics on Arrays
NumPy also offers common summary statistics like sum, mean and max. You would call these as methods on an array.
``` python
arr = np.arange(0,10)
arr
```
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
``` python
arr.sum()
```
45
``` python
arr.mean()
```
4.5
``` python
arr.max()
```
9
#### Other summary statistics include:

arr.min() returns 0                   minimum
arr.var() returns 8.25                variance
arr.std() returns 2.8722813232690143  standard deviation
### Axis Logic
When working with 2-dimensional arrays (matrices) we have to consider rows and columns. This becomes very important when we get to the section on pandas. In array terms, axis 0 (zero) is the vertical axis (rows), and axis 1 is the horizonal axis (columns). These values (0,1) correspond to the order in which arr.shape values are returned.

Let's see how this affects our summary statistic calculations from above.
``` python
arr_2d = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
arr_2d
```
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]])
       ``` python
arr_2d.sum(axis=0)
```
array([15, 18, 21, 24])
By passing in axis=0, we're returning an array of sums along the vertical axis, essentially [(1+5+9), (2+6+10), (3+7+11), (4+8+12)]

### Image
``` python
arr_2d.shape
```



 # `MATPLOTLIB`

Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It is widely used for producing high-quality plots, charts, and figures for various scientific, engineering, and data analysis applications. Matplotlib provides a MATLAB-like interface and is highly customizable, allowing users to create a wide range of visualizations with ease.

## Key Features and Components of Matplotlib:

### Pyplot Interface:
The `matplotlib.pyplot` module provides a MATLAB-like interface for creating plots. It simplifies the process of creating common types of plots such as line plots, scatter plots, histograms, bar charts, etc.

### Object-Oriented Interface:
Matplotlib also offers an object-oriented interface, which gives more control and flexibility over the plots. Users can directly work with figure and axis objects to create and customize plots in a more granular manner.

### Supported Plot Types:
Matplotlib supports a wide range of plot types, including:
- Line plots
- Scatter plots
- Histograms
- Bar charts
- Pie charts
- Box plots
- Contour plots
- Heatmaps
- 3D plots (with mplot3d toolkit)

### Customization:
Matplotlib allows extensive customization of plots. Users can control aspects such as colors, line styles, markers, labels, titles, axis limits, grid lines, legends, annotations, and more.

### Multiple Output Formats:
Matplotlib can save plots in various formats such as PNG, PDF, SVG, and EPS, making it easy to integrate plots into documents, reports, or presentations.

### Integration with NumPy:
Matplotlib seamlessly integrates with NumPy, a fundamental library for numerical computing in Python. This integration allows users to plot NumPy arrays directly and perform mathematical operations on data before plotting.

### Support for LaTeX:
Matplotlib supports LaTeX for mathematical expressions in labels, titles, annotations, and other text elements, enabling high-quality typesetting of mathematical symbols and equations.

### Interactive Features:
Matplotlib can be used in interactive environments such as Jupyter notebooks, allowing users to dynamically explore and manipulate plots.

### Extensibility:
Matplotlib is highly extensible and customizable. Users can create custom plot types, styles, and functionalities by extending Matplotlib's functionality or by utilizing additional toolkits and libraries built on top of Matplotlib, such as Seaborn, pandas plotting, and mpld3 for interactive web-based visualizations.

Matplotlib is a powerful and versatile library for data visualization in Python, making it a popular choice among data scientists, researchers, engineers, and analysts for exploring and communicating data insights.
