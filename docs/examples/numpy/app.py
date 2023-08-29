import streamlit as st
from pathlib import Path
import base64
import requests


# Initial page config
st.set_page_config(
    page_title='NumPy Cheat Sheet',
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    """
    Main function to set up the Streamlit app layout.
    """
    cs_sidebar()
    cs_body()
    return None

# Define img_to_bytes() function
def img_to_bytes(img_url):
    response = requests.get(img_url)
    img_bytes = response.content
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

# Define the cs_sidebar() function
def cs_sidebar():
    """
    Populate the sidebar with various content sections related to NumPy.
    """
    st.sidebar.markdown(
        '''[<img src='data:image/png;base64,{}' class='img-fluid' width=95 >](https://streamlit.io/)'''.format(
            img_to_bytes("https://raw.githubusercontent.com/fralfaro/DS-Cheat-Sheets/main/docs/examples/numpy/numpy2.png")), unsafe_allow_html=True)

    st.sidebar.header('NumPy Cheat Sheet')
    st.sidebar.markdown('''
<small>[NumPy](https://numpy.org/) is the core library for scientific computing in
Python. It provides a high-performance multidimensional array
object, and tools for working with these arrays.</small>
    ''', unsafe_allow_html=True)

    # NumPy installation and import
    st.sidebar.markdown('__Install and import NumPy__')
    st.sidebar.code('$ pip install numpy')
    st.sidebar.code('''
# Import NumPy convention
>>> import numpy as np
''')

    # NumPy array creation
    st.sidebar.markdown('__NumPy Arrays__')
    st.sidebar.markdown(
        '''[<img src='data:image/png;base64,{}' class='img-fluid' width=300 >](https://streamlit.io/)'''.format(
            img_to_bytes("https://raw.githubusercontent.com/fralfaro/DS-Cheat-Sheets/main/docs/examples/numpy/np_02.png")), unsafe_allow_html=True)

    st.sidebar.code('''
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
                ''')
    return None


# Define the cs_body() function
def cs_body():
    """
    Create content sections for the main body of the Streamlit cheat sheet with NumPy examples.
    """
    col1, col2, col3 = st.columns(3)  # Create columns for layout

    #######################################
    # COLUMN 1
    #######################################

    # Initial Placeholders
    col1.subheader('Initial Placeholders')
    col1.code('''
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

    # Create a 2X2 identity matrix
    g = np.eye(2)

    # Create an array with random values
    random_arr = np.random.random((2, 2))

    # Create an empty array
    empty_arr = np.empty((3, 2))
        ''')

    # Saving & Loading On Disk
    col1.subheader('Saving & Loading On Disk')
    col1.code('''
    # Save a NumPy array to a file
    a = np.array([1, 2, 3])
    np.save('my_array', a)

    # Save multiple NumPy arrays to a compressed file
    b = np.array([
        (1.5, 2, 3), 
        (4, 5, 6)
        ], dtype=float)
    np.savez('array.npz', a, b)

    # Load a NumPy array from a file
    loaded_array = np.load('my_array.npy')
        ''')

    # Saving & Loading Text Files
    col1.subheader('Saving & Loading Text Files')
    col1.code('''
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
        ''')

    # NumPy data types
    col1.subheader('NumPy Data Types')
    col1.code('''
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
        ''')



    # Asking for help
    col1.subheader('Asking for Help')
    col1.code('''
    # Get information about a NumPy function or object
    np.info(np.ndarray.dtype)
        ''')

    #######################################
    # COLUMN 2
    #######################################

    # Inspecting array properties
    col2.subheader('Inspecting Array Properties')
    col2.code('''
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
        ''')



    # Arithmetic operations
    col2.subheader('Arithmetic Operations')
    col2.code('''
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
    dot_product_result = e.dot(f)
        ''')


    # Aggregate functions
    col2.subheader('Aggregate Functions')
    col2.code('''
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
    array_median = b.median()

    # Correlation coefficient
    corr_coefficient = a.corrcoef()

    # Standard deviation
    std_deviation = np.std(b)
        ''')

    #######################################
    # COLUMN 3
    #######################################

    # Comparison operations
    col3.subheader('Comparison Operations')
    col3.code('''
    # Element-wise comparison for equality
    equality_comparison = a == b

    # Element-wise comparison for less than
    less_than_comparison = a < 2

    # Array-wise comparison using np.array_equal
    np_equal = np.array_equal(a, b)
        ''')

    # Copying arrays
    col3.subheader('Copying Arrays')
    col3.code('''
    # Create a view of the array with the same data
    array_view = a.view()

    # Create a copy of the array
    array_copy = np.copy(a)

    # Create a deep copy of the array
    array_deep_copy = a.copy()
        ''')

    # Sorting arrays
    col3.subheader('Sorting Arrays')
    col3.code('''
    # Sort an array
    a.sort()

    # Sort the elements of an array's axis
    c.sort(axis=0)
        ''')


    # Subsetting, Slicing, and Indexing
    col3.subheader('Subsetting, Slicing, and Indexing')
    col3.code('''
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
        ] # array([ 4. , 2. , 6. , 1.5])
    fancy_indexing_subset = b[[1, 0, 1, 0]][:, [0, 1, 2, 0]] 
        ''')

    # Array Manipulation
    col3.subheader('Array Manipulation')
    col3.code('''
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
        ''')

# Run the main function if the script is executed directly
if __name__ == '__main__':
    main()
