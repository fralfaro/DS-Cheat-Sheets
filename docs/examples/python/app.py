import streamlit as st
from pathlib import Path
import base64
import requests


# Initial page config
st.set_page_config(
    page_title='Python Cheat Sheet',
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
    Populate the sidebar with various content sections related to Python.
    """
    st.sidebar.markdown(
        '''[<img src='data:image/png;base64,{}' class='img-fluid' width=95 >](https://streamlit.io/)'''.format(
            img_to_bytes("https://raw.githubusercontent.com/fralfaro/DS-Cheat-Sheets/main/docs/examples/python/python.png")), unsafe_allow_html=True)

    st.sidebar.header('Python Cheat Sheet')
    st.sidebar.markdown('''
<small>[Python](https://www.python.org/) is a high-level, interpreted programming
 language known for its simplicity, readability, and versatility. 
 Created by Guido van Rossum and first released in 1991,
  Python has gained immense popularity in various domains,
   including web development, data science, automation, and more.</small>
    ''', unsafe_allow_html=True)

    # why python ?
    st.sidebar.markdown('__Why Python?__')
    st.sidebar.markdown('''
    <small>  Python has experienced a remarkable surge in popularity over the years and has become one of the most 
    widely used programming languages across various fields. </small> ''', unsafe_allow_html=True)

    st.sidebar.markdown(
        '''[<img src='data:image/png;base64,{}' class='img-fluid' width=300 >](https://streamlit.io/)'''.format(
            img_to_bytes("https://raw.githubusercontent.com/fralfaro/DS-Cheat-Sheets/main/docs/examples/python/survey2.png")), unsafe_allow_html=True)

    return None


# Define the cs_body() function
def cs_body():
    """
    Create content sections for the main body of the Streamlit cheat sheet with Python examples.
    """
    col1, col2, col3 = st.columns(3)  # Create columns for layout

    #######################################
    # COLUMN 1
    #######################################

    # Hello, World!
    col1.subheader('Hello, World!')
    col1.code('''
    print("Hello, World!")
        ''')

    # Comments
    col1.subheader('Comments')
    col1.code('''
    # This is a comment
        ''')

    # Variables and Data Types
    col1.subheader('Variables and Data Types')
    col1.code('''
    x = 5               # Integer
    y = 3.14            # Float
    name = "John"       # String
    is_student = True   # Boolean
        ''')

    # Basic Operations
    col1.subheader('Basic Operations')
    col1.code('''
    # Perform basic arithmetic operations
    sum_result = x + y
    sub_result = x - y
    mul_result = x * y
    div_result = x / y
        ''')


    # String Operations
    col1.subheader('String Operations')
    col1.code('''
    # String concatenation
    full_name = name + " Doe"

    # Formatted string
    formatted_string = f"Hello, {name}!"
        ''')

    # Input and Output
    col1.subheader('Input and Output')
    col1.code('''
    # Get user input and display it
    user_input = input("Enter a number: ")
    print("You entered:", user_input)
        ''')

    # File Handling
    col1.subheader('File Handling')
    col1.code('''
    # Read content from a file
    with open("file.txt", "r") as file:
        content = file.read()

    # Write content to a new file
    with open("new_file.txt", "w") as new_file:
        new_file.write("Hello, world!")
        ''')

    # Conditional Statements
    col1.subheader('Conditional Statements')
    col1.code('''
    # Check if x is greater than y
    if x > y:
        print("x is greater")
    # If not, check if x is less than y
    elif x < y:
        print("y is greater")
    # If neither condition is true, they must be equal
    else:
        print("x and y are equal")
        ''')

    #######################################
    # COLUMN 2
    #######################################

    # Creating lists
    col2.subheader('Creating Lists')
    col2.code('''
    # Create lists with [], elements separated by commas
    x = [1, 3, 2]
        ''')

    # List functions and methods
    col2.subheader('List Functions and Methods')
    col2.code('''
    # Return a sorted copy of the list e.g., [1,2,3]
    sorted_list = sorted(x)

    # Sorts the list in place (replaces x)
    x.sort()

    # Reverse the order of elements in x e.g., [2,3,1]
    reversed_list = list(reversed(x))

    # Reverse the list in place
    x.reverse()

    # Count the number of element 2 in the list
    count_2 = x.count(2)
        ''')

    # Selecting list elements
    col2.subheader('Selecting List Elements')
    col2.code('''
    # Select the 0th element in the list
    element_0 = x[0]

    # Select the last element in the list
    last_element = x[-1]

    # Select 1st (inclusive) to 3rd (exclusive)
    subset_1_to_3 = x[1:3]

    # Select the 2nd to the end
    subset_2_to_end = x[2:]

    # Select 0th to 3rd (exclusive)
    subset_0_to_3 = x[:3]
        ''')

    # Concatenating lists
    col2.subheader('Concatenating Lists')
    col2.code('''
    # Define the x and y lists
    x = [1, 3, 6]
    y = [10, 15, 21]

    # Concatenate lists using '+'
    concatenated_list = x + y

    # Replicate elements in a list using '*'
    replicated_list = 3 * x
        ''')

    # Creating dictionaries
    col2.subheader('Creating Dictionaries')
    col2.code('''
    # Create a dictionary with {}
    my_dict = {'a': 1, 'b': 2, 'c': 3}

        ''')

    # Dictionary functions and methods
    col2.subheader('Dictionary Functions and Methods')
    col2.code('''
    # Get the keys of a dictionary
    my_dict.keys()  # Returns dict_keys(['a', 'b', 'c'])

    # Get the values of a dictionary
    my_dict.values()  # Returns dict_values([1, 2, 3])
    
    # Get a value from a dictionary by specifying the key
    my_dict['a']  # Returns 1  
        ''')

    #######################################
    # COLUMN 3
    #######################################

    # Loops
    col3.subheader('Loops')
    col3.code('''
    # For loop
    numbers = [1, 2, 3, 4, 5]
    for num in numbers:
        print(num)

    # While loop
    x = 5
    while x > 0:
        print(x)
        x -= 1
        ''')

    # Error Handling
    col3.subheader('Error Handling')
    col3.code('''
    # Try to perform division
    try:
        result = x / y  # Attempt to divide x by y
    except ZeroDivisionError as e:  # If a ZeroDivisionError occurs
        print("Cannot divide by zero")  
        print(e)
        ''')

    # List Comprehensions
    col3.subheader('List Comprehensions')
    col3.code('''
    # Create a list of squared numbers using a list comprehension
    squared_numbers = [num**2 for num in numbers]

    # Create a list of even numbers using a list comprehension with condition
    even_numbers = [num for num in numbers if num % 2 == 0]
        ''')

    # Functions
    col3.subheader('Functions')
    col3.code('''
    # Define a function that takes a name parameter
    def greet(name):
        return f"Hello, {name}!"

    # Call the greet function with the argument "Alice"
    greeting = greet("Alice")
        ''')

    # Built-in Functions
    col3.subheader('Built-in Functions')
    col3.code('''
    # Get the length of a list
    len_fruits = len(fruits)

    # Find the maximum value in a list
    max_number = max(numbers)

    # Find the minimum value in a list
    min_number = min(numbers)
        ''')

    # Importing Modules
    col3.subheader('Importing Modules')
    col3.code('''
    import math
    sqrt_result = math.sqrt(x)  # Calculate square root using math module

    from random import randint
    random_number = randint(1, 10)  # Generate a random number between 1 and 10

    import math
    sqrt_result = math.sqrt(x)  # Reusing the math module for another calculation
        ''')

    # Classes and Objects
    col3.subheader('Classes and Objects')
    col3.code('''
    class Dog:
        def __init__(self, name, age):
            self.name = name
            self.age = age

        def bark(self):
            return "Woof!"

    my_dog = Dog("Buddy", 3)
        ''')


# Run the main function if the script is executed directly
if __name__ == '__main__':
    main()
