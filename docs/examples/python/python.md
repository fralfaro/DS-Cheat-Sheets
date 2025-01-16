<div style="display: flex; align-items: center;">
    <a href="https://colab.research.google.com/github/fralfaro/DS-Cheat-Sheets/blob/main/docs/examples/python/python.ipynb" target="_parent">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
</div>

# Python 


<img src="https://raw.githubusercontent.com/fralfaro/DS-Cheat-Sheets/main/docs/examples/python/python.png" alt="numpy logo" width = "200">

[Python](https://www.python.org/) is a high-level, interpreted programming
 language known for its simplicity, readability, and versatility. 
 Created by Guido van Rossum and first released in 1991,
  Python has gained immense popularity in various domains,
   including web development, data science, automation, and more.

## Why Python?

Python has experienced a remarkable surge in popularity over the years and has become one of the most 
    widely used programming languages across various fields. 

<img src="https://raw.githubusercontent.com/fralfaro/DS-Cheat-Sheets/main/docs/examples/python/survey2.png" alt="numpy logo" width = "400">

## Hello, World!



```python
print("Hello, World!")
```

    Hello, World!
    

## Comments


```python
# This is a comment
```

## Variables and Data Types



```python
# Define an integer variable
x = 5               # Integer

# Define a float variable
y = 3.14            # Float

# Define a string variable
name = "John"       # String

# Define a boolean variable
is_student = True   # Boolean

# Print the values of the variables
print("x:", x)
print("y:", y)
print("name:", name)
print("is_student:", is_student)
```

    x: 5
    y: 3.14
    name: John
    is_student: True
    

## Basic Operations



```python
# Perform basic arithmetic operations
sum_result = x + y  # Add x and y
sub_result = x - y  # Subtract y from x
mul_result = x * y  # Multiply x and y
div_result = x / y  # Divide x by y

# Print the results of the arithmetic operations
print("sum_result:", sum_result)
print("sub_result:", sub_result)
print("mul_result:", mul_result)
print("div_result:", div_result)
```

    sum_result: 8.14
    sub_result: 1.8599999999999999
    mul_result: 15.700000000000001
    div_result: 1.592356687898089
    

## String Operations




```python
# String concatenation
full_name = name + " Doe"  # Concatenate 'name' and " Doe"

# Formatted string
formatted_string = f"Hello, {name}!"  # Create a formatted string using 'name'

# Print the results of string operations
print("full_name:", full_name)
print("formatted_string:", formatted_string)
```

    full_name: John Doe
    formatted_string: Hello, John!
    

## Conditional Statements



```python
# Define the values of x and y
x = 7
y = 3.5

# Check if x is greater than y
if x > y:
    print("x is greater")
# If not, check if x is less than y
elif x < y:
    print("y is greater")
# If neither condition is true, they must be equal
else:
    print("x and y are equal")
```

    x is greater
    

## Lists

### Creating Lists



```python
# Create a list with elements 1, 3, and 2
x = [1, 3, 2]

# Print the list
print("x:", x)
```

    x: [1, 3, 2]
    

### List Functions and Methods



```python
# Return a sorted copy of the list
sorted_list = sorted(x)  # Creates a new list with elements in sorted order

# Sorts the list in place (replaces x)
x.sort()  # Modifies the existing list x to be sorted

# Reverse the order of elements in x
reversed_list = list(reversed(x))  # Creates a new list with elements in reversed order

# Reverse the list in place
x.reverse()  # Modifies the existing list x to be reversed

# Count the number of element 2 in the list
count_2 = x.count(2)  # Counts the occurrences of element 2 in the list x

# Print the results of list operations
print("sorted_list:", sorted_list)
print("reversed_list:", reversed_list)
print("count_2:", count_2)
```

    sorted_list: [1, 2, 3]
    reversed_list: [3, 2, 1]
    count_2: 1
    

### Selecting List Elements



```python
# Select the 0th element in the list
element_0 = x[0]  # Assigns the first element of x to element_0

# Select the last element in the list
last_element = x[-1]  # Assigns the last element of x to last_element

# Select 1st (inclusive) to 3rd (exclusive)
subset_1_to_3 = x[1:3]  # Creates a subset containing elements from index 1 to 2

# Select the 2nd to the end
subset_2_to_end = x[2:]  # Creates a subset containing elements from index 2 to the end

# Select 0th to 3rd (exclusive)
subset_0_to_3 = x[:3]  # Creates a subset containing elements from index 0 to 2

# Print the selected elements and subsets
print("element_0:", element_0)
print("last_element:", last_element)
print("subset_1_to_3:", subset_1_to_3)
print("subset_2_to_end:", subset_2_to_end)
print("subset_0_to_3:", subset_0_to_3)
```

    element_0: 3
    last_element: 1
    subset_1_to_3: [2, 1]
    subset_2_to_end: [1]
    subset_0_to_3: [3, 2, 1]
    

### Concatenating Lists



```python
# Define the x and y lists
x = [1, 3, 6]
y = [10, 15, 21]

# Concatenate lists using '+'
concatenated_list = x + y  # Creates a new list by concatenating x and y

# Replicate elements in a list using '*'
replicated_list = 3 * x  # Creates a new list with elements of x replicated 3 times

# Print the results of list operations
print("concatenated_list:", concatenated_list)
print("replicated_list:", replicated_list)
```

    concatenated_list: [1, 3, 6, 10, 15, 21]
    replicated_list: [1, 3, 6, 1, 3, 6, 1, 3, 6]
    

## Dictionaries

### Creating Dictionaries



```python
# Create a dictionary with key-value pairs
my_dict = {'a': 1, 'b': 2, 'c': 3}

# Print the dictionary
print("my_dict:", my_dict)
```

    my_dict: {'a': 1, 'b': 2, 'c': 3}
    

### Dictionary Functions and Methods



```python
# Get the keys of a dictionary
keys = my_dict.keys()  # Returns dict_keys(['a', 'b', 'c'])

# Get the values of a dictionary
values = my_dict.values()  # Returns dict_values([1, 2, 3])

# Get a value from a dictionary by specifying the key
value_a = my_dict['a']  # Returns 1

# Print the results of dictionary operations
print("keys:", keys)
print("values:", values)
print("value_a:", value_a)
```

    keys: dict_keys(['a', 'b', 'c'])
    values: dict_values([1, 2, 3])
    value_a: 1
    

## Loops


```python
# Define a list of numbers
numbers = [1, 2, 3, 4, 5]

# For loop
print("Using a for loop:")
for num in numbers:
    print(num)
```

    Using a for loop:
    1
    2
    3
    4
    5
    


```python
# While loop
x = 5
print("Using a while loop:")
while x > 0:
    print(x)
    x -= 1
```

    Using a while loop:
    5
    4
    3
    2
    1
    

## Functions


```python
# Define a function that takes a name parameter
def greet(name):
    return f"Hello, {name}!"

# Call the greet function with the argument "Alice"
greeting = greet("Alice")  # Calls the greet function and stores the result in greeting

# Print the greeting
print("greeting:", greeting)
```

    greeting: Hello, Alice!
    

## Built-in Functions



```python
# Define a list of fruits and numbers
fruits = ['apple', 'banana', 'orange', 'kiwi']
numbers = [14, 27, 8, 42, 5]

# Get the length of the list 'fruits'
len_fruits = len(fruits)

# Find the maximum value in the list 'numbers'
max_number = max(numbers)

# Find the minimum value in the list 'numbers'
min_number = min(numbers)

# Print the results
print("Length of fruits list:", len_fruits)
print("Maximum value in numbers list:", max_number)
print("Minimum value in numbers list:", min_number)
```

    Length of fruits list: 4
    Maximum value in numbers list: 42
    Minimum value in numbers list: 5
    

## Importing Modules



```python
import math

# Calculate square root using math module
sqrt_result = math.sqrt(x)

# Generate a random number between 1 and 10
from random import randint
random_number = randint(1, 10)

# Reusing the math module for another calculation
sqrt_result_reuse = math.sqrt(x)

# Print the results
print("sqrt_result:", sqrt_result)
print("random_number:", random_number)
print("sqrt_result_reuse:", sqrt_result_reuse)
```

    sqrt_result: 0.0
    random_number: 6
    sqrt_result_reuse: 0.0
    

## Classes and Objects



```python
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bark(self):
        return "Woof!"

# Create an instance of the Dog class
my_dog = Dog("Buddy", 3)

# Print the attributes of the instance
print("my_dog name:", my_dog.name)
print("my_dog age:", my_dog.age)

# Call the bark method of the instance
bark_result = my_dog.bark()
print("bark_result:", bark_result)
```

    my_dog name: Buddy
    my_dog age: 3
    bark_result: Woof!
    

## Input/Output and File Handling

**Input/Output**

```python
# Get user input and display it
user_input = input("Enter a number: ")
print("You entered:", user_input)
```


**File Handling**

```python
# Read content from a file
with open("file.txt", "r") as file:
    content = file.read()

# Write content to a new file
with open("new_file.txt", "w") as new_file:
    new_file.write("Hello, world!")
```
