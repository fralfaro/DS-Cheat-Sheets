import streamlit as st
from pathlib import Path
import base64
import requests

# Initial page config
st.set_page_config(
    page_title='Pandas Cheat Sheet',
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
    Populate the sidebar with various content sections related to Pandas.
    """
    st.sidebar.markdown(
        '''[<img src='data:image/png;base64,{}' class='img-fluid' width=200 >](https://streamlit.io/)'''.format(
            img_to_bytes("https://raw.githubusercontent.com/fralfaro/DS-Cheat-Sheets/main/docs/examples/pandas/pandas.png")), unsafe_allow_html=True)

    st.sidebar.header('Pandas Cheat Sheet')
    st.sidebar.markdown('''
<small>[Pandas](https://pandas.pydata.org/) is built on NumPy and provides easy-to-use
data structures and data analysis tools for the Python
programming language.</small>
    ''', unsafe_allow_html=True)

    # Pandas installation and import
    st.sidebar.markdown('__Install and import Pandas__')
    st.sidebar.code('$ pip install pandas')
    st.sidebar.code('''
# Import Pandas convention
>>> import pandas as pd
''')

    # Pandas array creation
    st.sidebar.subheader('Pandas Data Structures')
    st.sidebar.markdown('__Series__')
    st.sidebar.markdown(
        '''[<img src='data:image/png;base64,{}' class='img-fluid' width=100 >](https://streamlit.io/)'''.format(
            img_to_bytes("https://raw.githubusercontent.com/fralfaro/DS-Cheat-Sheets/main/docs/examples/pandas/serie.png")), unsafe_allow_html=True)

    st.sidebar.markdown('''
    <small>A **one-dimensional** labeled array a capable of holding any data type.</small>
        ''', unsafe_allow_html=True)

    st.sidebar.code('''
    # Create a pandas Series
    s = pd.Series(
        [3, -5, 7, 4],
        index=['a', 'b', 'c', 'd']
    )
    ''')

    st.sidebar.markdown('__DataFrame__')
    st.sidebar.markdown('''
        <small>A **two-dimensional** labeled data structure with columns of potentially different types.</small>
            ''', unsafe_allow_html=True)
    st.sidebar.markdown(
        '''[<img src='data:image/png;base64,{}' class='img-fluid' width=300 >](https://streamlit.io/)'''.format(
            img_to_bytes("https://raw.githubusercontent.com/fralfaro/DS-Cheat-Sheets/main/docs/examples/pandas/df.png")), unsafe_allow_html=True)

    st.sidebar.code('''
    # Create a pandas DataFrame
    data = {
        'Country': ['Belgium', 'India', 'Brazil'],
        'Capital': ['Brussels', 'New Delhi', 'Brasília'],
        'Population': [11190846, 1303171035, 207847528]
    }
    df = pd.DataFrame(
        data,
        columns=['Country', 'Capital', 'Population']
    )
    ''')

    return None


# Define the cs_body() function
def cs_body():
    """
    Create content sections for the main body of the Streamlit cheat sheet with Pandas examples.
    """
    col1, col2, col3 = st.columns(3)  # Create columns for layout

    #######################################
    # COLUMN 1
    #######################################

    # Read and Write to CSV
    col1.subheader('Read and Write to CSV')
    col1.code('''
    # Read from CSV
    df_read = pd.read_csv(
        'file.csv',
         header=None, 
         nrows=5
    )

    # Write to CSV
    df.to_csv('myDataFrame.csv')
        ''')

    # Read and Write to Excel
    col1.subheader('Read and Write to Excel')
    col1.code('''
    # Read from Excel
    df_read_excel = pd.read_excel('file.xlsx')

    # Write to Excel
    df.to_excel(
        'dir/myDataFrame.xlsx', 
        sheet_name='Sheet1'
    )

    # Read multiple sheets from the same file
    xlsx = pd.ExcelFile('file.xls')
    df_from_sheet1 = pd.read_excel(xlsx, 'Sheet1')
        ''')

    # Read and Write to SQL Query or Database Table
    col1.subheader('Read and Write to SQL Query or Database Table')
    col1.code('''
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///:memory:')

    # Read from SQL Query
    pd.read_sql("SELECT * FROM my_table;", engine)

    # Read from Database Table
    pd.read_sql_table('my_table', engine)

    # Read from SQL Query using read_sql_query()
    pd.read_sql_query("SELECT * FROM my_table;", engine)

    # Write DataFrame to SQL Table
    pd.to_sql('myDf', engine)
        ''')


    # Getting Elements from Series and DataFrame
    col1.subheader('Getting Elements')
    col1.code('''
    # Get one element from a Series
    s['b']
    # Output: -5

    # Get subset of a DataFrame
    df[1:]
    # Output:
    #    Country     Capital  Population
    # 1    India   New Delhi  1303171035
    # 2   Brazil    Brasília   207847528
        ''')

    # Asking For Help
    col1.subheader('Asking For Help')
    col1.code('''
    # Display help for a function or object
    help(pd.Series.loc)
        ''')

    #######################################
    # COLUMN 2
    #######################################

    # Selecting, Boolean Indexing & Setting
    col2.subheader('Selecting, Boolean Indexing & Setting')
    col2.code('''
    # Select single value by row & 'Belgium' column
    df.iloc[[0],[0]]
    # Output: 'Belgium'

    # Select single value by row & 'Belgium' column labels
    df.loc[[0], ['Country']]
    # Output: 'Belgium'

    # Select single row of subset of rows
    df.loc[2]
    # Output:
    # Country     Brazil
    # Capital    Brasília
    # Population 207847528

    # Select a single column of subset of columns
    df.loc[:,'Capital']
    # Output:
    # 0     Brussels
    # 1    New Delhi
    # 2     Brasília

    # Boolean indexing - Series s where value is not > 1
    s[~(s > 1)]

    # Boolean indexing - s where value is <-1 or >2
    s[(s < -1) | (s > 2)]

    # Use filter to adjust DataFrame
    df[df['Population'] > 1200000000]

    # Setting index a of Series s to 6
    s['a'] = 6
        ''')

    # Dropping
    col2.subheader('Dropping')
    col2.code('''
    # Drop values from rows (axis=0)
    s.drop(['a', 'c'])

    # Drop values from columns (axis=1)
    df.drop('Country', axis=1)
        ''')

    # Sort & Rank
    col2.subheader('Sort & Rank')
    col2.code('''
    # Sort by labels along an axis
    df.sort_index()

    # Sort by the values along an axis
    df.sort_values(by='Country')

    # Assign ranks to entries
    df.rank()
        ''')

    # Applying Functions
    col2.subheader('Applying Functions')
    col2.code('''
    # Define a function
    f = lambda x: x*2

    # Apply function to DataFrame
    df.apply(f)

    # Apply function element-wise
    df.applymap(f)
        ''')

    #######################################
    # COLUMN 3
    #######################################

    # Basic Information
    col3.subheader('Basic Information')
    col3.code('''
    # Get the shape (rows, columns)
    df.shape

    # Describe index
    df.index

    # Describe DataFrame columns
    df.columns

    # Info on DataFrame
    df.info()

    # Number of non-NA values
    df.count()
        ''')

    # Summary
    col3.subheader('Summary')
    col3.code('''
    # Sum of values
    df[col].sum()

    # Cumulative sum of values
    df[col].cumsum()

    # Minimum/maximum values
    df[col].min()
    df[col].max()

    # Index of minimum/maximum values
    df[col].idxmin()
    df[col].idxmax()

    # Summary statistics
    df[col].describe()

    # Mean of values
    df[col].mean()

    # Median of values
    df[col].median()
        ''')

    # Internal Data Alignment
    col3.subheader('Internal Data Alignment')
    col3.code('''
    # Create Series with different indices
    s3 = pd.Series([7, -2, 3], index=['a', 'c', 'd'])

    # Add two Series with different indices
    result = s + s3
        ''')

    # Arithmetic Operations with Fill Methods
    col3.subheader('Arithmetic Operations with Fill Methods')
    col3.code('''
    # Perform arithmetic operations with fill methods
    result_add = s.add(s3, fill_value=0)
    result_sub = s.sub(s3, fill_value=2)
    result_div = s.div(s3, fill_value=4)
    result_mul = s.mul(s3, fill_value=3)
        ''')

# Run the main function if the script is executed directly
if __name__ == '__main__':
    main()
