import streamlit as st
from pathlib import Path
import base64
import requests


# Initial page config
st.set_page_config(
    page_title='Dplyr Cheat Sheet',
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
    Populate the sidebar with various content sections related to dplyr.
    """
    st.sidebar.markdown(
        '''[<img src='data:image/png;base64,{}' class='img-fluid' width=95 >](https://streamlit.io/)'''.format(
            img_to_bytes("https://rstudio.github.io/cheatsheets/html/images/logo-dplyr.png")), unsafe_allow_html=True)

    st.sidebar.header('Dplyr Cheat Sheet')
    st.sidebar.markdown('''
<small>[dplyr](https://dplyr.tidyverse.org/) functions work with pipes and expect **tidy data**. In tidy data:

*   Each **variable** is in its own **column**
*   Each **observation**, or **case**, is in its own **row**
*   **pipes** `x |> f(y)` becomes `f(x,y)`</small>
    ''', unsafe_allow_html=True)

    # dplyr installation and import
    st.sidebar.markdown('__Install and import dplyr__')
    st.sidebar.code('''$ install.packages('dplyr')''')
    st.sidebar.code('''
# Import dplyr 
>>> library(dplyr)
''')


    return None


# Define the cs_body() function
def cs_body():
    """
    Create content sections for the main body of the Streamlit cheat sheet with dplyr examples.
    """
    col1, col2, col3 = st.columns(3)  # Create columns for layout

    #######################################
    # COLUMN 1
    #######################################

    # Summarize Cases
    col1.subheader('Summarize Cases')
    col1.code('''
    # Compute table of summaries.
    mtcars |> summarize(avg = mean(mpg))
        ''')

    col1.code('''
    # Count number of rows in each group.
    mtcars |> count(cyl)
        ''')

    # Group Cases
    col1.subheader('Group Cases')
    col1.code('''
    # created a “grouped” copy of a table grouped by columns in `...`
    mtcars |>
          group_by(cyl) |>
          summarize(avg = mean(mpg))
        ''')
    col1.code('''
    # to group data into individual rows
    starwars |>
          rowwise() |>
          mutate(film_count = length(films))
        ''')
    col1.code('''
    # Returns ungrouped copy of table.
    ungroup(x, ...)
        ''')


    # Extract Cases
    col1.subheader('Extract Cases')
    col1.code('''
     # Extract rows that meet logical criteria.
     mtcars |> filter(mpg > 20)
         ''')
    col1.code('''
     # Remove rows with duplicate values.
     mtcars |> distinct(gear)
         ''')
    col1.code('''
     # Select rows by position.
      mtcars |> slice(10:15)
         ''')
    col1.code('''
     # Randomly select rows.
      mtcars |> slice_sample(n = 5, replace = TRUE)
         ''')
    col1.code('''
     # Select rows with the lowest and highest values.
      mtcars |> slice_min(mpg, prop = 0.25)
         ''')
    col1.code('''
     # Select the first or last rows.
      mtcars |> slice_head(n = 5)
         ''')
    col1.subheader('''Logical and boolean operations to use with `filter()`''')
    col1.markdown('''
        *   `==`
    *   `<`
    *   `<=`
    *   `is.na()`
    *   `%in%`
    *   `|`
    *   `xor()`
    *   `!=`
    *   `>`
    *   `>=`
    *   `!is.na()`
    *   `!`
    *   `&`
    *   See `?base::Logic` and `?Comparison` for help.
            ''')

    # Arrange Cases
    col1.subheader('Arrange Cases')
    col1.code('''
     # Order rows by values of a column or columns (low to high)
     mtcars |> arrange(mpg)
     mtcars |> arrange(desc(mpg))
         ''')

    # Add Cases
    col1.subheader('Add Cases')
    col1.code('''
     # Add one or more rows to a table.
     cars |> add_row(speed = 1, dist = 1)
         ''')


    #######################################
    # COLUMN 2
    #######################################

    # Extract Variables
    col1.subheader('Extract Variables')
    col1.code('''
    # Extract column values as a vector, by name or index.
    mtcars |> pull(wt)
        ''')
    col1.code('''
    # Extract columns as a table.
    mtcars |> select(mpg, wt)
        ''')
    col1.code('''
    # Move columns to new position.
    mtcars |> relocate(mpg, cyl, after = last_col())
        ''')

    # Logical
    col2.subheader('''More about: `select()` and `across()`''')
    col2.code('''
            mtcars |> select(mpg:cyl)
                ''')
    col2.markdown('''
    *   `contains(match)`
    *   `num_range(prefix, range)`
    *   `:`, e.g., `mpg:cyl`
    *   `ends_with(match)`
    *   `all_of(x)` or `any_of(x, ..., vars)`
    *   `!`, e.g., `!gear`
    *   `starts_with(match)`
    *   `matches(match)`
    *   `everything()`
                ''')

    # Manipulate Multiple Variables at Once
    col2.subheader('Manipulate Multiple Variables at Once')
    col2.code('''
        df <- tibble(x_1 = c(1, 2), x_2 = c(3, 4), y = c(4, 5))
            ''')
    col2.code('''
        # summarize or mutate multiple columns in the same way.
        df |> summarize(across(everything(), mean))
            ''')
    col2.code('''
        # Compute across columns in row-wise data.
        df |> 
          rowwise() |>
          mutate(x_total = sum(c_across(1:2)))
            ''')

    # Make New Variables
    col2.subheader('Make New Variables')
    col2.code('''
        # Compute new column(s). Also add_column().
        mtcars |> mutate(gpm = 1 / mpg)
        mtcars |> mutate(mtcars, gpm = 1 / mpg, .keep = "none")
            ''')
    col2.code('''
        # Rename columns. Use rename_with() to rename with a function.
        mtcars |> rename(miles_per_gallon = mpg)
            ''')



    # Logical
    col2.subheader('''Vectorized Functions''')
    col2.markdown('''
    ### To Use with `mutate()`[](#to-use-with-mutate)

    `mutate()` applies vectorized functions to columns to create new columns. Vectorized functions take vectors as input and return vectors of the same length as output.

    ### Offset[](#offset)

    *   `dplyr::lag()`: offset elements by 1
    *   `dplyr::lead()`: offset elements by -1

    ### Cumulative Aggregate[](#cumulative-aggregate)

    *   `dplyr::cumall()`: cumulative `all()`
    *   `dply::cumany()`: cumulative `any()`
    *   `cummax()`: cumulative `max()`
    *   `dplyr::cummean()`: cumulative `mean()`
    *   `cummin()`: cumulative `min()`
    *   `cumprod()`: cumulative `prod()`
    *   `cumsum()`: cumulative `sum()`

    ### Ranking[](#ranking)

    *   `dplyr::cume_dist()`: proportion of all values <=
    *   `dplyr::dense_rank()`: rank with ties = min, no gaps
    *   `dplyr::min_rank()`: rank with ties = min
    *   `dplyr::ntile()`: bins into n bins
    *   `dplyr::percent_rank()`: `min_rank()` scaled to \[0,1\]
    *   `dplyr::row_number()`: rank with ties = “first”

    ### Math[](#math)

    *   `+`, `-`, `/`, `^`, `%/%`, `%%`: arithmetic ops
    *   `log()`, `log2()`, `log10()`: logs
    *   `<`, `<=`, `>`, `>=`, `!=`, `==`: logical comparisons
    *   `dplyr::between()`: x >= left & x <= right
    *   `dplyr::near()`: safe `==` for floating point numbers

    ### Miscellaneous[](#miscellaneous)

    *   `dplyr::case_when()`: multi-case `if_else()`

            starwars |>
              mutate(type = case_when(
                height > 200 | mass > 200 ~ "large",
                species == "Droid" ~ "robot",
                TRUE ~ "other"
              ))

    *   `dplyr::coalesce()`: first non-NA values by element across a set of vectors

    *   `dplyr::if_else()`: element-wise if() + else()

    *   `dplyr::na_if()`: replace specific values with NA

    *   `pmax()`: element-wise max()

    *   `pmin()`: element-wise min()
                    ''')




    #######################################
    # COLUMN 3
    #######################################

    # Row Names
    col3.subheader('Row Names')
    col3.code('''
        # Move row names into col.
        a <- rownames_to_column(mtcars, var = "C")
            ''')
    col3.code('''
        # Move col into row names.
        column_to_rownames(a, var = "C")
            ''')

    # Summary Functions
    col3.subheader('''Summary Functions''')
    col3.markdown('''
    ### To Use with `summarize()`[](#to-use-with-summarize)

    `summarize()` applies summary functions to columns to create a new table. Summary functions take vectors as input and return single values as output.

    ### Count[](#count)

    *   `dplyr::n()`: number of values/rows
    *   `dplyr::n_distinct()`: # of uniques
    *   `sum(!is.na())`: # of non-NAs

    ### Position[](#position)

    *   `mean()`: mean, also `mean(!is.na())`
    *   `median()`: median

    ### Logical[](#logical)

    *   `mean()`: proportion of TRUEs
    *   `sum()`: # of TRUEs

    ### Order[](#order)

    *   `dplyr::first()`: first value
    *   `dplyr::last()`: last value
    *   `dplyr::nth()`: value in the nth location of vector

    ### Rank[](#rank)

    *   `quantile()`: nth quantile
    *   `min()`: minimum value
    *   `max()`: maximum value

    ### Spread[](#spread)

    *   `IQR()`: Inter-Quartile Range
    *   `mad()`: median absolute deviation
    *   `sd()`: standard deviation
    *   `var()`: variance
                ''')

    # Relational Data
    col3.subheader('''Relational Data''')
    col3.markdown('''
        Use a **“Mutating Join”** to join one table to columns from another, matching values with the rows that the correspond to.

    *   `left_join(x, y, by = NULL, copy = FALSE, suffix = c(".x", ".y"), ..., keep = FALSE, na_matches = "na")`: Join matching values from `y` to `x`.
    *   `right_join(x, y, by = NULL, copy = FALSE, suffix = c(".x", ".y"), ..., keep = FALSE, na_matches = "na")`: Join matching values from `x` to `y`.
    *   `inner_join(x, y, by = NULL, copy = FALSE, suffix = c(".x", ".y"), ..., keep = FALSE, na_matches = "na")`: Join data. retain only rows with matches.
    *   `full_join(x, y, by = NULL, copy = FALSE, suffix = c(".x", ".y"), ..., keep = FALSE, na_matches = "na")`: Join data. Retain all values, all rows.

    Use a **“Filtering Join”** to filter one table against the rows of another.

    *   `semi_join(x, y, by = NULL, copy = FALSE, ..., na_matches = "na")`: Return rows of `x` that have a match in `y`. Use to see what will be included in a join.
    *   `anti_join(x, y, by = NULL, copy = FALSE, ..., na_matches = "na")`: Return rows of `x` that do not have a match in `y`. Use to see what will not be included in a join.

    Use a **“Nest Join”** to inner join one table to another into a nested data frame.

    *   `nest_join(x, y, by = NULL, copy = FALSE, keep = FALSE, name = NULL, ...)`: Join data, nesting matches from `y` in a single new data frame column.

                    ''')

    # Column Matching for Joins
    col3.subheader('Column Matching for Joins')
    col3.code('''
        # Use by = join_by(col1, col2, …) to specify one or more common columns to match on.
        left_join(x, y, by = join_by(A))
        left_join(x, y, by = join_by(A, B))
            ''')
    col3.code('''
        # Use a logical statement, by = join_by(col1 == col2), to match on columns that have different names in each table.
        left_join(x, y, by = join_by(C == D))
            ''')
    col3.code('''
        # Use suffix to specify the suffix to give to unmatched columns that have the same name in both tables.
        left_join(x, y, by = join_by(C == D), suffix = c("1", "2"))
            ''')





   # Set Operations
    col3.subheader('''Set Operations''')
    col3.markdown('''
*   `intersect(x, y, ...)`: Rows that appear in both `x` and `y`.
*   `setdiff(x, y, ...)`: Rows that appear in `x` but not `y`.
*   `union(x, y, ...)`: Rows that appear in x or y, duplicates removed. `union_all()` retains duplicates.
*   Use `setequal()` to test whether two data sets contain the exact same rows (in any order).
            ''')






# Run the main function if the script is executed directly
if __name__ == '__main__':
    main()
