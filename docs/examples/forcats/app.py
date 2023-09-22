import streamlit as st
from pathlib import Path
import base64
import requests


# Initial page config
st.set_page_config(
    page_title='Forcats Cheat Sheet',
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
    Populate the sidebar with various content sections related to forcats.
    """
    st.sidebar.markdown(
        '''[<img src='data:image/png;base64,{}' class='img-fluid' width=95 >](https://streamlit.io/)'''.format(
            img_to_bytes("https://rstudio.github.io/cheatsheets/html/images/logo-forcats.png")), unsafe_allow_html=True)

    st.sidebar.header('Forcats Cheat Sheet')
    st.sidebar.markdown('''
<small>The [forcats](https://forcats.tidyverse.org/) package provides tools for working with factors, which are R’s data structure for categorical data.
</small>
    ''', unsafe_allow_html=True)

    # forcats installation and import
    st.sidebar.markdown('__Install and import forcats__')
    st.sidebar.code('''$ install.packages('forcats')''')
    st.sidebar.code('''
# Import forcats 
>>> library(forcats)
''')


    return None


# Define the cs_body() function
def cs_body():
    """
    Create content sections for the main body of the Streamlit cheat sheet with forcats examples.
    """
    col1, col2, col3 = st.columns(3)  # Create columns for layout

    #######################################
    # COLUMN 1
    #######################################

    # Factors
    col1.subheader('Factors')
    col1.markdown('''
    R represents categorical data with factors. A **factor** is an integer vector with a **levels** attribute that stores a set of mappings between integers and categorical values. When you view a factor, R displays not the integers but the levels associated with them.
    
    For example, R will display `c("a", "c", "b", "a")` with levels `c("a", "b", "c")` but will store `c(1, 3, 2, 1)` where 1 = a, 2 = b, and 3 = c.
        ''')
    col1.code('''
        # R will display:
        [1] a c b a
        Levels: a b c
        ''')
    col1.code('''
        # R will store:
        [1] 1 3 2 1
        attr(,"levels")
        [1] "a" "b" "c"
        ''')


    col1.subheader('''Create a factor with `factor()`''')
    col1.code('''
    # Convert a vector to a factor. Also as_factor().
    f <- factor(c("a", "c", "b", "a"), levels = c("a", "b", "c"))
    ''')
    col1.code('''
    # Return/set the levels of a factor.
    levels(f)
    levels(f) <- c("x", "y", "z")
    ''')

    # Inspect Factors
    col1.subheader('''Inspect Factors''')
    col1.code('''
    # Count the number of values with each level.
    fct_match(f, "a")
    ''')
    col1.code('''
    # Check for lvls in f.
    fct_count(f)
    ''')
    col1.code('''
    # Return the unique values, removing duplicates.
    fct_unique(f)
    ''')


    #######################################
    # COLUMN 2
    #######################################

    # Combine Factors
    col2.subheader('''Combine Factors''')
    col2.code('''
            # Combine factors with different levels. Also fct_cross().
            f1 <- factor(c("a", "c"))
            f2 <- factor(c("b", "a"))
            fct_c(f1, f2)
                ''')
    col2.code('''
            # Standardize levels across a list of factors.
            fct_unify(list(f2, f1))
                ''')

    # Change the order of levels
    col2.subheader('''Change the order of levels''')
    col2.code('''
            # Manually reorder factor levels.
            fct_relevel(f, c("b", "c", "a"))
                ''')
    col2.code('''
            # Reorder levels by the frequency in which they appear in the data (highest frequency first).
            f3 <- factor(c("c", "c", "a"))
            fct_infreq(f3)
                ''')
    col2.code('''
            #  Reverse level order.
            f4 <- factor(c("a","b","c"))
            fct_rev(f4)
                ''')
    col2.code('''
            # Shift levels to left or right, wrapping around end.
            fct_shift(f4)
                ''')
    col2.code('''
            # Randomly permute order of factor levels.
            fct_shuffle(f4)
                ''')
    col2.code('''
            # Reorder levels by their relationship with another variable.
            boxplot(PlantGrowth, weight ~ fct_reorder(group, weight))
                ''')
    col2.code('''
            # Reorder levels by their final values when plotted with two other variables.
            ggplot(
                diamonds,aes(carat, price, 
                color = fct_reorder2(color, carat, price))
                ) + geom_smooth()
                ''')

    #######################################
    # COLUMN 3
    #######################################

    # Change the value of levels
    col3.subheader('Change the value of levels')
    col3.code('''
        # Manually change levels.
        fct_recode(f, v = "a", x = "b", z = "c")
        fct_relabel(f, ~ paste0("x", .x))
            ''')
    col3.code('''
        # Anonymize levels with random integers.
        fct_anon(f)
            ''')
    col3.code('''
        # Collapse levels into manually defined groups.
        fct_collapse(f, x = c("a", "b"))
            ''')
    col3.code('''
        #  Lumps together factors that appear fewer than min times.
        fct_lump_min(f, min = 2)
            ''')
    col3.code('''
        # Replace levels with “other.”
        fct_other(f, keep = c("a", "b"))
            ''')

    # Add or drop levels
    col3.subheader('''Add or drop levels''')
    col3.code('''
        # Drop unused levels.
        f5 <- factor(c("a","b"),c("a","b","x"))
        f6 <- fct_drop(f5)
            ''')
    col3.code('''
        # Add levels to a factor.
        fct_expand(f6, "x")
            ''')
    col3.code('''
        # Assigns a level to NAs to ensure they appear in plots, etc.
        f <- factor(c("a", "b", NA))
        fct_na_value_to_level(
            f, 
            level = "(Missing)"
        )
            ''')





# Run the main function if the script is executed directly
if __name__ == '__main__':
    main()
