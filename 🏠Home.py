import streamlit as st
import base64

import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Cheat Sheets for Data Science Learning",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Page Title
# Ruta de la imagen PNG generada
image_url = "https://raw.githubusercontent.com/fralfaro/DS-Cheat-Sheets/refs/heads/main/docs/images/rpython.svg" 

# Título con imagen
st.markdown(
    f"""
    <h1 style="display: flex; align-items: center;">
        <img src="{image_url}" alt="RPython Logo" style="margin-right: 10px; height: 50px;">
        Cheat Sheets for Data Science Learning
        
    </h1>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
Welcome to the **Ultimate Data Science Cheat Sheet Repository**, thoughtfully designed for Python and R enthusiasts.  
Whether you're just starting out or are an experienced professional, these cheat sheets offer a **comprehensive and flexible learning experience** in three convenient formats:

- 🚀 **Streamlit**: Interactive and user-friendly.
- 📄 **PDF**: Downloadable and easy to reference.
- 💻 **Google Colab**: Ready-to-run for hands-on learning.
"""
)





# Section: Python
st.markdown("")
st.markdown("")

tab1,tab2 = st.tabs(["📗 Python Cheat Sheets", "📘 R Cheat Sheets"])

with tab1:
    # Python Cards
    st.markdown("### Popular Python Topics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image("docs/images/python.png", width=120)
        st.write("**Python**")
        st.write(
            "Python, an easily readable high-level programming language, was created by Guido van Rossum in 1991."
        )
        if st.button("Learn Python", key="python"):
            st.write("🔗 [Go to Python Cheat Sheet](./python)")

    with col2:
        st.image("docs/images/numpy.png", width=120)
        st.write("**NumPy**")
        st.write(
            "NumPy is Python's essential library for scientific computing, offering high-performance multidimensional arrays."
        )
        if st.button("Learn NumPy", key="numpy"):
            st.write("🔗 [Go to NumPy Cheat Sheet](./numpy)")

    with col3:
        st.image("docs/images/pandas.png", width=120)
        st.write("**Pandas**")
        st.write(
            "Pandas, built on top of NumPy, offers powerful and user-friendly data structures and analysis tools for Python."
        )
        if st.button("Learn Pandas", key="pandas"):
            st.write("🔗 [Go to Pandas Cheat Sheet](./pandas)")

    # Python Cards
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.image("docs/images/matplotlib.png", width=120)
        st.write("**Matplotlib**")
        st.write(
            "Matplotlib is a Python library for creating high-quality 2D plots in various formats and interactive platforms."
        )
        if st.button("Learn Matplotlib", key="matplotlib"):
            st.write("🔗 [Go to Matplotlib Cheat Sheet](./matplotlib)")

    with col6:
        st.image("docs/images/sklearn.png", width=120)
        st.write("**Scikit-Learn**")
        st.write(
            "Scikit-learn is an open-source Python library that offers a unified interface for machine learning algorithms."
        )
        if st.button("Learn Scikit-Learn", key="sklearn"):
            st.write("🔗 [Go to Scikit-Learn Cheat Sheet](./sklearn)")

    with col7:
        st.image("docs/images/polars.png", width=120)
        st.write("**Polars**")
        st.write(
            "Polars is a highly efficient and versatile DataFrame library for working with structured data (written in Rust)."
        )
        if st.button("Learn Polars", key="polars"):
            st.write("🔗 [Go to Polars Cheat Sheet](./polars)")

with tab2:
    # Section: R
    st.markdown("## 📘 R Cheat Sheets")

    # R Cards
    st.markdown("### Popular R Topics")
    col9, col10, col11, col12 = st.columns(4)

    with col9:
        st.image("docs/images/dplyr.png", width=120)
        st.write("**dplyr**")
        st.write(
            "dplyr is a tool that provides a consistent set of actions for common data manipulation tasks."
        )
        if st.button("Learn dplyr", key="dplyr"):
            st.write("🔗 [Go to dplyr Cheat Sheet](./dplyr)")

    with col10:
        st.image("docs/images/ggplot2.png", width=120)
        st.write("**ggplot2**")
        st.write(
            "ggplot2 is a system for declaratively creating graphics, based on The Grammar of Graphics."
        )
        if st.button("Learn ggplot2", key="ggplot2"):
            st.write("🔗 [Go to ggplot2 Cheat Sheet](./ggplot2)")

    with col11:
        st.image("docs/images/forcats.png", width=120)
        st.write("**forcats**")
        st.write(
            "forcats package provides tools to solve common problems with factors in R."
        )
        if st.button("Learn forcats", key="forcats"):
            st.write("🔗 [Go to forcats Cheat Sheet](./forcats)")

# Footer
st.markdown(
    """
---
> 🔑 **Note**: The PDF cheat sheets in this repository are created by various contributors and have inspired the content presented here.
"""
)



css = '''
    <style>
        /* Ajusta el tamaño del texto en las pestañas (Tabs) */
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.5rem; /* Tamaño del texto en las pestañas */
        }

        /* Opción adicional: Ajusta el tamaño de los encabezados dentro de los expanders */
        .st-expander h1, .st-expander h2, .st-expander h3 {
            font-size: 4rem; /* Tamaño de los encabezados dentro de los expanders */
        }

        /* Ajustar el tamaño del texto del selectbox en el sidebar */
        .sidebar .stSelectbox label {
            font-size: 1.5rem; /* Ajusta este valor para cambiar el tamaño del texto */
        }
        .square-image {
            width: 150px;
            height: 150px;
            object-fit: cover; /* Ensures the image fits into a square */
        }
        /* Adjust tab text size */
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.5rem;
        }
    </style>
    '''

st.markdown(css, unsafe_allow_html=True)