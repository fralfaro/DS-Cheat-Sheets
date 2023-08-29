import streamlit as st
from pathlib import Path
import base64
import requests


# Initial page config
st.set_page_config(
    page_title='Matplotlib Cheat Sheet',
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
    Populate the sidebar with various content sections related to Matplotlib.
    """
    st.sidebar.markdown(
        '''[<img src='data:image/png;base64,{}' class='img-fluid' width=150 >](https://streamlit.io/)'''.format(
            img_to_bytes("https://raw.githubusercontent.com/fralfaro/DS-Cheat-Sheets/main/docs/examples/matplotlib/matplotlib.png")), unsafe_allow_html=True)

    st.sidebar.header('Matplotlib Cheat Sheet')
    st.sidebar.markdown('''
<small>[Matplotlib](https://matplotlib.org/) is a Python 2D plotting library which produces publication-quality 
figures in a variety of hardcopy formats and interactive environments across platforms.</small>
    ''', unsafe_allow_html=True)

    # Matplotlib installation and import
    st.sidebar.markdown('__Install and import Matplotlib__')
    st.sidebar.code('$ pip install matplotlib')
    st.sidebar.code('''
# Import Matplotlib convention
>>> import matplotlib.pyplot as plt
''')

    # Anatomy of a figure
    st.sidebar.markdown('__Anatomy of a figure__')
    st.sidebar.markdown(
        '''[<img src='data:image/png;base64,{}' class='img-fluid' width=450 >](https://streamlit.io/)'''.format(
            img_to_bytes("https://raw.githubusercontent.com/fralfaro/DS-Cheat-Sheets/main/docs/examples/matplotlib/mlp_01.png")), unsafe_allow_html=True)

    st.sidebar.markdown('''
    <small>In Matplotlib, a figure refers to the overall canvas or window that contains one or more individual plots or subplots. 
    Understanding the anatomy of a Matplotlib figure is crucial for creating and customizing your visualizations effectively. </small>
        ''', unsafe_allow_html=True)


    # Example code for the workflow
    st.sidebar.code('''
    # Workflow
    import matplotlib.pyplot as plt
    
    # Step 1: Prepare Data
    x = [1, 2, 3, 4]  
    y = [10, 20, 25, 30] 

    # Step 2: Create Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Step 3: Plot
    ax.plot(x, y, color='lightblue', linewidth=3)

    # Step 4: Customized Plot
    ax.scatter([2, 4, 6], [5, 15, 25], color='darkgreen', marker='^')
    ax.set_xlim(1, 6.5)

    # Step 5: Save Plot
    plt.savefig('foo.png')

    # Step 6: Show Plot
    plt.show()
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

    # Prepare the Data
    col1.subheader('Basic Plots ')

    ## Create a scatter plot
    col1.code('''
    # Create a scatter plot
    X = np.random.uniform(0, 1, 100)
    Y = np.random.uniform(0, 1, 100)
    plt.scatter(X, Y)
        ''')

    ## Create a bar plot
    col1.code('''
    # Create a bar plot
    X = np.arange(10)
    Y = np.random.uniform(1, 10, 10)
    plt.bar(X, Y)
        ''')

    ## Create an image plot using imshow
    col1.code('''
    # Create an image plot using imshow
    Z = np.random.uniform(0, 1, (8, 8))
    plt.imshow(Z)
        ''')

    ## Create a contour plot
    col1.code('''
    # Create a contour plot
    Z = np.random.uniform(0, 1, (8, 8))
    plt.contourf(Z)
    plt.show()
        ''')

    ## Create a pie chart
    col1.code('''
    # Create a pie chart
    Z = np.random.uniform(0, 1, 4)
    plt.pie(Z)
        ''')

    ## Create a histogram
    col1.code('''
    # Create a histogram
    Z = np.random.normal(0, 1, 100)
    plt.hist(Z)
        ''')

    ## Create an error bar plot
    col1.code('''
    # Create an error bar plot
    X = np.arange(5)
    Y = np.random.uniform(0, 1, 5)
    plt.errorbar(X, Y, Y / 4)
        ''')

    ## Create a box plot
    col1.code('''
    # Create a box plot
    Z = np.random.normal(0, 1, (100, 3))
    plt.boxplot(Z)
        ''')

    # Tweak
    col1.subheader('Tweak')

    ## Create a plot with a black solid line
    col1.code('''
    # Create a plot with a black solid line
    X = np.linspace(0, 10, 100)
    Y = np.sin(X)
    plt.plot(X, Y, color="black")
        ''')

    ## Create a plot with a dashed line
    col1.code('''
    # Create a plot with a dashed line
    X = np.linspace(0, 10, 100)
    Y = np.sin(X)
    plt.plot(X, Y, linestyle="--")
        ''')

    ## Create a plot with a thicker line
    col1.code('''
    # Create a plot with a thicker line
    X = np.linspace(0, 10, 100)
    Y = np.sin(X)
    plt.plot(X, Y, linewidth=5)
        ''')

    ## Create a plot with markers
    col1.code('''
    # Create a plot with markers
    X = np.linspace(0, 10, 100)
    Y = np.sin(X)
    plt.plot(X, Y, marker="o")
        ''')

    # Save
    col1.subheader('Save')
    col1.code('''
    # Save the figure as a PNG file with higher resolution (300 dpi)
    fig.savefig("my-first-figure.png", dpi=300)

    # Save the figure as a PDF file
    fig.savefig("my-first-figure.pdf")
        ''')

    #######################################
    # COLUMN 2
    #######################################

    # Markers
    col2.subheader('Organize')

    ## Create a plot with two lines on the same axes
    col2.code('''
    # Create a plot with two lines on the same axes
    X = np.linspace(0, 10, 100)
    Y1, Y2 = np.sin(X), np.cos(X)
    plt.plot(X, Y1, X, Y2)
        ''')

    ## Create a figure with two subplots (vertically stacked)
    col2.code('''
    # Create a figure with two subplots (vertically stacked)
    X = np.linspace(0, 10, 100)
    Y1, Y2 = np.sin(X), np.cos(X)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(X, Y1, color="C1")
    ax2.plot(X, Y2, color="C0")
        ''')

    ## Create a figure with two subplots (horizontally aligned)
    col2.code('''
    # Create a figure with two subplots (horizontally aligned)
    X = np.linspace(0, 10, 100)
    Y1, Y2 = np.sin(X), np.cos(X)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(Y1, X, color="C1")
    ax2.plot(Y2, X, color="C0")
        ''')

    # Label
    col2.subheader('Label')

    ## Create data and plot a sine wave
    col2.code('''
    # Create data and plot a sine wave
    X = np.linspace(0, 10, 100)
    Y = np.sin(X)
    plt.plot(X, Y)
        ''')

    ## Modify plot properties
    col2.code('''
    # Modify plot properties
    X = np.linspace(0, 10, 100)
    Y = np.sin(X)
    plt.plot(X, Y)
    plt.title("A Sine wave")
    plt.xlabel("Time")
    plt.ylabel(None)
        ''')

    # Figure, axes & spines
    col2.subheader('Figure, axes & spines')
    col2.code('''
    # Create a 3x3 grid of subplots
    fig, axs = plt.subplots(3, 3)

    # Set face colors for specific subplots
    axs[0, 0].set_facecolor("#ddddff")
    axs[2, 2].set_facecolor("#ffffdd")
        ''')

    col2.code('''
    # Create a 3x3 grid of subplots
    fig, axs = plt.subplots(3, 3)

    # Add a grid specification and set face color for a specific subplot
    gs = fig.add_gridspec(3, 3)
    ax = fig.add_subplot(gs[0, :])
    ax.set_facecolor("#ddddff")
        ''')

    col2.code('''
    # Create a figure with a single subplot
    fig, ax = plt.subplots()

    # Remove top and right spines from the subplot
    ax.spines["top"].set_color("None")
    ax.spines["right"].set_color("None")
        ''')

    # Colors
    col2.subheader('Colors')
    col2.code('''
    # Get a list of named colors
    named_colors = plt.colormaps()  
    print("Colors:",named_colors)
        ''')

    #######################################
    # COLUMN 3
    #######################################

    # Ticks & labels
    col3.subheader('Ticks & labels')
    col3.code('''
    from matplotlib.ticker import MultipleLocator as ML
    from matplotlib.ticker import ScalarFormatter as SF

    # Create a figure with a single subplot
    fig, ax = plt.subplots()
    
    # Set minor tick locations and formatter for the x-axis
    ax.xaxis.set_minor_locator(ML(0.2))
    ax.xaxis.set_minor_formatter(SF())
    
    # Rotate minor tick labels on the x-axis
    ax.tick_params(axis='x', which='minor', rotation=90)
        ''')

    # Lines & markers
    col3.subheader('Lines & markers')
    col3.code('''
    # Generate data and create a plot
    X = np.linspace(0.1, 10 * np.pi, 1000)
    Y = np.sin(X)
    plt.plot(X, Y, "C1o:", markevery=25, mec="1.0")
        ''')

    # Scales & projections
    col3.subheader('Scales & projections')
    col3.code('''
    # Create a figure with a single subplot
    fig, ax = plt.subplots()
    
    # Set x-axis scale to logarithmic
    ax.set_xscale("log")
    
    # Plot data with specified formatting
    ax.plot(X, Y, "C1o-", markevery=25, mec="1.0")
        ''')

    # Text & ornaments
    col3.subheader('Scales & projections')
    col3.code('''
    # Create a figure with a single subplot
    fig, ax = plt.subplots()
    
    # Fill the area between horizontal lines with a curve
    ax.fill_betweenx([-1, 1], [0], [2*np.pi])
    
    # Add a text annotation to the plot
    ax.text(0, -1, r" Period $\Phi$")
        ''')

    # Legend
    col3.subheader('Legend')
    col3.code('''
    # Create a figure with a single subplot
    fig, ax = plt.subplots()
    
    # Plot sine and cosine curves with specified colors and labels
    ax.plot(X, np.sin(X), "C0", label="Sine")
    ax.plot(X, np.cos(X), "C1", label="Cosine")
    
    # Add a legend with customized positioning and formatting
    ax.legend(bbox_to_anchor=(0, 1, 1, 0.1), ncol=2, mode="expand", loc="lower left")
        ''')

    # Annotation
    col3.subheader('Annotation')
    col3.code('''
    # Create a figure with a single subplot
    fig, ax = plt.subplots()
    
    ax.plot(X, Y, "C1o:", markevery=25, mec="1.0")
    
    # Add an annotation "A" with an arrow
    ax.annotate("A", (X[250], Y[250]), (X[250], -1),
                ha="center", va="center",
                arrowprops={"arrowstyle": "->", "color": "C1"})
        ''')





# Run the main function if the script is executed directly
if __name__ == '__main__':
    main()
