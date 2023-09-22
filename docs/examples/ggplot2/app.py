import streamlit as st
from pathlib import Path
import base64
import requests


# Initial page config
st.set_page_config(
    page_title='Ggplot2 Cheat Sheet',
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
    Populate the sidebar with various content sections related to ggplot2.
    """
    st.sidebar.markdown(
        '''[<img src='data:image/png;base64,{}' class='img-fluid' width=95 >](https://streamlit.io/)'''.format(
            img_to_bytes("https://rstudio.github.io/cheatsheets/html/images/logo-ggplot2.png")), unsafe_allow_html=True)

    st.sidebar.header('Ggplot2 Cheat Sheet')
    st.sidebar.markdown('''
<small>[ggplot2](https://ggplot2.tidyverse.org/) is based on the **grammar of graphics**, the idea that you can build every graph from the same components: a **data** set, a **coordinate system**, and **geoms**—visual marks that represent data points.</small>
    ''', unsafe_allow_html=True)

    # ggplot2 installation and import
    st.sidebar.markdown('__Install and import ggplot2__')
    st.sidebar.code('''$ install.packages('ggplot2')''')
    st.sidebar.code('''
# Import ggplot2 
>>> library(ggplot2)
''')




    return None


# Define the cs_body() function
def cs_body():
    """
    Create content sections for the main body of the Streamlit cheat sheet with ggplot2 examples.
    """
    col1, col2, col3 = st.columns(3)  # Create columns for layout

    #######################################
    # COLUMN 1
    #######################################
    col1.subheader('Basic')
    col1.code('''
        # Complete the template below to build a graph.
        ggplot(data = <Data>) +
            <Geom_Function>(mapping = aes(<Mappings>),
            stat = <Stat>,
            position = <Position>) +
            <Coordinate_Function> +
            <Facet_Function> +
            <Scale_Function> +
            <Theme_Function>
        ''')

    col1.markdown('''
    *   `ggplot(data = mpg, aes(x = cty, y = hwy))`: Begins a plot that you finish by adding layers to. Add one geom function per layer.

    *   `last_plot()`: Returns the last plot.

    *   `ggsave("plot.png", width = 5, height = 5)`: Saves last plot as 5’ x 5’ file named “plot.png” in working directory. Matches file type to file extension.
                ''')

    # Aes
    col1.subheader('Aes')
    col1.markdown('''
Common aesthetic values.

*   `color` and `fill`: String (`"red"`, `"#RRGGBB"`).
    
*   `linetype`: Integer or string (0 = `"blank"`, 1 = `"solid"`, 2 = `"dashed"`, 3 = `"dotted"`, 4 = `"dotdash"`, 5 = `"longdash"`, 6 = `"twodash"`).
    
*   `size`: Integer (line width in mm for outlines).
    
*   `linewidth`: Integer (line width in mm for lines).
    
*   `shape`: Integer/shape name or a single character (`"a"`).
        ''')

    # Graphical Primitives
    col1.subheader('Graphical Primitives')
    col1.code('''
    a <- ggplot(economics, aes(date, unemploy))
    b <- ggplot(seals, aes(x = long, y = lat))
    ''')


    col1.subheader('''One Variable - Continuous''')
    col1.code('''
           c <- ggplot(mpg, aes(hwy))
           c2 <- ggplot(mpg)
                ''')


    col1.subheader('''One Variable - Discrete''')
    col1.code('''
               d <- ggplot(mpg, aes(fl))
                    ''')


    col1.subheader('''Two Variables - Both Continuous''')
    col1.code('''
               e <- ggplot(mpg, aes(cty, hwy))
                    ''')

    col1.subheader('''Two Variables - One Discrete, One Continuous''')
    col1.code('''
               f <- ggplot(mpg, aes(class, hwy))
                    ''')


    #######################################
    # COLUMN 2
    #######################################


    # Logical
    col2.subheader('''Two Variables - Both Discrete''')
    col2.code('''
               g <- ggplot(diamonds, aes(cut, color))
                    ''')

    col2.subheader('''Two Variables - Continuous Bivariate Distribution''')
    col2.code('''
              h <- ggplot(diamonds, aes(carat, price))
                    ''')

    col2.subheader('''Two Variables - Continuous Function''')
    col2.code('''
             i <- ggplot(economics, aes(date, unemploy))
                    ''')

    col2.subheader('''Two Variables - Visualizing Error''')
    col2.code('''
             Two Variables - Visualizing Error
                    ''')

    col2.subheader('''Two Variables - Maps''')
    col2.code('''
             murder_data <- data.frame(
                    murder = USArrests$Murder, 
                    state = tolower(rownames(USArrests))
            )
            map <- map_data("state")
            k <- ggplot(murder_data, aes(fill = murder))
                    ''')

    col2.subheader('''Three Variables''')
    col2.code('''
             seals$z <- with(seals, sqrt(delta_long^2 + delta_lat^2))
             l <- ggplot(seals, aes(long, lat))
                    ''')

    col2.subheader('''Stats''')
    col2.markdown('''
    An alternative way to build a layer. A stat builds new variables to plot (e.g., count, prop).
    ''')
    col2.code('''
             i + stat_density_2d(aes(fill = after_stat(level)), geom = "polygon")
                    ''')

    col2.subheader('''Scales''')
    col2.markdown('''
    Scales map data values to the visual values of an aesthetic. To change a mapping, add a new scale.
    ''')
    col2.code('''
n <- d + geom_bar(aes(fill = fl))

n + scale_fill_manual(
  value = c(),
  limits = c(), 
  breaks = c(),
  name = "fuel", 
  labels = c("D", "E", "P", "R")
)
                    ''')



    #######################################
    # COLUMN 3
    #######################################

    # Row Names
    col3.subheader('''Color and Fill Scales (Continuous)''')
    col3.code('''
o <- c + geom_dotplot(aes(fill = ..x..))
                    ''')

    col3.subheader('''Shape and Size Scales''')
    col3.code('''
p <- e + geom_point(aes(shape = fl, size = cyl))
                        ''')

    col3.subheader('''Coordinate Systems''')
    col3.code('''
u <- d + geom_bar()
                        ''')

    col3.subheader('''Position Adjustments''')
    col3.markdown('''
    Position adjustments determine how to arrange geoms that would otherwise occupy the same space.
    ''')
    col3.code('''
s <- ggplot(mpg, aes(fl, fill = drv))
                        ''')
    col3.code('''
    s + geom_bar(position = position_dodge(width = 1))
                            ''')

    col3.subheader('''Themes''')

    col3.code('''
r + ggtitle("Title") + theme(plot.title.postion = "plot")

r + theme(panel.background = element_rect(fill = "blue"))
                            ''')

    col3.subheader('''Faceting''')
    col3.markdown('''
    Facets divide a plot into subplots based on the values of one or more discrete variables.
    ''')
    col3.code('''
t <- ggplot(mpg, aes(cty, hwy)) + geom_point()                        ''')

    col3.subheader('''Labels and Legends''')
    col3.markdown('''
        Use `labs()` to label elements of your plot.
        ''')
    col3.code('''
    t + labs(x = "New x axis label", 
        y = "New y axis label",
        title ="Add a title above the plot",
        subtitle = "Add a subtitle below title",
        caption = "Add a caption below plot",
        alt = "Add alt text to the plot",
        <Aes> = "New <Aes> legend title")
                         ''')

    col3.subheader('''Zooming''')
    col3.markdown('''
*   `t + coord_cartesian(xlim = c(0, 100), ylim = c(10,20))`: Zoom without clipping (preferred).
    
*   `t + xlim(0, 100) + ylim(10, 20)` or `t + scale_x_continuous(limits = c(0, 100)) + scale_y_continuous(limits = c(0, 100))`: Zoom with clipping (removes unseen data points).
    
        ''')


# Run the main function if the script is executed directly
if __name__ == '__main__':
    main()
