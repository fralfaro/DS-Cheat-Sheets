import streamlit as st
from pathlib import Path
import base64

# Initial page config
st.set_page_config(
    page_title='Streamlit cheat sheet',
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


def img_to_bytes(img_path):
    """
    Converts an image to base64 encoded bytes.

    Parameters:
        img_path (str): Path to the image file.

    Returns:
        str: Base64 encoded image bytes.
    """
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


# Define the function for the sidebar
def cs_sidebar():
    """
    Populate the sidebar with various content sections.
    """
    # Display Streamlit logo with link
    st.sidebar.markdown(
        '''[<img src='data:image/png;base64,{}' class='img-fluid' width=36 >](https://streamlit.io/)'''.format(
            img_to_bytes("docs/examples/streamlit/streamlit.png")), unsafe_allow_html=True)

    # Header and summary
    st.sidebar.header('Streamlit cheat sheet')
    st.sidebar.markdown('''
<small>Summary of the [docs](https://docs.streamlit.io/), as of [Streamlit v1.25.0](https://www.streamlit.io/).</small>
    ''', unsafe_allow_html=True)

    # Installation information
    st.sidebar.markdown('__Install and import__')
    st.sidebar.code('$ pip install streamlit')
    st.sidebar.code('''
# Import convention
>>> import streamlit as st
''')

    # Adding widgets to sidebar
    st.sidebar.markdown('__Add widgets to sidebar__')
    st.sidebar.code('''
# Just add it after st.sidebar:
>>> a = st.sidebar.radio(\'Choose:\',[1,2])
    ''')

    # Magic commands
    st.sidebar.markdown('__Magic commands__')
    st.sidebar.code('''
'_This_ is some __Markdown__'
a = 3
'dataframe:', data
''')

    # Command line commands
    st.sidebar.markdown('__Command line__')
    st.sidebar.code('''
$ streamlit --help
$ streamlit run your_script.py
$ streamlit hello
$ streamlit config show
$ streamlit cache clear
$ streamlit docs
$ streamlit --version
    ''')

    # Pre-release features
    st.sidebar.markdown('__Pre-release features__')
    st.sidebar.code('''
pip uninstall streamlit
pip install streamlit-nightly --upgrade
    ''')
    st.sidebar.markdown(
        '<small>Learn more about [experimental features](https://docs.streamlit.io/library/advanced-features/prerelease#beta-and-experimental-features)</small>',
        unsafe_allow_html=True)

    # Attribution and footer
    st.sidebar.markdown('''<hr>''', unsafe_allow_html=True)
    st.sidebar.markdown(
        '''<small>[Cheat sheet v1.25.0](https://github.com/daniellewisDL/streamlit-cheat-sheet) </small>''',
        unsafe_allow_html=True)

    return None


# Define the main body of the cheat sheet
def cs_body():
    """
    Create content sections for the main body of the Streamlit cheat sheet.
    """
    col1, col2, col3 = st.columns(3)  # Create columns for layout

    #######################################
    # COLUMN 1
    #######################################

    # Display text
    col1.subheader('Display text')
    col1.code('''
# Display fixed width text
st.text('Fixed width text')
# Display Markdown formatted text
st.markdown('_Markdown_')
# Display captions and LaTeX equations
st.caption('Balloons. Hundreds of them...')
st.latex(r\'\'\' e^{i\pi} + 1 = 0 \'\'\')
# Display various objects using st.write
st.write('Most objects')  # df, err, func, keras!
st.write(['st', 'is <', 3])  # Display list as text
st.title('My title')
st.header('My header')
st.subheader('My sub')
st.code('for i in range(8): foo()')
    ''')

    # Display data
    col1.subheader('Display data')
    col1.code('''
# Display DataFrame and table
st.dataframe(my_dataframe)
st.table(data.iloc[0:10])
# Display JSON and metric
st.json({'foo':'bar','fu':'ba'})
st.metric(label="Temp", value="273 K", delta="1.2 K")
    ''')

    # Display media
    col1.subheader('Display media')
    col1.code('''
# Display image, audio, and video
st.image('./header.png')
st.audio(data)
st.video(data)
    ''')

    # Columns
    col1.subheader('Columns')
    col1.code('''
# Create columns for layout
col1, col2 = st.columns(2)
col1.write('Column 1')
col2.write('Column 2')

# Create columns with different widths
col1, col2, col3 = st.columns([3,1,1])
# col1 is wider

# Use 'with' notation to write content within columns
with col1:
    st.write('This is column 1')
    ''')

    # Tabs
    col1.subheader('Tabs')
    col1.code('''
# Create tabs to organize content
tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
tab1.write("this is tab 1")
tab2.write("this is tab 2")

# Use 'with' notation to write content within tabs
with tab1:
    st.radio('Select one:', [1, 2])
    ''')

    # Control flow
    col1.subheader('Control flow')
    col1.code('''
# Control flow tools
st.stop()  # Stop execution immediately
st.experimental_rerun()  # Rerun script immediately

# Group widgets together using st.form
with st.form(key='my_form'):
    username = st.text_input('Username')
    password = st.text_input('Password')
    st.form_submit_button('Login')
    ''')

    # Personalize apps for users
    col1.subheader('Personalize apps for users')
    col1.code('''
# Display different content based on user
if st.user.email == 'jane@email.com':
    display_jane_content()
elif st.user.email == 'adam@foocorp.io':
    display_adam_content()
else:
    st.write("Please contact us to get access!")
    ''')

    #######################################
    # COLUMN 2
    #######################################

    # Display interactive widgets
    col2.subheader('Display interactive widgets')
    col2.code('''
# Display various interactive widgets
st.button('Hit me')
st.data_editor('Edit data', data)
st.checkbox('Check me out')
st.radio('Pick one:', ['nose','ear'])
st.selectbox('Select', [1,2,3])
st.multiselect('Multiselect', [1,2,3])
st.slider('Slide me', min_value=0, max_value=10)
st.select_slider('Slide to select', options=[1,'2'])
st.text_input('Enter some text')
st.number_input('Enter a number')
st.text_area('Area for textual entry')
st.date_input('Date input')
st.time_input('Time entry')
st.file_uploader('File uploader')
st.download_button('On the dl', data)
st.camera_input("一二三,茄子!")
st.color_picker('Pick a color')
    ''')

    col2.code('''
# Capture returned widget values in variables
for i in range(int(st.number_input('Num:'))):
    foo()
if st.sidebar.selectbox('I:', ['f']) == 'f':
    b()
my_slider_val = st.slider('Quinn Mallory', 1, 88)
st.write(slider_val)
    ''')
    col2.code('''
# Disable widgets to remove interactivity
st.slider('Pick a number', 0, 100, disabled=True)
    ''')

    # Build chat-based apps
    col2.subheader('Build chat-based apps')
    col2.code('''
# Create chat message container
with st.chat_message("user"):
    st.write("Hello 👋")
    st.line_chart(np.random.randn(30, 3))

# Display chat input widget
st.chat_input("Say something")
    ''')

    col2.markdown(
        '<small>Learn how to [build chat-based apps](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps)</small>',
        unsafe_allow_html=True)

    # Mutate data
    col2.subheader('Mutate data')
    col2.code('''
# Add rows to DataFrame and chart after showing
element = st.dataframe(df1)
element.add_rows(df2)
element = st.line_chart(df1)
element.add_rows(df2)
    ''')

    # Display code
    col2.subheader('Display code')
    col2.code('''
st.echo()
with st.echo():
    st.write('Code will be executed and printed')
    ''')

    # Placeholders, help, and options
    col2.subheader('Placeholders, help, and options')
    col2.code('''
# Replace any single element
element = st.empty()
element.line_chart(...)
element.text_input(...)  # Replaces previous.

# Insert elements out of order
elements = st.container()
elements.line_chart(...)
st.write("Hello")
elements.text_input(...)  # Appears above "Hello".

st.help(pandas.DataFrame)
st.get_option(key)
st.set_option(key, value)
st.set_page_config(layout='wide')
st.experimental_show(objects)
st.experimental_get_query_params()
st.experimental_set_query_params(**params)
    ''')

    #######################################
    # COLUMN 3
    #######################################

    # Connect to data sources
    col3.subheader('Connect to data sources')
    col3.code('''
# Connect to various data sources
st.experimental_connection('pets_db', type='sql')
conn = st.experimental_connection('sql')
conn = st.experimental_connection('snowpark')

class MyConnection(ExperimentalBaseConnection[myconn.MyConnection]):
    def _connect(self, **kwargs) -> MyConnection:
        return myconn.connect(**self._secrets, **kwargs)
    def query(self, query):
        return self._instance.query(query)
    ''')

    # Optimize performance
    col3.subheader('Optimize performance')
    col3.write('Cache data objects')
    col3.code('''
# Cache expensive function calls (data)
@st.cache_data
def foo(bar):
    return data
d1 = foo(ref1)  # Executes foo
d2 = foo(ref1)  # Does not execute foo, returns cached value
d3 = foo(ref2)  # Different arg, so foo executes

# Clear cached entries
foo.clear()  # Clear cached entries for specific function
st.cache_data.clear()  # Clear all cached data entries
    ''')
    col3.write('Cache global resources')
    col3.code('''
# Cache expensive function calls (non-data)
@st.cache_resource
def foo(bar):
    return session
s1 = foo(ref1)  # Executes foo
s2 = foo(ref1)  # Does not execute foo, returns cached value
s3 = foo(ref2)  # Different arg, so foo executes

# Clear cached entries
foo.clear()  # Clear cached entries for specific function
st.cache_resource.clear()  # Clear all cached resource entries
    ''')
    col3.write('Deprecated caching')
    col3.code('''
@st.cache
def foo(bar):
    return data
d1 = foo(ref1)  # Executes foo
d2 = foo(ref1)  # Does not execute foo, returns cached value
d3 = foo(ref2)  # Different arg, so foo executes
    ''')

    # Display progress and status
    col3.subheader('Display progress and status')
    col3.code('''
# Show spinner and progress bar
with st.spinner(text='In progress'):
    time.sleep(3)
    st.success('Done')
bar = st.progress(50)
time.sleep(3)
bar.progress(100)

st.balloons()
st.snow()
st.toast('Mr Stay-Puft')
st.error('Error message')
st.warning('Warning message')
st.info('Info message')
st.success('Success message')
st.exception(e)
    ''')

    return None

# Run the main function if the script is executed directly
if __name__ == '__main__':
    main()
