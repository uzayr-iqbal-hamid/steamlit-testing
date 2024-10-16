# Learning streamlit

import streamlit as st
st.title("Hello, Streamlit!")
st.write("This is a basic Streamlit app.")

st.header("This is a header")
st.subheader("This is a subheader")
st.text("This is plain text")

st.markdown("**This is bold text** and _this is italic text_.")

st.video("C:/Users/hamid/Videos/minecraft1.mkv")

st.button("Click Me")
st.slider("Select a range", 0, 100)
text = st.text_input("Enter some text")
st.write(text)

st.checkbox("Check me out")
st.radio("Choose an option", ["Option 1", "Option 2"])

st.selectbox("Pick one", ["Choice 1", "Choice 2"])
st.multiselect("Pick multiple", ["Choice A", "Choice B", "Choice C"])


import pandas as pd

df = pd.DataFrame({
    "Column A": [1, 2, 3],
    "Column B": [4, 5, 6]
})

st.table(df)
st.dataframe(df)

st.metric(label="Price", value="560", delta="-1.2%")

import plotly.express as px

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
st.plotly_chart(fig)


# Sample data: Monthly sales data
data = pd.DataFrame({
    "Month": ["January", "February", "March", "April", "May", "June", "July", 
              "August", "September", "October", "November", "December"],
    "Sales": [1200, 1500, 1700, 1300, 1600, 1800, 2100, 2000, 1900, 2300, 2200, 2400]
})

# Displaying the DataFrame
st.write("Monthly Sales Data")
st.write(data)


st.line_chart(data)
st.bar_chart(data)
st.area_chart(data)

col1, col2 = st.columns(2)
col1.write("Column 1")
col2.write("Column 2")


with st.expander("aur dekhein"):
    st.write("mat dekhein")

# Define a function to increment the counter
def increment_count():
    st.session_state.count += 1

# Initialize the session state if it doesn't exist
if "count" not in st.session_state:
    st.session_state.count = 0

# Use the increment_count function with the button
st.button("Increment", on_click=increment_count)
st.write(st.session_state.count)


uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # or pd.read_excel(uploaded_file)
    st.write(df)


import pydeck as pdk

data = pd.DataFrame({
    'lat': [13.39, 13.3161],
    'lon': [74.7443, 75.7720]
})

st.map(data)

# More advanced mapping with pydeck
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/dark-v9',
    initial_view_state=pdk.ViewState(
        latitude=12.9716,
        longitude=77.5946,
        zoom=11,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
            'HexagonLayer',
            data=data,
            get_position='[lon, lat]',
            radius=200,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
        ),
    ],
))


st.code('import pandas as pd\n\n# Load data\n df = pd.read_csv("data.csv")', language='python', line_numbers=True, wrap_lines=True)


@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='processed_data.csv',
    mime='text/csv',
)

sidebar_choice = st.sidebar.selectbox("Choose a plot", ["Scatter", "Bar", "Line"])
# st.sidebar.button("Scatter")
# st.sidebar.button("Box")
# st.sidebar.button("Line")

import time

with st.spinner('Processing...'):
    time.sleep(2)
st.success('Done!')

# Progress bar
progress = st.progress(0)
for i in range(100):
    time.sleep(0.1)
    progress.progress(i + 1)


import streamlit as st
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [24, 27, 22],
    "City": ["New York", "San Francisco", "Los Angeles"]
})

# Use st.data_editor to create an editable table
edited_df = st.data_editor(df, num_rows="dynamic")
st.write("Edited Data:", edited_df)

