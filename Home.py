import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Algorithms",
)

st.write("# Welcome!")

st.sidebar.success("Select een demo")

st.markdown(
    """
In this application you can you at the EDA of the data, and at a few different algorithms.
"""
)
st.markdown(
    """
I have used the dataset 'Car Evaluation'. (https://archive.ics.uci.edu/dataset/19/car+evaluation)
"""
)