import streamlit as st

st.set_page_config(page_title="Projects by Dimitri", layout="wide", page_icon='⚡')

main_page = st.Page(
    page='views/About_me.py',
    title = 'Projects',
    icon='📘',
    default=True
)

project1 = st.Page(
    page="views/House_prices.py",
    title = 'House prices',
    icon='🏠'
)

project2 = st.Page(
    page="views/Donors.py",
    title = 'Finding the right donor(s)',
    icon='🤝'
)

project3 = st.Page(
    page="views/Pets.py",
    title = 'My Pet 🔜  🔧 🔨	',
    icon='🐶'
)

project4 = st.Page(
    page="views/Supermarkets.py",
    title = 'Grocery shopping 🔜  🔧 🔨',
    icon='🛒'
)

project5 = st.Page(
    page="views/Movies.py",
    title = 'Netflix Movies 🔜  🔧 🔨',
    icon='🍿'
)


pg= st.navigation({
    "Info": [main_page],
     "Supervised Machine Learning": [project1, project2],
     "Unsupervised Machine Learning": [ project3, project5],
     "Deep Learning": [ project4] })

pg.run()






