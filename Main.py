import streamlit as st

st.set_page_config(page_title="Projects by Dimitri", layout="centered", page_icon='⚡')

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
    title = 'Netflix 🆕',
    icon='🍿'
)

project6 = st.Page(
    page="views/DW.py",
    title = 'DW with Azure Synapse 🔜  🔧 🔨',
    icon='🏢'
)

project7 = st.Page(
    page="views/Website.py",
    title = 'A better website 🔜  🔧 🔨',
    icon='👩‍💻'
)

project8 = st.Page(
    page="views/Titanic.py",
    title = 'Titanic (Decision 🌳🌳  algo)  🆕',
    icon='🚢'
)

pg= st.navigation({
    "Info": [main_page],
     "Supervised Machine Learning": [project1, project2],
     "Unsupervised Machine Learning": [ project4, project5],
     "Deep Learning": [ project3] ,
     "Data Engineering with Microsoft Azure": [project6],
     "A/B Test - Null Hypothesis": [project7],
     "Mini-Projects": [project8]})

pg.run()






