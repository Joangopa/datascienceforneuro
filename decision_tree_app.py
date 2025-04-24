import streamlit as st
import joblib
import pandas as pd

# Carregar o modelo salvo
model = joblib.load('decision_tree_model.pkl')

# Título do app
st.title("Classificação de Demência (CDR)")

st.subheader("Preencha os dados do paciente:")

# Layout em duas colunas
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Sexo (M/F)", ['M', 'F'])
    educ = st.selectbox("Nível educacional", [1, 2, 3, 4, 5])
    age = st.number_input("Idade", min_value=0, max_value=120, value=75)
    etiv = st.number_input("Volume Total Intracraniano Estimado", value=1500.0)

with col2:
    ses = st.selectbox("Status socioeconômico", [1, 2, 3, 4, 5])
    mmse = st.number_input("Mini-Exame do Estado Mental", min_value=0, max_value=30, value=28)
    nwbv = st.number_input("Volume Normalizado de Matéria Branca", value=0.75)

# Montar DataFrame com os dados inseridos
input_df = pd.DataFrame({
    'M/F': [gender],
    'Educ': [educ],
    'SES': [ses],
    'Age': [age],
    'MMSE': [mmse],
    'eTIV': [etiv],
    'nWBV': [nwbv]
})

# Botão para acionar a previsão
if st.button("Classificar"):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)
    st.success(f"Resultado previsto (CDR): {prediction[0]}")
