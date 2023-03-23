import pandas as pd
import streamlit as st
import numpy as np

def get_correlation_equation(df, predict_col_idx):
    col1 = df.columns[predict_col_idx]
    col2 = df.columns[predict_col_idx+1]
    corr = df[col1].corr(df[col2])
    equation = f"{col2} = {corr:.2f} * {col1} + {df[col2].mean()-corr*df[col1].mean():.2f}"
    return equation

st.title("Predicción de columnas en dataset")

# Cargar dataset
uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file,delimiter=";").iloc[:, 1:].astype(float)
    st.write(df)

    # Input para predicción de columnas
    st.header("Predicción de columnas")
    col_names = df.columns[1:]
    col_dict = {name: i for i, name in enumerate(col_names)}
    predict_col = st.selectbox("Seleccione una columna para predecir:", col_names, key="predict_col")
    predict_col_idx = col_dict[predict_col]

   # Datos que ingresa el usuario 
    
    #leads = st.number_input("Ingrese el valor de leads o valor de M:", key="leads")
    #conversion_MO = st.number_input("Ingrese la tasa de conversión M0 (Si no es leads, ingrese 1):", key="conversion_MO")
    leads_matrix = np.array(st.text_input("Ingrese los valores de leads separados por comas:", key="leads").split(','), dtype=np.float).reshape((-1, 1))
    conversion_matrix = np.array(st.text_input("Ingrese los valores de conversion_M0 separados por comas:", key="conversion_MO").split(','), dtype=np.float).reshape((-1, 1))
   
    # Predicción de columnas
    st.write(get_correlation_equation(df, predict_col_idx))

    predict_input_matrix = leads_matrix * conversion_matrix

    #predict_input = leads * conversion_MO

    st.write("Valor MO:",predict_input_matrix)

    predicted_values_matrix = np.empty((leads_matrix.shape[0], len(col_names)-predict_col_idx))


    #predict_input = st.number_input(f"Ingrese el valor de {df.columns[predict_col_idx]} para predecir {predict_col}:", key="predict_input")
    #predicted_values = []
    #for i in range(predict_col_idx, len(col_names)):
     #   col = col_names[i]
     #   corr = df[col_names[i-1]].corr(df[col])
     #   predicted_value = corr * float(predict_input) + df[col].mean() - corr * df[col_names[i-1]].mean()
     #   predicted_values.append(predicted_value)
     #   predict_input = predicted_value  # Actualizar predict_input con el valor predicho
    #st.write(f"Predicción para {col_names[predict_col_idx:]}: {predicted_values}")
    for i in range(predict_col_idx, len(col_names)):
        col = col_names[i]
        corr = df[col_names[i-1]].corr(df[col])
        predicted_values = corr * predict_input_matrix[:, i-predict_col_idx] + df[col].mean() - corr * df[col_names[i-1]].mean()
        predicted_values_matrix[:, i-predict_col_idx] = predicted_values
        predict_input_matrix[:, i-predict_col_idx+1:] = predicted_values.reshape((-1, 1))  # Actualizar predict_input_matrix con los valores predichos
    
st.write(f"Predicciones para {col_names[predict_col_idx:]}:\n{predicted_values_matrix}")


