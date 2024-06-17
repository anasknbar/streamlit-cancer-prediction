import streamlit as st
import pickle
import pandas as pd

import plotly.graph_objects as go

import numpy as np


def get_clean_data():
  data = pd.read_csv("dataset/data.csv")
  data = data.drop(['Unnamed: 32','id'],axis=1)
  data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
  return data 

def add_sidebar():
  st.sidebar.header("Cell Nuclei Measurements")
  data = get_clean_data()
  
  realtime_input_dic = {}
  
  slider_labels = [
        ("Radius (mean)", "radius_mean"),# "Radius(mean)" >> slider_label | "radius_mean" >> column name for max/min value
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
  for label,key in slider_labels:
    realtime_input_dic[key] = st.sidebar.slider(
      label=label,
      min_value=data[key].min(),
      max_value=data[key].max(),
      value=data[key].mean() # defalut value
    )
  return realtime_input_dic

def get_scaled_values(input_dic):
  data = get_clean_data()
  X = data.drop(['diagnosis'],axis=1)
  
  scaled_dic = {}
  for key,value in input_dic.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value-min_val)/(max_val-min_val)
    scaled_dic[key] = scaled_value
    
  return scaled_dic
   
def get_radar_chart(input_data):
  input_data = get_scaled_values(input_data)
  categories = ['Radius','Texture','Perimeter','Area',
                'Smoothness','Compactness','Concavity',
              'Concave points','Symmetry','Fractal dimension']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], 
          input_data['perimeter_mean'],input_data['area_mean'], 
          input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], 
          input_data['symmetry_mean'],input_data['fractal_dimension_mean']
          ],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[ 
          input_data['radius_se'], input_data['texture_se'], 
          input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], 
          input_data['concavity_se'],input_data['concave points_se'], 
          input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], 
          input_data['perimeter_worst'],input_data['area_worst'], 
          input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], 
          input_data['symmetry_worst'],input_data['fractal_dimension_worst']
          ],

        theta=categories,
        fill='toself',
        name='Worst'
  ))




  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  return fig


def add_predictions(input_data):
  model = pickle.load(open("model/model.pkl",'rb'))
  scaler = pickle.load(open("model/scalar.pkl",'rb'))
  
  input_array = np.array(list(input_data.values())).reshape(1,-1)
  # st.write(input_data)
  input_array_scaled = scaler.transform(input_array)
  prediction = model.predict(input_array_scaled)
  st.subheader('Cell cluster prediction')
  st.write("The cell cluster is: ")

  if prediction[0] == 0:
    
    st.write("<span class= 'diagnosis benign' >Benign<span>",unsafe_allow_html=True)
  else:
    st.write("<span class= 'diagnosis melicious' >Melicious<span>",unsafe_allow_html=True)
    
    
    
  st.write("Probability of being benign: ",
             model.predict_proba(input_array_scaled)[0][0])
  st.write("Probability of being malignant: ",
             model.predict_proba(input_array_scaled)[0][1])
  st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


    



def main():
  
  st.set_page_config(page_title="breast Cancer Predictor",
  page_icon="👩‍⚕️",
  layout="wide",
  initial_sidebar_state="expanded")
  
  input_data = add_sidebar()
  
  
  with open("assets/style.css") as file:
    st.markdown('<style>{}</style>'.format(file.read()), unsafe_allow_html=True)

  
  with st.container():
    st.title("Breast Cancer Predictor")
    st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.")
    
  col_1,col_2 = st.columns([3,1])
  
  with col_1:
    radar_chart = get_radar_chart(input_data)
    st.plotly_chart(radar_chart)
  with col_2:
    add_predictions(input_data)






if __name__ == '__main__':
  main()