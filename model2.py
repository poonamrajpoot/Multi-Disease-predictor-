import numpy as np
import pickle
import streamlit as st


loaded_model=pickle.load(open(r"C:\Users\HP\Downloads\ml_model\trained_model.sav",'rb'))




def diabetes_prediction(input_data):
    input_data=(4,110,92,0,0,37.6,0.191,30)
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    std_data=scaler.transform(input_data_reshaped)
    print(std_data)
    prediction=classifier.predict(std_data)
    print(prediction)
    if (prediction[0]==0):
      return'The person does not have diabetes'
    else:
      return'the person id diabetic'
      
      
def main():
    
    #giving a 
    st.title('Diabetes prediction web app')
    
    #getting inputrs from the user
    
    Pregnancies=st.text_input('NUmber of pregnancies')
    Glucose=st.text_input('Glucise level')
    BloodPressure=st.text_input('BP level')
    SkinThickness=st.text_input('Thickness of patients skin')
    Insulin=st.text_input('Insulin level')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('Diabetes pedi. Function') 
    Age=st.text_input('Age of patients')
    
    #code for prediction
    
    diagnosis=''
    
    #creating a button
    
    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    
    st.success(diagnosis)
    
    
if _name=='main_':
    main()
