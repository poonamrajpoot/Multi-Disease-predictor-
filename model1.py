import numpy as np
import pickle

loaded_model=pickle.load(open(r"C:\Users/HP/Downloads/ml_model/trained_model.sav",'rb'))

input_data=(4,110,92,0,0,37.6,0.191,30)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
   print('The person does not have diabetes')
else:
   print('the person id diabetic')
