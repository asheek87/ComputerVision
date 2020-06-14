#%%
from ImgManager import ImgManager
from CNNModel import CNNModel
from Analyser import Analyser
import time as t
import pandas as pd
import tensorflow as tf
output='output/'
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns', 51)

#%%
#Create Image manager and perform analysis on labels
imgMan=ImgManager()
df_analyse=imgMan.analyseLabel('trainLabels_bird.csv')
df_analyse
#%%
#Base on analysis above there are images less than 10 occurrances
#For images which has 9 or 10  occurances, add new images and lables so that the total is 11. 
# New images has file name begins with "new_XXX.jpg"
# The rest will not be considered 
# for deep learning due to insuffient images
#Create Image manager and perform analysis on labels
df_analyse=imgMan.analyseLabel('trainLabels_bird_updated.csv')
df_analyse
#%%
# if number of bird images  per class is less than 10, it will not be used.
df_approve=imgMan.filterApprovedBirds(df_analyse,10)
df_approve

#%%
#Save  label to a txt file a output/Labels.txt
imgMan.saveLabelToTxt(output)
#%%
#Read images from 'birds' folder and select approve bird labels 
start=t.time()
imgMan.readImages()
end=t.time()
print('Time taken for reading images is '+str(end-start)+" seconds")

#%%
# processing all approve image objects. Each image object 
# will return a resize image . ImageManager will add and 
# filter this images into a list of train and test data
# If an image has more than 10, only a max of 10 is used for training
# to ensure a balanced dataset
#10 images will be used for tarining, 1 image for testing. 
start=t.time()
imgMan.procesImages()
end=t.time()
print('Time taken for image processing is '+str(end-start)+" seconds")
#%%
#Return train and test data.
X_Train,y_train,X_Test,y_test=imgMan.get_Train_Test_Data()
print('X_train shape: '+str(X_Train.shape))
print('y_train shape: '+str(y_train.shape))
print('X_test shape: '+str(X_Test.shape))
print('y_test shape: '+str(y_test.shape))

#%%
# create the model
cnnMod=CNNModel(imgMan.getSideDimension(),X_Train,y_train,X_Test,y_test,imgMan.getNoOfUniqueOutputs())
#%%
#Find the best hyper paramters to get best results
epoch=[8,10]
batSize=[10,50] 
optimizers=['adam']
outAct=['softmax']
hiddenUnit=[256]

dictParam={'epochs':epoch,'batch_size':batSize,'anOptimizer':optimizers,'outActivation':outAct,'hidUnit':hiddenUnit}
start=t.time()
df_full,df_result,bestParam,bestScore,model=cnnMod.findOptimizeParamCV(dictParam,fold=3)
end=t.time()
#%%
print('Time taken for grid search is '+str(end-start)+" seconds") # Time taken is approx 2. hrs to find optimize params
#%%
#Print full results of searching for optimize param to output/DF_Full_Result.xlsx
df_full.to_excel(output+"DF_Full_Result.xlsx")
#%%
#Print partial results of searching for optimize param to output/DF_Partial_Result.xlsx
df_result.to_excel(output+"DF_Partial_Result.xlsx")
df_result.head()
#%%
# Show the best parameter to be used after grid search
bestParam
df_param=pd.DataFrame([bestParam])
df_param
#%%
# Show the best score after grid search
print('Best accuracy after grid search on training data: '+str(bestScore))
#%%
# Evaluating the best model found in grid search using Test data
res=model.score(X_Test,y_test)
print('Accuracy of grid search model on test data: '+str(res))
#%%
# Train new model with best parameters using full data set using kfold. 
start=t.time()
df_metrics,network,hist=cnnMod.trainModel(bestParam,X_Train,y_train)
end=t.time()
print('Time taken for training model is '+str(end-start)+" seconds")
#%%
#Show metrics after training with best parameters
df_metrics
#%%
# Display loss vs epoch graph for train Data. See \output\Loss.png
analyser=Analyser(output)
analyser.plot_loss(hist,'Loss') 
#%%
# Display loss vs epoch graph for train Data. See \output\Accuracy.png
analyser.plot_accuracy(hist,'Accuracy')
#%%
#Evaluate trained network with test data
param= network.evaluate(X_Test, y_test,batch_size=bestParam.get('batch_size'))
#%%
#Print results of test data
print('Test loss:', param[0])
print('Test accuracy:', param[1]*100)
print('Test precision:', param[2]*100)
print('Test recall:', param[3]*100)
# print('Test false negative:', param[4])
# print('Test false positive:', param[5])
# print('Test true negative:', param[6])
# print('Test true positive:', param[7])

#%%
# Validate KERAS model using tst img
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
wrong,right=0,0
for index,anImage in enumerate(X_Test):
    
    actual = np.argmax(y_test[index])
    actualStr=imgMan.getStrKeyFromVal(actual)
    
    prob = network.predict(np.expand_dims(anImage, axis=0))
    predictIndx = np.argmax(prob)
    predictStr=imgMan.getStrKeyFromVal(predictIndx)
    
    if actual !=predictIndx:
        plt.imshow(anImage)
        plt.show()
        print('Wrong:')
        print('Actual img is: '+ actualStr)
        print('Predicted img is '+predictStr +'\n')
        wrong+=1
    if actual ==predictIndx:
        print('Correct:')
        print('Actual img is: '+ actualStr)
        print('Predicted img is '+predictStr +'\n')
        right+=1
print('Wrong '+str(wrong))
print('Correct '+str(right))

# %%
import tensorflow as tf
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(network)
tflite_model = converter.convert()
# Save the TF Lite model.
with tf.io.gfile.GFile(output+'model.tflite', 'wb') as f:
  f.write(tflite_model)
#%%
import matplotlib.pyplot as plt
import numpy as np

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=output+'model.tflite') 
interpreter.allocate_tensors()
# Get input and output tensors
input_details = interpreter.get_input_details() 
output_details = interpreter.get_output_details()
# Test tflite model file on test data
wrong,right=0,0
for index,anImage in enumerate(X_Test):
    actual = np.argmax(y_test[index])
    actualStr=imgMan.getStrKeyFromVal(actual)

    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(anImage, axis=0))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted = np.argmax(output_data)
    predictStr=imgMan.getStrKeyFromVal(predicted)

    if actual !=predicted:
        plt.imshow(anImage)
        plt.show()
        print('Wrong:')
        print('Actual img is: '+ actualStr)
        print('Predicted img is '+predictStr +'\n')
        wrong+=1
    if actual ==predicted:
        print('Correct:')
        print('Actual img is: '+ actualStr)
        print('Predicted img is '+predictStr +'\n')
        right+=1

print('Wrong '+str(wrong))
print('Correct '+str(right))
#%%
import pickle
#Save Image manager and network model
data = [X_Test,y_test,X_Train,y_train]
lossFunc=cnnMod.getLossFunction()

with open(output+'Train_Test_Data.pickle', 'wb+') as out_file:
    pickle.dump(data, out_file)

with open(output+'ImageManager.pickle', 'wb+') as out_file:
    pickle.dump(imgMan, out_file)

with open(output+'BestParam.pickle', 'wb+') as out_file:
    pickle.dump(bestParam, out_file)

with open(output+'LossFunc.pickle', 'wb+') as out_file:
    pickle.dump(lossFunc, out_file)

network.save(output+"CNN_model.h5")
#%%
import pickle
import json
import h5py
output='output/'

infile = open(output+"ImageManager.pickle",'rb')
imgMan = pickle.load(infile)
infile.close()

#https://github.com/keras-team/keras/issues/10417
def fix_layer0(filename, batch_input_shape, dtype):
    with h5py.File(filename, 'r+') as f:
        model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
        layer0 = model_config['config']['layers'][0]['config']
        layer0['batch_input_shape'] = batch_input_shape
        layer0['dtype'] = dtype
        f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')

# Example
fix_layer0(output+"CNN_model.h5", [None, imgMan.getSideDimension(), imgMan.getSideDimension(),3], 'float32')

# %%
# https://stackoverflow.com/questions/61513447/load-keras-model-h5-unknown-metrics
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
output='output/'

infile = open(output+"BestParam.pickle",'rb')
bestParam = pickle.load(infile)
infile.close()

infile = open(output+"LossFunc.pickle",'rb')
loss = pickle.load(infile)
infile.close()

model = load_model(output+"CNN_model2.h5", compile = False)

METRICS=CNNModel.metrics

model.compile(optimizer = bestParam.get('anOptimizer'),
              loss = loss,
              metrics = METRICS
             )
#%%
model.summary()

# %%
param= model.evaluate(X_Test, y_test,batch_size=100)
#Print results of test data
print('Test loss:', param[0])
print('Test accuracy:', param[1]*100)
print('Test precision:', param[2]*100)
print('Test recall:', param[3]*100)
# print('Test false negative:', param[4])
# print('Test false positive:', param[5])
# print('Test true negative:', param[6])
# print('Test true positive:', param[7])

# %%