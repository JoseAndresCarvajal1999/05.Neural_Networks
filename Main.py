from Neural_Networks import Multilayer_Perceptron
from Activation_Function import Sigmoid, Tanh
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt 





data = pd.read_csv('heart.csv')
features_Name = ['cp', 'oldpeak', 'trtbps', 'age', 'restecg']
features = data[features_Name]
label_Name = ['chol']
label = data[label_Name]


#Mapping to Hipyercube (0,1)
sc = RobustScaler()
features = sc.fit_transform(features)
label = sc.fit_transform(label)
#Split dataset labes and features 
x_train, x, y_train, y = train_test_split(features,label, test_size = 0.4)
x_test, x_val, y_test, y_val = train_test_split(x,y, test_size = 0.5)
 
#Neural Network Regressor 
arq = [5,5,5,1]
activation = [Sigmoid, Tanh] 
activation_name = ['Sigmoid', 'Tanh']
learning_rate = [0.2, 0.9]


for act in activation: 
    i = 0
    for rate  in learning_rate:
        regressor = Multilayer_Perceptron(arq, act, rate)
        error, local_gradients = regressor.fit(x_train,y_train)
        y_pred = regressor.predict(x_val)
        
        plt.plot(y_pred, 'r')
        plt.plot(y_val,'b')
        plt.legend(['y_pred', 'y'])
        plt.title('rate: {}, activation. {}'.format(rate, activation_name[i]))
        plt.show()
        
        plt.plot(error)
        plt.title('Loss function, learning rate {}, activation: {}'
                  .format(rate,activation_name[i]))

        plt.show()        
        plt.plot(local_gradients)
        plt.title('Lacals gradient, learning learning rate {}, activation: {}'
                  .format(rate, activation_name[i]))
        plt.show()
        i +=1 


rate = 0.2 
activation = Sigmoid 

regressor_final =  Multilayer_Perceptron(arq, activation, rate)
error, local_gradients = regressor_final.fit(x_train, y_train)

y_pred = regressor_final.predict(x_test)

plt.plot(y_pred)
plt.plot(y_test)
plt.legend(['y_pred', 'y'])
plt.title('Pronostico del conjutno de testeo')
plt.show()


