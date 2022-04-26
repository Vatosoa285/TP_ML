# Multi-Layer Perceptron
## Introduction

The objective of this lab is to dive into particular kind of neural network: the *Multi-Layer Perceptron* (MLP).

To start, let us take the dataset from the previous lab (hydrodynamics of sailing boats) and use scikit-learn to train a MLP instead of our hand-made single perceptron.
The code below is already complete and is meant to give you an idea of how to construct an MLP with scikit-learn. You can execute it, taking the time to understand the idea behind each cell.


```python
# Importing the dataset
import numpy as np
dataset = np.genfromtxt("yacht_hydrodynamics.data", delimiter='')
X = dataset[:, :-1]
Y = dataset[:, -1]
```


```python
# Preprocessing: scale input data 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
```


```python
# Split dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=1, test_size = 0.20)
```


```python
# Define a multi-layer perceptron (MLP) network for regression
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(max_iter=3000, random_state=1) # define the model, with default params
mlp.fit(x_train, y_train) # train the MLP
```




    MLPRegressor(max_iter=3000, random_state=1)




```python
# Evaluate the model
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

print('Train score: ', mlp.score(x_train, y_train))
print('Test score:  ', mlp.score(x_test, y_test))
plt.plot(mlp.loss_curve_)
plt.xlabel("Iterations")
plt.ylabel("Loss")

fig = plt.gcf()
fig.set_size_inches(5, 4)
plt.show()
```

    Train score:  0.9940765369322633
    Test score:   0.9899773031580283



    
![png](mlp_files/mlp_5_1.png)
    



```python
# Plot the results
num_samples_to_plot = 20
plt.plot(y_test[0:num_samples_to_plot], 'ro', label='y')
yw = mlp.predict(x_test)
plt.plot(yw[0:num_samples_to_plot], 'bx', label='$\hat{y}$')
plt.legend()
plt.xlabel("Examples")
plt.ylabel("f(examples)")

fig = plt.gcf()
fig.set_size_inches(5, 4)
plt.show()
```


    
![png](mlp_files/mlp_6_0.png)
    


### Analyzing the network

Many details of the network are currently hidden as default parameters.

Using the [documentation of the MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html), answer the following questions.

- What is the structure of the network?
- What it is the algorithm used for training? Is there algorithm available that we mentioned during the courses?
- How does the training algorithm decides to stop the training?

*What is the structure of the network?*
* 3 layers, the hidden layer has 100 percetrons


*What it is the algorithm used for training? Is there algorithm available that we mentioned during the courses?*
* The default value ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba

*How does the training algorithm decides to stop the training?*
* It stops the training after `max_iter` itterations are done, there is no early stopping by default.

## Onto a more challenging dataset: house prices

For the rest of this lab, we will use the (more challenging) [California Housing Prices dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices).


```python
# clean all previously defined variables for the sailing boats
%reset -f

# Import the required modules
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import copy

```


```python
num_samples = 3000 # only use the first N samples to limit training time

cal_housing = fetch_california_housing()
X = pd.DataFrame(cal_housing.data,columns=cal_housing.feature_names)[:num_samples]
Y = cal_housing.target[:num_samples]

X.head(10) # print the first 10 values
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.0368</td>
      <td>52.0</td>
      <td>4.761658</td>
      <td>1.103627</td>
      <td>413.0</td>
      <td>2.139896</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.6591</td>
      <td>52.0</td>
      <td>4.931907</td>
      <td>0.951362</td>
      <td>1094.0</td>
      <td>2.128405</td>
      <td>37.84</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.1200</td>
      <td>52.0</td>
      <td>4.797527</td>
      <td>1.061824</td>
      <td>1157.0</td>
      <td>1.788253</td>
      <td>37.84</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.0804</td>
      <td>42.0</td>
      <td>4.294118</td>
      <td>1.117647</td>
      <td>1206.0</td>
      <td>2.026891</td>
      <td>37.84</td>
      <td>-122.26</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3.6912</td>
      <td>52.0</td>
      <td>4.970588</td>
      <td>0.990196</td>
      <td>1551.0</td>
      <td>2.172269</td>
      <td>37.84</td>
      <td>-122.25</td>
    </tr>
  </tbody>
</table>
</div>



Note that each row of the dataset represents a **group of houses** (one district). The `target` variable denotes the average house value in units of 100.000 USD. Median Income is per 10.000 USD.

### Extracting a subpart of the dataset for testing

- Split the dataset between a training set (75%) and a test set (25%)

Please use the conventional names `X_train`, `X_test`, `y_train` and `y_test`.


```python
X_train, X_test, Y_train,Y_test = train_test_split(X, Y,random_state=1, test_size = 0.25)
```

### Scaling the input data


A step of **scaling** of the data is often useful to ensure that all input data centered on 0 and with a fixed variance.

Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance). The function `StandardScaler` from `sklearn.preprocessing` computes the standard score of a sample as:

```
z = (x - u) / s
```

where `u` is the mean of the training samples, and `s` is the standard deviation of the training samples.

Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean and standard deviation are then stored to be used on later data using transform.

 - Apply the standard scaler to both the training dataset (`X_train`) and the test dataset (`X_test`).
 - Make sure that **exactly the same transformation** is applied to both datasets.

[Documentation of standard scaler in scikit learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)




```python
scaler = StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test =scaler.transform(X_test) #we only use scaler.transform to keep the same parametter 
```

## Overfitting

In this part, we are only interested in maximizing the **train score**, i.e., having the network memorize the training examples as well as possible.

- Propose a parameterization of the network (shape and learning parameters) that will maximize the train score (without considering the test score).

While doing this, you should (1) remain within two minutes of training time, and (2) obtain a score that is greater than 0.90.

- Is the **test** score substantially smaller than the **train** score (indicator of overfitting) ?
- Explain how the parameters you chose allow the learned model to overfit.


```python
h_layer=[]
train_score_layers=[]
test_score_layers=[]
nb_layers=[i for i in range(2,12)]
for i in range(10):
    h_layer.append(100)
    mlp = MLPRegressor(max_iter=3000, random_state=1,hidden_layer_sizes=tuple(h_layer))
    mlp.fit(X_train, Y_train)
    train_score_layers.append(mlp.score(X_train, Y_train))
    test_score_layers.append(mlp.score(X_test, Y_test))
```


```python
iter=np.linspace(500,2000,10).astype(int)
train_score_iterations=[]
test_score_iterations=[]
for nb_iter in iter:
    mlp = MLPRegressor(max_iter=nb_iter, random_state=1, hidden_layer_sizes=(50,50))
    mlp.fit(X_train, Y_train)
    train_score_iterations.append(mlp.score(X_train, Y_train))
    test_score_iterations.append(mlp.score(X_test, Y_test))
```


```python
fig, axs = plt.subplots(1, 2, constrained_layout=True)

fig.set_size_inches(10,4)
axs.flat[0].plot(nb_layers,train_score_layers,'go',label='training score')
axs.flat[0].plot(nb_layers,test_score_layers,'ro',label='tests score')
axs.flat[0].legend()
axs.flat[0].set_title('Impact of the number of layers')
axs.flat[0].set_xlabel("number of layers")
axs.flat[0].set_ylabel('score')

axs.flat[1].plot(iter, train_score_iterations,'g',label='training score')
axs.flat[1].plot(iter, test_score_iterations,'r',label='tests score')
axs.flat[1].legend()
axs.flat[1].set_title('Impact of the number of iterrations')
axs.flat[1].set_xlabel("number of layers")
axs.flat[1].set_ylabel('score')

plt.show()
```


    
![png](mlp_files/mlp_20_0.png)
    


### Remarks
* We remark that the test score is alway smaller than teh train score 
* Using a great number of layers cause overfitting &#8658; The train scrore increase while the test score decrease

    A large number of layers allow the model to be closer to the train data 
* The number of itteration doesn't seams to have an impact on overfitting

## Hyperparameter tuning

In this section, we are now interested in maximizing the ability of the network to predict the value of unseen examples, i.e., maximizing the **test** score.
You should experiment with the possible parameters of the network in order to obtain a good test score, ideally with a small learning time.

Parameters to vary:

- number and size of the hidden layers
- activation function
- stopping conditions
- maximum number of iterations
- initial learning rate value

Results to present for the tested configurations:

- Train/test score
- training time


Present in a table the various parameters tested and the associated results. You can find in the last cell of the notebook a code snippet that will allow you to plot tables from python structure.
Be methodical in the way your run your experiments and collect data. For each run, you should record the parameters and results into an external data structure.

(Note that, while we encourage you to explore the solution space manually, there are existing methods in scikit-learn and other learning framework to automate this step as well, e.g., [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html))


```python
parameters= [
    {'activation': 'tanh', 'max_iter': 4000, 'early_stopping': False, 'hidden_layer_sizes':(50,50), 'learning_rate_init':0.001, 'learning_rate':'constant', 'val_score': None, 'test_score': None, 'train_score': None, 'training_time':None},
    {'activation': 'relu', 'max_iter': 1000, 'early_stopping': True, 'hidden_layer_sizes':(50, 50,  50), 'learning_rate_init':0.001, 'learning_rate':'constant', 'val_score': None, 'test_score': None, 'train_score': None, 'training_time':None},
    {'activation': 'tanh', 'max_iter': 10000, 'early_stopping': False, 'hidden_layer_sizes':(50,), 'learning_rate_init':0.001, 'learning_rate':'adaptive', 'val_score': None, 'test_score': None, 'train_score': None, 'training_time':None},
    {'activation': 'relu', 'max_iter': 1000, 'early_stopping': False, 'hidden_layer_sizes':(50,50), 'learning_rate_init':0.001, 'learning_rate':'constant', 'val_score': None, 'test_score': None, 'train_score': None, 'training_time':None},
    {'activation': 'logistic', 'max_iter': 3000, 'early_stopping': False, 'hidden_layer_sizes':(50,50), 'learning_rate_init':0.001, 'learning_rate':'adaptive', 'val_score': None, 'test_score': None, 'train_score': None, 'training_time':None},
    {'activation': 'tanh', 'max_iter': 1000, 'early_stopping': False, 'hidden_layer_sizes':(50,), 'learning_rate_init':0.001, 'learning_rate':'constant', 'val_score': None, 'test_score': None, 'train_score': None, 'training_time':None},
    {'activation': 'tanh', 'max_iter': 1000, 'early_stopping': False, 'hidden_layer_sizes':(50,), 'learning_rate_init':0.001, 'learning_rate':'constant', 'val_score': None, 'test_score': None, 'train_score': None, 'training_time':None},
    {'activation': 'tanh', 'max_iter': 1000, 'early_stopping': False, 'hidden_layer_sizes':(50,), 'learning_rate_init':0.001, 'learning_rate':'constant', 'val_score': None, 'test_score': None, 'train_score': None, 'training_time':None},
    {'activation': 'logistic', 'max_iter': 600, 'early_stopping': False, 'hidden_layer_sizes':(50,), 'learning_rate_init':0.01, 'learning_rate':'adaptive', 'val_score': None, 'test_score': None, 'train_score': None, 'training_time':None},
    {'activation': 'tanh', 'max_iter': 1000, 'early_stopping': False, 'hidden_layer_sizes':(50,), 'learning_rate_init':0.001, 'learning_rate':'constant', 'val_score': None, 'test_score': None, 'train_score': None, 'training_time':None},
]
```


```python
def find_best_parametter(X_train,X_val,Y_train, Y_val, parameters):
    Parameters = copy.deepcopy(parameters)
    for prm in Parameters:
        mlp = MLPRegressor(
            activation=prm['activation'],
            max_iter=prm['max_iter'],
            hidden_layer_sizes=prm['hidden_layer_sizes'],
            learning_rate_init=prm['learning_rate_init'],
            learning_rate=prm['learning_rate'],
            early_stopping=prm['early_stopping']
        )
        
        st = time.time()
        mlp.fit(X_train, Y_train)
        et = time.time() - st
        prm['train_score'] = mlp.score(X_train, Y_train)
        prm['val_score'] = mlp.score(X_val, Y_val)
        prm['training_time'] = et
    return Parameters
```


```python
# Code snippet to display a nice table in jupyter notebooks  (remove from report)
checked_prms = find_best_parametter(X_train,X_test,Y_train, Y_test, parameters)

table = pd.DataFrame.from_dict(checked_prms)
table = table.replace(np.nan, '-')
table = table.sort_values(by='val_score', ascending=False)
table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>activation</th>
      <th>max_iter</th>
      <th>early_stopping</th>
      <th>hidden_layer_sizes</th>
      <th>learning_rate_init</th>
      <th>learning_rate</th>
      <th>val_score</th>
      <th>test_score</th>
      <th>train_score</th>
      <th>training_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tanh</td>
      <td>4000</td>
      <td>False</td>
      <td>(50, 50)</td>
      <td>0.001</td>
      <td>constant</td>
      <td>0.865599</td>
      <td>-</td>
      <td>0.875062</td>
      <td>3.314819</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tanh</td>
      <td>1000</td>
      <td>False</td>
      <td>(50,)</td>
      <td>0.001</td>
      <td>constant</td>
      <td>0.865209</td>
      <td>-</td>
      <td>0.863621</td>
      <td>1.573738</td>
    </tr>
    <tr>
      <th>8</th>
      <td>logistic</td>
      <td>600</td>
      <td>False</td>
      <td>(50,)</td>
      <td>0.010</td>
      <td>adaptive</td>
      <td>0.861801</td>
      <td>-</td>
      <td>0.858189</td>
      <td>0.471554</td>
    </tr>
    <tr>
      <th>6</th>
      <td>tanh</td>
      <td>1000</td>
      <td>False</td>
      <td>(50,)</td>
      <td>0.001</td>
      <td>constant</td>
      <td>0.859807</td>
      <td>-</td>
      <td>0.862771</td>
      <td>1.577773</td>
    </tr>
    <tr>
      <th>7</th>
      <td>tanh</td>
      <td>1000</td>
      <td>False</td>
      <td>(50,)</td>
      <td>0.001</td>
      <td>constant</td>
      <td>0.856682</td>
      <td>-</td>
      <td>0.858054</td>
      <td>1.211859</td>
    </tr>
    <tr>
      <th>9</th>
      <td>tanh</td>
      <td>1000</td>
      <td>False</td>
      <td>(50,)</td>
      <td>0.001</td>
      <td>constant</td>
      <td>0.855972</td>
      <td>-</td>
      <td>0.859859</td>
      <td>1.329147</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tanh</td>
      <td>10000</td>
      <td>False</td>
      <td>(50,)</td>
      <td>0.001</td>
      <td>adaptive</td>
      <td>0.855211</td>
      <td>-</td>
      <td>0.849904</td>
      <td>1.293586</td>
    </tr>
    <tr>
      <th>4</th>
      <td>logistic</td>
      <td>3000</td>
      <td>False</td>
      <td>(50, 50)</td>
      <td>0.001</td>
      <td>adaptive</td>
      <td>0.848674</td>
      <td>-</td>
      <td>0.847081</td>
      <td>5.032111</td>
    </tr>
    <tr>
      <th>3</th>
      <td>relu</td>
      <td>1000</td>
      <td>False</td>
      <td>(50, 50)</td>
      <td>0.001</td>
      <td>constant</td>
      <td>0.839223</td>
      <td>-</td>
      <td>0.893153</td>
      <td>2.051382</td>
    </tr>
    <tr>
      <th>1</th>
      <td>relu</td>
      <td>1000</td>
      <td>True</td>
      <td>(50, 50, 50)</td>
      <td>0.001</td>
      <td>constant</td>
      <td>0.831973</td>
      <td>-</td>
      <td>0.843345</td>
      <td>0.538187</td>
    </tr>
  </tbody>
</table>
</div>



## Evaluation

- From your experiments, what seems to be the best model (i.e. set of parameters) for predicting the value of a house?

Unless you used cross-validation, you have probably used the "test" set to select the best model among the ones you experimented with.
Since your model is the one that worked best on the "test" set, your selection is *biased*.

In all rigor the original dataset should be split in three:

- the **training set**, on which each model is trained
- the **validation set**, that is used to pick the best parameters of the model 
- the **test set**, on which we evaluate the final model


Evaluate the score of your algorithm on a test set that was not used for training nor for model selection.




```python
X_tr, X_test, Y_tr,Y_test = train_test_split(X, Y,random_state=1, test_size = 0.20) # 20% for testing
X_train, X_val, Y_train,Y_val = train_test_split(X_tr, Y_tr,random_state=1, test_size = 0.25) # 20% for validation

scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_val =scaler.transform(X_val) #we only use scaler.transform to keep the same parametter 
X_test =scaler.transform(X_test) #we only use scaler.transform to keep the same parametter 

df =  pd.DataFrame.from_dict(find_best_parametter(X_train,X_val,Y_train, Y_val,parameters))
df = df.replace(np.nan, '-')

df = df.sort_values(by='val_score', ascending=False)
df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>activation</th>
      <th>max_iter</th>
      <th>early_stopping</th>
      <th>hidden_layer_sizes</th>
      <th>learning_rate_init</th>
      <th>learning_rate</th>
      <th>val_score</th>
      <th>test_score</th>
      <th>train_score</th>
      <th>training_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tanh</td>
      <td>4000</td>
      <td>False</td>
      <td>(50, 50)</td>
      <td>0.001</td>
      <td>constant</td>
      <td>0.827788</td>
      <td>-</td>
      <td>0.866743</td>
      <td>2.036587</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tanh</td>
      <td>1000</td>
      <td>False</td>
      <td>(50,)</td>
      <td>0.001</td>
      <td>constant</td>
      <td>0.818846</td>
      <td>-</td>
      <td>0.868004</td>
      <td>1.533499</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tanh</td>
      <td>10000</td>
      <td>False</td>
      <td>(50,)</td>
      <td>0.001</td>
      <td>adaptive</td>
      <td>0.818130</td>
      <td>-</td>
      <td>0.863551</td>
      <td>1.300022</td>
    </tr>
    <tr>
      <th>8</th>
      <td>logistic</td>
      <td>600</td>
      <td>False</td>
      <td>(50,)</td>
      <td>0.010</td>
      <td>adaptive</td>
      <td>0.817825</td>
      <td>-</td>
      <td>0.868591</td>
      <td>0.545273</td>
    </tr>
    <tr>
      <th>9</th>
      <td>tanh</td>
      <td>1000</td>
      <td>False</td>
      <td>(50,)</td>
      <td>0.001</td>
      <td>constant</td>
      <td>0.815315</td>
      <td>-</td>
      <td>0.855915</td>
      <td>0.964416</td>
    </tr>
    <tr>
      <th>3</th>
      <td>relu</td>
      <td>1000</td>
      <td>False</td>
      <td>(50, 50)</td>
      <td>0.001</td>
      <td>constant</td>
      <td>0.813041</td>
      <td>-</td>
      <td>0.889887</td>
      <td>1.635100</td>
    </tr>
    <tr>
      <th>7</th>
      <td>tanh</td>
      <td>1000</td>
      <td>False</td>
      <td>(50,)</td>
      <td>0.001</td>
      <td>constant</td>
      <td>0.811582</td>
      <td>-</td>
      <td>0.848408</td>
      <td>0.905025</td>
    </tr>
    <tr>
      <th>6</th>
      <td>tanh</td>
      <td>1000</td>
      <td>False</td>
      <td>(50,)</td>
      <td>0.001</td>
      <td>constant</td>
      <td>0.809922</td>
      <td>-</td>
      <td>0.845506</td>
      <td>0.769561</td>
    </tr>
    <tr>
      <th>1</th>
      <td>relu</td>
      <td>1000</td>
      <td>True</td>
      <td>(50, 50, 50)</td>
      <td>0.001</td>
      <td>constant</td>
      <td>0.792135</td>
      <td>-</td>
      <td>0.854870</td>
      <td>0.656084</td>
    </tr>
    <tr>
      <th>4</th>
      <td>logistic</td>
      <td>3000</td>
      <td>False</td>
      <td>(50, 50)</td>
      <td>0.001</td>
      <td>adaptive</td>
      <td>0.787650</td>
      <td>-</td>
      <td>0.839308</td>
      <td>2.576903</td>
    </tr>
  </tbody>
</table>
</div>




```python
best_params = df[df.val_score == df.val_score.max()].to_dict('records')[0]
```


```python
mlp = MLPRegressor(
            activation=best_params['activation'],
            max_iter=best_params['max_iter'],
            hidden_layer_sizes=best_params['hidden_layer_sizes'],
            learning_rate_init=best_params['learning_rate_init'],
            learning_rate=best_params['learning_rate'],
            early_stopping=best_params['early_stopping']
        )

st = time.time()
mlp.fit(X_train, Y_train)
et = time.time() - st
# best_params['train_score'] = mlp.score(X_train.values, Y_train)
best_params['test_score'] = mlp.score(X_test, Y_test)
best_params['training_time'] = et
results = pd.DataFrame.from_dict([best_params])
results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>activation</th>
      <th>max_iter</th>
      <th>early_stopping</th>
      <th>hidden_layer_sizes</th>
      <th>learning_rate_init</th>
      <th>learning_rate</th>
      <th>val_score</th>
      <th>test_score</th>
      <th>train_score</th>
      <th>training_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tanh</td>
      <td>4000</td>
      <td>False</td>
      <td>(50, 50)</td>
      <td>0.001</td>
      <td>constant</td>
      <td>0.827788</td>
      <td>0.86979</td>
      <td>0.866743</td>
      <td>2.12866</td>
    </tr>
  </tbody>
</table>
</div>



### Remarks
* In the current version of our code we train our model two times
    * For selecting the best parameters
    * For the final test of th model
    
    It will be better if our function `find_best_parametter` returned the model that performed best during selection phase along side with the parameters.
