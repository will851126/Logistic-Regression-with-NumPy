# Logistic Regression  with NumPy 

## Describe

In this blog , we will be implementing from scratch is **Logistic Regression**. Alongside its belowed sister algorithm linear regression, this one is higly used in machine learning as well, due to its simplicity and robustness. Even though its called logistic regression, it's actually a classification algorithm that is used to classify input data into its classes 

This powerful machine learning model can be used to answer some questions such as;

* Whether an e-mail is spam or not
* If the customer will churn
* Whether a tumor is benign or malignant

All of the questions above are simply yes-no questions, therefore they can be used to classify input data into two classes. Hence, the term binary classification is used when the data can be categorized into two distinct classes. Obviously, multi-class classification deals with data that has more than two classes. After grasping the ins and outs of logistic regression to make binary classification, transition to a multi-class classification is pretty straight-forward, as a consequence, here, we will deal with data that has two classes only.

Remember, that in linear regression we predict numerical values based on the input values and parameters of the model.Here, in logistic regression we can also approach the model as we are trying to predict numbers but this time these numbers correspond to the probability values of input data belonging to a particular class.

The term logistic in logistic regression is used because this time we are applying another function to the weighted sum of input data and paratemeters of the model and this function is called logit (sigmoid) function.

Sigmoid function always outputs values between 0 and 1, and thus can be used to calculate probabilities of input data belonging to a certain class:

<img src="images/sigmoid.png"
     style="float: left; margin-right: 10px;" />



## Implement

1. We start off by importing necessary libraries. As always, **NumPy** is the only package that we will use in order to implement the logistic regression algorithm. All the others will only help us with minor issues such as visualizing the data at hand or creating a dataset. Hence, we won't be using already implemented package solutions for logistic regression.

```def sigmoid(x): 
     return 1 / (1 + np.exp(-x))
```

Here, we write the code for the aforementioned sigmoid (logit) function. It is important to note that this function can be applied to all of the elements of a numpy array individually, simply because we make use of the exponential function from the NumPy package.

```
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
    return cost
```

Next, we write the cost function for logistic regression. Note that the cost function used in logistic regression is different than the one used in linear regression.

Remember, in linear regression we calculated the weighted sum of input data and parameters and fed that sum to the cost function to calculate the cost. When we plotted the cost function it was seen to be convex, hence a local minimum was also the global minimum.

However, in logistic regression, we apply sigmoid function to the weighted sum which makes the resulting outcome non-linear. If we feed that non-linear outcome to the cost function, what we get would be a non-convex function and we wouldn't be assured to find only one local minimum that is also the global minimum.

As a result, we use another cost function to calculate the cost which is guaranteed to give one local minimum during the optimization.


```
def gradient_descent(X, y, params, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros((iterations,1))

    for i in range(iterations):
        params = params - (learning_rate/m) * (X.T @ (sigmoid(X @ params) - y)) 
        cost_history[i] = compute_cost(X, y, params)

    return (cost_history, params)
```

Gradient descent implementation  is not so different than the one we used in linear regression. Only difference to be noted is the sigmoid function that is applied to the weighted sum.


```
def predict(X, params):
    return np.round(sigmoid(X @ params))
```

While writing out the prediction function, let's not forget that we are dealing with probabilities here.  if the resulting value is above 0.50, we round it up to 1, meaning the data point belongs to the class 1. Consequently, if the probability of a data point belonging to the class 1 is below 0.50, it simply means that it is part of the other class (class 0). Remember that this is binary classification, so we have only two classes (class 1 and class 0).


<img src="images/classification.png"
     style="float: left; margin-right: 10px;" />

After coding out the necessary functions, let's create our own dataset with `make_classification`  function from sklearn.datasets. We create 500 sample points with two classes and plot the dataset with the help of seaborn library.



<img src="images/cost-function.png"
     style="float: left; margin-right: 10px;" />


Now, let's run our algorithm and calculate the parameters of our model. Seeing plot, we can now be sure that we have implemented the logistic regression algorithm without a fault, since it decreases with every iteration until the decrease is so minimal that the cost converges to a minimum which is what we want indeed.


<img src="images/visualization.png"
     style="float: left; margin-right: 10px;" />
    


Now, for the sake of visualization lets plot our dataset along with the decision boundary of our model. We simply calculate the intercept and slope values using the optimal parameters and plot the boundary that classifies the data into two classes. We can see from the plot that the classification is not 100% correct since the separation of classes is not linear naturally.




