# Implementation-of-Ridge-Regression

## Objective:
This project implements the linear regression with Tikhonov regularization from scratch (without using any existing machine learning libraries e.g. sklearn) for predicting 2D surface sampled data.   

The hyperparameter lambda (the weight of penalty) is fine-tuned using 10-Fold Cross Validataion which is also implemented from scratch.

Note: sklearn packages are used in this project for verification and comparision purposes.

## Dataset:
The data used for this project corresponds to samples from a 2D surface.

### Format: 
There is one row per data instance and one column per attribute. The targets are real values. The training set is already divided into 10 subsets for 10-fold cross validation.

## Mean Squared Error Results from 10-Fold Cross Validation:

Lambda increases from 0 to 4 with stepsize of 0.1.

![Capture](https://user-images.githubusercontent.com/29167705/63799500-39396800-c8da-11e9-8b77-80067e660550.JPG)


## Visualization:
![Capture](https://user-images.githubusercontent.com/29167705/63799601-68e87000-c8da-11e9-8770-634b24c37b3e.JPG)

## Analysis of Weighted Loss Function:
The loss functions used in this project assumes that the error contributed by each data point have the same importance. If we consider a
scenario where we would like to give more weight to some data points. The goal is to fit the data points (xn, yn) in proportion to their weights rn by minimizing the following objective:

![Capture](https://user-images.githubusercontent.com/29167705/63800217-ad284000-c8db-11e9-9a9b-d30c8df021f2.JPG)

Deriviation of the closed-form expression for the estimates of w and b that minimize the above objective:

![Capture](https://user-images.githubusercontent.com/29167705/63800399-055f4200-c8dc-11e9-936a-c73fa48ff6e6.JPG)

The above objective is equivalent to the negative log-likelihood for linear regression where each data point may have a different Gaussian measurement noise.

Proof:

![Capture](https://user-images.githubusercontent.com/29167705/63800608-86b6d480-c8dc-11e9-895d-44fb8b6a27da.JPG)
