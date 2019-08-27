import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


class LinearRegression:
    # The default Lambda is set to 2
    def __init__(self, Lambda=2):
        self.Lambda = Lambda

    def fit(self, train_data, train_label):
        # Add ones to all sample points
        x0 = np.ones(train_data.shape[0])
        x_bar = np.c_[x0, train_data.values]
        # Ridge regression: A = (xtx)^(-1) + (Lambda*I), b = xty, Aw = b
        A = np.matmul(np.transpose(x_bar), x_bar) + np.eye(x_bar.shape[1]) * self.Lambda
        A_inv = np.linalg.inv(A)
        b = np.matmul(np.transpose(x_bar), train_label.values)
        self.w = np.matmul(A_inv, b)

    def predict(self, test_data):
        x0 = np.ones(test_data.shape[0])
        x_bar = np.c_[x0, test_data.values]
        pred = np.matmul(x_bar, self.w)
        return pred


# Merge multiple csv file into one data and one label data frames. Optionally, we can exclude certain files
def merge_train_files(num_of_files, skip=None):
    df_train_data = pd.DataFrame()
    df_train_label = pd.DataFrame()
    for k in range(num_of_files):
        if k == skip:
            continue
        data = pd.read_csv('regression-dataset/trainInput' + str(k + 1) + '.csv', header=None)
        df_train_data = df_train_data.append(data, ignore_index=True)
        label = pd.read_csv('regression-dataset/trainTarget' + str(k + 1) + '.csv', header=None)
        df_train_label = df_train_label.append(label, ignore_index=True)
    return df_train_data, df_train_label


def MSE(p1, p2):
    return np.linalg.norm(p1 - p2) ** 2 / p1.shape[0]


def cross_validation(k_fold=10, max_Lambda=4):
    MSEs = []
    for i in [x / 10 for x in range(max_Lambda * 10 + 1)]:
        MSE_per_run = []
        for j in range(k_fold):
            df_validate_data = pd.read_csv('regression-dataset/trainInput' + str(j + 1) + '.csv', header=None)
            df_validate_label = pd.read_csv('regression-dataset/trainTarget' + str(j + 1) + '.csv', header=None)
            df_train_data, df_train_label = merge_train_files(k_fold, skip=j)
            # Create a linear regression classifier
            clf = LinearRegression(Lambda=i)
            clf.fit(df_train_data, df_train_label)
            pred = clf.predict(df_validate_data)
            MSE_per_run.append(MSE(df_validate_label, pred))
            # At the end of each k-fold cv, calculate the average MSE
            if j == k_fold - 1:
                avg_MSE = np.mean(np.array(MSE_per_run))
                MSEs.append(avg_MSE)
                print('Lambda = {}, MSE = {:8.6f}'.format(i, avg_MSE))
    # Find the index of the minimum MSE so we can get optimal Lambda by multiplying 0.1
    optimal_Lambda = np.argmin(np.array(MSEs)) * 0.1
    print('The best Lambda = ', optimal_Lambda)
    return optimal_Lambda, MSEs


optimal_Lambda, y = cross_validation()
# Find MSE for the test set
df_train_data, df_train_label = merge_train_files(10)
df_test_data = pd.read_csv('regression-dataset/testInput.csv', header=None)
df_test_label = pd.read_csv('regression-dataset/testTarget.csv', header=None)
clf = LinearRegression(Lambda=2)
clf.fit(df_train_data, df_train_label)
pred = clf.predict(df_test_data)
print('The MSE for the test set = {:8.6f}'.format(MSE(df_test_label, pred)))
# Try linear regression model from sklearn
clf_ = linear_model.LinearRegression()
clf_.fit(df_train_data, df_train_label)
pred_ = clf.predict(df_test_data)
print('sklearn: The MSE for the test set = {:8.6f}'.format(MSE(df_test_label, pred_)))


# Plot the relationship between Lambda and MSE
x = [i/10 for i in range(41)]
plt.plot(x, y)
plt.xlabel('Lambda', fontsize=14)
plt.ylabel('10-Fold Cross Validation MSE', fontsize=14)
plt.title('Lambda vs Mean Squared Error', fontsize=18)
plt.show()

