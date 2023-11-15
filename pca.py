#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)


# In[ ]:





# In[2]:



data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

fig=plt.figure(figsize=(13, 6))

plt.subplot(1, 2, 1) # row 1, col 2 index 1
x = df["petal length (cm)"]
y = df["petal width (cm)"]

mask = x < 2
plt.scatter(x[mask], y[mask], c="orange", s=7)

mask2 = x > 2
plt.scatter(x[mask2], y[mask2], c="blue", s=7)
plt.title('(a)')

plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")

plt.subplot(1, 2, 2)

x = df["sepal length (cm)"]
y = df["sepal width (cm)"]

mask = (x < 6) 
plt.scatter(x[mask], y[mask], c="red", s=7)

mask2 = (x > 5.5) & (y < 3.5)
plt.scatter(x[mask2], y[mask2], c="green", s=7)

plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.title('(b)')

plt.savefig('dataexploration.svg')
plt.show()


# In[3]:


def standardise(y):
    mean_value = np.mean(y)
    std_value = np.std(y)
    for i in range(len(y)):
        y[i] = (y[i] - mean_value) / std_value
    return y


# In[4]:


print(np.mean(standardise(df["petal length (cm)"]))) #check if mean is 0
print(np.std(standardise(df["petal length (cm)"]))) #check if standard deviation is 1


# In[5]:


fig=plt.figure(figsize=(13, 6))

plt.subplot(1, 2, 1) # row 1, col 2 index 1
x = standardise(df["petal length (cm)"])
y = standardise(df["petal width (cm)"])

mask = x < -1
plt.scatter(x[mask], y[mask], c="orange", s=7)

mask2 = x > 0
plt.scatter(x[mask2], y[mask2], c="blue", s=7)

plt.xlabel("petal length (σ)")
plt.ylabel("petal width (σ)")
plt.title('(a)')
plt.subplot(1, 2, 2)

x = standardise(df["sepal length (cm)"])
y = standardise(df["sepal width (cm)"])

mask = (x < 0) & (y > -.5)
plt.scatter(x[mask], y[mask], c="red", s=7)

mask2 = (x > -.5) & (y < 2)
plt.scatter(x[mask2], y[mask2], c="green", s=7)

plt.xlabel("sepal length (σ)")
plt.ylabel("sepal width (σ)")
plt.title('(b)')

plt.savefig('standardisedplot.svg')
plt.show()


# In[6]:


def calculate_covariance_matrix(df, standardized=False):
    # Standardize the data if specified
    if standardized:
        df = (df - df.mean()) / df.std()

    # Number of observations
    N = len(df)

    # Number of variables
    num_variables = len(df.columns)

    # Initialize a NumPy array for the covariance matrix
    cov_matrix = np.zeros((num_variables, num_variables))

    # Calculate Covariance Matrix
    for i in range(num_variables):
        for j in range(num_variables):
            mean_i = df.iloc[:, i].mean()
            mean_j = df.iloc[:, j].mean()
            cov_matrix[i, j] = sum((df.iloc[:, i] - mean_i) * (df.iloc[:, j] - mean_j)) / N

    return cov_matrix

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Calculate covariance matrix for unstandardized variables
cov_matrix_unstd = calculate_covariance_matrix(df, standardized=False)

# Calculate covariance matrix for standardized variables
cov_matrix_std = calculate_covariance_matrix(df, standardized=True)

# Display results
print("Covariance Matrix for Unstandardized Variables:")
print(cov_matrix_unstd)

print("\nCovariance Matrix for Standardized Variables:")
print(cov_matrix_std)


# In[7]:


from numpy.linalg import eig

eigenvalue,eigenvector=eig(cov_matrix_std)
# prints the eigenvalues (a 2x1 array)
print('Eigenvalues:', eigenvalue)
# prints the eigenvectors (a 2x2 array)
print('Eigenvectors:', eigenvector)

def calculate_magnitudes(eigenvector):
    # Calculate magnitudes using NumPy array methods
    magnitudes = np.sqrt(np.sum(eigenvector**2, axis=1))
    return magnitudes
magnitudes = calculate_magnitudes(eigenvector)
print('List of the magnitudes of the eigenvectors:', magnitudes)


# In[8]:


import numpy as np
import pandas as pd
from numpy.linalg import eig
from sklearn.datasets import load_iris

def calculate_covariance_matrix(df, standardized=False):
    # Standardize the data if specified
    if standardized:
        df = (df - df.mean()) / df.std()

    # Number of observations
    N = len(df)

    # Number of variables
    num_variables = len(df.columns)

    # Initialize a NumPy array for the covariance matrix
    cov_matrix = np.zeros((num_variables, num_variables))

    # Calculate Covariance Matrix
    for i in range(num_variables):
        for j in range(num_variables):
            mean_i = df.iloc[:, i].mean()
            mean_j = df.iloc[:, j].mean()
            cov_matrix[i, j] = sum((df.iloc[:, i] - mean_i) * (df.iloc[:, j] - mean_j)) / N

    return cov_matrix

# Load Iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Calculate covariance matrix for unstandardized variables
cov_matrix_unstd = calculate_covariance_matrix(df, standardized=False)

# Calculate covariance matrix for standardized variables
cov_matrix_std = calculate_covariance_matrix(df, standardized=True)

# Display results
print("Covariance Matrix for Unstandardized Variables:")
print(cov_matrix_unstd)

print("\nCovariance Matrix for Standardized Variables:")
print(cov_matrix_std)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = eig(cov_matrix_std)

# Print eigenvalues and eigenvectors
print('\nEigenvalues:', eigenvalues)
print('Eigenvectors:')
print(eigenvectors)

# Calculate magnitudes
magnitudes = np.sqrt(np.sum(eigenvectors**2, axis=1))

# Print magnitudes
print('\nList of the magnitudes of the eigenvectors:', magnitudes)


# In[9]:


def sort_eigens(eigenvalues, eigenvectors):
    # creates a pandas dataframe out of the eigenvectors
    df_eigen = pd.DataFrame(eigenvectors)

    # adds a column for the eigenvalues
    df_eigen['Eigenvalues'] = eigenvalues

    # sorts the dataframe in place by eigenvalue
    df_eigen.sort_values("Eigenvalues", inplace=True, ascending=False)

    # makes a numpy array out of the sorted eigenvalue column
    sorted_eigenvalues = np.array(df_eigen['Eigenvalues'])
    # makes a numpy array out of the rest of the sorted dataframe
    sorted_eigenvectors = np.array(df_eigen.drop(columns="Eigenvalues"))

    # returns the sorted values
    return sorted_eigenvalues, sorted_eigenvectors
#You should then be able to use matrix multiplication to reorient your data to be aligned with the unit vectors of your principal components.

def reorient_data(df,eigenvectors):
    # turns the dataframe into a numpy array to enable matrix multiplication
    numpy_data = np.array(df)

    # mutiplies the data by the eigenvectors to get the data in terms of pca features
    pca_features = np.dot(numpy_data, eigenvectors)

    # turns the new array back into a dataframe for plotting
    pca_df = pd.DataFrame(pca_features)

    return pca_df


# In[10]:


#Now use the above functions to sort the iris dataset by principal components


# In[11]:


eigenvalues, eigenvectors = eig(cov_matrix_std)

# Sort eigenvectors based on eigenvalues
sorted_eigenvalues, sorted_eigenvectors = sort_eigens(eigenvalues, eigenvectors)

# Reorient the data using matrix multiplication
pca_data = reorient_data(df, sorted_eigenvectors)
print(pca_data)


# In[21]:


fig=plt.figure(figsize=(13, 6))

plt.subplot(1, 2, 1) # row 1, col 2 index 1
x = standardise(pca_data[2])
y = standardise(pca_data[3])

mask = x < -1
plt.scatter(x[mask], y[mask], c="orange", s=7)

mask2 = x > 0
plt.scatter(x[mask2], y[mask2], c="blue", s=7)

plt.xlabel("petal length (σ)")
plt.ylabel("petal width (σ)")
plt.title('(a)')
plt.subplot(1, 2, 2)

x = standardise(pca_data[0])
y = standardise(pca_data[1])

mask = x < -.5
plt.scatter(x[mask], y[mask], c="red", s=7)

mask2 = x > -.5
plt.scatter(x[mask2], y[mask2], c="green", s=7)

plt.xlabel("sepal length (σ)")
plt.ylabel("sepal width (σ)")
plt.title('(b)')

plt.savefig('pcaplot1.svg')
plt.show()


# In[19]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Fit k-means clustering with k=3 (assuming there are three species)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(pca_data.iloc[:, [0, 1]])

fig = plt.figure(figsize=(13, 6))

# Plot for petal length and width (PC2 vs. PC3)
plt.subplot(1, 2, 1)
x = standardise(pca_data[2])
y = standardise(pca_data[3])

# Plot points with cluster colors
plt.scatter(x, y, c=clusters, cmap='viridis', s=7)

plt.xlabel("petal length (σ)")
plt.ylabel("petal width (σ)")
plt.title('(a)')

# Plot for sepal length and width (PC1 vs. PC2)
plt.subplot(1, 2, 2)
x = standardise(pca_data[0])
y = standardise(pca_data[1])

# Plot points with cluster colors
plt.scatter(x, y, c=clusters, cmap='viridis', s=7)

plt.xlabel("sepal length (σ)")
plt.ylabel("sepal width (σ)")
plt.title('(b)')

plt.show()

