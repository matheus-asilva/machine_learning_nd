#%%
import os
os.chdir('/home/matheus/Documents/GitHub/machine_learning_nd/unsupervised_learning/project')

#%% [markdown]
# # Machine Learning Engineer Nanodegree
# ## Unsupervised Learning
# ## Project: Creating Customer Segments
#%% [markdown]
# Welcome to the third project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
#%% [markdown]
# ## Getting Started
# 
# In this project, you will analyze a dataset containing data on various customers' annual spending amounts (reported in *monetary units*) of diverse product categories for internal structure. One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with. Doing so would equip the distributor with insight into how to best structure their delivery service to meet the needs of each customer.
# 
# The dataset for this project can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers). For the purposes of this project, the features `'Channel'` and `'Region'` will be excluded in the analysis — with focus instead on the six product categories recorded for customers.
# 
# Run the code block below to load the wholesale customers dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.

#%%
# Import libraries necessary for this project
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
%matplotlib inline

# Load the wholesale customers dataset
data = pd.read_csv('./data/customer.csv')
data.drop(['Region', 'Channel'], axis=1, inplace=True)
print('Wholesale customers dataset has {} samples with {} features each.'.format(*data.shape))
display(data.head())

#%% [markdown]
# ## Data Exploration
# In this section, you will begin exploring the data through visualizations and code to understand how each feature is related to the others. You will observe a statistical description of the dataset, consider the relevance of each feature, and select a few sample data points from the dataset which you will track through the course of this project.
# 
# Run the code block below to observe a statistical description of the dataset. Note that the dataset is composed of six important product categories: **'Fresh'**, **'Milk'**, **'Grocery'**, **'Frozen'**, **'Detergents_Paper'**, and **'Delicatessen'**. Consider what each category represents in terms of products you could purchase.

#%%
# Display a description of the dataset
display(data.describe())

#%% [markdown]
# ### Implementation: Selecting Samples
# To get a better understanding of the customers and how their data will transform through the analysis, it would be best to select a few sample data points and explore them in more detail. In the code block below, add **three** indices of your choice to the `indices` list which will represent the customers to track. It is suggested to try different sets of samples until you obtain customers that vary significantly from one another.

#%%
# Select three indices of your choice you wish to sample from the dataset
indices = [3,10,100]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print("Chosen samples of wholesale customers dataset:")
display(samples)

plt.axes().set_title('Heatmap vs 75% percentile')
sns.heatmap((samples-data.quantile(q=.75))/data.std(ddof=0), annot=True, square=True);

#%% [markdown]
# ### Question 1
# Consider the total purchase cost of each product category and the statistical description of the dataset above for your sample customers.  
# 
# * What kind of establishment (customer) could each of the three samples you've chosen represent?
# 
# **Hint:** Examples of establishments include places like markets, cafes, delis, wholesale retailers, among many others. Avoid using names for establishments, such as saying *"McDonalds"* when describing a sample customer as a restaurant. You can use the mean values for reference to compare your samples with. The mean values are as follows:
# 
# * Fresh: 12000.2977
# * Milk: 5796.2
# * Grocery: 7951.3
# * Detergents_paper: 2881.4
# * Delicatessen: 1524.8
# 
# Knowing this, how do your samples compare? Does that help in driving your insight into what kind of establishments they might be? 
# 

#%%
print('Offset from mean of whole dataset')
display(samples - np.around(data.mean().values, 1))

print('Offset from median of whole dataset')
display(samples - np.around(data.median().values, 1))

#%%
print('Samples vs Dataset Values')
sns.heatmap((samples - np.around(data.mean().values))/data.std(ddof=0), annot=True); # ddof: degrees of freedom
#%% [markdown]
# **Answer:**
# * Sample 1 - Fresh, Frozen and Delicatessen are well above the mean and median of the dataset for these features.
# It looks like a small restaurant which serves dishes or snacks, hence needs more of fresh and frozen stuff.
# * Sample 2 - Grocery, Frozen and Detergents_Paper are well above the mean and median of the dataset for these features.]
# It looks like a grocery store or a supermarket, that's why he do not have so many Fresh like sample 1.
# * Sample 3 - All the products are well above the mean and median of the dataset for these features.
# It looks like a kind of supermarket, which sells everything and can serve some snacks inside of it.
 
#%% [markdown]
# ### Implementation: Feature Relevance
# One interesting thought to consider is if one (or more) of the six product categories is actually relevant for understanding customer purchasing. That is to say, is it possible to determine whether customers purchasing some amount of one category of products will necessarily purchase some proportional amount of another category of products? We can make this determination quite easily by training a supervised regression learner on a subset of the data with one feature removed, and then score how well that model can predict the removed feature.
# 
# In the code block below, you will need to implement the following:
#  - Assign `new_data` a copy of the data by removing a feature of your choice using the `DataFrame.drop` function.
#  - Use `sklearn.model_selection.train_test_split` to split the dataset into training and testing sets.
#    - Use the removed feature as your target label. Set a `test_size` of `0.25` and set a `random_state`.
#  - Import a decision tree regressor, set a `random_state`, and fit the learner to the training data.
#  - Report the prediction score of the testing set using the regressor's `score` function.

#%%
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
for col in data.keys():
    # Make a copy of the DataFrame, using the 'drop' function to drop the given feature
    new_data = data.drop(col, axis=1)

    # Split the data into training and testing sets using the given feature as the target
    X_train, X_test, y_train, y_test = train_test_split(new_data, data[col], test_size=.25, random_state=0)

    # Create a decision tree regressor and fit it to the training set
    regressor = DecisionTreeRegressor(random_state=0).fit(X_train, y_train)

    # Report the score of prediction using the testing set
    score = regressor.score(X_test, y_test)
    print('{}: {}'.format(col, score))

# Using 'Detergents_Paper'
print('-----')
print('Using Detergents_Paper...')
new_data = data.drop('Detergents_Paper', axis=1)

X_train, X_test, y_train, y_test = train_test_split(new_data, data['Detergents_Paper'], test_size=.25, random_state=0)

regressor = DecisionTreeRegressor(random_state=0).fit(X_train, y_train)

score = regressor.score(X_test, y_test)
print('Detergents_Paper: {}'.format(score))

#%% [markdown]
# ### Question 2
# 
# * Which feature did you attempt to predict? 
# * What was the reported prediction score? 
# * Is this feature necessary for identifying customers' spending habits?
# 
# **Hint:** The coefficient of determination, `R^2`, is scored between 0 and 1, with 1 being a perfect fit. A negative `R^2` implies the model fails to fit the data. If you get a low score for a particular feature, that lends us to beleive that that feature point is hard to predict using the other features, thereby making it an important feature to consider when considering relevance.
#%% [markdown]
# **Answer:** 
#
# I attempt to predict `Detergents_Paper` because this feature gives 0.73 for R². This feature can be a good choice
# to identify the customers' spending habits due to a high value to R², saying that others features correlate well with `Detergents_Paper`. 

#%% [markdown]
# ### Visualize Feature Distributions
# To get a better understanding of the dataset, we can construct a scatter matrix of each of the six product features present in the data. If you found that the feature you attempted to predict above is relevant for identifying a specific customer, then the scatter matrix below may not show any correlation between that feature and the others. Conversely, if you believe that feature is not relevant for identifying a specific customer, the scatter matrix might show a correlation between that feature and another feature in the data. Run the code block below to produce a scatter matrix.

#%%
# Produce a scatter matrix for each pair of features in the data
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

#%% [markdown]
# ### Question 3
# * Using the scatter matrix as a reference, discuss the distribution of the dataset, specifically talk about the normality, outliers, large number of data points near 0 among others. If you need to sepearate out some of the plots individually to further accentuate your point, you may do so as well.
# * Are there any pairs of features which exhibit some degree of correlation? 
# * Does this confirm or deny your suspicions about the relevance of the feature you attempted to predict? 
# * How is the data for those features distributed?
# 
# **Hint:** Is the data normally distributed? Where do most of the data points lie? You can use [corr()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html) to get the feature correlations and then visualize them using a [heatmap](http://seaborn.pydata.org/generated/seaborn.heatmap.html)(the data that would be fed into the heatmap would be the correlation values, for eg: `data.corr()`) to gain further insight.

#%%
sns.heatmap(data.corr(), annot=True);

#%%
sns.pairplot(data);

#%% [markdown]
# **Answer:** 
#
# While observing correlation between 'Detergents_Paper' and other features, we can see:
# * `Detergents_Paper` and `Grocery` have a very strong correlation looking through its .92 R² value and a linear correlation in the scatterplot.
# It is a good choice use `Grocery` in the model to predict `Detergents_Paper`.
# * `Detergents_Paper` and `Milk` also have a strong correlation but less than DP ~ Grocery, with a .66 value for R².
# * We can see another strong correlation in the heatmap when we look to `Milk` and `Grocery` with a .73 value for R².
#%% [markdown]
# ## Data Preprocessing
# In this section, you will preprocess the data to create a better representation of customers by performing a scaling on the data and detecting (and optionally removing) outliers. Preprocessing data is often times a critical step in assuring that results you obtain from your analysis are significant and meaningful.
#%% [markdown]
# ### Implementation: Feature Scaling
# If data is not normally distributed, especially if the mean and median vary significantly (indicating a large skew), it is most [often appropriate](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics) to apply a non-linear scaling — particularly for financial data. One way to achieve this scaling is by using a [Box-Cox test](http://scipy.github.io/devdocs/generated/scipy.stats.boxcox.html), which calculates the best power transformation of the data that reduces skewness. A simpler approach which can work in most cases would be applying the natural logarithm.
# 
# In the code block below, you will need to implement the following:
#  - Assign a copy of the data to `log_data` after applying logarithmic scaling. Use the `np.log` function for this.
#  - Assign a copy of the sample data to `log_samples` after applying logarithmic scaling. Again, use `np.log`.

#%%
# Scale the data using the natural logarithm
log_data = np.log(data)

# Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
pd.plotting.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

#%% [markdown]
# ### Observation
# After applying a natural logarithm scaling to the data, the distribution of each feature should appear much more normal. For any pairs of features you may have identified earlier as being correlated, observe here whether that correlation is still present (and whether it is now stronger or weaker than before).
# 
# Run the code below to see how the sample data has changed after having the natural logarithm applied to it.

#%%
# Display the log-transformed sample data
display(log_samples)

#%% [markdown]
# ### Implementation: Outlier Detection
# Detecting outliers in the data is extremely important in the data preprocessing step of any analysis. The presence of outliers can often skew results which take into consideration these data points. There are many "rules of thumb" for what constitutes an outlier in a dataset. Here, we will use [Tukey's Method for identfying outliers](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/): An *outlier step* is calculated as 1.5 times the interquartile range (IQR). A data point with a feature that is beyond an outlier step outside of the IQR for that feature is considered abnormal.
# 
# In the code block below, you will need to implement the following:
#  - Assign the value of the 25th percentile for the given feature to `Q1`. Use `np.percentile` for this.
#  - Assign the value of the 75th percentile for the given feature to `Q3`. Again, use `np.percentile`.
#  - Assign the calculation of an outlier step for the given feature to `step`.
#  - Optionally remove data points from the dataset by adding indices to the `outliers` list.
# 
# **NOTE:** If you choose to remove any outliers, ensure that the sample data does not contain any of these points!  
# Once you have performed this implementation, the dataset will be stored in the variable `good_data`.

#%%
# Select the indices for data points you wish to remove
outliers  = []

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    
    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3 - Q1) * 1.5
    print('Step:', step)
    
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    feature_outliers = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    display(feature_outliers)

    outliers += feature_outliers.index.tolist()
    
# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
print('Number of outliers (inc duplicates):', len(outliers))
print('New dataset with removed outliers has {} samples with {} features each.'.format(*good_data.shape))

#%% [markdown]
# ### Question 4
# * Are there any data points considered outliers for more than one feature based on the definition above? 
# * Should these data points be removed from the dataset? 
# * If any data points were added to the `outliers` list to be removed, explain why.
# 
# ** Hint: ** If you have datapoints that are outliers in multiple categories think about why that may be and if they warrant removal. Also note how k-means is affected by outliers and whether or not this plays a factor in your analysis of whether or not to remove them.
#%% [markdown]
# **Answer:** 
#
# Several datapoints were outliers for more than one feature. You can see some points below:
# * `65` was an outlier for Fresh and Frozen.
# * `66` was an outlier for Delicatessen and Fresh.
# * `75` was an outlier for Detergents_Paper and Grocery.
# * `154` was an outlier for Milk, Delicatessen and Grocery.
#
# We see outliers because some datapoints fall well below the IQR, skewing data in some features. The above outliers are more anomalous, given that, they cause 
# skewing data in multiple features.
#
# Considering that they have different magnitude to the rest dataset, it seems reasonable removing these points from the overall data preventing skewing.
#
# Hence, all these data points were added to the `outliers` list.

#%% [markdown]
# ## Feature Transformation
# In this section you will use principal component analysis (PCA) to draw conclusions about the underlying structure of the wholesale customer data. Since using PCA on a dataset calculates the dimensions which best maximize variance, we will find which compound combinations of features best describe customers.
#%% [markdown]
# ### Implementation: PCA
# 
# Now that the data has been scaled to a more normal distribution and has had any necessary outliers removed, we can now apply PCA to the `good_data` to discover which dimensions about the data best maximize the variance of features involved. In addition to finding these dimensions, PCA will also report the *explained variance ratio* of each dimension — how much variance within the data is explained by that dimension alone. Note that a component (dimension) from PCA can be considered a new "feature" of the space, however it is a composition of the original features present in the data.
# 
# In the code block below, you will need to implement the following:
#  - Import `sklearn.decomposition.PCA` and assign the results of fitting PCA in six dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

#%%
from sklearn.decomposition import PCA
# Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA(n_components=6).fit(good_data)

# Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)

#%% [markdown]
# ### Question 5
# 
# * How much variance in the data is explained* **in total** *by the first and second principal component? 
# * How much variance in the data is explained by the first four principal components? 
# * Using the visualization provided above, talk about each dimension and the cumulative variance explained by each, stressing upon which features are well represented by each dimension(both in terms of positive and negative variance explained). Discuss what the first four dimensions best represent in terms of customer spending.
# 
# **Hint:** A positive increase in a specific dimension corresponds with an *increase* of the *positive-weighted* features and a *decrease* of the *negative-weighted* features. The rate of increase or decrease is based on the individual feature weights.
#%% [markdown]
# **Answer:**
# * The **first principal component** (49.93%) and the **second principal component** (22.59%) explain, **in total**, 72.52% of variance
# * The first four principal components explain 92.79% of variance.
# * **1st PC** represents a wide variety of the featureset. Most prominently it represents Detergents_Paper, but also provides information Gain for Milk, Grocery and Delicatassen to some extent. However, it badly predicts Fresh and Frozen categories and needs another component to help. This could represent the 'convenience' or 'supermarket' spending category.
# * **2nd PC** allows for the recovery of Information Gain for Fresh and Frozen features, and supplements Delicatessen. It provides small gains for Milk and Grocery, and a very small loss of Detergents_Paper. This could represent customers who are in the hospitality or restaurant industry.
# * **3rd PC** represents gains for Fresh and Detergents_Paper, and minimal or losses for other categories. This could represent smaller corner shops, with convenience items and small amounts of groceries.
# * **4th PC** represents Frozen and Detergents_Paper, and losses for other categories. This could represent bulk buyers of frozen goods, such as fish importers.
#%% [markdown]
# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it in six dimensions. Observe the numerical value for the first four dimensions of the sample points. Consider if this is consistent with your initial interpretation of the sample points.

#%%
# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))

#%% [markdown]
# ### Implementation: Dimensionality Reduction
# When using principal component analysis, one of the main goals is to reduce the dimensionality of the data — in effect, reducing the complexity of the problem. Dimensionality reduction comes at a cost: Fewer dimensions used implies less of the total variance in the data is being explained. Because of this, the *cumulative explained variance ratio* is extremely important for knowing how many dimensions are necessary for the problem. Additionally, if a signifiant amount of variance is explained by only two or three dimensions, the reduced data can be visualized afterwards.
# 
# In the code block below, you will need to implement the following:
#  - Assign the results of fitting PCA in two dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `good_data` using `pca.transform`, and assign the results to `reduced_data`.
#  - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

#%%
# Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2).fit(good_data)

# Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
vs.pca_results(good_data, pca)

#%% [markdown#]
# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it using only two dimensions. Observe how the values for the first two dimensions remains unchanged when compared to a PCA transformation in six dimensions.

#%%
# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))

#%% [markdown]
# ## Visualizing a Biplot
# A biplot is a scatterplot where each data point is represented by its scores along the principal components. The axes are the principal components (in this case `Dimension 1` and `Dimension 2`). In addition, the biplot shows the projection of the original features along the components. A biplot can help us interpret the reduced dimensions of the data, and discover relationships between the principal components and original features.
# 
# Run the code cell below to produce a biplot of the reduced-dimension data.

#%%
# Create a biplot
vs.biplot(good_data, reduced_data, pca)

#%% [markdown]
# ### Observation
# 
# Once we have the original feature projections (in red), it is easier to interpret the relative position of each data point in the scatterplot. For instance, a point the lower right corner of the figure will likely correspond to a customer that spends a lot on `'Milk'`, `'Grocery'` and `'Detergents_Paper'`, but not so much on the other product categories. 
# 
# From the biplot, which of the original features are most strongly correlated with the first component? What about those that are associated with the second component? Do these observations agree with the pca_results plot you obtained earlier?
#%% [markdown]
# ## Clustering
# 
# In this section, you will choose to use either a K-Means clustering algorithm or a Gaussian Mixture Model clustering algorithm to identify the various customer segments hidden in the data. You will then recover specific data points from the clusters to understand their significance by transforming them back into their original dimension and scale. 
#%% [markdown]
# ### Question 6
# 
# * What are the advantages to using a K-Means clustering algorithm? 
# * What are the advantages to using a Gaussian Mixture Model clustering algorithm? 
# * Given your observations about the wholesale customer data so far, which of the two algorithms will you use and why?
# 
# ** Hint: ** Think about the differences between hard clustering and soft clustering and which would be appropriate for our dataset.
#%% [markdown]
# **Answer:**
# * **K_Means:** has fewer parameters, each cluster has traditional assignments of Z and $\mu$. As a result, it is much faster and is well suited towards
# situations with lots of data, and where clusters are clearly separable and non-uniform.
# * **Gaussian Mixture Model:** has many more parameters Z, $\mu$, $pi$ and $\sigma$, and it is a method of `soft clustering`. By using Gaussian distributions
# and probabilities, data points do not necessarilyhave to be assigned ridigly, and ones with lower probability could be assigned to multiple clusters at once.
# It is able to assign non-spherical clusters.
#
# Given the above scatterplot and looking to the log distributions, the data appears to be quite uniform. A lot of data points don't clearly 
# belong to one particular cluster or another, so it seems more logical to use a Gaussian Mixture Model here.

#%% [markdown]
# ### Implementation: Creating Clusters
# Depending on the problem, the number of clusters that you expect to be in the data may already be known. When the number of clusters is not known *a priori*, there is no guarantee that a given number of clusters best segments the data, since it is unclear what structure exists in the data — if any. However, we can quantify the "goodness" of a clustering by calculating each data point's *silhouette coefficient*. The [silhouette coefficient](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) for a data point measures how similar it is to its assigned cluster from -1 (dissimilar) to 1 (similar). Calculating the *mean* silhouette coefficient provides for a simple scoring method of a given clustering.
# 
# In the code block below, you will need to implement the following:
#  - Fit a clustering algorithm to the `reduced_data` and assign it to `clusterer`.
#  - Predict the cluster for each data point in `reduced_data` using `clusterer.predict` and assign them to `preds`.
#  - Find the cluster centers using the algorithm's respective attribute and assign them to `centers`.
#  - Predict the cluster for each sample data point in `pca_samples` and assign them `sample_preds`.
#  - Import `sklearn.metrics.silhouette_score` and calculate the silhouette score of `reduced_data` against `preds`.
#    - Assign the silhouette score to `score` and print the result.

#%%
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def clusterGMM(k):
    global clusterer, preds, centers, sample_preds
    # Apply your GMM algorithm to the reduced data 
    clusterer = GaussianMixture(k, random_state=0).fit(reduced_data)

    # Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)

    # Find the cluster centers
    centers = clusterer.means_

    # Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds)
    return score

results = pd.DataFrame(columns=['Silhouette Score'])
results.columns.name = 'Number of Clusters'

for k in range(2,16):
    score = clusterGMM(k)
    results = results.append(pd.DataFrame([score], columns=['Silhouette Score'], index=[k]))
results

#%% [markdown]
# ### Question 7
# 
# * Report the silhouette score for several cluster numbers you tried. 
# * Of these, which number of clusters has the best silhouette score?
#%% [markdown]
# **Answer:** 
#
# Overall, a GMM with two clusters has the best silhouette score.
#%% [markdown]
# ### Cluster Visualization
# Once you've chosen the optimal number of clusters for your clustering algorithm using the scoring metric above, you can now visualize the results by executing the code block below. Note that, for experimentation purposes, you are welcome to adjust the number of clusters for your clustering algorithm to see various visualizations. The final visualization provided should, however, correspond with the optimal number of clusters. 

#%%
# Display the results of the clustering from implementation for two clusters
clusterGMM(2)
vs.cluster_results(reduced_data, preds, centers, pca_samples)

#%% [markdown]
# ### Implementation: Data Recovery
# Each cluster present in the visualization above has a central point. These centers (or means) are not specifically data points from the data, but rather the *averages* of all the data points predicted in the respective clusters. For the problem of creating customer segments, a cluster's center point corresponds to *the average customer of that segment*. Since the data is currently reduced in dimension and scaled by a logarithm, we can recover the representative customer spending from these data points by applying the inverse transformations.
# 
# In the code block below, you will need to implement the following:
#  - Apply the inverse transform to `centers` using `pca.inverse_transform` and assign the new centers to `log_centers`.
#  - Apply the inverse function of `np.log` to `log_centers` using `np.exp` and assign the true centers to `true_centers`.
# 

#%%
# Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)

#%% [markdown]
# ### Question 8
# 
# * Consider the total purchase cost of each product category for the representative data points above, and reference the statistical description of the dataset at the beginning of this project(specifically looking at the mean values for the various feature points). What set of establishments could each of the customer segments represent?
# 
# **Hint:** A customer who is assigned to `'Cluster X'` should best identify with the establishments represented by the feature set of `'Segment X'`. Think about what each segment represents in terms their values for the feature points chosen. Reference these values with the mean values to get some perspective into what kind of establishment they represent.

#%%
sns.heatmap((true_centers - data.mean())/data.std(ddof=0), square=True, annot=True, cbar=False);

#%%
for i in range(2):
    plt.figure()
    plt.axes().set_title('Segment ' + str(i))
    sns.barplot(true_centers.iloc[i], true_centers.columns.values);

#%% [markdown]
# **Answer:** 
# * **Cluster 0:** This most likely to represent cafes or restaurants serving fresh food due to a high value for Fresh category.
# * **Cluster 1:** Milk and Grocery are predominant here comparing to the other categories. It could be bulk distributors or large resellers such as supermarkets.

#%% [markdown]
# ### Question 9
# 
# * For each sample point, which customer segment from* **Question 8** *best represents it? 
# * Are the predictions for each sample point consistent with this?*
# 
# Run the code block below to find which cluster each sample point is predicted to be.

#%%
# Display the predictions
for i, pred in enumerate(sample_preds):
    print("Sample point", i, "predicted to be in Cluster", pred)

#%% [markdown]
# **Answer:**
# * **Index 3 (0)**
#   * **Previous Assessment:** Restaurant
#   * **Model Assessment:** Restaurants or Cafes
#   * **Comments:** The model agreeded with the original predictions. I thought the predominance of Frozen and Fresh as a 
# characteristic of restaurants and the model seems to suggest this as well.
# 
# * **Index 10 (1)**
#   * **Previous Assessment:** Grocery Shop or Supermarket
#   * **Model Assessment:** Bulk Distributor or Supermarket
#   * **Comments:** The model agreeded with the original predictions. I thought the predominance of Grocery as a characteristic of supermarkets and the
# model seems to suggest this as well.
#
# * **Index 100 (2)**
#   * **Previous Assessment:** Supermarket
#   * **Model Assessment:** Bulk Distributor or Supermarket
#   * **Comments:** The model agreeded with the original predictions. I thought the predominance of all features as a characteristic of supermarkets
#  with small cafes inside and model seems to suggest this as well.
#%% [markdown]
# ## Conclusion
#%% [markdown]
# In this final section, you will investigate ways that you can make use of the clustered data. First, you will consider how the different groups of customers, the ***customer segments***, may be affected differently by a specific delivery scheme. Next, you will consider how giving a label to each customer (which *segment* that customer belongs to) can provide for additional features about the customer data. Finally, you will compare the ***customer segments*** to a hidden variable present in the data, to see whether the clustering identified certain relationships.
#%% [markdown]
# ### Question 10
# Companies will often run [A/B tests](https://en.wikipedia.org/wiki/A/B_testing) when making small changes to their products or services to determine whether making that change will affect its customers positively or negatively. The wholesale distributor is considering changing its delivery service from currently 5 days a week to 3 days a week. However, the distributor will only make this change in delivery service for customers that react positively. 
# 
# * How can the wholesale distributor use the customer segments to determine which customers, if any, would react positively to the change in delivery service?*
# 
# **Hint:** Can we assume the change affects all customers equally? How can we determine which group of customers it affects the most?
#%% [markdown]
# **Answer:**
# 
# The model has established two main customer types - Cluster 0 `Restaurants/Cafes` that stocks fresh food and Cluster 1 `Supermarket/Bulk Distributors`
# that stocks lots of different items. We can imagine two scenarios:
# * It is likely that customers from Cluster 0, that serves lots of fresh food, are going to want 5-day weeks in order to keep food as fresh as possible.
# * Cluster 1 could be more flexible - they buy a more wide variety of perishable and non-perishable goods, so they do not necessarily need a daily delivery.
# So, the Company could run `A/B Tests` and generalize. By picking a subset customers from each cluster, they can evaluate feedback separately. It could be 
# established whether changing the delivery service can be critical to each segment, and whether customers are happy with the change.
#
# If a trend is found in a particular cluster, it allows a business to make targeted decisions to benefit customers depending on their profile.

#%% [markdown]
# ### Question 11
# Additional structure is derived from originally unlabeled data when using clustering techniques. Since each customer has a ***customer segment*** it best identifies with (depending on the clustering algorithm applied), we can consider *'customer segment'* as an **engineered feature** for the data. Assume the wholesale distributor recently acquired ten new customers and each provided estimates for anticipated annual spending of each product category. Knowing these estimates, the wholesale distributor wants to classify each new customer to a ***customer segment*** to determine the most appropriate delivery service.  
# * How can the wholesale distributor label the new customers using only their estimated product spending and the **customer segment** data?
# 
# **Hint:** A supervised learner could be used to train on the original customers. What would be the target variable?
#%% [markdown]
# **Answer:**
#
# We can use semi-supervised learning approach to classify new customers:
# * By the first running an unsupervised clustering technique, such as Gaussian Mixture Model, we first establish clusters and use this as a new feature - 
# which cluster they are in. We label it as `Customer Segment` and they could be assigned arbitrary enumerating values.
# * Then, we create new data points for each new customer, with all of their spending estimates. After that, we can use a supervised learnin technique,
# for example a Support Vector Machine (which does very well to separate classified clusters) with a target variable of `Customer Segment`.
# * Standard Supervised Learning optimizations could be used to tune the model: *boosting*, *cross-validation* etc.
#%% [markdown]
# ### Visualizing Underlying Distributions
# 
# At the beginning of this project, it was discussed that the `'Channel'` and `'Region'` features would be excluded from the dataset so that the customer product categories were emphasized in the analysis. By reintroducing the `'Channel'` feature to the dataset, an interesting structure emerges when considering the same PCA dimensionality reduction applied earlier to the original dataset.
# 
# Run the code block below to see how each data point is labeled either `'HoReCa'` (Hotel/Restaurant/Cafe) or `'Retail'` the reduced space. In addition, you will find the sample points are circled in the plot, which will identify their labeling.

#%%
# Display the clustering results based on 'Channel' data
vs.channel_results(reduced_data, outliers, pca_samples)

#%% [markdown]
# ### Question 12
# 
# * How well does the clustering algorithm and number of clusters you've chosen compare to this underlying distribution of Hotel/Restaurant/Cafe customers to Retailer customers? 
# * Are there customer segments that would be classified as purely 'Retailers' or 'Hotels/Restaurants/Cafes' by this distribution? 
# * Would you consider these classifications as consistent with your previous definition of the customer segments?
#%% [markdown]
# **Answer:**
#%% [markdown]
# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.