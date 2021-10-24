#!/usr/bin/env python
# coding: utf-8

# (file-types:notebooks)=
# 
# 
# # Introduction
# 

# library(tidyverse)
# library(skimr)
# library(GGally)
# library(ggmap)
# library(tidymodels)
# library(visdat)
# library(corrr)
# library(ggsignif)
# library(gt)
# 
# theme_set(theme_classic())
# 

# In this chapter, we'll build the following models:

# * lasso, 
# * natural spline, 
# * random forest,
# * XGBoost (extreme gradient boosted trees)
# * K-nearest neighbor

# *Lasso* performs a so called L1 regularization (a process of introducing additional information in order to prevent overfitting). In particular, it adds a penalty equivalent to the absolute value of the magnitude of coefficients. See @James2000 for more details about lasso regression.

# A *natural spline* is an advancement of a piecewise polynomial regression spline which involves fitting separate low-degree polynomials over different regions of our predictor space X. In particular, a natural spline is a regression spline with additional boundary constraints: the function is required to be linear at the boundary (in the region where X is smaller than the smallest knot, or larger than the largest knot). This additional constraint means that natural splines generally produce more stable estimates at the boundaries. See @James2000 for more details about piecewise polynomial regression splines and natural splines.

# # Business understanding
# 
# :::note
# In business understanding, you:
# 
# - Define your (business) goal
# - Frame the problem (regression, classification,...)
# - Choose a performance measure
# - Show the data processing components
# :::
# 
# First of all, we take a look at the big picture and define the objective of our data science project in business terms.
# 
# In our example, the goal is to build a model of housing prices in California. In particular, the model should learn from California census data and be able to predict the median house price in any district (population of 600 to 3000 people), given some predictor variables. Hence, we face a **supervised learning** situation and should use a **regression model** to predict the numerical outcomes. Furthermore, we use the **root mean square error (RMSE)** as a performance measure for our regression problem.
# 
# Let's assume that the model’s output (a prediction of a district’s median housing price) will be fed to another analytics system, along with other data. This downstream system will determine whether it is worth investing in a given area or not. The **data processing components** (also called data pipeline) are shown in \@ref(fig:datapipeline) (you can use [Google's architectural templates](https://docs.google.com/presentation/d/1vjm5YdmOH5LrubFhHf1vlqW2O9Z2UqdWA8biN3e8K5U/edit#slide=id.g19b41f69d7_2_265) to draw the data pipeline).

# 
# knitr::include_graphics("css/data-pipeline.png")
# 

# # Data understanding
# 
# :::note
# In data understanding, you:
# 
# - Import data 
# - Clean data
# - Format data properly
# - Create new variables
# - Get an overview about the complete data
# - Split data into training and test set using stratified sampling
# - Discover and visualize the data to gain insights 
# :::
# 
# ## Imort Data
# 
# First of all, let's import the data:

# library(tidyverse)
# 
# LINK <- "https://raw.githubusercontent.com/kirenz/datasets/master/housing_unclean.csv"
# housing_df <- read_csv(LINK)
# 

# ## Clean data
# 
# To get a first impression of the data we take a look at the top 4 rows: 

# library(gt)
# 
# housing_df %>% 
#   slice_head(n = 4) %>% 
#   gt() # print output using gt
# 

# Notice the values in the first row of the variables `housing_median_age` and `median_house_value`. We need to remove the strings "years" and "$" with the function `str_remove_all` from the `stringr` package. Since there could be multiple wrong entries of the same type, we apply our corrections to all of the rows of the corresponding variable:

# library(stringr)
# 
# housing_df <- 
#   housing_df %>% 
#   mutate(
#     housing_median_age = str_remove_all(housing_median_age, "[years]"),
#     median_house_value = str_remove_all(median_house_value, "[$]")
#   )
# 

# We don't cover the phase of data cleaning in detail in this tutorial. However, in a real data science project data cleaning is usually a very time consuming process.
# 
# ## Format data
# 
# Next, we take a look at the data structure and check wether all data formats are correct:
# 
# * Numeric variables should be formatted as integers (`int`) or double precision floating point numbers (`dbl`).
# 
# * Categorical (nominal and ordinal) variables should usually be formatted as factors (`fct`) and not characters (`chr`). Especially, if they don't have many levels.

# glimpse(housing_df)
# 

# The package `visdat` helps us to explore the data class structure visually:

# library(visdat)
# 
# vis_dat(housing_df)
# 

# We can observe that the numeric variables `housing_media_age` and `median_house_value` are declared as characters (`chr`) instead of numeric. We choose to format the variables as `dbl`, since the values could be floating-point numbers. 
# 
# Furthermore, the categorical variable `ocean_proximity` is formatted as character instead of factor. Let's take a look at the levels of the variable: 

# housing_df %>% 
#   count(ocean_proximity,
#         sort = TRUE)
# 

# The variable has only 5 levels and therefore should be formatted as a factor.
# 
# Note that it is usually a good idea to first take care of the numerical variables. Afterwards, we can easily convert all remaining character variables to factors using the function `across` from the dplyr package (which is part of the tidyverse).

# # convert to numeric
# housing_df <- 
#   housing_df %>% 
#   mutate(
#     housing_median_age = as.numeric(housing_median_age),
#     median_house_value = as.numeric(median_house_value)
#   )
# 
# # convert all remaining character variables to factors 
# housing_df <- 
#   housing_df %>% 
#   mutate(across(where(is.character), as.factor))
# 

# ## Missing data
# 
# Now let's turn our attention to missing data. Missing data can be viewed with the function `vis_miss` from the package `visdat`. We arrange the data by columns with most missingness: 

# vis_miss(housing_df, sort_miss = TRUE)
# 

# Here an alternative method to obtain missing data: 

# is.na(housing_df) %>% colSums()
# 

# We have a missing rate of 0.1% (207 cases) in our variable `total_bedroms`. This can cause problems for some algorithms. We will take care of this issue during our data preparation phase. 

# ## Create new variables 
# 
# One very important thing you may want to do at the beginning of your data science project is to create new variable combinations. For example:
# 
# * the *total number of rooms* in a district is not very useful if you don’t know how many households there are. What you really want is the *number of rooms per household*. 
# 
# * Similarly, the total number of bedrooms by itself is not very useful: you probably want to compare it to the number of rooms. 
# 
# * And the *population per household* also seems like an interesting attribute combination to look at. 
# 
# Let’s create these new attributes:

# 
# housing_df <- 
#   housing_df %>% 
#   mutate(rooms_per_household = total_rooms/households,
#         bedrooms_per_room = total_bedrooms/total_rooms,
#         population_per_household = population/households)
# 

# ## Data overview  
# 
# After we took care of our data problems, we can obtain a data summary of all numerical and categorical attributes using a function from the package `skimr`:

# 
# skim(housing_df)
# 

# We have `r nrow(housing_df)` observations and `r ncol(housing_df)` columns in our data.
# 
# * The `sd` column shows the standard deviation, which measures how dispersed the values are. 
# 
# * The p0, p25, p50, p75 and p100 columns show the corresponding percentiles: a percentile indicates the value below which a given percentage of observations in a group of observations fall. For example, 25% of the districts have a `housing_median_age` lower than 18, while 50% are lower than 29 and 75% are lower than 37. These are often called the 25th percentile (or first quartile), the median, and the 75th percentile.
# 
# * Further note that the **median income** attribute does not look like it is expressed in US dollars (USD). Actually the data has been scaled and capped at 15 (actually, 15.0001) for higher median incomes, and at 0.5 (actually, 0.4999) for lower median incomes. The numbers represent roughly tens of thousands of dollars (e.g., 3 actually means about $30,000).

# Another quick way to get an overview of the type of data you are dealing with is to plot a histogram for each numerical attribute. A histogram shows the number of instances (on the vertical axis) that have a given value range (on the horizontal axis). You can either plot this one attribute at a time, or you can use `ggscatmat` from the package `GGally` on the whole dataset (as shown in the following code example), and it will plot a histogram for each numerical attribute as well as  correlation coefficients (Pearson is the default). We just select the most promising variabels for our plot:
# 

# library(GGally)
# 
# housing_df %>% 
#   select(
#     median_house_value, housing_median_age, 
#     median_income, bedrooms_per_room, rooms_per_household, 
#     population_per_household) %>% 
#   ggscatmat(alpha = 0.2)
# 

# Another option is to use `ggpairs`, where we even can integrate our categorical variable `ocean_proximity` in the output:

# library(GGally)
# 
# housing_df %>% 
#   select(
#     median_house_value, housing_median_age, 
#     median_income, bedrooms_per_room, rooms_per_household, 
#     population_per_household,
#     ocean_proximity) %>% 
#   ggpairs()
# 

# There are a few things you might notice in these histograms:
# 
# * The variables *median income*, *housing median age* and the *median house value* were capped. The latter may be a serious problem since it is our target attribute (your y label). Our Machine Learning algorithms may learn that prices never go beyond that limit. This will be a serious problem if we need predictions beyond 500,000. We take care of this issue in our data preparation phase and only use districts below 500,000.
# 
# * Note that our attributes have very different scales. We will take care of this issue later in data preparation, when we use feature scaling (data normalization).
# 
# * Finally, many histograms are tail-heavy: they extend much farther to the right of the median than to the left. This may make it a bit harder for some Machine Learning algorithms to detect patterns. We will transform these attributes later on to have more bell-shaped distributions. For our right-skewed data (i.e., tail is on the right, also called positive skew), common transformations include square root and log (we will use the log). 

# ## Data splitting
# 
# Before we get started with our in-depth data exploration, let’s split our single dataset into two: a training set and a testing set. The training data will be used to fit models, and the testing set will be used to measure model performance. We perform data exploration only on the training data.
# 
# A **training dataset** is a dataset of examples used during the learning process and is used to fit the models. A **test dataset** is a dataset that is independent of the training dataset and is used to evaluate the performance of the final model. If a model fit to the training dataset also fits the test dataset well, minimal *overfitting* has taken place. A better fitting of the training dataset as opposed to the test dataset usually points to overfitting.
# 
# In our data split, we want to ensure that the training and test set is representative of the various categories of median house values in the whole dataset. Take a look at \@ref(fig:hist-med-value)
# 

# 
# housing_df %>% 
#   ggplot(aes(median_house_value)) +
#   geom_histogram(bins = 4) 
# 

# In general, we would like to have instances for each *stratum*, or else the estimate of a stratum's importance may be biased. A *stratum* (plural strata) refers to a subset (part) of the whole data from which is being sampled. You should not have too many strata, and each stratum should be large enough. We use 4 strata in our example. 
# 
# To actually split the data, we can use the `rsample` package (included in `tidymodels`) to create an object that contains the information on how to split the data (which we call `data_split`), and then two more `rsample` functions to create data frames for the training and testing sets:

# 
# # Fix the random numbers by setting the seed 
# # This enables the analysis to be reproducible 
# set.seed(123)
# 
# # Put 3/4 of the data into the training set 
# data_split <- initial_split(housing_df, 
#                            prop = 3/4, 
#                            strata = median_house_value, 
#                            breaks = 4)
# 
# # Create dataframes for the two sets:
# train_data <- training(data_split) 
# test_data <- testing(data_split)
# 

# ## Data exploration
# 
# The point of data exploration is to gain insights that will help you select important variables for your model and to get ideas for feature engineering in the data preparation phase. Ususally, data exploration is an iterative process: once you get a prototype model up and running, you can analyze its output to gain more insights and come back to this exploration step. It is important to note that we perform data exploration only with our training data.

# ### Create data copy 
# 
# We first make a copy of the training data since we don't want to alter our data during data exploration. 

# In[1]:



#data_explore <- train_data


# Next, we take a closer look at the relationships between our variables. In particular, we are interested in the relationships between our *dependent* variable `median_house_value` and all other variables. The goal is to identify possible *predictor variables* which we could use in our models to predict the `median_house_value`. 

# ### Geographical overview
# 
# Since our data includes information about `longitude` and `latitude`, we start our data exploration with the creation of a geographical scatterplot of the data to get some first insights: 

# In[2]:



#data_explore %>% 
#  ggplot(aes(x = longitude, y = latitude)) +
#  geom_point(color = "cornflowerblue")


# A better visualization that highlights high-density areas (with parameter `alpha = 0.1` ):

# In[3]:



#data_explore %>% 
#  ggplot(aes(x = longitude, y = latitude)) +
#  geom_point(color = "cornflowerblue", alpha = 0.1) 
  


# Overview about California housing prices: 
# 
# - red is expensive, 
# - purple is cheap and 
# - larger circles indicate areas with a larger population.
# 

# In[4]:



#data_explore %>% 
#  ggplot(aes(x = longitude, y = latitude)) +
#  geom_point(aes(size = population, color = median_house_value), 
#             alpha = 0.4) +
#  scale_colour_gradientn(colours=rev(rainbow(4)))


# Lastly, we add a map to our data:

# In[5]:


#library(ggmap)

#qmplot(x = longitude, 
#       y = latitude, 
#       data = data_explore, 
#       geom = "point", 
#       color = median_house_value, 
#       size = population,
#       alpha = 0.4) +
#  scale_colour_gradientn(colours=rev(rainbow(4))) +
#  scale_alpha(guide = 'none') # don't show legend for alpha


# This image tells you that the housing prices are very much related to the location (e.g., close to the ocean) and to the population density. Hence our `ocean_proximity` variable may be a useful predictor of median housing prices, although in Northern California the housing prices in coastal districts are not too high, so it is not a simple rule.
# 
# ### Boxplots
# 
# We can use boxplots to check, if we actually find differences in the median house value for the different levels of the *categorical variable* `ocean_proximity`. Additionally, we use the package `ggsignif` to calculate the significance of the difference between two of our groups and add the annotation to the plot in a single line.   

# In[6]:


#library(ggsignif)

#data_explore %>% 
#  ggplot(aes(ocean_proximity, median_house_value)) +
#  geom_boxplot(fill="steelblue") +
#  xlab("Ocean proximity") +
#  ylab("Median house value") +
#  geom_signif(comparisons = list(c("<1H OCEAN", "INLAND")), # calculate significance
#               map_signif_level=TRUE) 
  


# We can observe a difference in the median house value for the different levels of our categorical variable (except between "NEAR BAY" and "NEAR OCEAN") why we should include this variable in our model. Furthermore, the difference between "<1H OCEAN" and "INLAND" is  statistically significant.

# ### Correlations
# 
# Now let's analyze our numerical variables: To obtain the correlations of our numerical data, we can use the function `vis_cor` from the `visdat` package. We use Spearman's correlation coefficient since this measure is more insensitive to outliers than Pearson's correlation coefficient:

# In[7]:


#library(visdat)#

#data_explore %>% 
 # select(where(is.numeric)) %>% # only select numerical data
 # vis_cor(cor_method = "spearman", na_action = "pairwise.complete.obs")


# Now we take a closer look at the correlation coefficients with the package `corrr`. 

# In[8]:


#library(corrr)

# calculate all correlations
#cor_res <- 
#  data_explore %>%
#  select(where(is.numeric)) %>% 
#  correlate(method = "spearman", use = "pairwise.complete.obs") 

# show correlations
#cor_res %>% 
#  select(term, median_house_value) %>% 
#  filter(!is.na(median_house_value)) %>% # focus on dependent variable 
#  arrange(median_house_value) %>% # sort values
#  fashion() # print tidy correlations
  


# Furthermore, the function `network_plot` outputs a nice network plot of our data in which 
# 
# * variables that are more highly correlated appear closer together and are joined by stronger paths. 
# * Paths are also colored by their sign (blue for positive and red for negative). 
# * The proximity of the points are determined using clustering
# 

# In[9]:



#data_explore %>%
#  select(where(is.numeric)) %>%  
#  correlate() %>% 
#  network_plot(min_cor = .15)


# Summary of our findings for the correlation analysis: 
# 
# * `median_income` has a strong positive correlation with `median_house_value`.
# 
# * the new `bedrooms_per_room` attribute is negatively correlated with the `median_house_value`. Apparently houses with a lower bedroom/room ratio tend to be more expensive. 
# 
# * `rooms_per_household` is also a bit more informative than the total number of rooms (`total_rooms`) in a district. Obviously the larger the houses, the more expensive they are (positive correlation).
# 
# * `population_per_household` is negatively correlated with our dependent variable.

# As a last step in our correlation analysis, we check the statistical significance of Spearman's rank correlations. In our example, we only obtain significant p-values:

# In[10]:



#cor.test(data_explore$median_house_value, 
#         data_explore$population_per_household, 
#         method = "spearman", 
#         exact=FALSE)$p.value

#cor.test(data_explore$median_house_value, 
#         data_explore$bedrooms_per_room, 
#         method = "spearman", 
#         exact=FALSE)$p.value

#cor.test(data_explore$median_house_value, 
#         data_explore$rooms_per_household, 
#         method = "spearman", 
#         exact=FALSE)$p.value

#cor.test(data_explore$median_house_value, 
#         data_explore$population_per_household, 
#         method = "spearman", 
#         exact=FALSE)$p.value


# Consequently we will use this four numerical variables as well as `ocean_proximity` as predictors in our model.
# 
# ### Visual inspections
# 
# Now let's analyze the choosen variables in more detail. The function `ggscatmat` from the package `GGally` creates a matrix with scatterplots, densities and correlations for numeric columns. In our code we choose an alpha level of 0.2 (for transparency).

# In[11]:



#data_explore %>% 
#  select(median_house_value, ocean_proximity, 
#         median_income, bedrooms_per_room, rooms_per_household, 
#         population_per_household) %>% 
#  ggscatmat(corMethod = "spearman",
#            alpha=0.2)


# We can also add a color column for our categorical variable `ocean_proximity` to get even more insights about the :

# In[12]:



#data_explore %>% 
#  select(median_house_value, ocean_proximity, 
#         median_income, bedrooms_per_room, rooms_per_household, 
#         population_per_household) %>% 
#  ggscatmat(color="ocean_proximity", # add a categorical variable
#            corMethod = "spearman",
#            alpha=0.2)


# We can observe that our ocean proximity variable is indeed a good predictor for our different median house values. Another promising attribute to predict the median house value is the median income, so let’s zoom in:

# In[13]:



#data_explore %>% 
#  ggplot(aes(median_income, median_house_value)) +
#  geom_jitter(color = "steelblue", alpha = 0.2) + 
#  xlab("Median income") +
#  ylab("Median house value") +
 # scale_y_continuous(labels = scales::dollar)


# This plot reveals a few things. First, the correlation is indeed very strong; you can clearly see the upward trend, and the points are not too dispersed. Second, the price cap that we noticed earlier is clearly visible as a horizontal line at 500,000 dollars. But this plot reveals other less obvious straight lines: a horizontal line around 450,000 dollars, another around 350,000 dollars, perhaps one around $280,000 dollars, and a few more below that. Hence, in our data preparation phase we will remove districts with 500,000 dollars to prevent our algorithms from learning to reproduce these data quirks.
# 
# # Data preparation

# :::note
# Data preparation:
# 
# - Handle missing values
# - Fix or remove outliers  
# - Feature selection
# - Feature engineering
# - Feature scaling
# - Create a validation set
# :::
# 
# Next, we’ll preprocess our data before training the models. We mainly use the tidymodels packages `recipes` and `workflows` for this steps. `Recipes` are built as a series of optional data preparation steps, such as:
# 
# * *Data cleaning*: Fix or remove outliers, fill in missing values (e.g., with zero, mean, median…) or drop their rows (or columns).
# 
# * *Feature selection*: Drop the attributes that provide no useful information for the task.
# 
# * *Feature engineering*: Discretize continuous features, decompose features (e.g., the weekday from a date variable, etc.), add promising transformations of features (e.g., log(x), sqrt(x), x2 , etc.) or aggregate features into promising new features (like we already did).
# 
# * *Feature scaling*: Standardize or normalize features.
# 
# We will want to use our recipe across several steps as we train and test our models. To simplify this process, we can use a *model workflow*, which pairs a model and recipe together. 
# 
# ## Data preparation
# 
# Before we create our `recipes`, we first select the variables which we will use in the model. We also remove specific cases with a price in median house value equal to or greater as 500000 dollars. Note that we keep `longitude` and `latitude` to be able to map the data in a later stage but we will not use the variables in our model. 

# In[14]:



#housing_df_new <-
#  housing_df %>% 
#  filter(median_house_value < 500000) %>% # only use houses with a value below 500000
#  select( # select our predictors
#    longitude, latitude, 
#    median_house_value, 
#    median_income, 
#    ocean_proximity, 
#    bedrooms_per_room, 
#    rooms_per_household, 
 #   population_per_household
 #        )

#glimpse(housing_df_new)


# Furthermore, we need to make a new data split since we updated the original data. 

# In[15]:


#set.seed(123)

#data_split <- initial_split(housing_df_new, # updated data
#                           prop = 3/4, 
#                           strata = median_house_value, 
#                           breaks = 4)

#train_data <- training(data_split) 
#test_data <- testing(data_split)


# ## Data prepropecessing recipe 
# 
# The type of data preprocessing is dependent on the data and the type of model being fit. The excellent book "Tidy Modeling with R" provides an [appendix with recommendations for baseline levels of preprocessing](https://www.tmwr.org/pre-proc-table.html) that are needed for various model functions (@Kuhn2021) 

# Let’s create a base `recipe` for all of our regression models (for one of the models, we need to add a new recipe step at a later stage). Note that the sequence of steps matter:
# 
# * The `recipe()` function has two arguments: 
#  * *A formula*. Any variable on the left-hand side of the tilde (`~`) is considered the model outcome (here, `median_house_value`). On the right-hand side of the tilde are the predictors. Variables may be listed by name (separated by a `+`), or you can use the dot (`.`) to indicate all other variables as predictors. 
#  * *The data*. A recipe is associated with the data set used to create the model. This will typically be the training set, so `data = train_data` here. 

# * `update_role()`: This step of adding roles to a recipe is optional; the purpose of using it here is that those two variables can be retained in the data but not included in the model. This can be convenient when, after the model is fit, we want to investigate some poorly predicted value. These ID columns will be available and can be used to try to understand what went wrong. 
# 
# * `step_naomit()` removes observations (rows of data) if they contain NA or NaN values. We use `skip = TRUE` because we don't want to perform this part to new data so that the number of samples in the assessment set is the same as the number of predicted values (even if they are NA).
# 
# * `step_novel()` converts all nominal variables to factors and takes care of other issues related to categorical variables.
# 
# * `step_log()` will log transform data (since some of our numerical variables are right-skewed). Note that this step can not be performed on negative numbers.
# 
# * `step_normalize()` normalizes (center and scales) the numeric variables to have a standard deviation of one and a mean of zero. (i.e., z-standardization). 
# 
# * `step_dummy()` converts our factor column `ocean_proximity` into numeric binary (0 and 1) variables.

# * `step_zv()`: removes any numeric variables that have zero variance.
# 
# * `step_corr()`: will remove predictor variables that have large correlations with other predictor variables.
# 

# In[16]:



#housing_rec <-
#  recipe(median_house_value ~ .,
#         data = train_data) %>%
#  update_role(longitude, latitude, 
#              new_role = "ID") %>% 
#  step_log(
#    median_house_value, median_income,
#    bedrooms_per_room, rooms_per_household, 
#    population_per_household
#    ) %>% 
#  step_naomit(everything(), skip = TRUE) %>% 
#  step_novel(all_nominal(), -all_outcomes()) %>%
#  step_normalize(all_numeric(), -all_outcomes(), 
#                 -longitude, -latitude) %>% 
#  step_dummy(all_nominal()) %>%
#  step_zv(all_numeric(), -all_outcomes()) %>%
#  step_corr(all_predictors(), threshold = 0.7, method = "spearman") 


# To view the current set of variables and roles, use the `summary()` function:

# In[17]:



#summary(housing_rec)


# If we would like to check if all of our preprocessing steps from above actually worked, we can  proceed as follows:

# In[18]:



#prepped_data <- 
#  housing_rec %>% # use the recipe object
#  prep() %>% # perform the recipe on training data
#  juice() # extract only the preprocessed dataframe 


# Take a look at the data structure:

# In[19]:



#glimpse(prepped_data)


# Visualize the data:

# In[20]:


#prepped_data %>% 
#  select(median_house_value, 
#         median_income, 
#         rooms_per_household, 
#         population_per_household) %>% 
#  ggscatmat(corMethod = "spearman",
#            alpha=0.2)


# You should notice that:
# 
# * the variables `longitude` and `latitude` did not change. 
# 
# * `median_income`, `rooms_per_household` and `population_per_household` are now z-standardized and the distributions are a bit less right skewed (due to our log transformation)
# 
# * `ocean_proximity` was replaced by dummy variables. 

# ## Validation set
# 
# Remember that we already partitioned our data set into a *training set* and *test set*. This lets us judge whether a given model will generalize well to new data. However, using only two partitions may be insufficient when doing many rounds of hyperparameter tuning (which we don't perform in this tutorial but it is always recommended to use a validation set).
# 
# Therefore, it is usually a good idea to create a so called `validation set`. Watch this short [video from Google's Machine Learning crash course](https://developers.google.com/machine-learning/crash-course/validation/video-lecture) to learn more about the value of a validation set.  
# 
# We use k-fold crossvalidation to build a set of 5 validation folds with the function `vfold_cv`. We also use stratified sampling:

# In[21]:



#set.seed(100)

#cv_folds <-
# vfold_cv(train_data, 
#          v = 5, # number of folds
#          strata = median_house_value,
#          breaks = 4) 


# We will come back to the *validation set* after we specified our models. 
# 
# # Model building
# 
# ## Specify models
# 
# The process of specifying our models is always as follows:
# 
# 1. Pick a `model type` 
# 2. set the `engine`
# 3. Set the `mode`: regression or classification
# 
# You can choose the `model type` and `engine` from this [list](https://www.tidymodels.org/find/parsnip/).
# 
# ### Lasso regression 

# In[22]:



#lasso_spec <- # your model specification
#  linear_reg(penalty = 0.1, mixture = 1) %>%  # model type and some options
#  set_engine(engine = "glmnet") %>%  # model engine
#  set_mode("regression") # model mode

# Show your model specification
#lasso_spec


# * `penalty`: The total amount of regularization in the model. Higher values imply a higher penalty. If you choose a penalty of 0 you fit a standard linear regression model. 
# 
# * `mixture`: The mixture amounts of different types of regularization. A number between zero and one (inclusive) that is the proportion of L1 regularization (i.e. lasso) in the model. When mixture = 1, it is a pure lasso model while mixture = 0 indicates that ridge regression is being used (this works only for engines "glmnet" and "spark").
# 
# Note that for the lasso regression to work properly it is very important to always add a data normalization step. 
# 
# ### Natural spline  

# In[ ]:





# To use this model correctly, we also need to add a data normalization step as well as a step to declare the degree of freedom in our model. We will include the degrees of freedom at a later step (when we create the workflows).
# 
# ### Random forest

# In[ ]:





# ### Boosted tree (XGBoost)

# In[ ]:





# ### K-nearest neighbor   

# In[ ]:





# ## Create workflows
# 
# To combine the data preparation recipe with the model building, we use the package [workflows](https://workflows.tidymodels.org). A workflow is an object that can bundle together your pre-processing recipe, modeling, and even post-processing requests (like calculating the RMSE). 
# 
# ### Lasso
# 
# Bundle recipe and model with `workflows`:

# In[ ]:





# ### Natural spline
# 
# We need to declare the degrees of freedom -with `step_ns()`- for our natural spline. In our example, we just add the new step to our `housing_rec` recipe and create a new recipe which we will only use for ourse natural spline.

# In[ ]:





# The higher the degree of freedom, the more complex the resulting model.
# 
# Now we bundle the recipe and our model:

# In[ ]:





# ### Random forest
# 
# Bundle recipe and model:
# 

# In[ ]:





# ### XGBoost
# 
# Bundle recipe and model:
# 

# In[ ]:





# ### K-nearest neighbor
# 
# Bundle recipe and model:

# In[ ]:





# ## Evaluate models
# 
# Now we can use our validation set (`cv_folds`) to estimate the performance of our models using the `fit_resamples()` function to fit the models on each of the folds and store the results. 
# 
# Note that `fit_resamples()` will fit our model to each resample and evaluate on the heldout set from each resample. The function is usually only used for computing performance metrics across some set of resamples to evaluate our models (like RMSE) - the models are not even stored. However, in our example we save the predictions in order to visualize the model fit and residuals with `control_resamples(save_pred = TRUE)`.
# 
# Finally, we collect the performance metrics with `collect_metrics()` and pick the model that does best on the validation set.
# 
# ### Lasso regression
# 

# In[ ]:





# Show average performance over all folds:

# In[ ]:





# Show performance for every single fold:

# In[ ]:





# To assess the model predictions, we plot the predictions on the y-axis and the real median house value at the x-axis. Note that the red line is not our model. If our model would have made no mistakes at all, all points would lie on the red diagonal line (where the prediction equals the real value). 

# In[ ]:





# Let`s look at the 10 districts where our model produced the greatest residuals: 

# In[ ]:





# Show the observations in the training data.

# In[ ]:





# In this tutorial, we don't further investigate the reasons for the wrong predictions. In reality, we would check wether some of the districts are outliers in comparision to the rest of our data and we would need to decide if we should drop some of the cases from the data (if there are good reasons to do so).    
# 
# ### Natural spline
# 
# We don't repeat all of the steps shown in lasso regression and just focus on the performance metrics.

# In[ ]:





# ### Random forest
# 
# We don't repeat all of the steps shown in lasso regression and just focus on the performance metrics.

# In[ ]:





# ### XGBoost
# 
# We don't repeat all of the steps shown in lasso regression and just focus on the performance metrics.

# In[ ]:





# ### K-nearest neighbor
# 
# We don't repeat all of the steps shown in lasso regression and just focus on the performance metrics.

# In[ ]:





# ### Compare models
# 
# Extract the RMSE from our models to compare them:

# In[ ]:





# Note that the model results are quite similar. 

# In[ ]:





# Now it's time to fit the best model (in our case the XGBoost model) one last time to the full *training set* and evaluate the resulting final model on the *test set*.
# 
# ## Last fit and evaluation on test set 
# 
# Tidymodels provides the function [`last_fit()`](https://tune.tidymodels.org/reference/last_fit.html) which fits a model to the *training data* and evaluates it on the *test set*. We just need to provide the workflow object of the best model as well as the **data split** object (not the training data).

# In[ ]:





# And this is our final result. Remember that if a model fit to the training dataset also fits the test dataset well, minimal *overfitting* has taken place. This seems to be also the case in our example.
