#!/usr/bin/env python
# coding: utf-8

# (file-types:notebooks)=
# 
# (l-lifecycle)=
# # Data Science Lifecycle
# 
# In our data science projects, we follow the data science lifecycle process proposed in the "cross industry standard process for data mining (CRISP-DM)" from {cite:t}`Wirth2000`
# 

# ```{note}
# To learn more about this framework, review this [presentation about the CRISP-DM](https://docs.google.com/presentation/d/1Y_6d-yv0Wq9WQvWkYS64KYkcSoswewm-7t2jfSz3aT4/edit?usp=sharing). 
# ```

# ```{image} ../_static/lecture_specific/01_crisp_dm/CRISP_DM.png
# :alt: crispdm
# :class: bg-primary mb-1
# :width: 500px
# :align: center
# ```

# 
# 
# Next, we show the most crucial steps of the framework. 
# 
# ## Business understanding
# 
# 1. Define your (business) goal
# 1. Frame the problem (regression, classification,...)
# 1. Choose a performance measure (RMSE, ...)
# 1. Show the data processing components (data pipeline)
# 
# ## Data understanding
# 
# 1. Import data 
# 1. Clean data
# 1. Format data properly (numeric or categorical)
# 1. Create new variables
# 1. Overview about the complete data
# 1. Split data into training and test set using stratified sampling
# 1. Discover and visualize the data to gain insights (on a copy of the training data)

# ## Data preparation
# 
# 1. Perform feature selection (choose predictor variables)
# 1. Do feature engineering (mainly with `recipes`)
# 1. Create a validation set from the training data (e.g., with k-fold crossvalidation) 

# ## Modeling
# 
# 1. Specify the models
# 1. Bundle the data preprocessing recipe and model in a `workflow` 
# 1. Compare model performance on the validation set 
# 1. Pick the model that does best on the validation set
# 1. Train your best model with all of the training data
# 1. Double-check that model against the test set.
