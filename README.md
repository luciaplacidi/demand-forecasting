# Demand Forecasting: Predicting Units Sold

<!-- using RandomForestRegressor because we're predicting units sold and it is a numerical, not a class label, and not trying to classify the information to a specific category. -->

## Project Overview

This project builds a machine learning model to predict product-level sales volume ("units sold") using transactional data including pricing, store and product identifiers, and promotional activity.

## Executive Summary

The goal was to forecast the number of units sold for a given SKU at a specific store using historical pricing and promotional data. The model achieved high predictive performance, with an R² score of 0.96 on the test set. Visual analysis showed that the predictions closely aligned with actual values and the residuals were tightly distributed around zero, indicating good generalization.

## Data Structure
This data set is from Kaggle: [Demand Forecasting](https://www.kaggle.com/datasets/aswathrao/demand-forecasting)

Columns:
- `sku_id`: product ID
- `store_id`: store ID
- `total_price` and `base_price`: promotional vs. regular prices
- `is_featured_sku`, `is_display_sku`: binary flags indicating promotional activity
- `week`: week of transaction (converted to day, month, year)
- `units_sold`: target variable (number of units sold)

### Preprocessing
- Parsed `week` into `day`, `month`, and `year`.
- One-hot encoded `sku_id` and `store_id` using pd.get_dummies().
- Dropped unused fields such as `record_ID`.

## Modelling
- Used `RandomForestRegressor` from scikit-learn
- Built an initial baseline model
- Tuned hyperparameters using GridSearchCV across:
-   `n_estimators`
-   `max_depth`
-   `min_samples_split`
- Compared baseline and tuned model predictions

## Evaluation
R² Score
- Baseline model: ~0.82
- Tuned model: 0.96

This indicates that the tuned model explains 96% of the variance in weekly units sold — a strong result given the simplicity of the input data.

### Predicted vs Actual
A scatterplot of actual vs. predicted units sold for both the baseline and tuned models showed that:
- The baseline model predictions are more dispersed
- The tuned model predictions cluster tightly along the 45° line

<img src="https://github.com/luciaplacidi/demand-forecasting/blob/main/actual_vs_predicted.png" width=800/>

### Residual Distribution
Overlaid histograms showed that:
- The tuned model’s errors are more tightly centered around zero
- The baseline model had a wider error spread

<img src="https://github.com/luciaplacidi/demand-forecasting/blob/main/residual_distribution.png" width=800/>

## Next Steps
- Promotional flags (`is_featured_sku`, `is_display_sku`) and pricing differences likely helped the model learn sales trends
- The model can detect the relationship between historical pricing and volume without needing external product/store information

## Summary

This notebook demonstrates how to forecast sales using only structured transactional data. By tuning a Random Forest model and interpreting its output visually and statistically, we were able to achieve high predictive performance.

## Next Steps
To extend the project and make the model more accessible for real-world use, the following next steps could be taken:

### 1. Build a Streamlit App
Develop an interactive web interface to make the model accessible to non-technical users such as sales analysts or store managers. Features could include:
- Form inputs for product, store, price, and promotion details
- Real-time prediction output for units_sold

### 2. Add a Model Insights Dashboard
Integrate a separate tab or page in the Streamlit app with key model insights:
- Feature importance plot: Bar chart showing the most influential variables in the model
- Error plots: Visualizations like residual distributions and predicted vs. actual

### 3. Add Explainability
Incorporate SHAP values to explain how each input feature contributes to the prediction, especially useful when:
- Justifying forecasts to business stakeholders
- Identifying which factors most affect sales predictions

