# Travel Package Purchase Prediction

A machine learning project that predicts customer likelihood to purchase travel packages, with a focus on identifying potential honeymoon package customers.

## Overview

This project uses a Random Forest classifier to predict whether customers will purchase travel packages based on their demographic information, travel history, and interaction data. The model includes feature engineering to identify honeymoon candidates and handles class imbalance using SMOTE (Synthetic Minority Oversampling Technique).

## Features

- **Predictive Modeling**: Random Forest classifier with balanced class weights
- **Data Preprocessing**: Handles missing values and categorical encoding
- **Feature Engineering**: Creates honeymoon candidate flags based on age and marital status
- **Class Imbalance Handling**: Uses SMOTE to balance the dataset
- **Customer Targeting**: Identifies top 10% of customers with highest purchase probability

## Dataset

The model expects a CSV file named `Travel.csv` with the following features:

### Numeric Features
- `Age`: Customer age
- `CityTier`: City tier classification
- `DurationOfPitch`: Duration of sales pitch
- `NumberOfPersonVisiting`: Number of people in travel group
- `NumberOfFollowups`: Number of follow-up contacts
- `PreferredPropertyStar`: Preferred hotel star rating
- `NumberOfTrips`: Historical number of trips
- `Passport`: Passport availability (0/1)
- `PitchSatisfactionScore`: Satisfaction score for sales pitch
- `OwnCar`: Car ownership status (0/1)
- `NumberOfChildrenVisiting`: Number of children in travel group
- `MonthlyIncome`: Customer monthly income

### Categorical Features
- `TypeofContact`: Contact method
- `Occupation`: Customer occupation
- `Gender`: Customer gender
- `ProductPitched`: Type of package pitched
- `MaritalStatus`: Marital status
- `Designation`: Job designation

### Target Variable
- `ProdTaken`: Whether customer purchased package (0/1)

## Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
uuid
```

Install dependencies:
```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

## Usage

1. **Prepare your data**: Ensure your dataset is named `Travel.csv` and contains all required columns

2. **Run the notebook**: Execute all cells in `DW travel.ipynb`

3. **Review results**: The model will output:
   - Classification report with precision, recall, and F1-scores
   - ROC-AUC score
   - Feature importance rankings
   - Target customer list saved to `target_honeymoon_customers.csv`

## Model Performance

The Random Forest classifier achieves:
- **Overall Accuracy**: ~90%
- **ROC-AUC Score**: 0.9360
- **Precision (Class 1)**: 0.82
- **Recall (Class 1)**: 0.58
- **F1-Score (Class 1)**: 0.68

## Key Features by Importance

1. **Passport**: Most important predictor
2. **Age**: Strong demographic indicator
3. **Duration of Pitch**: Sales interaction quality
4. **Monthly Income**: Economic capability
5. **Number of Followups**: Engagement level

## Feature Engineering

### Honeymoon Candidate Detection
The model creates an `IsHoneymoonCandidate` feature that flags customers who are:
- Aged 20-35 years
- Single, Unmarried, or Divorced

This helps identify potential honeymoon package buyers for targeted marketing.

## Output

The project generates `target_honeymoon_customers.csv` containing:
- `CustomerID`: Unique customer identifier
- `PurchaseProbability`: Predicted probability of purchase (0-1)

This file contains the top 10% of customers most likely to purchase, ideal for targeted marketing campaigns.

## Data Processing Pipeline

1. **Missing Value Handling**: Fills missing values with median for numeric features
2. **Preprocessing**: StandardScaler for numeric features, OneHotEncoder for categorical
3. **Class Balancing**: SMOTE oversampling to handle imbalanced target classes
4. **Model Training**: Random Forest with balanced class weights
5. **Evaluation**: Comprehensive metrics on test set
6. **Prediction**: Purchase probability scores for all customers

## File Structure

```
project/
├── DW travel.ipynb          # Main notebook
├── Travel.csv               # Input dataset (not included)
├── target_honeymoon_customers.csv  # Output predictions
└── README.md               # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Notes

- Ensure your dataset follows the expected schema
- The model is specifically tuned for travel package prediction
- Consider retraining the model periodically with new data
- The honeymoon candidate feature can be customized based on business requirements

## Future Enhancements

- Add cross-validation for more robust model evaluation
- Implement hyperparameter tuning
- Create visualization dashboard for model insights
- Add model persistence for production deployment
- Include time-series features for seasonal patterns
