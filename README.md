# Bank Customer Churn Prediction Project

## What is Bank Customer Churn Prediction?
Bank customer churn prediction is the analysis and modeling process of a bank to predict whether its customers will abandon its services in the future. "Churn" (customer churn) means that a customer stops using the bank's services or switches to another financial institution.

## Why is it important?

### 1. Preventing Customer Churn
Customer churn can cause serious revenue loss for banks. Gaining new customers is more costly than retaining existing customers. Therefore, it is important to identify customers with a high churn risk in advance and offer them special offers.

### 2. Better Marketing Strategies
Churn analysis helps marketing teams segment customers and create special campaigns. For example, loyalty can be increased by offering additional benefits to loyal customers.

### 3. Effective Use of Financial Resources
A bank can determine which customers it should invest in to prevent customer churn. In this way, the marketing and customer service budget can be used more efficiently.

### 4. Gaining Competitive Advantage
Banks can gain a stronger position in the market by reducing customer churn. Reducing the churn rate helps the bank increase customer satisfaction and stand out from its competitors.

### 5. Data-Driven Decision Making
With Churn forecasting, banks can optimize their operations by making data-based decisions instead of intuitive decisions.

### The following processes were followed in this project:

#### 1) Data Collection:
Data was taken from Kaggle.

#### 2) Data Preprocessing:
- Determination of categorical and numerical features
- Filling in missing data
- Feature engineering
- Digitization of categorical data (One-Hot Encoding etc.)
Undersampling and class-balanced loss (label weight) adjustment for unbalanced data set

#### 3) Model Selection and Training:

- XGBoost hyperparameter optimization
- LightGBM hyperparameter optimization
- CatBoost hyperparameter optimization
- Establishing an ensemble model

#### 4) Model Evaluation:
- Performance analysis with metrics such as Accuracy, Precision, Recall, F1-score
- "Recall" metric since classifying customers that will be lost as not to be lost is more costly than classifying the opposite

#### 5) API:
- Creating an API with Flask framework

#### 6) Real World Applications:
- Offering special campaigns to customers with high churn risk in line with model estimates
- Dealing with risky customers by special customer representatives
