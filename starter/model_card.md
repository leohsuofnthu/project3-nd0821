# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Developer: Ryan Hsu 
- Model date: 2021-12-26 
- Model Version: 1.0.0 
- Model type: DecisionTreeClassifier from scikit-learn 
- Parameters: max_depth=5 

## Intended Use
- Primary intended uses: Predict if income is over 50K or under 50K.
- Primary intended users: Economy or sociology researcher. 
- Out-of-scope use cases: Try to predict actual income or threshold that is not 50K. 

## Training Data
- Datasets: Census Income Data Set from UCI Machine Learning Repository 
- Motivation: Build a binary classification model to predict if a person's income is over 50K or not given several features. 
- Preprocessing: (can be checked in EDA.ipynb) \
&nbsp;&nbsp;- 80% of total samples \
&nbsp;&nbsp;- Drop NaN \
&nbsp;&nbsp;- Remove all extra space in string \
&nbsp;&nbsp;- All categorical column are encoded using OneHotEncoder from scikit-learn \
&nbsp;&nbsp;- Label column ('salary') is encoded using LabelBinarizer from scikit-learn 

## Evaluation Data
- Datasets: Census Income Data Set from UCI Machine Learning Repository 
- Motivation: Build a binary classification model to predict if a person's income is over 50K or not given several features. 
- Preprocessing: (can be checked in EDA.ipynb) \
&nbsp;&nbsp;- 20% of total samples \
&nbsp;&nbsp;- Drop NaN \
&nbsp;&nbsp;- Remove all extra space in string
&nbsp;&nbsp;- All categorical column are encoded using one hot encoding \
&nbsp;&nbsp;- All categorical column are encoded using OneHotEncoder from scikit-learn \
&nbsp;&nbsp;- Label column ('salary') is encoded using LabelBinarizer from scikit-learn 

## Metrics
- Model performance measures: \
&nbsp;&nbsp;- precision: 0.777 \
&nbsp;&nbsp;- recall: 0.565 \
&nbsp;&nbsp;- fbeta: 0.654 

## Ethical Considerations
- There are no sensitive infomation. 
- The data is not used  to inform decisions about matters central to human life or flourishing – e.g., health or safety. 

## Caveats and Recommendations 
- Accroding to data slicing analysis (can be checked in /screenshots/slice_output.txt), this model might be poor in performance: \
&nbsp;&nbsp;- native-country: Puerto-Rico, Cuba, China, Vietnam, Yugoslavia, Dominican-Republic, Honduras, Hongkong, Iran \
&nbsp;&nbsp;- marital-status: Separated \
&nbsp;&nbsp;- education: 5th-6th 