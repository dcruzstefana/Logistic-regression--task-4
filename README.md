# Logistic-regression--task-4
Logistic Regression
This project demonstrates how to build a binary classification model using logistic regression with Python. The objective is to classify whether a tumor is malignant or benign based on a set of diagnostic features. The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset, which includes 30 numeric features computed from digitized images of fine needle aspirate (FNA) of breast mass. These features describe characteristics such as radius, texture, perimeter, area, and smoothness of the cell nuclei.

The workflow begins by importing the dataset and performing initial data cleaning. Irrelevant columns such as id and unnamed columns are removed, and the target column diagnosis is mapped to binary values—1 representing malignant and 0 representing benign. After preprocessing, the dataset is split into training and testing subsets. Feature scaling is applied using standardization to ensure that all features contribute equally to the model’s performance.

A logistic regression model is then trained on the standardized data. Once trained, the model is evaluated using several performance metrics. These include the confusion matrix, classification report (which contains precision, recall, and F1-score), and the ROC-AUC score. The ROC curve is plotted to visualize the trade-off between the true positive rate and the false positive rate at various classification thresholds. In addition, a histogram of predicted probabilities is created to observe the distribution of the model's confidence in its predictions.

To better understand the effect of the decision threshold, the default threshold of 0.5 is adjusted to 0.3, allowing us to compare the resulting confusion matrix and observe the impact on recall and precision. A precision-recall curve is also plotted to further analyze model performance, particularly for imbalanced classes.

The logistic regression coefficients are visualized in a bar chart to highlight the most influential features in the classification decision. This helps in interpreting the model by identifying which features have the most significant impact on whether a tumor is predicted as malignant or benign.

Finally, the sigmoid function used in logistic regression is demonstrated by applying it to a logit (raw output) value from the model. This shows how the linear combination of features is converted into a probability, making logistic regression an interpretable and powerful tool for binary classification problems.
