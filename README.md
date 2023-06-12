# Mushroom-Classification-Adaboost
 Aimed at differentiating between poisonous and edible mushrooms.   Achieved  accuracy of 100% using the AdaBoost algorithm. Developed a web application using Flask to deploy the mushroom classification model, allowing users to determine the edibility of mushrooms.

#Problem Statement:
The Audubon Society Field Guide to North American Mushrooms contains descriptions
of hypothetical samples corresponding to 23 species of gilled mushrooms in the 
Agaricus and Lepiota Family Mushroom (1981). Each species is labelled as either 
definitely edible, definitely poisonous, or maybe edible but not recommended. This last 
category was merged with the toxic category. The Guide asserts unequivocally that 
there is no simple rule for judging a mushroom's edibility, such as "leaflets three, leave it 
be" for Poisonous Oak and Ivy.
The main goal is to predict which mushroom is poisonous & which is edible.

PROCESS FLOW

Data Collection: A comprehensive dataset of mushroom samples with associated parameters and edibility labels is gathered. The dataset should encompass a diverse range of mushroom species and their various characteristics.

Data Preprocessing: The collected dataset undergoes preprocessing steps to ensure its quality and suitability for training the classification model. This may involve handling missing values, normalizing features, and encoding categorical variables.

Model Selection: The AdaBoost algorithm is chosen as the classification algorithm due to its ability to combine multiple weak learners (decision trees) to create a strong classifier. Decision trees are suitable base models as they can capture complex relationships between mushroom features.

Training Phase: The dataset is divided into training and testing sets. The AdaBoost classifier is trained using the training set, where it iteratively adjusts the weights of the training samples to focus on difficult-to-classify instances. Decision trees are trained to minimize misclassification errors and optimize overall performance.

Evaluation Metrics: The trained model is evaluated using various evaluation metrics, such as accuracy, precision, recall, and F1 score, to assess its performance. The goal is to achieve high accuracy and ensure reliable predictions.

Web Application Development: The Flask web framework is utilized to develop the Mushroom Classification Web Application. The application provides a user-friendly interface where users can input mushroom parameters.

