#!/usr/bin/env python
# coding: utf-8

# # **Supervised Data Mining: Breast Cancer Classification Project**
# ## 1. Project Overview
# In this project, we are going to use three different classification algorithms, Random Forest, LSTM, and KNN to classify a dataset of breast cancer cases. The goal is to categorize the cases into two groups: malignant (cancerous) and benign (non-cancerous). We’ll walk through the steps of data preprocessing, visualizing the data, training the models, evaluating their performance, and comparing the results to see which algorithm works best.
# ### 2. Using KNN, RF and LSTM To categorize the cases into two groups: malignant (cancerous) and benign (non-cancerous

# ### 2.1 Importing the packages and libraries that are required for the project

# In[57]:


# Data manipulation and analysis
import pandas as pd  # Provides data structures and data analysis tools
import numpy as np   # Used for efficient numerical computations

# Data visualization
import matplotlib.pyplot as plt  # For basic plotting
import seaborn as sns  # For more advanced and attractive visualizations

# Machine learning tools and algorithms from scikit-learn
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, average_precision_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold  # For splitting data into train and test sets, and cross-validation
from sklearn.preprocessing import StandardScaler  # For normalizing data

# K-Nearest Neighbors and Random Forest classifiers
from sklearn.neighbors import KNeighborsClassifier  # KNN algorithm for classification
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier for ensemble learning

# Deep learning with TensorFlow and Keras
from tensorflow.keras.models import Sequential  # For building sequential neural networks
from tensorflow.keras.layers import Dense, LSTM  # For adding layers to the neural network
from tensorflow.keras.utils import to_categorical  # For one-hot encoding categorical data

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")


# ### 2.2 Loading Data from raw file (wbc.data)

# In[2]:


# Load the data
data = pd.read_csv('datasets/wdbc.data', header=None)


# ### 2.2.1 Assign Column Names

# In[3]:


# Assign Column Names
column_names = [
    "ID", "Diagnosis", 
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", 
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", 
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", 
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]
data.columns = column_names


# ### 2.2.2 To prepare for classification, convert the Diagnosis column to binary values (M = 1 for malignant, B = 0 for benign)

# In[4]:


#Encode the Target Column
data['Diagnosis'] = data['Diagnosis'].apply(lambda x: 1 if x == 'M' else 0)


# ### 2.2.3 Save as .csv File 

# In[5]:


# Save as .csv File 
data.to_csv('datasets/breast_cancer.csv', index=False)


# ### 2.2.4 Read as CSV file and Drop the ID Column

# In[6]:


# Read the CSV file
data = pd.read_csv('datasets/breast_cancer.csv')
# Drop the ID Column, Since ID is not needed for classification tasks, you may drop it.
data = data.drop(columns=['ID'])


# In[7]:


# Display the first few rows of the dataframe
data.describe()


# In[8]:


data.info()


# ### 2.3 Data Visualization
# **In this project, we use visualizations like pair plots, heatmaps, histograms, confusion matrices, ROC curves and more to look at the relationships between features and see how well the classification models are doing. These visualizations make it easier to interpret the data and compare the results from different models.**

# ### 2.3.1 Distribution of Diagnosis

# In[9]:


# Distribution of Diagnosis
plt.figure(figsize=(6, 4))
sns.countplot(data['Diagnosis'])
plt.title("Distribution of Diagnosis")
plt.xlabel("Diagnosis (1 = Malignant, 0 = Benign)")
plt.ylabel("Count")
plt.show()


# ### 2.3.2 Checking the Distribution of Diagnoses (Malignant vs Benign)

# In[10]:


# Check the unique values in the Diagnosis column to confirm both classes are present
print(data['Diagnosis'].unique())

# Plotting the counts of Diagnosis
plt.figure(figsize=(6, 4))
sns.countplot(x='Diagnosis', data=data, palette="viridis")
plt.title("Total Count of Diagnoses (Malignant=1, Benign=0)")
plt.xlabel("Diagnosis (1 = Malignant, 0 = Benign)")
plt.ylabel("Count")
plt.xticks([0, 1], ['Benign (0)', 'Malignant (1)'])
plt.show()


# ### 2.3.3 Visualizing Feature Correlations with a Heatmap

# In[11]:


plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


# ### 2.3.4 Visualizing Feature Distributions with Histograms

# In[12]:


data.hist(figsize=(15, 15), bins=20)
plt.suptitle("Feature Distributions")
plt.show()


# ### 2.3.5 Pair Plot of Radius, Texture, and Area by Diagnosis

# In[49]:


# Pair Plot of Radius, Texture, and Area by Diagnosis
pair_features = ["radius_mean", "texture_mean", "area_mean", "Diagnosis"]
sns.pairplot(data[pair_features], hue="Diagnosis", palette="Set1")
plt.suptitle("Pair Plot of Selected Features by Diagnosis", y=1.02)
plt.show()


# ### 2.3.6 Violin Plot of Compactness Mean by Diagnosis

# In[50]:


# Violin Plot for Compactness_mean by Diagnosis
plt.figure(figsize=(12, 6))
sns.violinplot(x="Diagnosis", y="compactness_mean", data=data, palette="Set2")
plt.title("Violin Plot of Compactness Mean by Diagnosis")
plt.show()


# ### 2.4 Train Test Data Split

# In[15]:


# Train-Test Split data into training and test sets using the train_test_split function and stratification for balanced class distribution

X = data.drop(columns=['Diagnosis'])
y = data['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=21, stratify=y)


# ### 2.4.1 Normalizing the dataset to have a mean of 0 and a standard deviation of 1.
# #### This is important for algorithms that are sensitive to the scale of data.

# #To standardize the features, subtract the mean and divide by the standard deviation
# X_train_std = (X_train - X_train.mean()) / X_train.std()
# X_test_std = (X_test - X_train.mean()) / X_train.std()  # Use train data stats for scaling
# 
# X_train_std .describe()

# ### 2.5 Performance Metrics and 10-Fold Cross-Validation
# ##### In this project, performance metrics and 10-fold cross-validation work together to assess model accuracy and reliability in predicting whether a case is malignant (cancerous) or benign (non-cancerous). Key metrics—accuracy, precision, recall (sensitivity), F1 score, and AUC help determine how well the model identifies true cases while minimizing false predictions. Using 10-fold cross-validation, the dataset is divided into 10 parts, rotating each as a test set to give a balanced average score. This approach ensures the model's performance is both accurate and consistent across different sections of data.

# ### 2.5.1 Function to Calculate Performance Metrics

# In[16]:


# Function to calculate performance metrics
def calculate_metrics(y_true, y_pred):
    # If y_true is one-hot encoded (for LSTM), convert it back to binary labels
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:  # One-hot encoded format
        y_true = np.argmax(y_true, axis=1)
    
    # If y_pred is in one-hot format, convert it back to binary labels
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    
    # Sensitivity and Specificity
    sensitivity = recall  # same as recall
    specificity = tn / (tn + fp)
    
    # Balanced Accuracy (BACC)
    bacc = (sensitivity + specificity) / 2
    
    # False Positive Rate (FPR) and False Negative Rate (FNR)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    # True Skill Statistic (TSS)
    tss = sensitivity + specificity - 1
    
    # Heidke Skill Score (HSS)
    hss = 2 * (tp * tn - fp * fn) / ((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn))
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC': auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'BACC': bacc,
        'FPR': fpr,
        'FNR': fnr,
        'TSS': tss,
        'HSS': hss
    }


# ### 2.5.2 Implementing and Evaluating the KNN Model

# In[51]:


# KNN Model
knn = KNeighborsClassifier(n_neighbors=5)  # You can choose different hyperparameters
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)


# In[18]:


# KNN performance metrics
knn_metrics = calculate_metrics(y_test, y_pred_knn)


# ### 2.5.3 Implementing and Evaluating the Random Forest Model

# In[52]:


#  Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# In[20]:


# Random Forest performance metrics
rf_metrics = calculate_metrics(y_test, y_pred_rf)


# ### 2.5.4 Implementing and Evaluating the LSTM Model

# In[53]:


#  LSTM Model
# Convert the DataFrames to NumPy arrays
X_train_lstm = X_train.values
X_test_lstm = X_test.values


# ##### 2.5.4.1 Reshaping data for LSTM 

# In[22]:


# Reshaping data for LSTM (samples, timesteps, features)
X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))  # Add the third dimension
X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))


# ##### 2.5.4.2 One-hot encoding of labels for LSTM

# In[23]:


# One-hot encoding of labels for LSTM
y_train_lstm = to_categorical(y_train, num_classes=2)
y_test_lstm = to_categorical(y_test, num_classes=2)


# #### 2.5.4.3 Building the LSTM Model for Classification

# In[58]:


# Build LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, activation='relu', input_shape=(X_train_lstm.shape[1], 1)))
lstm_model.add(Dense(2, activation='softmax'))
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# ### 2.6 Train,Predict and performance measues for LSTM model

# In[25]:


# Train the LSTM model
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, verbose=0)


# In[26]:


# Predict with LSTM
y_pred_lstm = np.argmax(lstm_model.predict(X_test_lstm), axis=1)


# In[27]:


# LSTM performance metrics
lstm_metrics = calculate_metrics(y_test, y_pred_lstm)


# ### 2.7 Display performance for each model

# In[28]:


# Convert each dictionary to a DataFrame
knn_df = pd.DataFrame(list(knn_metrics.items()), columns=['Metric', 'KNN'])


# In[29]:


# Display as tables
print("\nKNN Performance Metrics:")
print(knn_df.to_string(index=False))


# In[30]:


rf_df = pd.DataFrame(list(rf_metrics.items()), columns=['Metric', 'Random Forest'])


# In[31]:


print("\nRandom Forest Performance Metrics:")
print(rf_df.to_string(index=False))


# In[32]:


lstm_df = pd.DataFrame(list(lstm_metrics.items()), columns=['Metric', 'LSTM'])


# In[33]:


print("\nLSTM Performance Metrics:")
print(lstm_df.to_string(index=False))


# ### 2.8 Comparing the classifiers with selected parameters by using 10-Fold Cross-Validation
# #### 10-Fold Cross-Validation Function for Models Evaluation

# In[34]:


# Function to perform 10-Fold Cross-Validation for a model
def cross_val_model(model, X, y, is_lstm=False):
    # Initialize 10-fold cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    metrics = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # If it's an LSTM model, reshape the input data
        if is_lstm:
            X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_val = X_val.values.reshape((X_val.shape[0], X_val.shape[1], 1))
            y_train = to_categorical(y_train, num_classes=2)
            y_val = to_categorical(y_val, num_classes=2)

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0) if is_lstm else model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_val)
        if is_lstm:
            y_pred = np.argmax(y_pred, axis=1)  # Convert predictions back to class labels

        # Calculate metrics
        fold_metrics = calculate_metrics(y_val, y_pred)
        metrics.append(fold_metrics)

    # Convert the list of metrics into a DataFrame for better readability
    metrics_df = pd.DataFrame(metrics)
    metrics_df.loc['Average'] = metrics_df.mean()  # Calculate the average of metrics across all folds
    return metrics_df


# ### 2.8.1 KNN Model Performance with 10-Fold Cross-Validation

# In[54]:


# KNN Model (10-fold cross-validation)
knn = KNeighborsClassifier(n_neighbors=5)
knn_metrics = cross_val_model(knn, X, y)


# ### 2.8.2 Random Forest Model Performance with 10-Fold Cross-Validation

# In[55]:


#  Random Forest Model (10-fold cross-validation)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_metrics = cross_val_model(rf, X, y)


# ### 2.8.3 LSTM Model Performance with 10-Fold Cross-Validation

# In[56]:


#  LSTM Model (10-fold cross-validation)
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dense(2, activation='softmax'))
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

lstm_metrics = cross_val_model(lstm_model, X, y, is_lstm=True)


# ### 2.8.4 Display Metrics for Each Model

# In[38]:


print("Random Forest Cross-Validation Results:\n", rf_metrics)


# In[39]:


# 5. Display Metrics for Each Model
print("KNN Cross-Validation Results:\n", knn_metrics)


# In[40]:


print("LSTM Cross-Validation Results:\n", lstm_metrics)


# ### 2.9 Evaluating the performance of various algorithms by comparing their ROC curves and AUC scores on the test dataset.

# In[41]:


# ROC Curve for KNN
y_prob_knn = knn.predict_proba(X_test)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
auc_knn = auc(fpr_knn, tpr_knn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, color="blue", label=f'KNN (AUC = {auc_knn:.2f})')
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve for KNN")
plt.legend(loc="lower right")
plt.grid()
plt.show()


# In[42]:


# ROC Curve for Random Forest
y_prob_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color="green", label=f'Random Forest (AUC = {auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve for Random Forest")
plt.legend(loc="lower right")
plt.grid()
plt.show()


# In[43]:


# ROC Curve for LSTM
X_test_lstm = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))  # Convert to numpy array and reshape
y_prob_lstm = lstm_model.predict(X_test_lstm)[:, 1]  # Get probability for class 1
fpr_lstm, tpr_lstm, _ = roc_curve(y_test, y_prob_lstm)
auc_lstm = auc(fpr_lstm, tpr_lstm)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lstm, tpr_lstm, color="red", label=f'LSTM (AUC = {auc_lstm:.2f})')
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve for LSTM")
plt.legend(loc="lower right")
plt.grid()
plt.show()


# ### 2.10 Evaluating the performance of various algorithms by comparing their Precision-Recall curve scores on the test dataset.

# In[44]:


# Calculate Precision-Recall for each model
def plot_precision_recall_curve(y_test, y_pred_probs, model_name):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)
    average_precision = average_precision_score(y_test, y_pred_probs)
    f1 = f1_score(y_test, (y_pred_probs > 0.5).astype(int))  # Using 0.5 threshold for F1 Score

    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name} (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend(loc="best")
    plt.show()

    # Display F1 score
    print(f'{model_name} - F1 Score (threshold=0.5): {f1:.2f}')


# In[45]:


# Plot for KNN
y_pred_probs_knn = knn.predict_proba(X_test)[:, 1]
plot_precision_recall_curve(y_test, y_pred_probs_knn, "KNN")


# In[46]:


# Plot for Random Forest
y_pred_probs_rf = rf.predict_proba(X_test)[:, 1]
plot_precision_recall_curve(y_test, y_pred_probs_rf, "Random Forest")


# In[48]:


# Reshape X_test for LSTM model (samples, time steps, features)
X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# Get the predicted probabilities for the positive class
y_pred_probs_lstm = lstm_model.predict(X_test_reshaped)[:, 1]

# Plot Precision-Recall Curve for LSTM
plot_precision_recall_curve(y_test, y_pred_probs_lstm, "LSTM")

