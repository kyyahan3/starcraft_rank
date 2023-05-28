# import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plot_confusion_matrix', 'smooth_class_weights', 'three_sigma', 'get_filler', 'plot_roc']

# function for plotting confusion matrix
def plot_confusion_matrix(cm, title = ''):
    # Plot the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(' '.join([title, "Confusion Matrix"]))
    plt.colorbar()

    # Create class labels for the matrix
    classes = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"] 
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Fill the matrix with values
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        # remove the grid
    plt.grid(False)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


# Smoothen Weights: log 
def smooth_class_weights(train_df, target = 'LeagueIndex', mu=0.15):
    labels_dict = train_df[target].value_counts().to_dict()
    total = np.sum(list(labels_dict.values()))  # Convert dict_values to a list and sum the values
    keys = labels_dict.keys()
    weights = {}
    for i in keys:
        score = np.log(mu * total / float(labels_dict[i]))
        weights[i] = score if score > 1 else 1
    return weights

# calculate 3 sigma
def three_sigma(df, col):
    mean = df[col].mean()
    std = df[col].std()
    # lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    return upper_bound



from sklearn.linear_model import LinearRegression
def get_filler(df, feature):
    x_train = df.loc[df['LeagueIndex'] != 8, 'LeagueIndex'].values.reshape(-1, 1) 
    y_train = df.loc[df[feature] != '?', feature].values.reshape(-1, 1)
    x_test = df.loc[df['LeagueIndex'] == 8, 'LeagueIndex'].values.reshape(-1, 1) 
    # Fit a regression model on the complete data
    regression_model = LinearRegression()
    regression_model.fit(x_train, y_train)

    # Predict the missing values using the regression model
    return regression_model.predict([[8]])


# plot ROC for logistic regression for 8 classes
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle

def plot_roc(y_test, y_pred, title=''):
    y_test_bin = label_binarize(y_test, classes=[1, 2, 3, 4, 5, 6, 7, 8])
    y_pred_bin = label_binarize(y_pred, classes=[1, 2, 3, 4, 5, 6, 7, 8])

    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    fpr["macro"], tpr["macro"], _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.2f})'
                        ''.format(roc_auc["macro"]),
                        color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'yellow', 'purple', 'pink'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                            ''.format(i+1, roc_auc[i]))


    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(' '.join([title, 'ROC']), fontsize=16)
    plt.legend(loc="lower right")
    plt.show()
