# Fraud-Detection-Report
Credit Card Fraud Detection using Logistic Regression in Python

<hr>
You can find the credit card fraud dataset [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).


<hr>
Based on the correlation heatmap, there is a significant correlation between the Class (fraud or legit transaction) and some of the features (V1 to V28) in the dataset. In particular, V3, V4, V9, V10, V11, V12, V14, V16, and V17 show a relatively strong correlation with the Class variable, indicating that they may be useful in predicting whether a transaction is fraudulent or not. 
<hr>

<pre><code>
<div>
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# creating dataframe
data = pd.DataFrame({
    'Class': [0, 1],
    'Mean_Amount': [94838.202258, 80746.806911],
    'V1': [0.008258, -4.771948],
    'V2': [-0.006271, 3.623778],
    'V3': [0.012171, -7.033281],
    'V4': [-0.007860, 4.542029],
    'V5': [0.005453, -3.151225],
    'V6': [0.002419, -1.397737],
    'V7': [0.009637, -5.568731],
    'V8': [-0.000987, 0.570636],
    'V9': [0.004467, -2.581123],
    'V10': [0.009824, -5.676883],
    'V11': [-0.006576, 3.800173],
    'V12': [0.010832, -6.259393],
    'V13': [0.000189, -0.109334],
    'V14': [0.012064, -6.971723],
    'V15': [0.000161, -0.092929],
    'V16': [0.007164, -4.139946],
    'V17': [0.011535, -6.665836],
    'V18': [0.003887, -2.246308],
    'V19': [-0.001178, 0.680659],
    'V20': [-0.000644, 0.372319],
    'V21': [-0.001235, 0.713588],
    'V22': [-0.000024, 0.014049],
    'V23': [0.000070, -0.040308],
    'V24': [0.000182, -0.105130],
    'V25': [-0.000072, 0.041449],
    'V26': [-0.000089, 0.051648],
    'V27': [-0.000295, 0.170575],
    'V28': [-0.000131, 0.075667]
})

#correlation matrix
corr_matrix = data.corr()

#heatmap using seaborn package
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f')

plt.title('Correlation Heatmap for Fraud and Legit Transactions')

plt.show()
</div>
 </code></pre>
  

<hr>
The bar graph shows that the mean transaction amount for "Fraud" category is higher (122.21) compared to the "Normal" category (88.29). Therefore, we can conclude that on average, fraudulent transactions tend to involve larger amounts of money than normal transactions.
<pre><code>
<div>
import matplotlib.pyplot as plt
categories = ['Fraud', 'Normal']
means = [122.21, 88.29]
plt.bar(categories, means)
plt.xlabel('Transaction Category')
plt.ylabel('Mean Transaction Amount')
plt.title('Comparison of Mean Transaction Amounts')
plt.show()
</div>
</code></pre>
