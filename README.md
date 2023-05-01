# Fraud-Detection-Report
<hr>
Credit Card Fraud Detection using Logistic Regression in Python

<hr>

Hypothesis:

1.Fraudulent transactions tend to involve larger amounts of money than normal transactions.

2.Can the dataset help predict whether a transaction is fraudulent based on the correlation between fraudulent and legitimate transactions?

<hr>
Hypothesis and Analysis:

Fraudulent transactions tend to involve larger amounts of money than normal transactions:

The bar graph shows that the mean transaction amount for "Fraud" category is higher (122.21) compared to the "Normal" category (88.29). Therefore, we can conclude that on average, fraudulent transactions tend to involve larger amounts of money than normal transactions. Additionally, various studies such as the report by Kount and The Fraud Practice and the study by the Federal Reserve Bank of New York also support this hypothesis.

Can the dataset help predict whether a transaction is fraudulent based on the correlation between fraudulent and legitimate transactions:

There is a significant correlation between the Class (fraud or legit transaction) and some of the features (V1 to V28) in the dataset. In particular, V3, V4, V9, V10, V11, V12, V14, V16, and V17 show a relatively strong correlation with the Class variable, indicating that they may be useful in predicting whether a transaction is fraudulent or not. Therefore, the dataset can be helpful in predicting whether a transaction is fraudulent or not based on the correlation between fraudulent and legitimate transactions. 

<hr>


Identifying Fraudulent Transactions: Utilizing Correlation Heatmap Analysis of Features in Dataset for Prediction
<hr>
<div style="display: flex;">
    <div style="flex-basis: 50%;">
        <img src="https://github.com/KamoEllen/Fraud-Detection-Report/blob/main/heatmap.png" alt="Heatmap" width="400"/>
    </div>
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
Fraudulent Transactions Involve Larger Amounts of Money on Average
<hr>
<div style="display: flex;">
    <div style="flex-basis: 50%;">
        <img src="https://github.com/KamoEllen/Fraud-Detection-Report/blob/main/compare_mean.png" alt="Compare Mean" width="400"/>
    </div>
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
<hr>

<div style="display: flex;">
    <div style="flex-basis: 50%;">
        <img src="https://github.com/KamoEllen/Fraud-Detection-Report/blob/main/Fraudulent%20Transaction%20Comparison.png" alt="Fraudulent_20Transaction_20Comparison" width="400"/>
    </div>
<pre><code>
<div>
import numpy as np
import matplotlib.pyplot as plt
categories = ['Median Loss', 'Average Transaction Value', 'Average Fraudulent Transaction Size']
means = [150000, 254, 1.7*227] # converting the average fraudulent transaction size from USD 353 to USD 603 (70% larger than domestic fraudulent transactions)

# Create line chart
fig, ax = plt.subplots()
ax.plot(categories, means, marker='o')
ax.set_xlabel('Transaction Category')
ax.set_ylabel('Transaction Amount (USD)')
ax.set_title('Comparison of Fraudulent vs Normal Transactions')
plt.show()
</div>
</code></pre>
Source:
<hr>

Kaggle. (n.d.). Credit Card Fraud Detection. 

Kount. (2018). Fraud prevention benchmark report. 

The Fraud Practice. (2018). 2018 true cost of fraud℠ study: e-commerce/Retail Edition.

Federal Reserve Bank of New York. (2016). Cybersecurity and financial stability: Risks and resilience. 
