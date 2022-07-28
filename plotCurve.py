# Import Statements
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing our CSV database file into file and confirming

file_name = 'F1-Scores.csv'
df = pd.read_csv(file_name, index_col=0)
print(df.head())

ax = df.plot(kind='bar', figsize=(20, 8), width=0.8, color=['#5bc0de', '#d9534f'], fontsize=14)

plt.title("F1-Scores using Bag of Words", fontsize=16)  # add title to the plot

# Solution inspired in https://stackoverflow.com/questions/14088687/how-to-change-plot-background-color/23907866
ax.set_facecolor((1.0, 1.0, 1.0))

# Solution inspired in https://stackoverflow.com/questions/40705614/hide-axis-label-only-not-entire-axis-in-pandas-plot
y_axis = ax.axes.get_yaxis()
y_axis.set_visible(True)

# Solution inspired in https://stackoverflow.com/questions/25447700/annotate-bars-with-values-on-pandas-bar-plots
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.03))

plt.gca().spines['top'].set_color('none')
plt.gca().spines['left'].set_color('none')
plt.gca().spines['right'].set_color('none')
plt.show()
