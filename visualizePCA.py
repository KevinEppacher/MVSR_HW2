import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Laden Sie die Daten aus der CSV-Datei
df = pd.read_csv('/home/fhtw_user/catkin_ws/src/test/pca_reduced_data.csv')

# Erstellen Sie eine 3D-Figur
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotten Sie die Datenpunkte mit verschiedenen Farben für verschiedene Labels
for label in df['Label'].unique():
    subset = df[df['Label'] == label]
    ax.scatter(subset['Component1'], subset['Component2'], subset['Component3'], label=f'Label {label}')

# Beschriftungen und Legende
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
ax.legend()

plt.show()
