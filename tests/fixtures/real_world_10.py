# Source: https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_features.html
# Pie chart from matplotlib gallery with animatable startangle
import matplotlib.pyplot as plt

startangle = 90

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)

fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
       shadow=True, startangle=startangle)

plt.show()
