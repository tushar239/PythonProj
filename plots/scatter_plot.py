import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]  # X-axis values
y = [2, 4, 5, 4, 5]  # Y-axis values

# Create scatter plot
#plt.scatter(x, y)

#Change color and marker
#plt.scatter(x, y, color='red', marker='x')

# Add size to points (observations)
sizes = [20, 40, 60, 80, 100]
# plt.scatter(x, y, s=sizes, color='red', marker='x')

colors = [1, 2, 3, 4, 5]  # Used to color-code points
plt.scatter(x, y, c=colors, cmap='viridis', marker='x', s=sizes)

#  to show regression lines or model fits.
plt.plot(x, y) # line connecting observations

#for better readability
plt.grid(True)

# Add labels and title
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Scatter Plot Example")

# Use legend() when plotting multiple scatter plots.
# plt.legend()

# Show plot
plt.show()
