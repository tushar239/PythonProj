# Use legend() when plotting multiple scatter plots.


import matplotlib.pyplot as plt

# Sample data for Group A (e.g., passed)
x_a = [1, 2, 3, 4, 5]
y_a = [35, 45, 50, 55, 60]

# Sample data for Group B (e.g., failed)
x_b = [1, 2, 3, 4, 5]
y_b = [25, 30, 28, 27, 33]

# Create scatter plots for both groups
plt.scatter(x_a, y_a, color='green', marker='o', label='Passed')
plt.scatter(x_b, y_b, color='red', marker='x', label='Failed')

# Add labels and title
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Student Performance Scatter Plot")

# Use legend() when plotting multiple scatter plots.
# in scatter function, label is required to show it as a legend
plt.legend()

# Optional: Add grid for better readability
plt.grid(True)

# Show plot
plt.show()
