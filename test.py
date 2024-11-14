import matplotlib.pyplot as plt

# Sample data for multiple lines
x = [1, 2, 3, 4, 5]
y1 = [2, 3, 5, 7, 11]  # First line
y2 = [1, 4, 6, 8, 10]  # Second line
y3 = [2, 5, 8, 11, 14] # Third line

# Create the plot
plt.plot(x, y1, marker='o', linestyle='-', color='b', label='Line 1')
plt.plot(x, y2, marker='s', linestyle='--', color='r', label='Line 2')
plt.plot(x, y3, marker='^', linestyle='-.', color='g', label='Line 3')

# Add labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Multiple Line Chart Example')

# Add grid and legend
plt.grid(True)
plt.legend()

# Save the plot as an image file
plt.savefig('line_chart.png')  # Save as PNG
plt.savefig('line_chart.pdf')  # To save as PDF, use this instead

# Show the plot
plt.show()
