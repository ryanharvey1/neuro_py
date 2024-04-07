import matplotlib.pyplot as plt
import numpy as np

def plot_grid_with_circle_and_random_dots():
    # Create a 15x15 grid of dots
    x = np.linspace(0, 14, 15)
    y = np.linspace(0, 14, 15)
    X, Y = np.meshgrid(x, y)
    
    # Calculate the circle parameters with an offset
    radius = 6.8  # Radius of the circle
    circle_center = (7, 7)  # Center of the circle
    
    # Create a mask to display only the dots within the circle
    circle_mask = ((X - circle_center[0])**2 + (Y - circle_center[1])**2) <= radius**2
    
    # Plot the grid of dots
    plt.figure(figsize=(8, 8))
    plt.plot(X[circle_mask], Y[circle_mask], 'o', color='k', markersize=6)
    
    # Plot the circle
    circle = plt.Circle(circle_center, radius, color='black', fill=False)
    plt.gca().add_patch(circle)
    
    # Mask for the dots within the circle
    grid_mask = ((X - circle_center[0])**2 + (Y - circle_center[1])**2) <= (radius - 0.5)**2
    
    # Randomly pick 3 dots within the circle
    num_dots = 3
    chosen_indices = np.random.choice(np.sum(grid_mask), size=num_dots, replace=False)
    chosen_dots = np.argwhere(grid_mask)
    chosen_dots = chosen_dots[chosen_indices]
    
    # Ensure minimum separation of 4 dots between the randomly chosen dots
    min_separation = 4
    for i in range(num_dots):
        for j in range(i + 1, num_dots):
            while np.linalg.norm(chosen_dots[i] - chosen_dots[j]) < min_separation:
                chosen_indices[j] = np.random.choice(np.sum(grid_mask), size=1)[0]
                chosen_dots[j] = np.argwhere(grid_mask)[chosen_indices[j]]
    
    # Color the randomly chosen dots red and make them slightly larger
    for dot in chosen_dots:
        plt.plot(X[dot[0], dot[1]], Y[dot[0], dot[1]], 'o', color='red', markersize=9)

    # Draw a grey box at bottom
    plt.fill_between([6, 8], -1.8, 0.2, color='darkgray', alpha=1)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis("off")
    plt.show()

plot_grid_with_circle_and_random_dots()