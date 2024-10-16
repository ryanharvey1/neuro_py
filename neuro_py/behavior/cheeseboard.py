import matplotlib.pyplot as plt
import numpy as np


def plot_grid_with_circle_and_random_dots():
    """
    Plots a 15x15 grid of dots within a circle, highlights 3 randomly chosen dots 
    within the circle, and draws a grey box at the bottom.

    The function generates a grid of points within a circle of a specified radius 
    and randomly selects three points from within the circle. These points are 
    colored red and slightly enlarged. Additionally, a grey box is drawn at the 
    bottom of the plot.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    - The grid is plotted on a 15x15 layout, with points that fall within the 
      circle of radius 6.8 being displayed.
    - The randomly selected points must be at least 4 grid units apart.
    - A grey rectangular box is drawn near the bottom of the plot for aesthetic 
      purposes.

    Examples
    --------
    >>> plot_grid_with_circle_and_random_dots()
    # This will display a plot of a circle containing a grid of dots with 
    # 3 randomly chosen dots highlighted in red.
    """
    # Create a 15x15 grid of dots within the circle
    x = np.linspace(-7, 7, 17)
    y = np.linspace(-7, 7, 17)
    X, Y = np.meshgrid(x, y)

    # Calculate the circle parameters with an offset
    radius = 6.8  # Radius of the circle
    circle_center = (0, 0)  # Center of the circle

    # Create a mask to display only the dots within the circle
    circle_mask = (X**2 + Y**2) <= radius**2

    # Plot the grid of dots within the circle
    plt.figure(figsize=(8, 8))
    plt.plot(X[circle_mask], Y[circle_mask], "o", color="k", markersize=6)

    # Plot the circle
    circle = plt.Circle(circle_center, radius, color="black", fill=False)
    plt.gca().add_patch(circle)

    # Randomly pick 3 dots within the circle
    num_dots = 3
    chosen_indices = np.random.choice(np.sum(circle_mask), size=num_dots, replace=False)
    chosen_dots = np.argwhere(circle_mask)
    chosen_dots = chosen_dots[chosen_indices]

    # Ensure minimum separation of 4 dots between the randomly chosen dots
    min_separation = 4
    for i in range(num_dots):
        for j in range(i + 1, num_dots):
            while np.linalg.norm(chosen_dots[i] - chosen_dots[j]) < min_separation:
                chosen_indices[j] = np.random.choice(np.sum(circle_mask), size=1)[0]
                chosen_dots[j] = np.argwhere(circle_mask)[chosen_indices[j]]

    # Color the randomly chosen dots red and make them slightly larger
    for dot in chosen_dots:
        plt.plot(X[dot[0], dot[1]], Y[dot[0], dot[1]], "o", color="red", markersize=9)

    # Draw a grey box at the bottom
    plt.fill_between([-1.5, 1.5], -8.5, -6.5, color="darkgray", alpha=1)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.show()
