import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy.io import loadmat, savemat


class CircularTrackLinearizer:
    """
    Simple GUI for fitting a circle to circular track data and linearizing positions.

    Usage:
    - Left click and drag to move the circle center
    - Right click and drag to resize the circle radius
    - Press Enter to apply linearization and save results
    - Press 'r' to reset circle to auto-fit
    """

    def __init__(self, x_data, y_data, basepath=None, epoch=None, interval=None):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.basepath = basepath
        self.epoch = epoch
        self.interval = interval

        # Remove NaN values for fitting
        valid_mask = ~(np.isnan(self.x_data) | np.isnan(self.y_data))
        self.x_valid = self.x_data[valid_mask]
        self.y_valid = self.y_data[valid_mask]

        # Initial circle parameters (auto-fit)
        self.reset_circle()

        # Create figure and axis
        with plt.style.context("dark_background"):
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            self.ax.set_aspect("equal")

            # Plot data points
            self.data_scatter = self.ax.scatter(
                self.x_valid,
                self.y_valid,
                c="lightblue",
                s=1,
                alpha=0.6,
                label="Track data",
            )

            # Create circle patch
            self.circle = Circle(
                (self.center_x, self.center_y),
                self.radius,
                fill=False,
                color="red",
                linewidth=2,
                label="Fit circle",
            )
            self.ax.add_patch(self.circle)

            # Add center point
            self.center_point = self.ax.scatter(
                [self.center_x],
                [self.center_y],
                c="red",
                s=100,
                marker="x",
                linewidth=3,
                label="Center",
            )

            # Set up the plot
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlabel("X Position (cm)")
            self.ax.set_ylabel("Y Position (cm)")
            self.ax.set_title(
                "Circular Track Linearization\n"
                "Left click+drag: move center | Right click+drag: resize radius | Enter: linearize | r: reset"
            )

            # Event handling variables
            self.dragging = False
            self.drag_mode = None  # 'center' or 'radius'
            self.last_mouse_pos = None

            # Connect events
            self.fig.canvas.mpl_connect("button_press_event", self.on_mouse_press)
            self.fig.canvas.mpl_connect("button_release_event", self.on_mouse_release)
            self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
            self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

            plt.show()

    def reset_circle(self):
        """Reset circle to auto-fit the data"""
        # Auto-fit circle to data bounds
        x_center = np.nanmean([np.nanmin(self.x_valid), np.nanmax(self.x_valid)])
        y_center = np.nanmean([np.nanmin(self.y_valid), np.nanmax(self.y_valid)])

        # Estimate radius as average distance from center to data points
        distances = np.sqrt(
            (self.x_valid - x_center) ** 2 + (self.y_valid - y_center) ** 2
        )
        radius = np.nanmean(distances)

        self.center_x = x_center
        self.center_y = y_center
        self.radius = radius

    def on_mouse_press(self, event):
        """Handle mouse press events"""
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left click - move center
            self.dragging = True
            self.drag_mode = "center"
            self.last_mouse_pos = (event.xdata, event.ydata)
        elif event.button == 3:  # Right click - resize radius
            self.dragging = True
            self.drag_mode = "radius"
            self.last_mouse_pos = (event.xdata, event.ydata)

    def on_mouse_release(self, event):
        """Handle mouse release events"""
        self.dragging = False
        self.drag_mode = None
        self.last_mouse_pos = None

    def on_mouse_move(self, event):
        """Handle mouse move events"""
        if not self.dragging or event.inaxes != self.ax:
            return

        if self.drag_mode == "center":
            # Move circle center
            self.center_x = event.xdata
            self.center_y = event.ydata
            self.update_circle()

        elif self.drag_mode == "radius":
            # Resize circle radius based on distance from center
            if event.xdata is not None and event.ydata is not None:
                new_radius = np.sqrt(
                    (event.xdata - self.center_x) ** 2
                    + (event.ydata - self.center_y) ** 2
                )
                self.radius = new_radius
                self.update_circle()

    def on_key_press(self, event):
        """Handle key press events"""
        if event.key == "enter":
            self.linearize_and_save()
        elif event.key == "r":
            self.reset_circle()
            self.update_circle()

    def update_circle(self):
        """Update the circle visualization"""
        self.circle.center = (self.center_x, self.center_y)
        self.circle.radius = self.radius
        self.center_point.set_offsets([[self.center_x, self.center_y]])
        self.fig.canvas.draw()

        # Update title with current parameters
        self.ax.set_title(
            f"Circular Track Linearization\n"
            f"Center: ({self.center_x:.1f}, {self.center_y:.1f}), "
            f"Radius: {self.radius:.1f}\n"
            f"Left click+drag: move center | Right click+drag: resize radius | Enter: linearize | r: reset"
        )

    def linearize_positions(self):
        """Linearize positions using the fitted circle"""
        # Apply your linearization logic
        x_centered = self.x_data - self.center_x
        y_centered = self.y_data - self.center_y

        # Calculate theta (angle)
        theta = np.arctan2(y_centered, x_centered)
        theta[theta < 0] += 2 * np.pi

        # Convert to linear position (0 to 2*pi*radius)
        linear_position = theta * self.radius

        return linear_position, theta, x_centered, y_centered

    def linearize_and_save(self):
        """Apply linearization and save results"""
        print("Applying linearization...")

        # Get linearized positions
        linear_pos, theta, x_centered, y_centered = self.linearize_positions()

        # Create results DataFrame
        results_df = pd.DataFrame(
            {
                "x_original": self.x_data,
                "y_original": self.y_data,
                "x_centered": x_centered,
                "y_centered": y_centered,
                "theta": theta,
                "linear_position": linear_pos,
                "circle_center_x": self.center_x,
                "circle_center_y": self.center_y,
                "circle_radius": self.radius,
            }
        )

        # Show results plot
        self.plot_results(results_df)

        # Save results if basepath is provided
        if self.basepath is not None:
            self.save_to_behavior_file(results_df)

        print(f"Linearization complete!")
        print(f"Circle center: ({self.center_x:.2f}, {self.center_y:.2f})")
        print(f"Circle radius: {self.radius:.2f}")
        print(
            f"Linear position range: {np.nanmin(linear_pos):.2f} to {np.nanmax(linear_pos):.2f}"
        )

        return results_df

    def plot_results(self, results_df):
        """Plot linearization results"""
        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Circular Track Linearization Results", fontsize=14)

            # Plot 1: Original data with fitted circle
            ax1 = axes[0, 0]
            ax1.scatter(
                results_df["x_original"],
                results_df["y_original"],
                c=results_df["linear_position"],
                s=1,
                alpha=0.7,
                cmap="viridis",
            )
            circle_plot = Circle(
                (self.center_x, self.center_y),
                self.radius,
                fill=False,
                color="red",
                linewidth=2,
            )
            ax1.add_patch(circle_plot)
            ax1.scatter(
                [self.center_x],
                [self.center_y],
                c="red",
                s=100,
                marker="x",
                linewidth=3,
            )
            ax1.set_aspect("equal")
            ax1.set_title("Original Data with Fitted Circle")
            ax1.set_xlabel("X Position (cm)")
            ax1.set_ylabel("Y Position (cm)")
            ax1.grid(True, alpha=0.3)

            # Plot 2: Centered coordinates
            ax2 = axes[0, 1]
            ax2.scatter(
                results_df["x_centered"],
                results_df["y_centered"],
                c=results_df["linear_position"],
                s=1,
                alpha=0.7,
                cmap="viridis",
            )
            circle_centered = Circle(
                (0, 0), self.radius, fill=False, color="red", linewidth=2
            )
            ax2.add_patch(circle_centered)
            ax2.scatter([0], [0], c="red", s=100, marker="x", linewidth=3)
            ax2.set_aspect("equal")
            ax2.set_title("Centered Coordinates")
            ax2.set_xlabel("X - Center (cm)")
            ax2.set_ylabel("Y - Center (cm)")
            ax2.grid(True, alpha=0.3)

            # Plot 3: Linear position over time
            ax3 = axes[1, 0]
            valid_mask = ~np.isnan(results_df["linear_position"])
            ax3.scatter(
                np.arange(len(results_df))[valid_mask],
                results_df["linear_position"][valid_mask],
                s=1,
                alpha=0.7,
                color="blue",
            )
            ax3.set_title("Linear Position Over Time")
            ax3.set_xlabel("Time Point")
            ax3.set_ylabel("Linear Position")
            ax3.grid(True, alpha=0.3)

            # Plot 4: Theta (angle) distribution
            ax4 = axes[1, 1]
            valid_theta = results_df["theta"][~np.isnan(results_df["theta"])]
            ax4.hist(valid_theta, bins=50, alpha=0.7, color="green")
            ax4.set_title("Angle (Theta) Distribution")
            ax4.set_xlabel("Theta (radians)")
            ax4.set_ylabel("Count")
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    def save_to_behavior_file(self, results_df):
        """Save results to behavior file following the same epoch-aware logic as original code"""
        try:
            # Load existing behavior data
            behave_df = load_animal_behavior(self.basepath)

            # Determine which epochs/intervals to update (same logic as original)
            if self.epoch is not None:
                epochs = load_epoch(self.basepath)
                cur_epoch = (
                    ~np.isnan(behave_df.x)
                    & (behave_df.time >= epochs.iloc[self.epoch].startTime)
                    & (behave_df.time <= epochs.iloc[self.epoch].stopTime)
                )
            elif hasattr(self, "interval") and self.interval is not None:
                cur_epoch = (
                    ~np.isnan(behave_df.x)
                    & (behave_df.time >= self.interval[0])
                    & (behave_df.time <= self.interval[1])
                )
            else:
                cur_epoch = behave_df.index

            print("Saving to disk...")

            # Update only the specified epoch/interval data
            behave_df.loc[cur_epoch, "linearized"] = results_df[
                "linear_position"
            ].values
            behave_df.loc[cur_epoch, "theta"] = results_df["theta"].values
            behave_df.loc[cur_epoch, "x_centered"] = results_df["x_centered"].values
            behave_df.loc[cur_epoch, "y_centered"] = results_df["y_centered"].values

            # Load the .mat file and update it
            filename = os.path.join(
                self.basepath, os.path.basename(self.basepath) + ".animal.behavior.mat"
            )

            if os.path.exists(filename):
                data = loadmat(filename, simplify_cells=True)

                # Update the linearized data (epoch-aware)
                data["behavior"]["position"]["linearized"] = behave_df.linearized.values
                data["behavior"]["position"]["theta"] = behave_df.theta.values
                data["behavior"]["position"]["x_centered"] = behave_df.x_centered.values
                data["behavior"]["position"]["y_centered"] = behave_df.y_centered.values

                # Store circle parameters in behavior file (similar to nodes/edges storage)
                # data = self.save_circle_params_to_behavior(data, behave_df)

                # Save updated data
                savemat(filename, data, long_field_names=True)
                print(f"Results saved to {filename}")
            else:
                print(f"Behavior file not found: {filename}")

        except Exception as e:
            print(f"Error saving to behavior file: {e}")

        # Also save as CSV for easy access
        # csv_filename = os.path.join(self.basepath, "circular_linearization_results.csv")
        # results_df.to_csv(csv_filename, index=False)
        # print(f"Results also saved as CSV: {csv_filename}")

    def save_circle_params_to_behavior(
        self, data: dict, behave_df: pd.DataFrame
    ) -> dict:
        """
        Store circle parameters into behavior file.
        Similar to save_nodes_edges_to_behavior but for circular track parameters.
        """
        circle_params = {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "radius": self.radius,
            "method": "circular_track",
        }

        if self.epoch is None and not hasattr(self, "interval"):
            # Load epochs and add to all epochs with valid linearized coords
            epochs = load_epoch(self.basepath)
            for epoch_i, ep in enumerate(epochs.itertuples()):
                idx = behave_df.time.between(ep.startTime, ep.stopTime)
                if not all(np.isnan(behave_df[idx].linearized)) & (
                    behave_df[idx].shape[0] != 0
                ):
                    data["behavior"]["epochs"][epoch_i]["circle_params"] = circle_params
        elif hasattr(self, "interval") and self.interval is not None:
            # Add to epochs within the interval
            epochs = load_epoch(self.basepath)
            for epoch_i, ep in enumerate(epochs.itertuples()):
                start_overlap = max(self.interval[0], ep.startTime)
                end_overlap = min(self.interval[1], ep.stopTime)
                overlap = max(0, end_overlap - start_overlap)
                if overlap > 1:  # If overlap is greater than 1 second
                    data["behavior"]["epochs"][epoch_i]["circle_params"] = circle_params
        elif self.epoch is not None:
            # Add to specific epoch
            data["behavior"]["epochs"][self.epoch]["circle_params"] = circle_params
        else:
            pass

        return data


def load_epoch(basepath: str) -> pd.DataFrame:
    """Load epoch info from cell explorer basename.session and store in a DataFrame."""
    filename = os.path.join(basepath, os.path.basename(basepath) + ".session.mat")
    data = loadmat(filename, simplify_cells=True)
    try:
        return pd.DataFrame(data["session"]["epochs"])
    except Exception:
        return pd.DataFrame([data["session"]["epochs"]])


def load_animal_behavior(basepath: str) -> pd.DataFrame:
    """
    Load animal behavior data from a .mat file.

    Parameters
    ----------
    basepath : str
        The base path where the .mat file is located.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the animal behavior data.
    """
    filename = os.path.join(
        basepath, os.path.basename(basepath) + ".animal.behavior.mat"
    )
    data = loadmat(filename, simplify_cells=True)
    df = pd.DataFrame()
    df["time"] = data["behavior"]["timestamps"]
    try:
        df["states"] = data["behavior"]["states"]
    except Exception:
        pass
    for key in data["behavior"]["position"].keys():
        try:
            df[key] = data["behavior"]["position"][key]
        except Exception:
            pass
    return df


def run_circular_linearization(
    basepath: str,
    epoch: Optional[int] = None,
    interval: Optional[Tuple[float, float]] = None,
):
    """
    Run the circular track linearization GUI.

    Parameters
    ----------
    basepath : str
        Path to the data directory
    epoch : int, optional
        Specific epoch to process
    interval : tuple, optional
        Time interval (start, end) to process
    """
    print(f"Loading data from {basepath}")

    # Load behavior data
    behave_df = load_animal_behavior(basepath)

    # Filter by epoch or interval if specified
    if epoch is not None:
        epochs = load_epoch(basepath)
        print(f"Processing epoch {epoch}")
        behave_df = behave_df[
            behave_df["time"].between(
                epochs.iloc[epoch].startTime, epochs.iloc[epoch].stopTime
            )
        ]
    elif interval is not None:
        print(f"Processing interval {interval}")
        behave_df = behave_df[behave_df["time"].between(interval[0], interval[1])]

    # Extract x, y coordinates
    x_data = behave_df["x"].values
    y_data = behave_df["y"].values

    print(f"Loaded {len(x_data)} data points")
    print(f"X range: {np.nanmin(x_data):.1f} to {np.nanmax(x_data):.1f}")
    print(f"Y range: {np.nanmin(y_data):.1f} to {np.nanmax(y_data):.1f}")

    # Create and run the linearization GUI
    linearizer = CircularTrackLinearizer(
        x_data, y_data, basepath=basepath, epoch=epoch, interval=interval
    )

    return linearizer


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        basepath = sys.argv[1]
        epoch = int(sys.argv[2]) if len(sys.argv) >= 3 else None
        interval = (
            (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) >= 4 else None
        )

        run_circular_linearization(basepath, epoch=epoch, interval=interval)
    else:
        print(
            "Usage: python circular_linearization.py <basepath> [epoch] or [start_time end_time]"
        )

        # Demo with synthetic circular data
        print("Running demo with synthetic data...")

        # Generate synthetic circular track data
        n_points = 1000
        theta_true = np.linspace(0, 4 * np.pi, n_points)  # Two laps
        radius = 50
        center_x, center_y = 100, 150

        # Add some noise
        noise_level = 5
        x_demo = (
            center_x
            + radius * np.cos(theta_true)
            + np.random.normal(0, noise_level, n_points)
        )
        y_demo = (
            center_y
            + radius * np.sin(theta_true)
            + np.random.normal(0, noise_level, n_points)
        )

        # Create demo linearizer
        demo_linearizer = CircularTrackLinearizer(x_demo, y_demo)
