import os
import pickle
import sys
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist


class TrackGraph:
    """
    A simple track graph implementation for linearization.

    Parameters
    ----------
    node_positions : np.ndarray
        Array of node positions (n_nodes, 2)
    edges : list
        List of edge connections between nodes
    """

    def __init__(self, node_positions: np.ndarray, edges: List[List[int]]):
        self.node_positions = np.asarray(node_positions)
        self.edges = edges
        self.n_nodes = len(node_positions)

        # Create adjacency matrix
        self.adjacency_matrix = self._create_adjacency_matrix()

        # Calculate distances between connected nodes
        self.edge_distances = self._calculate_edge_distances()

        # Calculate cumulative distances for linearization
        self.cumulative_distances = self._calculate_cumulative_distances()

    def _create_adjacency_matrix(self) -> csr_matrix:
        """Create sparse adjacency matrix from edges."""
        row_indices = []
        col_indices = []

        for edge in self.edges:
            if len(edge) >= 2:
                for i in range(len(edge) - 1):
                    row_indices.extend([edge[i], edge[i + 1]])
                    col_indices.extend([edge[i + 1], edge[i]])

        data = np.ones(len(row_indices))
        return csr_matrix(
            (data, (row_indices, col_indices)), shape=(self.n_nodes, self.n_nodes)
        )

    def _calculate_edge_distances(self) -> dict:
        """Calculate distances between connected nodes."""
        distances = {}
        for edge in self.edges:
            if len(edge) >= 2:
                for i in range(len(edge) - 1):
                    node1, node2 = edge[i], edge[i + 1]
                    dist = np.linalg.norm(
                        self.node_positions[node1] - self.node_positions[node2]
                    )
                    distances[(node1, node2)] = dist
                    distances[(node2, node1)] = dist
        return distances

    def _calculate_cumulative_distances(self) -> np.ndarray:
        """Calculate cumulative distances along the track."""
        # Find the main path through the track
        # For simplicity, we'll use the first edge as the starting point
        if not self.edges or len(self.edges[0]) < 2:
            return np.zeros(self.n_nodes)

        cumulative = np.zeros(self.n_nodes)
        visited = set()

        # Start from the first edge
        current_edge = self.edges[0]
        if len(current_edge) >= 2:
            for i in range(len(current_edge) - 1):
                node1, node2 = current_edge[i], current_edge[i + 1]
                if node1 not in visited:
                    visited.add(node1)
                if node2 not in visited:
                    visited.add(node2)
                    cumulative[node2] = cumulative[node1] + self.edge_distances.get(
                        (node1, node2), 0
                    )

        return cumulative


def project_position_to_track(
    position: np.ndarray, track_graph: TrackGraph
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 2D positions onto the track graph.

    Parameters
    ----------
    position : np.ndarray
        Array of 2D positions (n_positions, 2)
    track_graph : TrackGraph
        Track graph object

    Returns
    -------
    linear_position : np.ndarray
        Linearized positions along the track
    track_segment_id : np.ndarray
        Track segment IDs for each position
    projected_position : np.ndarray
        Projected 2D positions on the track
    """
    n_positions = position.shape[0]
    linear_position = np.full(n_positions, np.nan)
    track_segment_id = np.full(n_positions, -1, dtype=int)
    projected_position = np.full((n_positions, 2), np.nan)

    for i, pos in enumerate(position):
        # Find closest node
        distances_to_nodes = np.linalg.norm(track_graph.node_positions - pos, axis=1)
        closest_node = np.argmin(distances_to_nodes)

        # Find closest edge
        min_distance = np.inf
        best_segment = -1
        best_projection = None

        for edge_idx, edge in enumerate(track_graph.edges):
            if len(edge) >= 2:
                for j in range(len(edge) - 1):
                    node1, node2 = edge[j], edge[j + 1]
                    p1 = track_graph.node_positions[node1]
                    p2 = track_graph.node_positions[node2]

                    # Project point onto line segment
                    v = p2 - p1
                    u = pos - p1
                    t = np.dot(u, v) / np.dot(v, v)
                    t = np.clip(t, 0, 1)

                    projection = p1 + t * v
                    distance = np.linalg.norm(pos - projection)

                    if distance < min_distance:
                        min_distance = distance
                        best_segment = edge_idx
                        best_projection = projection

        if best_segment >= 0 and best_projection is not None:
            # Calculate linear position
            edge = track_graph.edges[best_segment]
            for j in range(len(edge) - 1):
                node1, node2 = edge[j], edge[j + 1]
                p1 = track_graph.node_positions[node1]
                p2 = track_graph.node_positions[node2]

                # Check if projection is on this segment
                v = p2 - p1
                u = best_projection - p1
                t = np.dot(u, v) / np.dot(v, v)

                if 0 <= t <= 1:
                    # Linear position is cumulative distance to node1 + distance along segment
                    linear_position[i] = track_graph.cumulative_distances[
                        node1
                    ] + t * np.linalg.norm(v)
                    track_segment_id[i] = best_segment
                    projected_position[i] = best_projection
                    break

    return linear_position, track_segment_id, projected_position


def get_linearized_position(
    position: np.ndarray,
    track_graph: TrackGraph,
    edge_order: Optional[List[List[int]]] = None,
    use_HMM: bool = False,
) -> pd.DataFrame:
    """
    Get linearized position from 2D positions using track graph.

    Parameters
    ----------
    position : np.ndarray
        Array of 2D positions (n_positions, 2)
    track_graph : TrackGraph
        Track graph object
    edge_order : list, optional
        Order of edges to use (for compatibility)
    use_HMM : bool, optional
        Whether to use HMM (not implemented in this version)

    Returns
    -------
    pd.DataFrame
        DataFrame with linearized position, track segment ID, and projected positions
    """
    linear_position, track_segment_id, projected_position = project_position_to_track(
        position, track_graph
    )

    return pd.DataFrame(
        {
            "linear_position": linear_position,
            "track_segment_id": track_segment_id,
            "projected_x_position": projected_position[:, 0],
            "projected_y_position": projected_position[:, 1],
        }
    )


def make_track_graph(node_positions: np.ndarray, edges: List[List[int]]) -> TrackGraph:
    """
    Create a track graph from node positions and edges.

    Parameters
    ----------
    node_positions : np.ndarray
        Array of node positions (n_nodes, 2)
    edges : list
        List of edge connections between nodes

    Returns
    -------
    TrackGraph
        Track graph object
    """
    return TrackGraph(node_positions, edges)


class NodePicker:
    """
    Interactive creation of track graph by looking at video frames.

    Parameters
    ----------
    ax : plt.Axes, optional
        The matplotlib axes to draw on, by default None.
    basepath : str, optional
        The base path for saving data, by default None.
    node_color : str, optional
        The color of the nodes, by default "#177ee6".
    node_size : int, optional
        The size of the nodes, by default 100.
    epoch : int, optional
        The epoch number, by default None.
    interval : Tuple[float, float], optional
        Time interval to process, by default None.

    Attributes
    ----------
    ax : plt.Axes
        The matplotlib axes to draw on.
    canvas : plt.FigureCanvas
        The matplotlib figure canvas.
    cid : int
        The connection id for the event handler.
    _nodes : list
        The list of node positions.
    node_color : str
        The color of the nodes.
    _nodes_plot : plt.scatter
        The scatter plot of the nodes.
    edges : list
        The list of edges.
    basepath : str
        The base path for saving data.
    epoch : int
        The epoch number.
    use_HMM : bool
        Whether to use the hidden markov model.

    Methods
    -------
    node_positions
        Get the positions of the nodes.
    connect
        Connect the event handlers.
    disconnect
        Disconnect the event handlers.
    process_key
        Process key press events.
    click_event
        Process mouse click events.
    redraw
        Redraw the nodes and edges.
    remove_point
        Remove a point from the nodes.
    clear
        Clear all nodes and edges.
    format_and_save
        Format the data and save it to disk.
    save_nodes_edges
        Save the nodes and edges to a pickle file.
    save_nodes_edges_to_behavior
        Store nodes and edges into behavior file.

    Examples
    --------
    # in command line
    >>> python linearization.py path/to/session

    # for a specific epoch
    >>> python linearization.py path/to/session 1

    # for a specific interval
    >>> python linearization.py path/to/session 0 100
    """

    def __init__(
        self,
        ax: Optional[plt.Axes] = None,
        basepath: Optional[str] = None,
        node_color: str = "#177ee6",
        node_size: int = 100,
        epoch: Optional[int] = None,
        interval: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize the NodePicker.

        Parameters
        ----------
        ax : plt.Axes, optional
            The matplotlib axes to draw on, by default None.
        basepath : str, optional
            The base path for saving data, by default None.
        node_color : str, optional
            The color of the nodes, by default "#177ee6".
        node_size : int, optional
            The size of the nodes, by default 100.
        epoch : int, optional
            The epoch number, by default None.
        interval : Tuple[float, float], optional
            Time interval to process, by default None.
        """
        if ax is None:
            ax = plt.gca()
        self.ax = ax
        self.canvas = ax.get_figure().canvas
        self.cid = None
        self._nodes = []
        self.node_color = node_color
        self._nodes_plot = ax.scatter([], [], zorder=5, s=node_size, color=node_color)
        self.edges = [[]]
        self.basepath = basepath
        self.epoch = epoch
        self.interval = interval
        self.use_HMM = False

        if self.epoch is not None:
            self.epoch = int(self.epoch)

        ax.set_title(
            "Left click to place node.\nRight click to remove node."
            "\nShift+Left click to clear nodes.\nCntrl+Left click two nodes to place an edge"
            "\nEnter to save and exit.",
            fontsize=8,
        )

        self.canvas.draw()
        self.connect()

    @property
    def node_positions(self) -> np.ndarray:
        """
        Get the positions of the nodes.

        Returns
        -------
        np.ndarray
            An array of node positions.
        """
        return np.asarray(self._nodes)

    def connect(self) -> None:
        """Connect the event handlers."""
        print("Connecting to events")
        if self.cid is None:
            self.cid = self.canvas.mpl_connect("button_press_event", self.click_event)
            self.canvas.mpl_connect("key_press_event", self.process_key)
            print("Mouse click event connected!")

    def disconnect(self) -> None:
        """Disconnect the event handlers."""
        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
            self.cid = None

    def process_key(self, event: Any) -> None:
        """
        Process key press events.

        Parameters
        ----------
        event : Any
            The key press event.
        """
        if event.key == "enter":
            self.format_and_save()

    def click_event(self, event: Any) -> None:
        """
        Process mouse click events.

        Parameters
        ----------
        event : Any
            The mouse click event.
        """
        print(
            f"Mouse clicked at: {event.xdata}, {event.ydata}, button: {event.button}, key: {event.key}"
        )
        if not event.inaxes:
            return

        if event.key is None:  # Regular mouse clicks
            if event.button == 1:  # Left click
                self._nodes.append((event.xdata, event.ydata))
            elif event.button == 3:  # Right click
                self.remove_point((event.xdata, event.ydata))

        elif event.key == "shift" and event.button == 1:  # Shift + Left click
            self.clear()

        elif (
            event.key == "control" and event.button == 1
        ):  # Ctrl + Left click (Edge creation)
            if len(self._nodes) == 0:
                return
            point = (event.xdata, event.ydata)
            distance_to_nodes = np.linalg.norm(self.node_positions - point, axis=1)
            closest_node_ind = np.argmin(distance_to_nodes)
            if len(self.edges[-1]) < 2:
                self.edges[-1].append(closest_node_ind)
            else:
                self.edges.append([closest_node_ind])

        elif event.key == "enter":  # Pressing Enter
            self.format_and_save()

        self.redraw()

    def redraw(self) -> None:
        """Redraw the nodes and edges."""
        # Draw Node Circles
        if len(self.node_positions) > 0:
            self._nodes_plot.set_offsets(self.node_positions)
        else:
            self._nodes_plot.set_offsets([])

        # Draw Node Numbers
        for ind, (x, y) in enumerate(self.node_positions):
            self.ax.text(
                x,
                y,
                ind,
                zorder=6,
                fontsize=10,
                horizontalalignment="center",
                verticalalignment="center",
                clip_on=True,
                bbox=None,
                transform=self.ax.transData,
            )
        # Draw Edges
        for edge in self.edges:
            if len(edge) > 1:
                x1, y1 = self.node_positions[edge[0]]
                x2, y2 = self.node_positions[edge[1]]
                self.ax.plot(
                    [x1, x2], [y1, y2], color="#1f8e4f", linewidth=3, zorder=1000
                )
        self.canvas.draw()

    def remove_point(self, point: Tuple[float, float]) -> None:
        """
        Remove a point from the nodes.

        Parameters
        ----------
        point : Tuple[float, float]
            The point to remove.
        """
        if len(self._nodes) > 0:
            distance_to_nodes = np.linalg.norm(self.node_positions - point, axis=1)
            closest_node_ind = np.argmin(distance_to_nodes)
            self._nodes.pop(closest_node_ind)

    def clear(self) -> None:
        """Clear all nodes and edges."""
        self._nodes = []
        self.edges = [[]]
        self.redraw()

    def format_and_save(self) -> None:
        """Format the data and save it to disk."""
        behave_df = load_animal_behavior(self.basepath)

        if self.epoch is not None:
            epochs = load_epoch(self.basepath)

            cur_epoch = (
                ~np.isnan(behave_df.x)
                & (behave_df.time >= epochs.iloc[self.epoch].startTime)
                & (behave_df.time <= epochs.iloc[self.epoch].stopTime)
            )
        elif self.interval is not None:
            cur_epoch = (
                ~np.isnan(behave_df.x)
                & (behave_df.time >= self.interval[0])
                & (behave_df.time <= self.interval[1])
            )
        else:
            cur_epoch = ~np.isnan(behave_df.x)

        print("running linearization...")
        track_graph = make_track_graph(self.node_positions, self.edges)

        position = np.vstack(
            [behave_df[cur_epoch].x.values, behave_df[cur_epoch].y.values]
        ).T

        position_df = get_linearized_position(
            position=position,
            track_graph=track_graph,
            edge_order=self.edges,
            use_HMM=self.use_HMM,
        )

        print("saving to disk...")
        behave_df.loc[cur_epoch, "linearized"] = position_df.linear_position.values
        behave_df.loc[cur_epoch, "states"] = position_df.track_segment_id.values
        behave_df.loc[cur_epoch, "projected_x_position"] = (
            position_df.projected_x_position.values
        )
        behave_df.loc[cur_epoch, "projected_y_position"] = (
            position_df.projected_y_position.values
        )

        filename = os.path.join(
            self.basepath, os.path.basename(self.basepath) + ".animal.behavior.mat"
        )

        data = loadmat(filename, simplify_cells=True)

        data["behavior"]["position"]["linearized"] = behave_df.linearized.values
        data["behavior"]["states"] = behave_df.states.values
        data["behavior"]["position"]["projected_x"] = (
            behave_df.projected_x_position.values
        )
        data["behavior"]["position"]["projected_y"] = (
            behave_df.projected_y_position.values
        )

        # store nodes and edges within behavior file
        data = self.save_nodes_edges_to_behavior(data, behave_df)

        savemat(filename, data, long_field_names=True)

        self.save_nodes_edges()
        self.disconnect()
        plt.close()

    def save_nodes_edges(self) -> None:
        """Save the nodes and edges to a pickle file."""
        results = {"node_positions": self.node_positions, "edges": self.edges}
        save_file = os.path.join(self.basepath, "linearization_nodes_edges.pkl")
        with open(save_file, "wb") as f:
            pickle.dump(results, f)

    def save_nodes_edges_to_behavior(self, data: dict, behave_df: pd.DataFrame) -> dict:
        """
        Store nodes and edges into behavior file.
        Searches to find epochs with valid linearized coords.
        Nodes and edges are stored within behavior.epochs{n}.{node_positions and edges}

        Parameters
        ----------
        data : dict
            The behavior data dictionary.
        behave_df : pd.DataFrame
            The DataFrame containing behavior data.

        Returns
        -------
        dict
            The updated behavior data dictionary.
        """
        if self.epoch is None and self.interval is None:
            # load epochs
            epochs = load_epoch(self.basepath)
            # iter over each epoch
            for epoch_i, ep in enumerate(epochs.itertuples()):
                # locate index for given epoch
                idx = behave_df.time.between(ep.startTime, ep.stopTime)
                # if linearized is not all nan, add nodes and edges
                if not all(np.isnan(behave_df[idx].linearized)) & (
                    behave_df[idx].shape[0] != 0
                ):
                    # adding nodes and edges
                    data["behavior"]["epochs"][epoch_i]["node_positions"] = (
                        self.node_positions
                    )
                    data["behavior"]["epochs"][epoch_i]["edges"] = self.edges
        elif self.interval is not None:
            # if interval was used, add nodes and edges just the epochs within that interval
            epochs = load_epoch(self.basepath)
            for epoch_i, ep in enumerate(epochs.itertuples()):
                # amount of overlap between interval and epoch
                start_overlap = max(self.interval[0], ep.startTime)
                end_overlap = min(self.interval[1], ep.stopTime)
                overlap = max(0, end_overlap - start_overlap)

                # if overlap is greater than 1 second, add nodes and edges
                if overlap > 1:
                    data["behavior"]["epochs"][epoch_i]["node_positions"] = (
                        self.node_positions
                    )
                    data["behavior"]["epochs"][epoch_i]["edges"] = self.edges
        else:
            # if epoch was used, add nodes and edges just that that epoch
            data["behavior"]["epochs"][self.epoch]["node_positions"] = (
                self.node_positions
            )
            data["behavior"]["epochs"][self.epoch]["edges"] = self.edges

        return data


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


def load_epoch(basepath: str) -> pd.DataFrame:
    """
    Load epoch info from cell explorer basename.session and store in a DataFrame.

    Parameters
    ----------
    basepath : str
        The base path where the .session.mat file is located.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the epoch information.
    """
    filename = os.path.join(basepath, os.path.basename(basepath) + ".session.mat")
    data = loadmat(filename, simplify_cells=True)
    try:
        return pd.DataFrame(data["session"]["epochs"])
    except Exception:
        return pd.DataFrame([data["session"]["epochs"]])


def run(
    basepath: str,
    epoch: Optional[int] = None,
    interval: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Run the linearization pipeline.

    Parameters
    ----------
    basepath : str
        The base path where the data files are located.
    epoch : int, optional
        The epoch number to process, by default None.
    interval : Tuple[float, float], optional
        Time interval to process, by default None.

    Returns
    -------
    None
    """
    plt.close("all")
    print("here is the file,", basepath)

    with plt.style.context("dark_background"):
        plt.ioff()

        _, ax = plt.subplots(figsize=(5, 5))

        behave_df = load_animal_behavior(basepath)

        if epoch is not None:
            epochs = load_epoch(basepath)

            behave_df = behave_df[
                behave_df["time"].between(
                    epochs.iloc[epoch].startTime, epochs.iloc[epoch].stopTime
                )
            ]
        elif interval is not None:
            behave_df = behave_df[behave_df["time"].between(interval[0], interval[1])]

        ax.scatter(behave_df.x, behave_df.y, color="white", s=0.5, alpha=0.5)
        ax.axis("equal")
        ax.set_axisbelow(True)
        ax.yaxis.grid(color="gray", linestyle="dashed")
        ax.xaxis.grid(color="gray", linestyle="dashed")
        ax.set_ylabel("y (cm)")
        ax.set_xlabel("x (cm)")

        picker = NodePicker(ax=ax, basepath=basepath, epoch=epoch, interval=interval)
        picker.connect()  # Ensure connection

        plt.show(block=True)


if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) == 2:
        run(sys.argv[1])
    elif len(sys.argv) == 3:
        run(sys.argv[1], epoch=int(sys.argv[2]))
    elif len(sys.argv) == 4:
        run(sys.argv[1], interval=(float(sys.argv[2]), float(sys.argv[3])))
