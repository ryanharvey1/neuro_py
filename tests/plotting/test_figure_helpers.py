import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from neuro_py.plotting.figure_helpers import paired_lines


def test_paired_lines_basic():
    """Test basic paired_lines functionality with hue and units."""
    data = pd.DataFrame(
        {
            "trial_type": ["trial1", "trial1", "trial1", "trial1", "trial1", "trial1"],
            "condition": ["A", "B", "A", "B", "A", "B"],
            "value": [1, 2, 1.5, 2.5, 2, 3],
            "subject": ["S1", "S1", "S2", "S2", "S3", "S3"],
        }
    )

    fig, ax = plt.subplots()
    result_ax = paired_lines(
        data, x="trial_type", y="value", hue="condition", units="subject", ax=ax
    )

    assert isinstance(result_ax, plt.Axes)
    assert result_ax is ax
    # Check that lines were drawn (3 subjects = 3 lines)
    assert len(ax.lines) == 3
    plt.close(fig)


def test_paired_lines_no_hue():
    """Test paired_lines without hue parameter."""
    data = pd.DataFrame(
        {
            "condition": ["A", "B", "A", "B"],
            "value": [1, 2, 1.5, 2.5],
            "subject": ["S1", "S1", "S2", "S2"],
        }
    )

    fig, ax = plt.subplots()
    result_ax = paired_lines(data, x="condition", y="value", units="subject", ax=ax)

    assert isinstance(result_ax, plt.Axes)
    # Two subjects, each with A->B connection
    assert len(ax.lines) == 2
    plt.close(fig)


def test_paired_lines_no_hue_no_units():
    """Test paired_lines without hue or units connects across x categories."""
    data = pd.DataFrame(
        {
            "condition": ["A", "B", "C"],
            "value": [1.0, 2.0, 3.0],
        }
    )

    fig, ax = plt.subplots()
    result_ax = paired_lines(data, x="condition", y="value", ax=ax)

    assert isinstance(result_ax, plt.Axes)
    # One series across three x-categories yields two line segments
    assert len(ax.lines) == 2
    plt.close(fig)


def test_paired_lines_custom_color():
    """Test paired_lines with custom color."""
    data = pd.DataFrame(
        {
            "trial_type": ["trial1", "trial1", "trial1", "trial1"],
            "condition": ["A", "B", "A", "B"],
            "value": [1, 2, 1.5, 2.5],
            "subject": ["S1", "S1", "S2", "S2"],
        }
    )

    fig, ax = plt.subplots()
    result_ax = paired_lines(
        data,
        x="trial_type",
        y="value",
        hue="condition",
        units="subject",
        color="red",
        ax=ax,
    )

    assert isinstance(result_ax, plt.Axes)
    # Check that lines have the specified color
    for line in ax.lines:
        assert line.get_color() == "red"
    plt.close(fig)


def test_paired_lines_palette():
    """Test paired_lines with palette parameter."""
    data = pd.DataFrame(
        {
            "trial_type": ["trial1", "trial1", "trial1", "trial1", "trial1", "trial1"],
            "condition": ["A", "B", "A", "B", "A", "B"],
            "value": [1, 2, 1.5, 2.5, 2, 3],
            "subject": ["S1", "S1", "S2", "S2", "S3", "S3"],
        }
    )

    fig, ax = plt.subplots()
    palette = {"S1": "red", "S2": "blue", "S3": "green"}
    result_ax = paired_lines(
        data,
        x="trial_type",
        y="value",
        hue="condition",
        units="subject",
        palette=palette,
        ax=ax,
    )

    assert isinstance(result_ax, plt.Axes)
    assert len(ax.lines) == 3
    plt.close(fig)


def test_paired_lines_palette_list():
    """Test paired_lines with palette as a list."""
    data = pd.DataFrame(
        {
            "trial_type": ["trial1", "trial1", "trial1", "trial1"],
            "condition": ["A", "B", "A", "B"],
            "value": [1, 2, 1.5, 2.5],
            "subject": ["S1", "S1", "S2", "S2"],
        }
    )

    fig, ax = plt.subplots()
    palette = ["red", "blue"]
    result_ax = paired_lines(
        data,
        x="trial_type",
        y="value",
        hue="condition",
        units="subject",
        palette=palette,
        ax=ax,
    )

    assert isinstance(result_ax, plt.Axes)
    plt.close(fig)


def test_paired_lines_palette_seaborn():
    """Test paired_lines with seaborn palette name."""
    data = pd.DataFrame(
        {
            "trial_type": ["trial1", "trial1", "trial1", "trial1"],
            "condition": ["A", "B", "A", "B"],
            "value": [1, 2, 1.5, 2.5],
            "subject": ["S1", "S1", "S2", "S2"],
        }
    )

    fig, ax = plt.subplots()
    result_ax = paired_lines(
        data,
        x="trial_type",
        y="value",
        hue="condition",
        units="subject",
        palette="Set2",
        ax=ax,
    )

    assert isinstance(result_ax, plt.Axes)
    plt.close(fig)


def test_paired_lines_custom_dodge_width():
    """Test paired_lines with custom dodge width."""
    data = pd.DataFrame(
        {
            "trial_type": ["trial1", "trial1", "trial1", "trial1"],
            "condition": ["A", "B", "A", "B"],
            "value": [1, 2, 1.5, 2.5],
            "subject": ["S1", "S1", "S2", "S2"],
        }
    )

    fig, ax = plt.subplots()
    result_ax = paired_lines(
        data,
        x="trial_type",
        y="value",
        hue="condition",
        units="subject",
        dodge_width=0.5,
        ax=ax,
    )

    assert isinstance(result_ax, plt.Axes)
    plt.close(fig)


def test_paired_lines_kwargs():
    """Test paired_lines with additional matplotlib kwargs."""
    data = pd.DataFrame(
        {
            "trial_type": ["trial1", "trial1", "trial1", "trial1"],
            "condition": ["A", "B", "A", "B"],
            "value": [1, 2, 1.5, 2.5],
            "subject": ["S1", "S1", "S2", "S2"],
        }
    )

    fig, ax = plt.subplots()
    result_ax = paired_lines(
        data,
        x="trial_type",
        y="value",
        hue="condition",
        units="subject",
        linestyle="--",
        marker="o",
        ax=ax,
    )

    assert isinstance(result_ax, plt.Axes)
    # Check that linestyle was applied
    for line in ax.lines:
        assert line.get_linestyle() == "--"
        assert line.get_marker() == "o"
    plt.close(fig)


def test_paired_lines_sets_labels_when_missing():
    """Auto-set x/y labels when axes are unlabeled and set_labels is True."""
    data = pd.DataFrame(
        {
            "trial_type": ["trial1", "trial1"],
            "condition": ["A", "B"],
            "value": [1, 2],
            "subject": ["S1", "S1"],
        }
    )

    fig, ax = plt.subplots()
    # Clear labels to simulate blank axes
    ax.set_xlabel("")
    ax.set_ylabel("")

    result_ax = paired_lines(
        data,
        x="trial_type",
        y="value",
        hue="condition",
        units="subject",
        ax=ax,
    )

    assert isinstance(result_ax, plt.Axes)
    assert ax.get_xlabel() == "trial_type"
    assert ax.get_ylabel() == "value"
    plt.close(fig)


def test_paired_lines_custom_labels():
    """Pre-set x/y labels are preserved when set_labels is True."""
    data = pd.DataFrame(
        {
            "trial_type": ["trial1", "trial1"],
            "condition": ["A", "B"],
            "value": [1, 2],
            "subject": ["S1", "S1"],
        }
    )

    fig, ax = plt.subplots()
    # Pre-set custom labels; function should not override them
    ax.set_xlabel("Custom X")
    ax.set_ylabel("Custom Y")

    result_ax = paired_lines(
        data,
        x="trial_type",
        y="value",
        hue="condition",
        units="subject",
        ax=ax,
    )

    assert isinstance(result_ax, plt.Axes)
    assert ax.get_xlabel() == "Custom X"
    assert ax.get_ylabel() == "Custom Y"
    plt.close(fig)


def test_paired_lines_style_mapping_dict():
    """Style mapping via style column and style_map dict."""
    data = pd.DataFrame(
        {
            "trial_type": ["trial1", "trial1", "trial1", "trial1"],
            "condition": ["A", "B", "A", "B"],
            "value": [1, 2, 1.5, 2.5],
            "subject": ["S1", "S1", "S2", "S2"],
            "group": ["g1", "g1", "g2", "g2"],
        }
    )

    fig, ax = plt.subplots()
    result_ax = paired_lines(
        data,
        x="trial_type",
        y="value",
        hue="condition",
        units="subject",
        style="group",
        style_map={"g1": "--", "g2": ":"},
        ax=ax,
    )

    assert isinstance(result_ax, plt.Axes)
    # Each subject gets its style
    line_styles = {line.get_linestyle() for line in ax.lines}
    assert "--" in line_styles and ":" in line_styles
    plt.close(fig)


def test_paired_lines_style_mapping_list_cycle():
    """Style mapping cycles list when more levels than entries."""
    data = pd.DataFrame(
        {
            "trial_type": ["trial1"] * 6,
            "condition": ["A", "B", "A", "B", "A", "B"],
            "value": [1, 2, 1.5, 2.5, 1.2, 2.2],
            "subject": ["S1", "S1", "S2", "S2", "S3", "S3"],
            "group": ["g1", "g1", "g2", "g2", "g3", "g3"],
        }
    )

    fig, ax = plt.subplots()
    result_ax = paired_lines(
        data,
        x="trial_type",
        y="value",
        hue="condition",
        units="subject",
        style="group",
        style_map=["--", ":"],  # should cycle
        ax=ax,
    )

    assert isinstance(result_ax, plt.Axes)
    line_styles = {line.get_linestyle() for line in ax.lines}
    # Expect two provided styles present, cycling covers third
    assert "--" in line_styles and ":" in line_styles
    plt.close(fig)


def test_paired_lines_unknown_order_raises():
    """Unknown x category with order specified should raise ValueError."""
    data = pd.DataFrame(
        {
            "trial_type": ["T1", "T2"],
            "condition": ["A", "B"],
            "value": [1, 2],
            "subject": ["S1", "S1"],
        }
    )

    with pytest.raises(ValueError):
        paired_lines(
            data,
            x="trial_type",
            y="value",
            hue="condition",
            units="subject",
            order=["T1"],
        )


def test_paired_lines_unknown_hue_order_raises():
    """Unknown hue category with hue_order should raise ValueError."""
    data = pd.DataFrame(
        {
            "trial_type": ["trial1", "trial1"],
            "condition": ["A", "B"],
            "value": [1, 2],
            "subject": ["S1", "S1"],
        }
    )

    with pytest.raises(ValueError):
        paired_lines(
            data,
            x="trial_type",
            y="value",
            hue="condition",
            units="subject",
            hue_order=["A"],
        )


def test_paired_lines_order():
    """Test paired_lines with custom order parameter."""
    data = pd.DataFrame(
        {
            "trial_type": ["T1", "T2", "T3", "T1", "T2", "T3"],
            "condition": ["A", "A", "A", "B", "B", "B"],
            "value": [1, 2, 3, 1.5, 2.5, 3.5],
            "subject": ["S1", "S1", "S1", "S2", "S2", "S2"],
        }
    )

    fig, ax = plt.subplots()
    result_ax = paired_lines(
        data,
        x="trial_type",
        y="value",
        hue="condition",
        units="subject",
        order=["T3", "T2", "T1"],
        ax=ax,
    )

    assert isinstance(result_ax, plt.Axes)
    plt.close(fig)


def test_paired_lines_hue_order():
    """Test paired_lines with custom hue_order parameter."""
    data = pd.DataFrame(
        {
            "trial_type": ["trial1", "trial1", "trial1", "trial1", "trial1", "trial1"],
            "condition": ["A", "B", "C", "A", "B", "C"],
            "value": [1, 2, 3, 1.5, 2.5, 3.5],
            "subject": ["S1", "S1", "S1", "S2", "S2", "S2"],
        }
    )

    fig, ax = plt.subplots()
    result_ax = paired_lines(
        data,
        x="trial_type",
        y="value",
        hue="condition",
        units="subject",
        hue_order=["C", "B", "A"],
        ax=ax,
    )

    assert isinstance(result_ax, plt.Axes)
    plt.close(fig)


def test_paired_lines_no_dodge():
    """Test paired_lines with dodge=False."""
    data = pd.DataFrame(
        {
            "trial_type": ["trial1", "trial1", "trial1", "trial1"],
            "condition": ["A", "B", "A", "B"],
            "value": [1, 2, 1.5, 2.5],
            "subject": ["S1", "S1", "S2", "S2"],
        }
    )

    fig, ax = plt.subplots()
    result_ax = paired_lines(
        data,
        x="trial_type",
        y="value",
        hue="condition",
        units="subject",
        dodge=False,
        ax=ax,
    )

    assert isinstance(result_ax, plt.Axes)
    plt.close(fig)


def test_paired_lines_alpha_lw_zorder():
    """Test paired_lines with custom alpha, linewidth, and zorder."""
    data = pd.DataFrame(
        {
            "trial_type": ["trial1", "trial1", "trial1", "trial1"],
            "condition": ["A", "B", "A", "B"],
            "value": [1, 2, 1.5, 2.5],
            "subject": ["S1", "S1", "S2", "S2"],
        }
    )

    fig, ax = plt.subplots()
    result_ax = paired_lines(
        data,
        x="trial_type",
        y="value",
        hue="condition",
        units="subject",
        alpha=0.8,
        lw=2,
        zorder=5,
        ax=ax,
    )

    assert isinstance(result_ax, plt.Axes)
    # Check line properties
    for line in ax.lines:
        assert line.get_alpha() == 0.8
        assert line.get_linewidth() == 2
        assert line.get_zorder() == 5
    plt.close(fig)


def test_paired_lines_multiple_hue_values():
    """Test paired_lines with more than 2 hue values."""
    data = pd.DataFrame(
        {
            "trial_type": [
                "trial1",
                "trial1",
                "trial1",
                "trial1",
                "trial1",
                "trial1",
                "trial1",
                "trial1",
            ],
            "condition": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "value": [1, 2, 3, 4, 1.5, 2.5, 3.5, 4.5],
            "subject": ["S1", "S1", "S1", "S1", "S2", "S2", "S2", "S2"],
        }
    )

    fig, ax = plt.subplots()
    result_ax = paired_lines(
        data, x="trial_type", y="value", hue="condition", units="subject", ax=ax
    )

    assert isinstance(result_ax, plt.Axes)
    # Should have lines connecting A->B, B->C, C->D for each subject
    assert len(ax.lines) == 6  # 3 lines per subject * 2 subjects
    plt.close(fig)


def test_paired_lines_no_ax():
    """Test paired_lines without providing an axes object."""
    data = pd.DataFrame(
        {
            "trial_type": ["trial1", "trial1", "trial1", "trial1"],
            "condition": ["A", "B", "A", "B"],
            "value": [1, 2, 1.5, 2.5],
            "subject": ["S1", "S1", "S2", "S2"],
        }
    )

    result_ax = paired_lines(
        data, x="trial_type", y="value", hue="condition", units="subject"
    )

    assert isinstance(result_ax, plt.Axes)
    plt.close()


def test_paired_lines_duplicate_warning():
    """Warn when duplicate (x, units, hue) groups are present."""
    data = pd.DataFrame(
        {
            "trial_type": ["trial1", "trial1", "trial1", "trial1"],
            "condition": ["A", "A", "B", "B"],
            "value": [1.0, 1.1, 2.0, 2.1],
            "subject": ["S1", "S1", "S1", "S1"],
        }
    )

    fig, ax = plt.subplots()
    with pytest.warns(UserWarning, match="duplicate group"):
        result_ax = paired_lines(
            data,
            x="trial_type",
            y="value",
            hue="condition",
            units="subject",
            ax=ax,
        )

    assert isinstance(result_ax, plt.Axes)
    plt.close(fig)
