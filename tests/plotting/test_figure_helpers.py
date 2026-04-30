import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import neuro_py.plotting.figure_helpers as figure_helpers
from neuro_py.plotting.figure_helpers import (
    _build_scaled_image_html,
    _HTMLDisplay,
    figure_scale,
    paired_lines,
    scale_figsize,
    set_plotting_defaults,
    show_scaled,
)


def test_scale_figsize_scales_dimensions():
    """scale_figsize should multiply width and height by the scale."""
    assert scale_figsize((4.0, 2.5), 1.75) == pytest.approx((7.0, 4.375))


@pytest.mark.parametrize("scale", [0, -1, -0.5])
def test_scale_figsize_invalid_scale_raises(scale: float):
    """scale_figsize should reject non-positive scale values."""
    with pytest.raises(ValueError, match="greater than 0"):
        scale_figsize((4.0, 2.5), scale)


def test_figure_scale_scales_targeted_rcparams():
    """figure_scale should scale curated size-related rcParams."""
    original_font = plt.rcParams["font.size"]
    original_linewidth = plt.rcParams["lines.linewidth"]
    original_marker = plt.rcParams["lines.markersize"]
    original_tick_width = plt.rcParams["xtick.major.width"]

    with figure_scale(1.5):
        assert plt.rcParams["font.size"] == pytest.approx(original_font * 1.5)
        assert plt.rcParams["lines.linewidth"] == pytest.approx(
            original_linewidth * 1.5
        )
        assert plt.rcParams["lines.markersize"] == pytest.approx(original_marker * 1.5)
        assert plt.rcParams["xtick.major.width"] == pytest.approx(
            original_tick_width * 1.5
        )


def test_figure_scale_restores_rcparams_after_exit():
    """figure_scale should restore rcParams after normal context exit."""
    original_font = plt.rcParams["font.size"]

    with figure_scale(2.0):
        assert plt.rcParams["font.size"] == pytest.approx(original_font * 2.0)

    assert plt.rcParams["font.size"] == pytest.approx(original_font)


def test_figure_scale_restores_rcparams_after_exception():
    """figure_scale should restore rcParams even if the context errors."""
    original_font = plt.rcParams["font.size"]

    with pytest.raises(RuntimeError, match="boom"):
        with figure_scale(1.25):
            assert plt.rcParams["font.size"] == pytest.approx(original_font * 1.25)
            raise RuntimeError("boom")

    assert plt.rcParams["font.size"] == pytest.approx(original_font)


def test_figure_scale_respects_current_style_state():
    """figure_scale should scale the active style values rather than defaults."""
    plt.rcdefaults()
    try:
        set_plotting_defaults("nature")
        original_font = plt.rcParams["font.size"]
        original_tick_size = plt.rcParams["xtick.major.size"]

        with figure_scale(1.75):
            assert plt.rcParams["font.size"] == pytest.approx(original_font * 1.75)
            assert plt.rcParams["xtick.major.size"] == pytest.approx(
                original_tick_size * 1.75
            )
    finally:
        plt.rcdefaults()


def test_figure_scale_leaves_non_numeric_rcparams_unchanged():
    """figure_scale should not alter string-valued rcParams."""
    original_family = list(plt.rcParams["font.family"])

    with figure_scale(1.5):
        assert list(plt.rcParams["font.family"]) == original_family


@pytest.mark.parametrize("scale", [0, -1, -0.5])
def test_figure_scale_invalid_scale_raises(scale: float):
    """figure_scale should reject non-positive scale values."""
    with pytest.raises(ValueError, match="greater than 0"):
        with figure_scale(scale):
            pass


def test_show_scaled_rejects_invalid_scale():
    """show_scaled should reject non-positive scale values."""
    fig, _ = plt.subplots()
    with pytest.raises(ValueError, match="greater than 0"):
        show_scaled(fig, scale=0, backend="jupyter")
    plt.close(fig)


def test_show_scaled_rejects_invalid_format():
    """show_scaled should reject unsupported formats."""
    fig, _ = plt.subplots()
    with pytest.raises(ValueError, match="format must be 'png'"):
        show_scaled(fig, format="svg", backend="jupyter")
    plt.close(fig)


def test_show_scaled_rejects_invalid_backend():
    """show_scaled should reject unsupported backends."""
    fig, _ = plt.subplots()
    with pytest.raises(ValueError, match="backend must be 'auto', 'jupyter', or 'marimo'"):
        show_scaled(fig, backend="qt")
    plt.close(fig)


def test_show_scaled_jupyter_returns_html():
    """show_scaled should return an HTML wrapper for the Jupyter backend."""
    fig, _ = plt.subplots(figsize=(4.0, 2.0), dpi=100)
    display_obj = show_scaled(fig, scale=1.5, backend="jupyter")

    assert isinstance(display_obj, _HTMLDisplay)
    assert 'width: 600px' in display_obj.data
    assert "data:image/png;base64," in display_obj.data
    plt.close(fig)


def test_show_scaled_jupyter_displays_immediately(monkeypatch: pytest.MonkeyPatch):
    """show_scaled should use the IPython display hook when it is available."""
    fig, _ = plt.subplots(figsize=(4.0, 2.0), dpi=100)
    displayed: list[_HTMLDisplay] = []

    monkeypatch.setattr(
        figure_helpers,
        "_display_in_ipython",
        lambda display_obj: displayed.append(display_obj) or True,
    )

    display_result = show_scaled(fig, scale=1.5, backend="jupyter")

    assert len(displayed) == 1
    assert displayed[0].data.startswith('<img src="data:image/png;base64,')
    assert display_result is None


def test_show_scaled_jupyter_closes_pyplot_figure_after_display(
    monkeypatch: pytest.MonkeyPatch,
):
    """show_scaled should close the pyplot-managed figure after Jupyter display."""
    fig, _ = plt.subplots(figsize=(4.0, 2.0), dpi=100)
    figure_number = fig.number

    monkeypatch.setattr(figure_helpers, "_display_in_ipython", lambda display_obj: True)

    show_scaled(fig, scale=1.5, backend="jupyter")

    assert figure_number not in plt.get_fignums()
    assert fig.get_size_inches() == pytest.approx((4.0, 2.0))


def test_show_scaled_does_not_mutate_figure_size_or_dpi():
    """show_scaled should leave the original figure dimensions unchanged."""
    fig, ax = plt.subplots(figsize=(4.0, 2.0), dpi=100)
    original_size = tuple(fig.get_size_inches())
    original_dpi = fig.dpi

    ax.plot([0, 1], [0, 1])
    show_scaled(fig, scale=2.0, backend="jupyter")

    assert tuple(fig.get_size_inches()) == pytest.approx(original_size)
    assert fig.dpi == pytest.approx(original_dpi)
    plt.close(fig)


def test_show_scaled_preserves_savefig_dimensions():
    """show_scaled should not alter subsequent savefig dimensions."""
    fig, ax = plt.subplots(figsize=(4.0, 2.0), dpi=100)
    ax.plot([0, 1], [0, 1])
    show_scaled(fig, scale=2.0, backend="jupyter")

    html = _build_scaled_image_html(fig, scale=1.0, dpi=100)

    assert 'width: 400px' in html
    plt.close(fig)


def test_show_scaled_auto_resolves_active_jupyter(monkeypatch: pytest.MonkeyPatch):
    """show_scaled should prefer Jupyter when an active kernel is detected."""
    fig, _ = plt.subplots(figsize=(4.0, 2.0), dpi=100)

    monkeypatch.setattr(figure_helpers, "_in_active_ipython_session", lambda: True)
    monkeypatch.setattr(figure_helpers, "_display_in_ipython", lambda obj: True)

    display_result = show_scaled(fig, backend="auto")

    assert display_result is None


def test_show_scaled_auto_does_not_choose_marimo_when_only_installed(
    monkeypatch: pytest.MonkeyPatch,
):
    """show_scaled auto backend should not pick marimo based on installation alone."""
    fig, _ = plt.subplots()

    monkeypatch.setattr(figure_helpers, "_in_active_ipython_session", lambda: False)
    monkeypatch.setattr(figure_helpers, "_in_active_marimo_session", lambda: False)

    with pytest.raises(RuntimeError, match="could not detect a supported notebook backend"):
        show_scaled(fig, backend="auto")

    plt.close(fig)


def test_show_scaled_auto_without_backend_raises(monkeypatch: pytest.MonkeyPatch):
    """show_scaled should raise when auto backend cannot be resolved."""
    fig, _ = plt.subplots()

    monkeypatch.setattr(figure_helpers, "_in_active_ipython_session", lambda: False)
    monkeypatch.setattr(figure_helpers, "_in_active_marimo_session", lambda: False)

    with pytest.raises(RuntimeError, match="could not detect a supported notebook backend"):
        show_scaled(fig, backend="auto")

    plt.close(fig)


def test_show_scaled_explicit_jupyter_without_ipython_still_returns_html(
    monkeypatch: pytest.MonkeyPatch,
):
    """Explicit Jupyter backend should still return HTML when no display hook exists."""
    fig, _ = plt.subplots(figsize=(4.0, 2.0), dpi=100)

    monkeypatch.setattr(figure_helpers, "_in_active_ipython_session", lambda: False)

    display_obj = show_scaled(fig, scale=1.5, backend="jupyter")

    assert isinstance(display_obj, _HTMLDisplay)
    plt.close(fig)


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


def test_paired_lines_style_without_style_map():
    """Style parameter works without style_map - uses default line styles."""
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
        # Note: no style_map provided - should auto-assign default styles
        ax=ax,
    )

    assert isinstance(result_ax, plt.Axes)
    # Should have lines and they should have different line styles
    assert len(ax.lines) > 0
    line_styles = {line.get_linestyle() for line in ax.lines}
    # Default styles are: "-", "--", "-.", ":" so should have at least 3 different styles
    assert len(line_styles) >= 3
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


def test_paired_lines_hue_without_units():
    """Test hue without units parameter (lines 629-645)."""
    # When hue is provided without units, the function groups by x only,
    # and connects lines across different hue values within each x-category.
    data = pd.DataFrame(
        {
            "trial": ["A", "A", "A", "B", "B", "B"],
            "value": [1.0, 1.5, 1.2, 2.0, 2.5, 2.3],
            "device": ["X", "Y", "Z", "X", "Y", "Z"],
        }
    )

    fig, ax = plt.subplots()
    result_ax = paired_lines(data, x="trial", y="value", hue="device", ax=ax)

    assert isinstance(result_ax, plt.Axes)
    # With 3 devices per trial and 2 trials, we expect 2 line groups
    # (one for each trial, connecting the 3 device values)
    # Each group has 2 lines connecting 3 points: X-Y, Y-Z
    assert len(ax.lines) == 4
    plt.close(fig)
