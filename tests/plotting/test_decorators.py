import matplotlib.pyplot as plt

from neuro_py.plotting.decorators import AngleAnnotation


def test_angle_annotation_uses_current_axes_by_default():
    fig, ax = plt.subplots()
    plt.sca(ax)

    annotation = AngleAnnotation((0, 0), (1, 0), (0, 1), text="90")

    assert annotation.ax is ax
    assert annotation in ax.patches
    assert annotation.text.get_text() == "90"
    plt.close(fig)
