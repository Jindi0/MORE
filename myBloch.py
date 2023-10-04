import numpy as np
from qiskit.visualization.bloch import Bloch, Arrow3D

'''
    myBlochSphere sets a specific color for states from different classes
'''
class myBloch(Bloch):
    def __init__(self, color_list=None, fig=None, axes=None, view=None, figsize=None, background=False):
        super().__init__(fig, axes, view, figsize, background)
        self._color_list = color_list
        
    def plot_vectors(self):
        """Plot vector"""
        # -X and Y data are switched for plotting purposes
        for k in range(len(self.vectors)):

            xs3d = self.vectors[k][1] * np.array([0, 1])
            ys3d = -self.vectors[k][0] * np.array([0, 1])
            zs3d = self.vectors[k][2] * np.array([0, 1])

            # color = self.vector_color[np.mod(k, len(self.vector_color))]
            # pink, light blue, orange, purple, yellow
            # green, dark blue, red, grey, dark green
            self.vector_color = ["#dc267f", "#648fff", "#fe6100", "#785ef0", "#ffb000",
            "#8fce00", "#0b5394", "#f44336", "#25CED1", "#38761d", "#577590"]
            color = self.vector_color[self._color_list[k]]

            if self.vector_style == "":
                # simple line style
                self.axes.plot(
                    xs3d, ys3d, zs3d, zs=0, zdir="z", label="Z", lw=self.vector_width, color=color
                )
            else:
                # decorated style, with arrow heads
                arr = Arrow3D(
                    xs3d,
                    ys3d,
                    zs3d,
                    mutation_scale=self.vector_mutation,
                    lw=self.vector_width,
                    arrowstyle=self.vector_style,
                    color=color,
                )

                self.axes.add_artist(arr)