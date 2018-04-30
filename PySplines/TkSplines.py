"""
Plots cubic splines.
"""
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import tkinter as tk
import sys
import numpy as np


class CubicSpline:
    """ Cubic spline.

    S(X) = \sum_{j=0}^n \beta_{oj} x^j + \sum_{i=1}^K \beta_{in}(x - t_i)^n_+
    u_+ = u if u > 0
    u_+ = 0 if u \lef 0

    as defined in Flexible Regression Models with Cubic Splines, Durrleman and Simon.
    """
    def __init__(self):
        self.n = 3    # Cubic spline
        self.k = 5    # Knots
        self.beta_oj = np.array([[1, -2, 1, 3]])
        self.beta_in = np.array([-1, 1, 2, 3])
        self.ti = np.array([50, 100, 200, 300])

        x = np.arange(0, 300)
        self.fig = fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, self(x))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$S(x)$')
        fig.suptitle("$S(x) = \\sum_{j=0}^n \\beta_{oj} x^j + \\sum_{i=1}^K \\beta_{in}(x - t_i)^n_+$")

    def __call__(self, x):
        """
        Evaluates the spline function.
        :param x: one dimensional array with values where the spline will be evaluated
        :return: spline value at x
        """

        # \sum_{j=0}^n \beta_{oj} x^j
        x_powers = np.zeros((self.n + 1, len(x)))
        x_powers[0] = x
        for i in range(1, self.n):
            # i is a row
            x_powers[i] = x_powers[i-1] * x
        sx = np.matmul(self.beta_oj, x_powers)
        print(sx.shape)

        # TODO: \sum_{i=1}^K \beta_{in}(x - t_i)^n_+

        return sx.flatten()


class Footer(tk.Frame):
    """
    Footer toolbar where mouse position is displayed.
    """

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.label_mouse = tk.Label(self, text="[-1,-1]")
        self.label_mouse.grid(row=0, column=0)

    def display_mouse_coords(self, x, y):
        self.label_mouse['text'] = "({0}, {1})".format(x, y)


class TkSplines(tk.Frame):
    """
    GUI that plots the indicated spline.
    """

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.footer = footer = Footer(self)
        self.footer.pack(side="bottom", fill="x")

        def mouse_motion(event):
            x, y = event.x, event.y
            footer.display_mouse_coords(x, y)
            #print('{}, {}'.format(x, y))

        parent.bind('<Motion>', mouse_motion)

        # Spline
        self.spline = spline = CubicSpline()

        # Canvas para las gráficas de las distribuciones de probabilidad
        mpl_canvas = FigureCanvasTkAgg(spline.fig, master=self)
        mpl_canvas.show()
        mpl_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.mpl_canvas = mpl_canvas




if '__main__' == __name__:
    root = tk.Tk()
    root.wm_title("Splines")
    TkSplines(root).pack(side="top", fill="both", expand=True)
    root.protocol("WM_DELETE_WINDOW", sys.exit)
    #root.bind('<Motion>', motion)
    root.mainloop()