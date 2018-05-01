"""
Plots splines.
"""
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import tkinter as tk
import sys
import numpy as np


class Spline:
    """ Spline of degree n.

    S(X) = \sum_{j=0}^n \beta_{oj} x^j + \sum_{i=1}^K \beta_{in}(x - t_i)^n_+
    u_+ = u if u > 0
    u_+ = 0 if u \lef 0

    as defined in Flexible Regression Models with Cubic Splines, Durrleman and Simon.
    """
    def __init__(self, n = 3):
        self.n = n
        self.k = 5                                         # Knots
        self.beta_oj = np.array([1, -2, 1, 3])            # len = n + 1
        #self.beta_oj = np.array([1, -2, 0, 0])            # for a restricted cubic spline on the left side
        self.beta_in = np.array([[-100, 1, -200, 3, -2]])     # len = k
        self.ti = np.array([50, 150, 200, 300, 350])       # len = k

        self.fig = plt.figure()
        self.plot()

    def plot(self):
        """ Update the content of the figure. """
        fig = self.fig
        ti = self.ti
        x = np.arange(np.amin(ti), np.amax(ti))
        ax = fig.add_subplot(111)
        ax.plot(x, self(x))

        y0 = np.zeros(ti.shape)
        ax.plot(ti, y0, 'o', picker=7)
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
        sx = np.matmul(self.beta_oj.reshape((1, len(self.beta_oj))), x_powers)

        # \sum_{i=1}^K \beta_{in}(x - t_i)^n_+
        knot_terms = np.zeros((self.k, len(x)))
        for i in range(0, self.k - 1):
            knot_term = np.power(x - self.ti[i], self.n)
            knot_terms[i] = knot_term * np.heaviside(knot_term, 0) # heaviside is the step function
        sx += np.matmul(self.beta_in, knot_terms)

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
        # self.label_mouse['text'] = "({0:.2f}, {1:.2f})".format(x, y)
        if x and y:
            self.label_mouse['text'] = "({0:.2E}, {1:.2E})".format(x, y)
        else:
            self.label_mouse['text'] = "(-1, -1)"


class MatplotlibCanvas(FigureCanvasTkAgg):
    """
    Graph to plot the spline
    """

    def __init__(self, board, fig, *args, **kwargs):
        super().__init__(fig, *args, **kwargs)
        #FigureCanvasTkAgg(fig, *args, **kwargs)

        def mouse_motion(event):
            x, y = event.xdata, event.ydata
            board.display_mouse_coords(x, y)

        def on_pick(event):
            line = event.artist
            xdata, ydata = line.get_data()
            ind = event.ind
            print('on pick line {0}:'.format(event.name), np.array([xdata[ind], ydata[ind]]).T)

        def on_click(event):
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))

        #self.mpl_connect('button_press_event', on_click)
        self.mpl_connect('motion_notify_event', mouse_motion)
        self.mpl_connect('pick_event', on_pick)
        self.show()
        self.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)


class ControlsPanel(tk.Frame):
    def __init__(self, spline, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        current_row = 0
        self.n_label = tk.Label(self, text="n = ")
        self.n_label.grid(row=0, column=0)
        var = tk.StringVar(self)
        var.set("{0}".format(spline.n))
        self.n_box = tk.Spinbox(self, from_=3, to=10, textvariable=var, state=tk.DISABLED)
        self.n_box.grid(row=current_row, column=1)

        def on_change(*args):
            print(dir(*args))

        beta_oj_vars = []
        for i, beta_oj in enumerate(spline.beta_oj):
            current_row += 1
            #var = tk.StringVar()
            #var.trace("rwua", on_change)
            #control = tk.Entry(self, textvariable=var)
            control = tk.Entry(self)
            control.insert(0, "{0}".format(spline.beta_oj[i]))
            control.bind('<Key-Return>', on_change)
            control.grid(row=current_row, column=1)
            beta_oj_vars.append(var)

        current_row += 1
        self.k_label = tk.Label(self, text="k = ")
        self.k_label.grid(row=current_row, column=0)
        var_k = tk.StringVar(self)
        var_k.set("{0}".format(spline.k))
        self.k_box = tk.Spinbox(self, from_=2, to=10, textvariable=var_k, state=tk.DISABLED)
        self.k_box.grid(row=current_row, column=1, pady=5)



class TkSplines(tk.Frame):
    """
    GUI that plots the indicated spline.
    """

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.footer = footer = Footer(self)
        self.footer.pack(side="bottom", fill="x")

        self.spline = spline = Spline()

        self.mpl_canvas = MatplotlibCanvas(footer, spline.fig, master=self)

        self.controls = ControlsPanel(spline, self)
        self.controls.pack(side="right", fill="y")


if '__main__' == __name__:
    root = tk.Tk()
    root.wm_title("Splines")
    TkSplines(root).pack(side="top", fill="both", expand=True)
    root.protocol("WM_DELETE_WINDOW", sys.exit)
    #root.bind('<Motion>', motion)
    root.mainloop()