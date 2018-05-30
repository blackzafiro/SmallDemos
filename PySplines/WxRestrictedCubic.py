"""
Plots splines.
"""
# matplotlib requires wxPython 2.8+
# set the wxPython version in lib\site-packages\wx.pth file
# or if you have wxversion installed un-comment the lines below
#import wxversion
#wxversion.ensureMinimal('2.8')

import sys
import time
import os
import gc
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.cm as cm
import matplotlib.cbook as cbook
from matplotlib.backends.backend_wxagg import Toolbar, FigureCanvasWxAgg
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure
import numpy as np

import wx


class RestrictedCubicSpline:
    def __init__(self):
        self._n = 3
        self.beta_00 = 5
        self.beta_01 = -5
        self._knots = np.array([50, 150, 200, 300, 350])
        self._beta_i3 = np.array([20, -35, 4])
        self._k = len(self._knots)
        if len(self._beta_i3) != (self._k - 2):
            raise ValueError("There must be " + str(self._k - 2) + " beta_i3 parameters.")

    def knots(self):
        """ Returns the numpy array of knots' coordinates.
        The values can be modified but no knots must be added or removed.
        """
        return self._knots

    def __call__(self, x):
        """
        Evaluates the spline function.
        :param x: one dimensional array with values where the spline will be evaluated
        :return: spline value at x
        """
        n = self._n
        k = self._k
        ti = self._knots
        beta_i3 = self._beta_i3

        result = self.beta_00 + self.beta_01 * x

        delta_tk = ti[k-1] - ti[k-2]    # t_K - t_{K-1}
        for i in range(0, k - 3):
            x_i = np.piecewise(x,
                               [x <= ti[i],
                                np.logical_and(x > ti[i], x <= ti[k-2]),
                                np.logical_and(x > ti[k-2], x <= ti[k-1]),
                                x > ti[k - 1]],
                               [0,
                                lambda x: np.power(x - ti[i], n),
                                lambda x: np.power(x - ti[k - 2], n) * ((ti[k - 1] - ti[i]) / delta_tk) +
                                          np.power(x - ti[k - 1], n) * ((ti[k - 2] - ti[i]) / delta_tk),
                                0])
            result += beta_i3[i] * x_i

        return result



class PlotPanel(wx.Panel):
    def __init__(self, spline, parent):
        wx.Panel.__init__(self, parent)
        self.spline = spline

        self.fig = Figure((5, 4), 75, tight_layout=True)
        self.axes = ax = self.fig.add_subplot(111)
        self.fig.suptitle("$S(x) = \\beta_{00} + \\beta_{01}x + \\sum_{i=1}^{K-2}\\beta_{i3}x_i$", fontsize=10)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$S(x)$')

        self.canvas = FigureCanvasWxAgg(self, -1, self.fig)
        self.draw()

        #self.toolbar = Toolbar(self.canvas)  # matplotlib toolbar
        #self.toolbar.Realize()
        # self.toolbar.set_active([0,1])

        # Now put all into a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        # Best to allow the toolbar to resize!
        #sizer.Add(self.toolbar, 0, wx.GROW)
        self.SetSizer(sizer)
        self.Fit()

    def draw(self):
        knots = self.spline.knots()
        ax = self.axes

        t = np.arange(knots[0], knots[-1], 0.01)
        s = self.spline(t)
        ax.plot(t, s)

        y0 = np.zeros(knots.shape)
        ax.plot(knots, y0, 'o', picker=7)


class SplineFrame(wx.Frame):
    """ Window that plots a spline. """

    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title=title)

        self.spline = spline = RestrictedCubicSpline()

        self.mpl_panel = PlotPanel(spline, self)
        self.Show()


class CubicSplineApp(wx.App):
    """ Plots restricted cubic splines. """

    def OnInit(self):

        self.main_frame = SplineFrame(None, "Restricted cubic splines")
        self.SetTopWindow(self.main_frame)

        return True


if __name__ == '__main__':
    app = CubicSplineApp(0)
    app.MainLoop()
