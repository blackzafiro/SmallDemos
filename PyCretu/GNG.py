import random
import cv2
import numpy as np
from operator import add, sub, mul

#sudo apt install python3-tk
#sudo pip3 uninstall matplotlib
#sudo pip3 install matplotlib --no-binary :all: --no-cache-dir

import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

class OpenCVProgressBar:
    def __init__(self):
        self.img = blank_image = np.zeros((40, 400, 3), np.uint8)
        cv2.imshow('progress_bar', self.img)
        cv2.waitKey(1)
        
    def set_progress(self, dec_percentage):
        """ Receives the percentage of advance in decimal numbers. """
        end = int(self.img.shape[1] * dec_percentage)
        self.img[:, 0:end] = (200, 50, 190)
        self.img[:, end:] = (55, 105, 65)

        cv2.imshow('progress_bar', self.img)
        cv2.waitKey(1)
        
    def close(self):
        cv2.destroyWindow('progress_bar')

class SegmentationNodeData:
    def __init__(self):
        """ initialized with an empty list of edges.
        Edges are of the form [s, age] and zero local error.
        """
        self.local_error = 0
        self.edges = []
        self.in_foreground = False
        
    def add_edge_with(self, node):
        """ Adds an edge between self and node. """
        if node == self: raise Exception("Trying to connect node with itself.")
        edge = [node, 0]
        self.edges.append(edge)
        
    def remove_edge_with(self, node):
        for edge in self.edges:
            if edge[0] == node:
                self.edges.remove(edge)
                return
        
    def find_edge(self, other_node):
        """
        Returns a reference to the edge if this node is connected to other_node,
        otherwise None
        """
        for edge in self.edges:
            if edge[0] == other_node:
                return edge
        return None
        
class SegmentationGNG:
    def __init__(self, imageShape, max_age, lambda_steps, epsilon_beta, epsilon_eta, alfa, beta):
        """ Receives the shape of the image that will be segmented. """
        self.nodes = {}
        self.imageShape = imageShape
        self.max_age = max_age
        self.lambda_steps = lambda_steps
        self.epsilon_beta = epsilon_beta
        self.epsilon_eta = epsilon_eta
        self.alfa = alfa
        self.beta = beta
        
    def add_node(self, node):
        """ Receives tuple [H,S,V,x,y]
        If not already added, associates empty node data
        Returns whether the GNG changed as a result
        """
        if node not in self.nodes.keys():
            self.nodes[node] = SegmentationNodeData()
            return True
        else:
            return False
    
    def _find_closest_pair(self, datum):
        """ Finds the nodes closest to the sampled nparray datum.
        Returns their keys.
        """
        nodes = list(self.nodes.keys())
        if len(nodes) < 2:
            raise Exception("GNG must be initialized with two nodes first.")
        
        #print(type(datum), datum, type(nodes[0]), nodes[0])
        distP2_1 = datum - np.array(nodes[0])
        distP2_1 = np.sum(distP2_1 ** 2)
        distP2_2 = datum - np.array(nodes[1])
        distP2_2 = np.sum(distP2_2 ** 2)
        if distP2_1 <= distP2_2:
            s1 = nodes[0]
            s2 = nodes[1]
        else:
            s1 = nodes[1]
            s2 = nodes[0]
            distP2_1, distP2_2 = distP2_2, distP2_1
            
        for s in nodes[2:]:
            distP2 = np.sum((datum - np.array(s)) ** 2)
            if distP2 < distP2_2:
                if distP2 < distP2_1:
                    s1, s2 = s, s1
                    distP2_1, distP2_2 = distP2, distP2_1
                else:
                    s2 = s
                    distP2_2 = distP2
            
        return s1, s2, distP2_1
   
    def _shift_node(self, s, w):
        """ Adds w to s coordinates.
        Returns the new coordinates
        """
        new_s = tuple(map(add, s, w))
        added = self.add_node(new_s)
        if not added:
            # pass s' neighbours to the original node at new_s
            original_data = self.nodes[new_s]
            original_data.edges += self.nodes[s].edges
            if(len(self.nodes.keys()) < 3):
                raise Exception("Collapsed!")
        
        s_data = self.nodes[s]
        new_s_data = self.nodes[new_s]
        new_s_data.local_error = s_data.local_error
        new_s_data.edges = s_data.edges
        for edge in s_data.edges:
            neighbour_data = self.nodes[edge[0]]
            symmetric_edge = neighbour_data.find_edge(s)
            if symmetric_edge is None:
                raise Exception("Inconsistent edge: " + str(edge) + " <=> " + str(neighbour_data.edges) + " in node " + str(s))
            symmetric_edge[0] = new_s
        del self.nodes[s]
        return new_s
        
    def shift_closest_pair(self, datum):
        """ Receives sampled numpy vector and updates node's positions. """
        s1, s2, distP2_1 = self._find_closest_pair(datum)
        s1_data = self.nodes[s1]
        s2_data = self.nodes[s2]
        
        # connect
        edge1 = s1_data.find_edge(s2)
        edge2 = None
        if edge1 is None:
            s1_data.add_edge_with(s2)
            s2_data.add_edge_with(s1)
        else:
            # Set age to zero
            edge1[1] = 0
            edge2 = s2_data.find_edge(s1)
            edge2[1] = 0
            
        # add to local error
        s1_data.local_error += distP2_1
        #print('shift_closest_pair ', distP2_1, ' le = ', s1_data.local_error )
        
        # Shift s1 and neighbours
        new_s1 = self._shift_node(s1, self.epsilon_beta * (datum - np.array(s1)))
        for edge in s1_data.edges:
            self._shift_node(edge[0], self.epsilon_eta * (datum - np.array(edge[0])))
        # increment age of egdes around s1
        for edge in self.nodes[new_s1].edges:
            edge[1] += 1
            edge2 = self.nodes[edge[0]].find_edge(new_s1)
            edge2[1] += 1
        #print(len(self.nodes), "\t", len(list(self.nodes.values())[0].edges), '\t', len(list(self.nodes.values())[1].edges))
        
    def delete_old_edges(self):
        """ Deletes edges with age greater than max_age. """
        max_age = self.max_age
        to_delete = []
        for s, s_data in self.nodes.items():
            s_data.edges = [edge for edge in s_data.edges if edge[1] < max_age]
            if len(s_data.edges) == 0:
                to_delete.append(s)
        for s in to_delete:
            del self.nodes[s]
        
    def _find_max_err(self):
        """ Returns the node with the greatest error. """
        max_err = -1
        node = None
        for s, s_data in self.nodes.items():
            if s_data.local_error > max_err:
                node = s
                max_err = s_data.local_error
        return node
    
    def _find_max_err_neighbour(self, q):
        """ Finds the neighbour of q with greatest local error. """
        max_err = -1
        node = None
        for edge in self.nodes[q].edges:
            s = edge[0]
            s_data = self.nodes[s]
            if s_data.local_error > max_err:
                node = s
                max_err = s_data.local_error
        return node
        
    def insert_node(self):
        """ Inserts node at position of maximum error. """
        q = self._find_max_err()
        f = self._find_max_err_neighbour(q)
        
        # Insert in middle
        r = tuple(0.5 * (np.array(q) + np.array(f)))
        if self.add_node(r):
            q_data = self.nodes[q]
            f_data = self.nodes[f]
            r_data = self.nodes[r]
            
            q_data.add_edge_with(r)
            r_data.add_edge_with(q)
            f_data.add_edge_with(r)
            r_data.add_edge_with(f)
            
            q_data.remove_edge_with(f)
            f_data.remove_edge_with(q)
            
            # Set new local errors
            q_data.local_error *= (1 - self.alfa)
            f_data.local_error *= (1 - self.alfa)
            r_data.local_error = 0.5 * (q_data.local_error + f_data.local_error)
        #print(len(self.nodes))
    
    def decrease_all_errors(self):
        """ Decreases all local errors a fixed amount.
        And returns the total accumulated error in the GNG.
        """
        comp_beta = 1 - self.beta
        total_error = 0
        for s, s_data in self.nodes.items():
            s_data.local_error *= comp_beta 
            total_error += s_data.local_error
        return total_error
        
    def show(self):
        """ Shows an image with x, y pixel on the hsv selected color. """
        img = np.zeros(self.imageShape, np.uint8)
        print("show: There are ", len(self.nodes), " nodes in GNG")
        
        #for k in self.nodes.keys():
        #    img[int(k[4]), int(k[3])] = np.array(k[:3]).astype(int)
        for k, s_data in self.nodes.items():
            s = np.array(k).astype(int)
            img[s[4], s[3]] = s[:3]
            for edge in s_data.edges:
                q = np.array(edge[0]).astype(int)
                color = np.array(k[:3])
                cv2.line(img, tuple(s[3:]), tuple(q[3:]), color)
            
        cv2.imshow('gng', img)
        cv2.moveWindow('gng', self.imageShape[1], 0)
        cv2.waitKey(1)
        
    def plotHSV(self, with_edges = False):
        """ Plots hsv values of nodes in GNG. """
        nodes_list = list(self.nodes.keys())
        nodes = np.array(nodes_list)
        show = False
        if not hasattr(self, 'fig'):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            thismanager = plt.get_current_fig_manager()
            thismanager.window.wm_geometry("+0+" + str(self.imageShape[0]))
            self.fig = fig
            self.ax = ax
            show = True
        else:
            plt.cla()

        ax = self.ax
        if with_edges:
            # Plot edges
            for s, s_data in self.nodes.items():
                for edge in s_data.edges:
                    q = edge[0]
                    ax.plot((s[0], q[0]), (s[1], q[1]), (s[2], q[2]))

        # Color clusters
        foreg = []
        for no in nodes_list:
            # Add color and marker
            if self.nodes[no].in_foreground:
                foreg.append([1])
            else:
                foreg.append([0])
        foreg = np.array(foreg)
        nodes = np.hstack((nodes, foreg))
        nodes_p = nodes[nodes[:,-1] == 1]
        nodes_n = nodes[nodes[:,-1] == 0]

        ax.scatter(nodes_p[:,0], nodes_p[:,1], nodes_p[:,2], c='r', marker='o')
        ax.scatter(nodes_n[:, 0], nodes_n[:, 1], nodes_n[:, 2], c='b', marker='^')
        ax.set_xlabel("H")
        ax.set_ylabel("S")
        ax.set_zlabel("V")
        if show:
            self.fig.show()
        else:
            self.fig.canvas.draw()
            
    def calculate_foreground_mean(self):
        """ Takes the mean of all hsv components of nodes,
        to divide them in foreground and background, as suggested
        by "Deformable Object Segmentation and Contour Tracking
        in Image Sequences Using Unsupervised Networks"
        by Cretu et. al. 2010
        
        After splitting, calculates the mean of the lighter cluster.
        """
        mean = np.array([0,0,0], dtype = np.float64)
        for node in self.nodes.keys():
            mean += np.array(node[0:3])
        mean = mean / (3 * len(self.nodes))
        # Mean HSV value?
        mean = np.sum(mean) / len(mean)
        
        mean_foreground = np.array([0,0,0], dtype = np.float64)
        for node in self.nodes.keys():
            array = np.array(node[0:3])
            mu = np.sum(array) / 3
            if mu > mean:
                self.nodes[node].in_foreground = True
                mean_foreground += array
            else:
                self.nodes[node].in_foreground = False
        mean_foreground /= (3 * len(self.nodes))
        return mean_foreground
    
class ErrorPlot:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        thismanager = plt.get_current_fig_manager()
        thismanager.window.wm_geometry("+600+0")
        self.ax.set_ylabel("Total accumulated error")
        self.fig.show()
        
    def plot(self, errors):
        self.ax.plot(errors)
        self.fig.canvas.draw()
        
    def save_plot(self, file_name):
        self.fig.savefig(file_name, bbox_inches='tight')
        
def calibrateSegmentationGNG(hsv, segment_params):
    """ Receives one image to calibrate the GNG for segmentation.
    Returns the GNG
    """
    # Randomly select a and b
    shape = hsv.shape
    #print("Size of image: ", shape)
    xa, ya, xb, yb = 0, 0, 0, 0
    max_x = shape[1] - 1
    max_y = shape[0] - 1
    while(xa == xb and ya == yb):
        xa = random.randint(0, max_x)
        xb = random.randint(0, max_x)
        ya = random.randint(0, max_y)
        yb = random.randint(0, max_y)
        
    gng = SegmentationGNG(shape, **segment_params)
    gng.add_node(tuple(hsv[ya, xa].tolist() + [xa, ya]))
    gng.add_node(tuple(hsv[yb, xb].tolist() + [xb, yb]))
    
    lambda_steps = segment_params['lambda_steps']
    # Apply to input image, pixel by pixel.
    
    pbar = OpenCVProgressBar()
    cv2.imshow('progress_bar', pbar.img)
    step = 0
    last_step = shape[0] * shape[1]
    error_plot = ErrorPlot()
    errors = []
    # Repeat n times
    for i in range(0, 4):
        step = 0
        pbar.set_progress(0)
        for yrow in range(0, shape[0]):
            for xcol in range(0, shape[1]):
                # Sample x,y randomly from image
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                gng.shift_closest_pair(np.array(hsv[y, x].tolist() + [x, y]))
                gng.delete_old_edges()
                step += 1
                if step % lambda_steps  == 0:
                    gng.insert_node()
            if y % 20 == 0:
                gng.plotHSV()
                gng.show()
                error_plot.plot(errors)
            pbar.set_progress(step/last_step)
            errors.append(gng.decrease_all_errors())
    pbar.close()
    error_plot.save_plot('error_plot.png')
    return gng