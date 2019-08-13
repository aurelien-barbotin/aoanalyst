# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:51:54 2017

@author: Aurelien
"""
from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QWidget, QApplication,QListWidgetItem,QPushButton

from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QGridLayout,QGroupBox,QScrollArea,QCheckBox,QLabel
from PyQt5.QtWidgets import QListWidget,QFileDialog, QErrorMessage, QComboBox,QLineEdit
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pywt
import json

from aoanalyst.display import analyse_experiment,display_results, plot_ac_curve

from aoanalyst.graphics import plot_with_images,image_comparison,\
                    correction_comparison_plot,aberration_difference_plot
                    
                    
from aoanalyst.io import file_extractor

from aoanalyst.sensorless import metrics

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

import sensorless.fitting as fitters


path = "."

from scipy import ndimage
def crop(im):
    u,v = im.shape
    return im[round(0.1*u):round(0.9*u),round(0.1*v):round(0.9*v)]
def sobel(im):
    im = im.astype(np.float)
    vert = ndimage.sobel(im,axis=0)
    hori = ndimage.sobel(im,axis=1)
    return np.sum((crop(vert)**2+crop(hori)**2))/np.sum(im**2)

def decompose_dwt(img,lvl=4):
    out = []
    u,v = img.shape
    u = u//2**lvl*2**lvl
    v = v//2**lvl*2**lvl
    img = img[0:u,0:v]
    for i in range(lvl):
        cA,cD = pywt.dwt2(img,"db1")
        out.append(cA)
        img = cA
    return out

def dwt_metric(img):
    dec = decompose_dwt(img)
    dec = [sobel(x*1.0) for x in dec]
    return np.array(dec)

"""metrics = {"mvect": lambda x: 1-compute_mvector(x)[-1],
           "djcentral0123":lambda x:compute_mvector(x,djs=True,coeffs=[0,1,2,3]),
           "djcentral0123den":lambda x:compute_mvector(x,denoise=True,djs=True,coeffs=[0,1,2,3]),
           "djcentral012den":lambda x:compute_mvector(x,denoise=True,djs=True,coeffs=[0,1,2]),
           "pywt": wavelet_metric,
           "inte":lambda x: float(np.sum(x)),
           "all_coeffs" : compute_mvector,
           "pywt_3coeffs": lambda x: wlt_vector(x,lvl=3),
           "djs_noNorm" : lambda x:compute_mvector(x,djs=True,normalise=False),
           "all_pywt" : wlt_vector,
           "hyperspherial":lambda x:hyperspherical(compute_mvector(x)),
           "pywt_hyperspherical":lambda x:hyperspherical(wlt_vector(x)),
           "djs":lambda x:compute_mvector(x,djs=True),
           "djcentral12":lambda x:compute_mvector(x,djs=True,coeffs=[1,2]),
           "djcentral13":lambda x:compute_mvector(x,djs=True,coeffs=[1,3]),
           "djcentral123":lambda x:compute_mvector(x,djs=True,coeffs=[1,2,3]),
           "norm_w_last": lambda x: compute_mvector(x,normwlast=True),
           "pyw_nwl":lambda x: compute_mvector(x,normwlast=True,pyw=True),
           "pyw": lambda x: compute_mvector(x,pyw=True),
           "sobel": sobel,
           "mvector_edges_djs":lambda x: compute_mvector(x,djs=True,entr=True),
           "mvector_edges_ajs":lambda x: compute_mvector(x,djs=False,entr=True),
           "mvector_edges_pyw":lambda x: compute_mvector(x,entr=True,pyw=True),
           "mvector_edges_djs_nonorm":lambda x: compute_mvector(x,djs=True,entr=True,normalise=False),
           "mvector_edges_ajs_nonorm":lambda x: compute_mvector(x,djs=False,entr=True,normalise=False),
           "mvector_edges_pyw_nonorm":lambda x: compute_mvector(x,entr=True,pyw=True,normalise=False),
           "dwt_metric": dwt_metric,
           "mvector_edges_ajs_nonorm_2coeffs":lambda x: compute_mvector(x,djs=False,entr=True,normalise=False)[0:2],
           "sobelvert_nonorm_2coeffs":lambda x: compute_mvector(x,djs=False,entr=True,normalise=False,vertical=True)[0:2],
           "sobelhori_2coeffs":lambda x: compute_mvector(x,djs=False,entr=True,normalise=False,horizontal=True)[0:2]
           }"""

class DataLabel(object):
    def __init__(self,opt1=None,opt2=None,comments=None,error=False,mode=None):
        """Optimum 1 is the optimum that is considered the best. optimum 2
        is the second best (useful in case it is unclear. comments is a string.
        error in case """
        self.optimum1 = opt1
        self.optimum2 = opt2
        self.comments = comments
        self.error = error
        self.mode = mode
        self.dict={}
        
    def to_dict(self):
        self.dict["optimum1"] = self.optimum1
        self.dict["optimum2"] = self.optimum2
        self.dict["comments"] = self.comments
        self.dict["error"] = self.error
        self.dict["mode"] = int(self.mode)
        
    def save(self,fname):
        self.to_dict()
        
        if os.path.isfile(fname+".json"):
            current_dict = self.load_dict(fname)
            current_dict.update({self.mode:self.dict})
        else:
            current_dict = {self.mode:self.dict}
        with open(fname+".json","w") as f:
            json.dump(current_dict,f)
    
    def load_dict(self,fname):
        """Loads the entire dictionary"""
        with open(fname+".json","r") as f:
            modes_dict=json.load(f)
        return modes_dict
    
    def load(self,fname,current_mode):
        print("load")
        with open(fname+".json","r") as f:            
            new_dict=json.load(f)
            print(current_mode,new_dict.keys())
        if current_mode not in new_dict and str(current_mode) not in new_dict:
            return
        if type(list(new_dict.keys())[0])==str:
            self.dict = new_dict[str(current_mode)]
        else:
            self.dict = new_dict[current_mode]
        print("dict:",self.dict)
        self.optimum1 = self.dict["optimum1"]
        self.optimum2 = self.dict["optimum2"]
        self.comments = self.dict["comments"]
        self.error = self.dict["error"]
        self.mode = self.dict["mode"]
        
class ExperimentListWidget(QListWidget):
   """Class designed to contain the different correction rounds. Each correction
   element is stored in itemList. A correction item is a list of 
   [str modes,int number, str filename]"""
   to_analyse = QtCore.Signal(str)
   
   def __init__(self,*args,**kwargs):
       super().__init__(*args, **kwargs)
       #self.itemClicked.connect(self.Clicked)
       def itemChangedSlot(item):
           try:
               self.to_analyse.emit(item.data(QtCore.Qt.UserRole))
           except:
               self.to_analyse.emit(None)
       self.currentItemChanged.connect(itemChangedSlot)
      
   def addToList(self,file):
       item = QListWidgetItem( self)
       item.setText(os.path.split(file)[-1])
       item.setData(QtCore.Qt.UserRole, file)

   def fill(self,folder):
       index = self.currentRow()
       self.clear()
       files = glob.glob(folder+"/*.h5")
       files.extend(glob.glob(folder+"/*.npy"))
       files.extend(glob.glob(folder+"/*.SIN"))
       files.sort()
       for file in files:
            self.addToList(file)
       try:
            self.setCurrentRow(index)
       except Exception as e:
            print(e)
class MatplotlibWindow(QDialog):
    plot_clicked = QtCore.Signal(int)
    back_to_plot = QtCore.Signal()
    image_clicked = QtCore.Signal(np.ndarray)
    
    def __init__(self, parent=None):
        super().__init__(parent)

        # a figure instance to plot on
        self.figure = Figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.mpl_connect('pick_event', self.onpick)
        
    def onclick(self,event):
        try:
            print("On click")
            if event.dblclick and event.button==1:
                ns = self.figure.axes[0].get_subplotspec().get_gridspec().get_geometry()
                assert(ns[0]==ns[1])
                ns = ns[0]
                
                            
                #gets the number of the subplot that is of interest for us
                n_sub = event.inaxes.rowNum*ns + event.inaxes.colNum
                self.figure.clf()
                self.plot_clicked.emit(n_sub)
            elif event.button==3: 
               self.back_to_plot.emit()
        except:
            if event.dblclick and event.button==1:
                self.figure.clf()
                self.plot_clicked.emit(-1)
            elif event.button==3: 
               self.back_to_plot.emit()
       
    def onpick(self,event):
        artist = event.artist
        if isinstance(artist, AxesImage):
            im = artist
            A = im.get_array()
            print('image clicked', A.shape)
            self.image_clicked.emit(A)
            
    def plot(self):
        self.canvas.draw()
        

class AOAnalyst_GUI(QWidget):
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.folder = None
        
        self.newAnalystButton = QPushButton("New Analyst")
        self.refreshButton = QPushButton("Refresh")
        self.trashButton = QPushButton("Trash")
        
        self.newAnalystButton.clicked.connect(lambda :
            self.loadFiles(str(QFileDialog.getExistingDirectory(self, "Select Directory"))))
        self.refreshButton.clicked.connect(self.refreshFileList)
        self.trashButton.clicked.connect(self.trash_measurement)
        
        self.current_mode = None
        self.expListWidget = ExperimentListWidget()
        self.plotBox = MatplotlibWindow()
        
        self.make_metrics_tab()
        self.make_custom_plot_tab()
        # self.make_labelling_tab()
        
        self.imageComparisonWidget = QWidget()
        self.imageComparisonWidgetOn=False
        self.connects()
        
        self.expListWidget.fill(path)
            
        self.grid = QGridLayout()
        self.setLayout(self.grid)
        self.grid.addWidget(self.newAnalystButton,0,0,1,1)
        self.grid.addWidget(self.refreshButton,0,1,1,1)
        self.grid.addWidget(self.trashButton,0,2,1,1)
        self.grid.addWidget(self.expListWidget,1,0,4,1)
        self.grid.addWidget(self.plotBox,1,1,5,5)
        self.grid.addWidget(self.imageComparisonWidget,1,6,5,5)
        self.grid.addWidget(self.custom_plot_tab,6,1,1,5)
        self.grid.addWidget(self.metrics_tab,5,0,2,1)
        
        
    def dragEnterEvent(self, e):
        e.accept()
    
    def dropEvent(self,e):
        print("drop",e.mimeData().hasUrls)
        for url in e.mimeData().urls():
            url = str(url.toLocalFile())
            if url[-3:]==".h5" or url[-3:]=="npy" or url[-3:]=="SIN":
                url = "/".join(url.split("/")[:-1])
            self.loadFiles(url)
            
    def connects(self):
        #Connecting various signals
        self.expListWidget.to_analyse.connect(self.update_plot)
        self.plotBox.plot_clicked.connect(self.update_interactive_plot)
        self.plotBox.back_to_plot.connect(self.update_plot)
        self.plotBox.image_clicked.connect(self.compare_images)
        
    def disconnects(self):
        self.expListWidget.to_analyse.disconnect(self.update_plot)
        self.plotBox.plot_clicked.disconnect(self.update_interactive_plot)
        self.plotBox.back_to_plot.disconnect(self.update_plot)
        self.plotBox.image_clicked.disconnect(self.compare_images)
    
    def compare_images(self,im):
        if not self.imageComparisonWidgetOn:
            self.imageComparisonWidget.deleteLater()
            self.imageComparisonWidget = MatplotlibWindow()
            self.grid.addWidget(self.imageComparisonWidget,1,6,5,5)
            self.images_to_compare=[im,im]
            self.imageComparisonWidgetOn = True
        self.imageComparisonWidget.figure.clf()
        
        self.images_to_compare.pop(0)
        self.images_to_compare.append(im)
        image_comparison(self.images_to_compare[0],self.images_to_compare[1],
                         fig = self.imageComparisonWidget.figure)
        self.imageComparisonWidget.plot()
    
    def trash_measurement(self):
        try:
            file = self.expListWidget.currentItem().data(QtCore.Qt.UserRole)
        except:
            return
        path,name = os.path.split(file)
        trashDir = path+"/trash"
        if not os.path.isdir(trashDir):
            os.mkdir(trashDir)
        os.rename(file,trashDir+"/"+name)
        self.refreshFileList()
        
    def update_plot(self,file=None,extract=False):
        self.current_mode = None
        if file is None:
            try:
                file = self.expListWidget.currentItem().data(QtCore.Qt.UserRole)
            except:
                return 
        
        show_legend = self.displayLegendCheckBox.isChecked()
        show_experimental = self.showOriginalCheckBox.isChecked()
        
        bool_fit_data = self.fitterOnCheckBox.isChecked()
        fitter=None
        
        if bool_fit_data:
            fittername = str(self.fitterListComboBox.currentText())
            fitter=fitters.Fitter(fittername)

        
        supp_metrics=[]
        names=[]
        for i,metk in enumerate(metrics):
            if self.metrics_bool[i]:
                supp_metrics.append(metrics[metk])
                names.append(metk)
                
        fig = None
        if not extract:
            fig = self.plotBox.figure
        self.plotBox.figure.clf()
        
        if file.split(".")[-1]=="npy" or file.split(".")[-1]=="SIN":
            
            try:
                plot_ac_curve(file,fig = fig)
                if extract:
                        plt.show()
                else:
                    self.plotBox.plot()
            except Exception as e:
                print(e)
        else:
            try:
                self.osef=display_results(file,fig=fig,
                                   supp_metrics=supp_metrics,names=names,
                                   show_legend=show_legend,
                                   show_experimental=show_experimental,
                                   fitter=fitter)
        
                if extract:
                        plt.show()
                else:
                    self.plotBox.plot()
            except Exception as e:
                print(e)
    def plot_comparison(self,file,fig):
        self.plot_mode = "comparison"
        correction_comparison_plot(file,fig)
        
    def plot_sensorless(self,file,fig):
        self.plot_mode = "sensorless"
        show_legend = self.displayLegendCheckBox.isChecked()
        show_experimental = self.showOriginalCheckBox.isChecked()
        
        bool_fit_data = self.fitterOnCheckBox.isChecked()
        fitter=None
        
        try:
            if 'log_images' not in file_extractor(file):
                os.remove(file)
                self.refreshFileList()
        except:
            print("error when removing file")
            
        if bool_fit_data:
            fittername = str(self.fitterListComboBox.currentText())
            fitter=fitters.Fitter(fittername)

        #try:
        supp_metrics=[]
        names=[]
        for i,metk in enumerate(metrics):
            if self.metrics_bool[i]:
                supp_metrics.append(metrics[metk])
                names.append(metk)
        try:
            analyse_experiment(file,fig=fig,
                               supp_metrics=supp_metrics,names=names,
                               show_legend=show_legend,
                               show_experimental=show_experimental,
                               fitter=fitter)

                
        except Exception as e:
            self.error_dialog = QErrorMessage()
            self.error_dialog.showMessage(str(e))
            print(e)
            
    def update_interactive_plot(self,mode):
        file = self.expListWidget.currentItem().data(QtCore.Qt.UserRole)
        self.plotBox.figure.clf()
        print("update interactive plot")
        self.plot_mode = "sensorless" #Temporary fix
        if self.plot_mode=="sensorless":
            self.current_mode = mode
            self.load_comments()
            #try:
            supp_metrics=[]
            names=[]
            for i,metk in enumerate(metrics):
                if self.metrics_bool[i]:
                    supp_metrics.append(metrics[metk])
                    names.append(metk)
            plot_with_images(file,mode,fig=self.plotBox.figure,supp_metrics=
                         supp_metrics)
            """except Exception as e:
                self.error_dialog = QErrorMessage()
                self.error_dialog.showMessage(str(e)+"mode:"+str(mode) )    
                print(e)"""
        elif self.plot_mode=="comparison":
            aberration_difference_plot(file,fig =self.plotBox.figure )
        
        self.plotBox.plot()
        
    def loadFiles(self,folder=None):
        if folder is None:
            folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if folder:
            self.disconnects()
            self.expListWidget.fill(folder)
            self.connects()
            self.folder = folder
            
    def refreshFileList(self):
        self.loadFiles(self.folder)
        self.update_plot()
        
    def make_metrics_tab(self):
        top = QGroupBox('Metrics')
        toplay = QGridLayout()
        top.setLayout(toplay)
        bool_rows=list()
        
        self.metrics_bool = np.zeros(len(metrics),dtype=bool)
        
        scroll = QScrollArea()
        toplay.addWidget(scroll, 1, 0, 1, 5)
        scroll.setWidget(QWidget())
        scrollLayout = QGridLayout(scroll.widget())
        scroll.setWidgetResizable(True)
        
        def update_mode(m):
            def f(value):
                self.metrics_bool[m] = value
                self.update_plot()
            return f
            
        for i,kn in enumerate(metrics):
            index_bool = QCheckBox(kn)
            upf = update_mode(i)
            index_bool.toggled.connect(upf)
            scrollLayout.addWidget(index_bool,i+1,0)
            bool_rows.append(index_bool)
            
        self.metrics_tab = top
     
    def make_custom_plot_tab(self):
        top = QGroupBox('Plot options')
        toplay = QGridLayout()
        top.setLayout(toplay)
        
        self.extractFigureButton = QPushButton("Extract figure")
        self.extractFigureButton.clicked.connect(lambda: self.update_plot(extract=True))
        
        self.displayLegendCheckBox = QCheckBox("Show legend")
        self.displayLegendCheckBox.setChecked(True)
        self.displayLegendCheckBox.toggled.connect(lambda:self.update_plot())
        
        self.showOriginalCheckBox = QCheckBox("Display experimental data")
        self.showOriginalCheckBox.setChecked(True)
        self.showOriginalCheckBox.toggled.connect(lambda:self.update_plot())
        
        self.fitterOnCheckBox = QCheckBox("Fit data")
        self.fitterOnCheckBox.toggled.connect(lambda:self.update_plot())
        
        self.dataSlicerPushButton = QPushButton("Fine analysis")
        self.dataSlicerPushButton.clicked.connect(self.open_data_slicer)
        
        self.fitterListComboBox = QComboBox()
        for fname in fitters.fitters_list:
            self.fitterListComboBox.addItem(fname)
        self.fitterListComboBox.currentIndexChanged.connect(lambda:self.update_plot())
                
        toplay.addWidget(self.extractFigureButton,0,0)
        toplay.addWidget(self.displayLegendCheckBox,0,1)
        toplay.addWidget(self.showOriginalCheckBox,0,2)
        
        toplay.addWidget(self.dataSlicerPushButton,1,0)
        toplay.addWidget(self.fitterOnCheckBox,1,1)
        toplay.addWidget(self.fitterListComboBox,1,2)
        
        self.custom_plot_tab = top
    
    def make_labelling_tab(self):
        # Not used anymore
        top = QGroupBox('Labelling')
        toplay = QGridLayout()
        top.setLayout(toplay)
    
        self.opt1 = QLineEdit("0")
        self.opt1Label = QLabel("Primary Optimum")
        self.opt2 = QLineEdit("0")
        self.opt2Label = QLabel("Scondary Optimum")
        
        self.errorCheckBox = QCheckBox()
        self.errorCheckBox.setChecked(False)
        
        self.saveButton = QPushButton("Save comments")
        self.saveButton.clicked.connect(self.save_comments)
        
        self.commentLineEdit = QLineEdit("")
        
        toplay.addWidget(self.opt1Label,0,0)
        toplay.addWidget(self.opt1,0,1)
        toplay.addWidget(self.opt2Label,1,0)
        toplay.addWidget(self.opt2,1,1)
        
        toplay.addWidget(QLabel("Error in data"),2,0)
        toplay.addWidget(self.errorCheckBox,2,1)
        toplay.addWidget(self.commentLineEdit,3,0,1,2)
        toplay.addWidget(self.saveButton,4,0)
        
        self.labelling_tab = top
        
    def save_comments(self):
        if self.current_mode is None:
            self.error_dialog = QErrorMessage()
            self.error_dialog.showMessage("No mode selected")
            return
        file = self.expListWidget.currentItem().data(QtCore.Qt.UserRole)
        
        fname = file[:-3]
        label = DataLabel(opt1 = self.opt1.text(),opt2 = self.opt2.text(),
                          comments = self.commentLineEdit.text(),
                          error = self.errorCheckBox.isChecked(),
                          mode = self.current_mode)
        label.save(fname)
        
    def load_comments(self):
        label = DataLabel()
        file = self.expListWidget.currentItem().data(QtCore.Qt.UserRole)
        fname = file[:-3]
        if not os.path.isfile(fname+".json"):
            return
        label.load(fname,self.current_mode)
        
        self.opt1.setText(label.optimum1)
        self.opt2.setText(label.optimum2)
        self.errorCheckBox.setChecked(label.error)
        self.commentLineEdit.setText(label.comments)
        
    def open_data_slicer(self):
        if self.current_mode is None:
            return
        import sys
        sys.path.append(r"C:\Users\Aurelien\Documents\Python Scripts\aotools\dataSlicer")
        sys.path.append(r"../dataSlicer")
        import dataSlicer
        
        mode = self.current_mode
        
        file = self.expListWidget.currentItem().data(QtCore.Qt.UserRole)
        ext = file_extractor(file)
        P = ext["P"]
        if "xdata" in ext:
            xdata = ext['xdata'][:,mode]
        else:
            xdata=np.arange(P)
        #ydata = ext['ydata'][:,mode]
        images = ext["log_images"][P*mode:P*(mode+1)]
        print(images.shape)
        cw = dataSlicer.DataSlicer(n_images=P)
        cw.show()
        for j in range(P):
            cw.imageWindows[j].set_image(images[j],str(xdata[j]) )
            
app = QApplication([])
win = AOAnalyst_GUI()
win.show()
app.exec_()
