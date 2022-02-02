# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:51:54 2017

@author: Aurelien
"""
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QDialog, QWidget, QApplication,QListWidgetItem,
                             QPushButton, QLineEdit, QLabel)

from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QGridLayout,QGroupBox,QScrollArea,QCheckBox
from PyQt5.QtWidgets import QListWidget,QFileDialog, QComboBox
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from aoanalyst.display_imfcs import interactive_plot_h5
from PyImFCS.class_imFCS import StackFCS
from PyImFCS.export import merge_fcs_results

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
       files.sort()
       for file in files:
            self.addToList(file)
       try:
            self.setCurrentRow(index)
       except Exception as e:
            print(e)
            
   def get_filenames(self):
        items = []
        for x in range(self.count()-1):
            items.append(self.item(x).data(QtCore.Qt.UserRole))
        return items
    
class MatplotlibWindow(QDialog):
    plot_clicked = QtCore.Signal(int)
    back_to_plot = QtCore.Signal()
    image_clicked = QtCore.Signal(np.ndarray)
    onclickf = None
    onclick_function = lambda x: print('no thing to display')
    def __init__(self, parent=None):
        super().__init__(parent)

        # a figure instance to plot on
        self.figure = Figure(figsize = (20,15))

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
        
    def make_axes(self,n=2):
        self.axes = [self.figure.add_subplot(2,1,1),
                      self.figure.add_subplot(2,1,2)]
        
    def onclick(self,event):
        if self.onclick_function is None:
            return
        self.onclick_function(event)
       
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
    onclick_function = None
    current_stack = None
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.folder = None
        
        self.newAnalystButton = QPushButton("New Analyst")
        self.refreshButton = QPushButton("Refresh")
        self.trashButton = QPushButton("Trash")
        self.exportButton = QPushButton("Export")
        
        self.newAnalystButton.clicked.connect(lambda :
            self.loadFiles(str(QFileDialog.getExistingDirectory(self, "Select Directory"))))
        self.refreshButton.clicked.connect(self.refreshFileList)
        self.trashButton.clicked.connect(self.trash_measurement)
        self.exportButton.clicked.connect(self.export_measurements)
        
        self.current_mode = None
        self.expListWidget = ExperimentListWidget()
        self.plotBox = MatplotlibWindow()
        # self.plotBox.resize(1600,800)
        
        self.make_metrics_tab()
        # self.make_labelling_tab()
        
        self.imageComparisonWidget = QWidget()
        self.imageComparisonWidgetOn=False
        self.connects()
        
        # TODO
        self.loaded = ""
        
        self.expListWidget.fill(".")
            
        self.grid = QGridLayout()
        self.setLayout(self.grid)
        self.grid.addWidget(self.newAnalystButton,0,0,1,1)
        self.grid.addWidget(self.refreshButton,0,1,1,1)
        self.grid.addWidget(self.trashButton,0,2,1,1)
        self.grid.addWidget(self.exportButton,0,3,1,1)
        self.grid.addWidget(self.expListWidget,1,0,9,1)
        
        self.grid.addWidget(self.plotBox,1,1,10,10)
        # self.grid.addWidget(self.imageComparisonWidget,1,1,1,5)
        # self.grid.addWidget(self.custom_plot_tab,6,1,1,5)
        self.grid.addWidget(self.metrics_tab,10,0,1,1)
        
        
    def dragEnterEvent(self, e):
        e.accept()
    
    def dropEvent(self,e):
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
        
    def disconnects(self):
        self.expListWidget.to_analyse.disconnect(self.update_plot)
        self.plotBox.plot_clicked.disconnect(self.update_interactive_plot)
        self.plotBox.back_to_plot.disconnect(self.update_plot)

    def export_measurements(self):
        filename = str(QFileDialog.getSaveFileName(self, "Select File name", filter="*.xlsx")[0])
        if filename == "":
            return
        filename = filename+".xlsx"
        files = self.expListWidget.get_filenames()
        thr = None
        tht  = self.thresholdLineEdit.text()
        if tht.replace('.','',1).isdigit():
            thr = float(tht)
        
        intensity_threshold = None
        intensity_threshold_tmp = self.intensityLineEdit.text()
        if intensity_threshold_tmp.replace('.','',1).isdigit():
            intensity_threshold = float(intensity_threshold_tmp)
        
        merge_fcs_results(files, filename, 
              intensity_threshold = intensity_threshold, chi_threshold = thr)
        
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
        
    def update_plot(self,file=None,extract=False, load_stack = True):
        print("Update plot")
        self.current_mode = None
        if file is None:
            try:
                file = self.expListWidget.currentItem().data(QtCore.Qt.UserRole)
            except:
                return 
        fig = None
        if not extract:
            fig = self.plotBox.figure
        self.plotBox.figure.clf()
        
        if load_stack:
            print('Load stack')
            self.current_stack = StackFCS(file, load_stack = False)
            self.current_stack.load()
            nsums = self.current_stack.parfit_dict.keys()
            _ = self.update_binnings(list(nsums))
            self.binningComboBox.setCurrentIndex(0)
        nsum = int(self.binningComboBox.currentText())
        
        vmax = None
        vt  = self.vmaxLineEdit.text()
        print(vt)
        if vt.replace('.','',1).isdigit():
            print('is digit')
            vmax = float(vt)
        
        
        thr = None
        tht  = self.thresholdLineEdit.text()
        if tht.replace('.','',1).isdigit():
            print('threshold is digit')
            thr = float(tht)
            
        self.onclick_function = interactive_plot_h5(self.current_stack, fig=fig, 
                                                nsum = nsum, vmax=vmax, chi_threshold = thr)
        self.plotBox.onclick_function = self.onclick_function
        
          
    def update_interactive_plot(self,mode):
        self.plotBox.figure.clf()
        print("update interactive plot")
        
        self.plotBox.plot()
    
    def update_binnings(self,nsums):
        print('update binning:',nsums)
        # self.binningComboBox.clear()
        print("number in cbox",type(self.binningComboBox.count()),self.binningComboBox.count())

        self.binningComboBox.disconnect()
        self.binningComboBox.clear()
        self.binningComboBox.addItems([str(w) for w in nsums])
        self.binningComboBox.currentIndexChanged.connect(lambda x: self.update_plot(load_stack=False))
        return 0
    
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
        top = QGroupBox('Binning')
        toplay = QGridLayout()
        top.setLayout(toplay)
        
        self.binningComboBox = QComboBox(self)
        self.binningComboBox.addItem("2")
        self.binningComboBox.addItem("4")
        self.binningComboBox.currentIndexChanged.connect(lambda x: self.update_plot(load_stack=False))
        
        self.vmaxLineEdit = QLineEdit("None")
        self.vmaxLineEdit.editingFinished.connect(lambda : self.update_plot(load_stack=False))
        
        self.thresholdLineEdit = QLineEdit("None")
        self.thresholdLineEdit.editingFinished.connect(lambda : self.update_plot(load_stack=False))
        
        self.intensityLineEdit = QLineEdit("None")
        # self.intensityLineEdit.editingFinished.connect(lambda : self.update_plot(load_stack=False))
        
        toplay.addWidget(self.binningComboBox,0,0)
        toplay.addWidget(self.vmaxLineEdit,1,1)
        toplay.addWidget(QLabel("Max diff. shown"),1,0)
        toplay.addWidget(self.thresholdLineEdit,2,1)
        toplay.addWidget(QLabel("Chi threshold"),2,0)
        toplay.addWidget(self.intensityLineEdit,3,1)
        toplay.addWidget(QLabel("Intensity threshold (0-1)"),3,0)
        self.metrics_tab = top
        
        
        
            
app = QApplication([])
win = AOAnalyst_GUI()
win.show()
app.exec_()
