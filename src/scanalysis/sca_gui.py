#!/usr/local/bin/python3

import tkinter as tk
from tkinter import filedialog, ttk

class sca_gui(tk.Tk):
    def __init__(self, parent):
        tk.Tk.__init__(self, parent)
        self.parent = parent
        self.initialize()

    def initialize(self):
        self.grid()
        self.vals = None
        self.currentPlot = None
        self.data = {}

        #set up menu bar
        self.menubar = tk.Menu(self)
        self.fileMenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.fileMenu)
        self.fileMenu.add_command(label="Load csv file", command=self.loadCSV)
       # self.fileMenu.add_command(label="Load sparse data file", command=self.loadMTX)
       # self.fileMenu.add_command(label="Load 10x file", command=self.load10x)
       # self.fileMenu.add_command(label="Load saved session from pickle file", command=self.loadPickle)
       # self.fileMenu.add_command(label="Save data", state='disabled', command=self.saveData)
       # self.fileMenu.add_command(label="Exit", command=self.quitMAGIC)

       # self.analysisMenu = tk.Menu(self.menubar, tearoff=0)
       # self.menubar.add_cascade(label="Analysis", menu=self.analysisMenu)
       # self.analysisMenu.add_command(label="Principal component analysis", state='disabled', command=self.runPCA)
       # self.analysisMenu.add_command(label="tSNE", state='disabled', command=self.runTSNE)
       # self.analysisMenu.add_command(label="Diffusion map", state='disabled', command=self.runDM)
       # self.analysisMenu.add_command(label="MAGIC", state='disabled', command=self.runMagic)

       # self.visMenu = tk.Menu(self.menubar, tearoff=0)
       # self.menubar.add_cascade(label="Visualization", menu=self.visMenu)
       # self.visMenu.add_command(label="Scatter plot", state='disabled', command=self.scatterPlot)
       # self.visMenu.add_command(label="PCA-variance plot", state='disabled', command=self.plotPCAVariance)
        
        self.config(menu=self.menubar)

        #intro screen
        tk.Label(self, text=u"SCAnalysis", font=('Helvetica', 48), fg="black", bg="white", padx=100, pady=20).grid(row=0)
        tk.Label(self, text=u"Single Cell Analysis", font=('Helvetica', 25), fg="black", bg="white", padx=100, pady=40).grid(row=1)
        tk.Label(self, text=u"Includes Wishbone, MAGIC, and Palantir", font=('Helvetica', 20), fg="black", bg="white", padx=100, pady=40).grid(row=2)
        tk.Label(self, text=u"To get started, select a data file by clicking File > Load Data", fg="black", bg="white", padx=100, pady=25).grid(row=3)

        #update
        self.protocol('WM_DELETE_WINDOW', self.quitGUI)
        self.grid_columnconfigure(0,weight=1)
        self.resizable(True,True)
        self.update()
        self.geometry(self.geometry())       
        self.focus_force()
    
    
    def loadCSV(self):
        self.dataFileName = filedialog.askopenfilename(title='Load data file', initialdir='~/.magic/data')
        if(self.dataFileName != ""):
            #pop up data options menu
            self.fileInfo = tk.Toplevel()
            self.fileInfo.title("Data options")
            tk.Label(self.fileInfo, text=u"File name: ").grid(column=0, row=0)
            tk.Label(self.fileInfo, text=self.dataFileName.split('/')[-1]).grid(column=1, row=0)

            tk.Label(self.fileInfo,text=u"Name:" ,fg="black",bg="white").grid(column=0, row=1)
            self.fileNameEntryVar = tk.StringVar()
            self.fileNameEntryVar.set('Data ' + str(len(self.data)))
            tk.Entry(self.fileInfo, textvariable=self.fileNameEntryVar).grid(column=1,row=1)

            tk.Label(self.fileInfo, text=u"Delimiter:").grid(column=0, row=2)
            self.delimiter = tk.StringVar()
            self.delimiter.set(',')
            tk.Entry(self.fileInfo, textvariable=self.delimiter).grid(column=1, row=2)

            tk.Label(self.fileInfo, text=u"Rows:", fg="black",bg="white").grid(column=0, row=3)
            self.rowVar = tk.IntVar()
            self.rowVar.set(0)
            tk.Radiobutton(self.fileInfo, text="Cells", variable=self.rowVar, value=0).grid(column=1, row=3)
            tk.Radiobutton(self.fileInfo, text="Genes", variable=self.rowVar, value=1).grid(column=2, row=3)

            tk.Label(self.fileInfo, text=u"Number of additional rows/columns to skip after gene/cell names").grid(column=0, row=4, columnspan=3)
            tk.Label(self.fileInfo, text=u"Number of rows:").grid(column=0, row=5)
            self.rowHeader = tk.IntVar()
            self.rowHeader.set(0)
            tk.Entry(self.fileInfo, textvariable=self.rowHeader).grid(column=1, row=5)

            tk.Label(self.fileInfo, text=u"Number of columns:").grid(column=0, row=6)
            self.colHeader = tk.IntVar()
            self.colHeader.set(0)
            tk.Entry(self.fileInfo, textvariable=self.colHeader).grid(column=1, row=6)


            tk.Button(self.fileInfo, text="Compute data statistics", command=partial(self.showRawDataDistributions, file_type='csv')).grid(column=1, row=7)

            #filter parameters
            self.filterCellMinVar = tk.StringVar()
            tk.Label(self.fileInfo,text=u"Filter by molecules per cell. Min:" ,fg="black",bg="white").grid(column=0, row=8)
            tk.Entry(self.fileInfo, textvariable=self.filterCellMinVar).grid(column=1,row=8)
            
            self.filterCellMaxVar = tk.StringVar()
            tk.Label(self.fileInfo, text=u" Max:" ,fg="black",bg="white").grid(column=2, row=8)
            tk.Entry(self.fileInfo, textvariable=self.filterCellMaxVar).grid(column=3,row=8)
            
            self.filterGeneNonzeroVar = tk.StringVar()
            tk.Label(self.fileInfo,text=u"Filter by nonzero cells per gene. Min:" ,fg="black",bg="white").grid(column=0, row=9)
            tk.Entry(self.fileInfo, textvariable=self.filterGeneNonzeroVar).grid(column=1,row=9)
            
            self.filterGeneMolsVar = tk.StringVar()
            tk.Label(self.fileInfo,text=u"Filter by molecules per gene. Min:" ,fg="black",bg="white").grid(column=0, row=10)
            tk.Entry(self.fileInfo, textvariable=self.filterGeneMolsVar).grid(column=1,row=10)

            #normalize
            self.normalizeVar = tk.BooleanVar()
            self.normalizeVar.set(True)
            tk.Checkbutton(self.fileInfo, text=u"Normalize by library size", variable=self.normalizeVar).grid(column=0, row=11, columnspan=4)

            #log transform
            self.logTransform = tk.BooleanVar()
            self.logTransform.set(False)
            tk.Checkbutton(self.fileInfo, text=u"Log-transform data", variable=self.logTransform).grid(column=0, row=12)

            self.pseudocount = tk.DoubleVar()
            self.pseudocount.set(0.1)
            tk.Label(self.fileInfo, text=u"Pseudocount (for log-transform)", fg="black",bg="white").grid(column=1, row=12)
            tk.Entry(self.fileInfo, textvariable=self.pseudocount).grid(column=2, row=12)

            tk.Button(self.fileInfo, text="Cancel", command=self.fileInfo.destroy).grid(column=1, row=13)
            tk.Button(self.fileInfo, text="Load", command=partial(self.processData, file_type='csv')).grid(column=2, row=13)

            self.wait_window(self.fileInfo)
            
    def quitGUI(self):
        self.quit()
        self.destroy()

        
def launch():
    app = sca_gui(None)
    
    app.title('SCAnalysis')
    try:
        app.mainloop()
    except UnicodeDecodeError:
        pass

if __name__ == "__main__":
    launch()
