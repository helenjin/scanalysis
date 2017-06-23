#!/usr/local/bin/python3

import tkinter as tk

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
        self.fileMenu.add_command(label="Load sparse data file", command=self.loadMTX)
        self.fileMenu.add_command(label="Load 10x file", command=self.load10x)
        self.fileMenu.add_command(label="Load saved session from pickle file", command=self.loadPickle)
        self.fileMenu.add_command(label="Save data", state='disabled', command=self.saveData)
        self.fileMenu.add_command(label="Exit", command=self.quitMAGIC)

        self.analysisMenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Analysis", menu=self.analysisMenu)
        self.analysisMenu.add_command(label="Principal component analysis", state='disabled', command=self.runPCA)
        self.analysisMenu.add_command(label="tSNE", state='disabled', command=self.runTSNE)
        self.analysisMenu.add_command(label="Diffusion map", state='disabled', command=self.runDM)
        self.analysisMenu.add_command(label="MAGIC", state='disabled', command=self.runMagic)

        self.visMenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Visualization", menu=self.visMenu)
        self.visMenu.add_command(label="Scatter plot", state='disabled', command=self.scatterPlot)
        self.visMenu.add_command(label="PCA-variance plot", state='disabled', command=self.plotPCAVariance)
        
        self.config(menu=self.menubar)

        #intro screen
        tk.Label(self, text=u"MAGIC", font=('Helvetica', 48), fg="black", bg="white", padx=100, pady=20).grid(row=0)
        tk.Label(self, text=u"Markov Affinity-based Graph Imputation of Cells", font=('Helvetica', 25), fg="black", bg="white", padx=100, pady=40).grid(row=1)
        tk.Label(self, text=u"To get started, select a data file by clicking File > Load Data", fg="black", bg="white", padx=100, pady=25).grid(row=2)

        #update
        self.protocol('WM_DELETE_WINDOW', self.quitGUI)
        self.grid_columnconfigure(0,weight=1)
        self.resizable(True,True)
        self.update()
        self.geometry(self.geometry())       
        self.focus_force()

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
