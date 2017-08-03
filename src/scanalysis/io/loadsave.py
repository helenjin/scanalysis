import numpy as np
import pandas as pd
import os.path
import fcsparser


def load(filename):
    """
    :parameter: str, name of .csv (csv file) or .p (pickle archive)
    :return: df, which is a pandas DataFrame object
    """
    
    # load single cell RNA-seq data from .csv file
    if (filename[len(filename)-4:] == ".csv"):
        try:
            df = pd.DataFrame.from_csv(filename)
            print("Successfully loaded " + filename)
            return df
        except FileNotFoundError:
            print("Loading failed. Check that you've chosen the right .csv file")
            return

    # load mass cytometry (mass-cyt) data from .fcs file
    if (filename[len(filename)-4:] == ".fcs"):
        try:
            # Parse the fcs file
            text, data = fcsparser.parse(filename)
            data = data.astype(np.float64)
            
            # Extract the S and N features (Indexing assumed to start from 1)
            # Assumes channel names are in S
            no_channels = text['$PAR']
            channel_names = [''] * no_channels
            for i in range(1, no_channels+1):
                # S name
                try:
                    channel_names[i - 1] = text['$P%dS' % i]
                except KeyError:
                    channel_names[i - 1] = text['$P%dN' % i]
            data.columns = channel_names
            
            # Metadata and data
            metadata_channels = data.columns.intersection(metadata_channels)
            data_channels = data.columns.difference( metadata_channels )
            metadata = data[metadata_channels]
            data = data[data_channels]
            return data
           # # Transform if necessary
           # if cofactor is not None and cofactor > 0:
           #     data = np.arcsinh(np.divide( data, cofactor ))
        except FileNotFoundError:
            print("Loading failed. Check that you've chosen the right .fcs file")
            return
    
    # load data from pickle file
    if (filename[len(filename)-2:] == ".p"):
        try:
            df = pd.read_pickle(filename)
            
            #check to see if it is a pickled dataframe
            if isinstance(df, pd.DataFrame):
                print("Successfully loaded " + filename)
                return df
                
        except FileNotFoundError:
            print("Loading failed. Check that you've chosen the right .p file")
            return
        
        #sparse data in the mtx format or 10x format
        #if (filename[len(filename)-4:] == ".mtx"):
         #   df =

#        if (filename[len(filename)-4:] == ".tsv"):


        #      warning...
    
    else:
        print("Loading failed. Please check that you've chosen the right file")
        return

    

def save(df, filename):
    """
    :parameter df: str, name of pandas DataFrame object
    :parameter filename: str, name of .csv (csv file) or .p (pickle archive) that data will be saved to
    :return: None

    NOTE: cannot overwrite an existing csv or pickle file
    """

    file = os.path.expanduser(filename)
    
    if (file[len(file)-4:] == ".csv"):
        if os.path.exists(file):
            input("WARNING: This file already exists!\nPress enter to overwrite.\nPress Ctrl-C to exit and try again with a different file name.")
            
        df.to_csv(file)
        print("Successfully saved as " + file)
        return
 #       except:
#            raise Exception("Check your inputs, especially the .csv file you chose.")

    ## FOR A PICKLE FILE, does the pickle file have to exist?

    if (file[len(file)-2:] == ".p"):
        if os.path.exists(file):
            input("WARNING: This file already exists!\nPress enter to overwrite.\nPress Ctrl-C to exit and try again with a different file name.")
            
        
        df.to_pickle(file)
        print("Successfully saved as " + file)
        return
    ## will there be a case where the data frame won't exist/something is wrong with the df?
        
#        except:
 #           raise Exception("Check your inputs, especially the .p file you chose.")
        
#        with open(filename, 'wb') as f:
#           dump(filename, f)
            
    else:
        #      warning...
        print("Saving failed. Please check that you've named the file correctly (.csv or .p)")
        return
