import pandas as pd
import os.path

def load(filename):
    """
    :parameter: str, name of .csv (csv file) or .p (pickle archive)
    :return: df, which is a pandas DataFrame object
    """
    if (filename[len(filename)-4:] == ".csv"):
 #       data = pd.read_csv(filename) <-- weird formatting after running multiple times
#       filepath_or_buffer = filename <--is it necessary?
        try:
            df = pd.DataFrame.from_csv(filename)
            print("Successfully loaded " + filename)
            return df
        except FileNotFoundError:
            print("Loading failed. Check that you've chosen the right .csv file")
            return

    if (filename[len(filename)-2:] == ".p"):
       # with open(filename, 'rb') as f:
#            loaded = pickle.load(f)
        try:
            df = pd.read_pickle(filename)
        #check to see if it is a pickled dataframe
            if isinstance(df, pd.DataFrame):
                print("Successfully loaded " + filename)
                return df
                #warning
        except FileNotFoundError:
            #print("Check that the pickle archive you've chosen is a pickled dataframe.")
            print("Loading failed. Check that you've chosen the right .p file")
            return
        #sparse data in the mtx format or 10x format
        #if (filename[len(filename)-4:] == ".mtx"):
         #   df =

#        if (filename[len(filename)-4:] == ".tsv"):


        #      warning...
    
    else:
        print("Loading failed. Please check that you've chosen the right file (.csv or .p)")
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



#a = load("t.csv")
#save(a, "t2.csv")
#print(a)
#save(a, "t1.p")
#b = load("t1.p")
#print(b)
#c = load("t.okk")
#f = save(b, 5) ##do we need to account for when filename isn't a string?


#d = load("../sample_scseq_data.csv")
#print(d)
#save(d, "s.csv")

#e = load("pickle_file.p")

