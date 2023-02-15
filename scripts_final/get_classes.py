
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

def assigning_classes(i):
    if i==1:
        return 1
    elif(2<=i<=12):
        return 2
    elif(13<=i<=32):
        return 3
    else:
        return 0

def create_label(path):
    '''
    This function returns labelled data according to our groundtruths (.dot files). 
    We will iterate over all of them and create a DataFrame, whose columns will be our features
    for retinophatic diabetes symptoms.
    We will create our classes according to the level of intensity defined on classes function.

    Parameters
    ----------
    path : string
        The path where the groundtruths of our dataset is located.

    Returns
    -------
    data : array
        Our labelled array.

    '''
	
    all_files = glob.glob(path + "/*.dot")

    li = []
    names=[]

    for filename in sorted(all_files):
    	df = pd.read_csv(filename, index_col=None, header=None,sep=' ')
    	li.append(df)
    	names.append(filename)

    frame = pd.concat(li, axis=0,sort=False, ignore_index=True)
    frame.columns=['redsmalldots','haemorrhages','hardexudates','softexudates','neovascularisation']
    frame=frame.notnull().astype('int')

    cols = frame.columns.tolist()
    cols=cols[0:2]+[cols[3]]+[cols[2]]+[cols[4]]
    data=frame[cols]

    intensity=data.values.dot(1 << np.arange(data.values.shape[-1]))

    classes=[]
    for i in intensity:
	    classes.append(assigning_classes(i))

    data.insert(loc=5,column='class',value=classes)
    #data.insert(loc=0,column='image',value=names)

    return data