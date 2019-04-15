import numpy as np
from PIL import Image

class LoadData:
    def __init__(self, image_size):
        self.image_size_x = image_size[0]
        self.image_size_y = image_size[1]
        
    def load_formatted_data(self,data):
        headers = data.columns.tolist()
        x_columns = headers[1:]
        y_columns = headers[0]
        #print("Train data Target variable = ", y_columns)
        #print("Train data Predictors:     " +str(x_columns))

        x_data = np.array(data[x_columns])
        y_data = np.array(data[y_columns])

        x_data = x_data.reshape((x_data.shape[0],self.image_size_x,self.image_size_y))

        print("x_data = ",x_data.shape)
        print("y_data = ",y_data.shape)
        return x_data, y_data

