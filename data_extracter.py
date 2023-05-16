import vtk
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridWriter
from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtk.util import numpy_support as converter
#from vtk.numpy_interface import dataset_adapter as dsa

import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time

class AirfoilData():
    def __init__(self, NACA_code):
        self.NACA_code = NACA_code
        
        ''' Airfoil geometry '''
        self.III = int(self.NACA_code[2:4]) / 100
        # get airfoil geometry from the vtk file and store it as a pandas dataframe
        vtk_airfoil_gemetry_data = self._get_airfoil_geometry_data()
        self._geometry = pd.DataFrame(converter.vtk_to_numpy(vtk_airfoil_gemetry_data.GetPoints().GetData()), 
                                      columns=["x", "y", "z"])
        
        ''' Flow field '''
        # get the field data from the vtk file and store it as pandas dataframe
        coords, pressure, velocity = self._get_flow_field_data()
        self._flow_field = pd.concat([pd.DataFrame(coords, columns=["x", "y", "z"]), pd.DataFrame(velocity, 
                                                                                                  columns=["u", "v", "w"])], axis=1)
        self._flow_field["p"] = pd.Series(pressure)

        # get 2D boundary points and flow field datasets
        self.boundary_points = self._geometry.copy()
        # remove third dimension
        self.boundary_points = self.boundary_points.loc[self.boundary_points['z'] == 0].copy()
        self.boundary_points = self.boundary_points.get(["x", "y"])
        
        #self.reduced_domain, self.train, self.test, self.extrap_test = self._get_datasets(-0.5, 1.5, 
        #                                                                                  -0.5*self.III-0.5, 0.5*self.III+0.5, 
        #                                                                                  10000, 2000, 2000)
        self.reduced_domain, self.train, self.test, self.extrap_test = self._get_datasets(-0.5, 0, -0.5, 0, 6000, 1000, 3000)
        #self.reduced_domain, self.train, self.test, self.extrap_test = self._get_datasets(0.5, 1, 
        #                                                                                  0, 0.5*self.III, 
        #                                                                                  35000, 2000, 2000)
        
    def _get_airfoil_geometry_data(self):
        # airfoil geometry file
        AGF = os.path.join("data", self.NACA_code, self.NACA_code+"_walls.vtk")
        if os.path.isfile(AGF):
            # read data from airfoil geometry file
            reader = vtkPolyDataReader()
            reader.SetFileName(AGF)
            reader.Update()
            return reader.GetOutput()
    
    # convert vtk flow field file data to numpy arrays
    def _get_flow_field_data(self):
        # field flow file
        FFF = os.path.join("data", self.NACA_code, self.NACA_code+".vtk")
        if os.path.isfile(FFF):
            # read in data from field flow file
            reader = vtkUnstructuredGridReader()
            reader.SetFileName(FFF)
            reader.Update()
            
            # We now need to convert this data into array
            #we can convert vtkDataArray into a numpy array using vtk_to_numpy method imported with the vtk.util.numpy_support module
            data = reader.GetOutput()
            
            # the data consists of point coordinates and point data and is stored as a vtkUnstructeredGrid:
             # point coordinates stored as a vtkDataArray in a vtkPoints class
            coords_data = converter.vtk_to_numpy(data.GetPoints().GetData())
            # point data stored as vtkPointData which is a subclass of vtkFieldData and consists of multiple vtkDataArrays            
            flow_field_data = data.GetPointData()
            
            
            # the point data consists of the pressure and velocity fields:
             # pressure field is the first vtkDataArray
            pressure_field = converter.vtk_to_numpy(flow_field_data.GetArray(0))
            # velocity field is the second vtkDataArray and is a 2D array
            velocity_field = converter.vtk_to_numpy(flow_field_data.GetArray(1))
            return coords_data, pressure_field, velocity_field
    
    
    def _get_datasets(self, min_x, max_x, min_y, max_y, train_size, test_size, extrap_test_size):
        # define original reduced and extrapolation domains
        original_domain = self._flow_field.copy()
        # remove third dimension
        original_domain = original_domain.loc[original_domain['z'] == 0]
        original_domain = original_domain.get(["x", "y", "p", "u", "v"])
        
        # get reduced domain
        reduced_domain = original_domain.loc[(self._flow_field['z'] == 0) &
                                              (self._flow_field["x"] > min_x) &
                                              (self._flow_field["x"] < max_x) &
                                              (self._flow_field["y"] > min_y) &
                                              (self._flow_field["y"] < max_y)].copy()
        
        
        # get extrapolation domain from
        extrap_domain = original_domain.loc[(self._flow_field['z'] == 0) &
                                              (self._flow_field["x"] > min_x-5) &
                                              (self._flow_field["x"] < max_x+5) &
                                              (self._flow_field["y"] > min_y-5) &
                                              (self._flow_field["y"] < max_y+5)].copy()
        extrap_domain = pd.concat([extrap_domain,reduced_domain]).drop_duplicates(keep=False).copy()
        # randomly sample ponts from extrapolation domain
        extrap_test = extrap_domain.sample(n=extrap_test_size, random_state=42).copy() # random state set to 42 for reproducability
        
        
        # randomly sample the necessary number of points from the reduced domain
        dataset_points = reduced_domain.sample(n=train_size + test_size, random_state=42).copy() # random state set to 42 for reproducability
        #split the points among the train, validation and test sets
        train = dataset_points.iloc[:train_size]
        test = dataset_points[train_size:]
        
        return reduced_domain, train, test, extrap_test
   
    
    # create new flow_field data using the PINN prediction and creat a vtk file with them
    def generate_vtk_file(self, model, coords_mins, coords_ranges, D_mins, D_ranges):
        # field flow file
        FFF = os.path.join("data", self.NACA_code, self.NACA_code+".vtk")
        if os.path.isfile(FFF):
            # read in data from field flow file
            reader = vtkUnstructuredGridReader()
            reader.SetFileName(FFF)
            reader.Update()
            
            # change the _flow_field data
            data = reader.GetOutput()
            data.GetPointData().Initialize()
            
            # make vtkDoubleArray for pressure
            new_p = vtk.vtkDoubleArray()
            new_p.SetNumberOfComponents(1)
            new_p.SetNumberOfTuples(len(self._flow_field))
            new_p.SetName("p")
            
            
            # make vtkDoubleArray for pressure
            new_velocity = vtk.vtkDoubleArray()
            new_velocity.SetNumberOfComponents(2)
            new_velocity.SetNumberOfTuples(len(self._flow_field))
            new_velocity.SetName("velocity")
            
            coords = (torch.tensor(np.array(self._flow_field[["x", "y"]])) - coords_mins) / coords_ranges
            D_hat = (model(coords)[:, 0:3] * D_ranges) + D_mins
            p_data, velocity_data = D_hat[:, 0].detach().numpy(), D_hat[:, 1:3].detach().numpy()
            
            for i, value in enumerate(p_data):
                new_p.SetValue(i, value)
            
            for i, value in enumerate(velocity_data):
                new_velocity.SetTuple(i, value)
                
            data.GetPointData().AddArray(new_p)
            data.GetPointData().AddArray(new_velocity)
            
            # write the grid
            writer = vtkUnstructuredGridWriter()
            writer.SetFileName(os.path.join("preds", self.NACA_code+"_pred"+".vtk"))
            writer.SetInputData(data)
            writer.Write()
         
    
    def visualise_airfoil(self):
        sns.set(style="whitegrid")
        sns.scatterplot(x=self.boundary_x,
                     y=self.boundary_y,
                     color="green")
        plt.ylabel("y")
        plt.xlabel("x")
        plt.ylim(-0.3,0.3)
    


if __name__ == "__main__":
    '''
    dataset = []
    for NACA_code in os.listdir("data"):
        dataset.append(AirfoilData(NACA_code))
    '''
    AF = AirfoilData("8646")
    #AF.visualise_airfoil()
    
        
    '''
    print(f"chord = {chord}")
    print(f"max thickness = {max_thickness}")
    print(f"max max_thickness_perc = {max_thickness_perc}")
    '''


