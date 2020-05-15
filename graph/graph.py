import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from matplotlib import animation
import collections

class SquareGrid:
    visited_cells = {}
    
    def __init__(self, bounds, step):
        self.xmin, self.ymin, self.xmax, self.ymax = bounds
        self.step = step
        self.bounds = bounds
        self.stations = {}
        
    def grid(self):
        try:
            return self.geo_grid
        except:
            self.columns = list(np.arange(int(np.floor(self.xmin)), int(np.ceil(self.xmax)), self.step))
            self.rows = list(np.arange(int(np.floor(self.ymin)), int(np.ceil(self.ymax)), self.step))
            self.rows.reverse()

            self.geo_grid = gpd.GeoDataFrame(columns = ['geometry'])
            for row in self.rows:
                for column in self.columns:
                    self.geo_grid.loc[len(self.geo_grid)]=Polygon([(column, row), (column + wide, row), (column + self.step, row - self.step), (column, row - self.step)])

            self.geo_grid['centroid'] = self.geo_grid.geometry.centroid
            self.geo_grid['num'] = self.geo_grid.index + 1
            self.geo_grid['coords'] = self.geo_grid['geometry'].apply(lambda x: x.representative_point().coords[:])
            self.geo_grid['coords'] = [coords[0] for coords in self.geo_grid['coords']]    

    def in_bounds(self, id):
        (x, y) = id
        return self.xmin <= x < self.xmax and self.ymin <= y < self.ymax
    
    def is_valid_point(self, id):
        (x, y) = id
        return (not bool(((x-self.step/2)%self.step) and bool((y-self.step/2)%self.step))) and self.in_bounds(id)
    
    def is_valid_point_err(self, id):
        (x, y) = id
        if not(bool((x-self.step/2)%self.step)) and not(bool((y-self.step/2)%self.step)) and self.in_bounds(id):
            return id
        else:
            raise ValueError("Point given is not valid!")
    
    def neighbors(self, id):
        if self.is_valid_point(id):
            (x, y) = id
            results = [(x+self.step, y), (x, y-self.step), (x-self.step, y), (x, y+self.step)]
            if (x + y) % 2 == 0: results.reverse() # ради эстетики
            results = filter(self.in_bounds, results)
            return results
        else:
            raise ValueError("Point given is not valid!")
    
    def add_station(self, id):
        # Единичная проверка на валидность узла в рамках сетки
        if id not in self.stations:
            self.stations[self.is_valid_point_err(id)] = Station(id, self.bounds, self.step)
            self.visited_cells[id] = id
            return self.stations[id]
        else:
            raise ValueError("Such station already exists!")
            
    def expand_borders(self):
        """Expand borders of each station"""
        raise NotImplementedError
        
    def plot(self):
        """Plot each artist"""
        raise NotImplementedError
        
    def set_grid(self):
        """Attribute to implement"""
        raise NotImplementedError

    def set_rows_columns(self):
        """Attributes to be implemented"""
        raise NotImplementedError


class Queue:
    def __init__(self):
        self.elements = collections.deque()
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, x):
        self.elements.append(x)
    
    def get(self):
        return self.elements.popleft()
		
class Station(SquareGrid):
    def __init__(self, id, bounds, step):
        self.xmin, self.ymin, self.xmax, self.ymax = bounds
        self.step = step
        if self.is_valid_point(id):
            self.coordinates = id
        else:
            raise ValueError("Point given is not valid!")
        self.queue = Queue()
        self.queue.put(id)
        self.visited_cells = SquareGrid.visited_cells
            
    def expand_border(self):
        if not self.queue.empty():
            for cell in list(self.queue.elements):
                current = self.queue.get()
                for neighbor in self.neighbors(current):
                    if neighbor not in self.visited_cells:
                        self.queue.put(neighbor)
                        self.visited_cells[neighbor] = self.coordinates

        else:
            raise ValueError("Queue is empty!")