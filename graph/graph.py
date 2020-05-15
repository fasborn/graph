import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from matplotlib import animation
import collections

class SquareGrid:

    def __init__(self, data, step: float):
        #if isinstance(data, GeoDataFrame):
        self.step = step
         
        if len(data)==4:
            self.xmin, self.ymin, self.xmax, self.ymax = data
            self.bounds = data
            
        elif len(data)==2:
            self.number_of_rows, self.number_of_columns = data
            self.xmin, self.ymin = 0, 0
            self.xmax, self.ymax = number_of_rows*step, number_of_columns*step
            self.bounds = [self.xmin, self.ymin, self.xmax, self.ymax]
            
        self.stations = {}
        self.geo_grid = self.grid()
        self.visited_cells = {}    
#    @attribute
#    self.geo_grid = selg.grid()
    
    def create_array(self):
        self.columns = list(np.arange(int(np.floor(self.xmin)), int(np.ceil(self.xmax)), self.step))
        self.rows = list(np.arange(int(np.floor(self.ymin)), int(np.ceil(self.ymax)), self.step))
        self.rows.reverse()
        """Method to be implemented"""
        raise NotImplementedError
        
        
    def grid(self):
        self.columns = list(np.arange(int(np.floor(self.xmin)), int(np.ceil(self.xmax)), self.step))
        self.rows = list(np.arange(int(np.floor(self.ymin)), int(np.ceil(self.ymax)), self.step)+self.step)
        self.rows.reverse()

        self.geo_grid = gpd.GeoDataFrame(columns = ['geometry'])
        for row in self.rows:
            for column in self.columns:
                self.geo_grid.loc[len(self.geo_grid)]=Polygon([(column, row), (column + wide, row), (column + self.step, row - self.step), (column, row - self.step)])

        self.geo_grid['centroid'] = self.geo_grid.geometry.centroid
        self.geo_grid['num'] = self.geo_grid.index + 1
        self.geo_grid['coords'] = self.geo_grid['geometry'].apply(lambda x: x.representative_point().coords[:])
        self.geo_grid['coords'] = [coords[0] for coords in self.geo_grid['coords']]

        return self.geo_grid

    def in_bounds(self, id):
        (x, y) = id
        return self.xmin <= x < self.xmax and self.ymin <= y < self.ymax
    
    # Depricated
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
        if self.is_valid_point_err(id):
            (x, y) = id
            results = [(x+self.step, y), (x, y-self.step), (x-self.step, y), (x, y+self.step)]
            if (x + y) % 2 == 0: results.reverse() # ради эстетики
            results = filter(self.in_bounds, results)
            return results
        else:
            raise ValueError("Point given is not valid!")
    
    def add_station(self,
                    x = None,
                    y = None,
                    id = None):
        # Единичная проверка на валидность узла в рамках сетки
        if id != None:
            if id not in self.stations:
                self.stations[self.is_valid_point_err(id)] = Station(id, self.bounds, self.step,
                                                                     geo_grid=self.geo_grid, 
                                                                     visited_cells=self.visited_cells)
                self.visited_cells[id] = id
                return self.stations[id]
            else:
                raise ValueError("Such station already exists!")
        elif x != None and y != None:
            id = (x, y)
            if id not in self.stations:
                self.stations[self.is_valid_point_err(id)] = Station(id, self.bounds, self.step, 
                                                                     geo_grid=self.geo_grid, 
                                                                     visited_cells=self.visited_cells)
                self.visited_cells[id] = id
                return self.stations[id]
            else:
                raise ValueError("Such station already exists!")
            
    def expand_borders(self):
        for station in self.stations.values():
            #station.expand_border()
            if station.expand_border():
                next
        print(" Queues are empty")
                
    def get_known_territories(self):
        frontier = []
        
        for station in self.stations.values():
            for cell in station.queue.elements:
                frontier.append(cell)
            
        known_territory = [cell for cell in self.visited_cells.keys() if p not in frontier]
        
        return known_territory, frontier
            
            
    def plot(self, axis):
        
#        self.grid()
        
        known_territory, frontier = self.get_known_territories()

        # known_territory
        self.geo_grid[self.geo_grid.coords.isin(known_territory)].plot(ax = ax, color = 'lightcyan', ec = 'bisque')

        # Frontier cells
        self.geo_grid[self.geo_grid.coords.isin(frontier)].plot(ax = ax, color = 'deepskyblue', ec = 'bisque')

        # Unknown cells
        self.geo_grid[~self.geo_grid.coords.isin(list(self.visited_cells.keys()))].plot(ax = ax, color = 'cyan', ec = 'bisque')

    def plot_animation(self, axis):
        art = []
        known_territory, frontier = self.get_known_territories()

        # known_territory
        for artist in self.geo_grid[self.geo_grid.coords.isin(known_territory)].plot(ax = axis,
                                                                                     color = 'lightcyan',
                                                                                     ec = 'bisque').get_children():
            art.append(artist)

        # Frontier cells
        for artist in self.geo_grid[self.geo_grid.coords.isin(frontier)].plot(ax = axis, 
                                                                              color = 'deepskyblue',
                                                                              ec = 'bisque').get_children():
            art.append(artist)


        # Other cells
        for artist in self.geo_grid[~self.geo_grid.coords.isin(list(self.visited_cells.keys()))].plot(ax = axis, 
                                                                                                      color = 'cyan', 
                                                                                                      ec = 'bisque').get_children():
            art.append(artist)
            
        return art
        
    def set_grid(self):
        """Attribute to be implemented"""
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

    def __init__(self,
                 id,
                 bounds,
                 step, 
                 geo_grid,
                 visited_cells):
        self.xmin, self.ymin, self.xmax, self.ymax = bounds
        self.step = step
        if self.is_valid_point(id):
            self.coordinates = id
        else:
            raise ValueError("Point given is not valid!")
        self.queue = Queue()
        self.queue.put(id)
        self.visited_cells = visited_cells
        self.geo_grid = geo_grid
        self.visited_cells = visited_cells
            
    def expand_border(self):
        if not self.queue.empty():
            for cell in list(self.queue.elements):
                current = self.queue.get()
                for neighbor in self.neighbors(current):
                    if neighbor not in self.visited_cells:
                        self.queue.put(neighbor)
                        self.visited_cells[neighbor] = self.coordinates
            return 0            
#         else:
#             raise ValueError(" Queue is emplty!")
            
        else:
            return 1
            
    def plot_step(self):
        visited = [p for p in self.visited_cells.keys() if p not in self.queue.elements]
        ax = plt.gca()

        # Visited cells
        self.geo_grid[self.geo_grid.coords.isin(visited)].plot(ax = ax, color = 'lightcyan', ec = 'bisque')

        # Frontier cells
        self.geo_grid[self.geo_grid.coords.isin(self.queue.elements)].plot(ax = ax, color = 'deepskyblue', ec = 'bisque')

        # Other cells
        self.geo_grid[~self.geo_grid.coords.isin(list(self.visited_cells.keys()))].plot(ax = ax, color = 'cyan', ec = 'bisque')

    