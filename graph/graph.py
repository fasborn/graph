import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from itertools import chain
import random
from math import isnan

from typing import (
    Iterable,
    List,
    Optional,
    Tuple)


class SquareGrid:

    def __init__(self, data, step: float, grid_color = 'c', method = 'rhombus'):
        #if isinstance(data, GeoDataFrame):
        self.step = step
         
        if len(data)==4:
            self.xmin, self.ymin, self.xmax, self.ymax = data
            self.bounds = data
            
        elif len(data)==2:
            self.number_of_rows, self.number_of_columns = data
            self.xmin, self.ymin = 0, 0
            self.xmax, self.ymax = self.number_of_rows*step, self.number_of_columns*step
            self.bounds = [self.xmin, self.ymin, self.xmax, self.ymax]
            
        self.stations = {}
        self.color = grid_color
        self.geo_grid = self.grid(grid_color)
        self.visited_cells = {}    
        self.method = method
        
    def create_array(self):
        self.columns = list(np.arange(int(np.floor(self.xmin)), int(np.ceil(self.xmax)), self.step))
        self.rows = list(np.arange(int(np.floor(self.ymin)), int(np.ceil(self.ymax)), self.step))
        self.rows.reverse()
        """Method to be implemented"""
        raise NotImplementedError
        
    '''
        
      3 ←-- 2
       |   ↑ 
     4 ↓   |1
    '''
    
    def create_polygon(self, params):
        column, row = params
        return Polygon([(column, row), # 1st point
                        (column + self.step, row), # 2nd point
                        (column + self.step, row - self.step), # 3rd point
                        (column, row - self.step)]) # 4th point
    
    def grid(self, grid_color):
        
        x = np.arange(int(np.floor(self.xmin)), int(np.ceil(self.xmax)), self.step)
        y = np.arange(int(np.floor(self.ymin)), int(np.ceil(self.ymax)), self.step)+self.step

        nx, ny = np.meshgrid(x, y)

        nx = nx.reshape((np.prod(nx.shape),))
        ny = ny.reshape((np.prod(ny.shape),))
        ny = np.flip(ny)
        
        df = pd.DataFrame()
        df['coords_rc'] = list(zip(nx, ny))
        df['geometry'] = df.coords_rc.apply(self.create_polygon)
        
        self.geo_grid = gpd.GeoDataFrame(df, geometry = 'geometry')
        self.geo_grid['centroid'] = self.geo_grid.geometry.centroid
        self.geo_grid['num'] = self.geo_grid.index + 1
        self.geo_grid['coords'] = self.geo_grid.centroid.apply(lambda p: (p.x, p.y))
        self.geo_grid['colors'] = grid_color
        
        return self.geo_grid
    
    def grid_stable(self):
        self.columns = list(np.arange(int(np.floor(self.xmin)), int(np.ceil(self.xmax)), self.step))
        self.rows = list(np.arange(int(np.floor(self.ymin)), int(np.ceil(self.ymax)), self.step)+self.step)
        self.rows.reverse()

        self.geo_grid = gpd.GeoDataFrame(columns = ['geometry'])
        for row in self.rows:
            for column in self.columns:
                self.geo_grid.loc[len(self.geo_grid)]=Polygon([(column, row), (column + self.step, row), (column + self.step, row - self.step), (column, row - self.step)])

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
            
    def check_circle(self, id, radius):
        if radius == None:
            raise ValueError('No radius given!')
        else:
            
    
    def neighbors(self, id, radius = None):
        if self.is_valid_point_err(id):
            (x, y) = id
            if self.method == 'rhombus':
                results = [(x+self.step, y), (x, y-self.step), (x-self.step, y), (x, y+self.step)]
            elif self.method == 'square':
                results = [(x+self.step, y),
                           (x, y-self.step),
                           (x-self.step, y), 
                           (x, y+self.step),
                           (x-self.step, y+self.step),
                           (x+self.step, y-self.step),
                           (x-self.step, y-self.step),
                           (x+self.step, y+self.step)]
            elif self.method == 'circle':
                results = self.get_circle(id, radius)
            else:
                raise ValueError('Bad method passed!')
            if (x + y) % 2 == 0: results.reverse() # ради эстетики
            results = filter(self.in_bounds, results)
            return results
        else:
            raise ValueError("Point given is not valid!")
    
    def add_station(self,
                    x = None,
                    y = None,
                    id : tuple = None):
        # Единичная проверка на валидность узла в рамках сетки
        if id != None:
            if id not in self.stations:
                self.stations[self.is_valid_point_err(id)] = Station(id, self.bounds, self.step,
                                                                     geo_grid=self.geo_grid, 
                                                                     visited_cells=self.visited_cells, 
                                                                     method = self.method)
                self.visited_cells[id] = id
                return self.stations[id]
            else:
                raise ValueError("Such station already exists!")
        elif x != None and y != None:
            id = (x, y)
            if id not in self.stations:
                self.stations[self.is_valid_point_err(id)] = Station(id, self.bounds, self.step, 
                                                                     geo_grid=self.geo_grid, 
                                                                     visited_cells=self.visited_cells,
                                                                     method = self.method)
                self.visited_cells[id] = id
                return self.stations[id]
            else:
                raise ValueError("Such station already exists!")
            
    def expand_borders(self, method = 'All', cat = None):
        if method == 'All':
            for station in self.stations.values():
                if station.expand_border():
                    next
        if method == 'list':
            if cat != None:
                for station in cat:
                    station.expand_border()
            else:
                raise ValueError('Pass station')
        
        
    def cumul_exp(self, plot = False):
        temp_stations = list(self.stations.values())
        
        if plot == False:
            while len(temp_stations):
                for station in temp_stations:
                    if station.queue.empty():
                        temp_stations.remove(station)

                self.expand_borders(method = 'list', cat = temp_stations)

            print('end')
            
        elif plot == True:
            container = []
            
            while len(temp_stations):
                for station in temp_stations:
                    if station.queue.empty():
                        temp_stations.remove(station)

                self.expand_borders(method = 'list', cat = temp_stations)
                container.append(self.plot_animation(ax))
            
            return container
            
        else:
            raise ValueError('Bad plot passed!')
  
        

        
    def create_from_virtual(self):
        """Method to be implemented"""
        raise NotImplementedError
    """
    Можно создать полигоны станций уже в конце, когда виртуальная сетка посчитается.
    Будет определено, к кому какие ячейки относятся и будут известны их геометрические параметры.
    Можно будет понять, какие из них являются граничным и создать полигон из их внешних координат.
    Можно даже подумать насчёт оптимизации расчета через проверку, лежать ли линии на одной линии, вопрос только в том, насколько целесообразна такая оптимизация
    """
           
    
    def random_color(self):
        r = lambda: random.randint(0,255)
        return '#%02X%02X%02X7f' % (r(),r(),r())
        
    def frontier_color(self, params):
        # If cell is front replace alpha parameter set as 0x7F
        if params[0]:
            return params[1][:-2]
        else:
            return params[1]        

    def plot_grid(self, axis, grid, ec = 'w'):
        grid.plot(ax = axis, ec = ec, color = grid.colors)
        
    def get_plot(self, axis, grid, ec = 'w'):
#         axis = plt.gca()
        #axis.cla()
#        art = []
#        for artist in grid.plot(ax = axis, ec = 'w', color = grid.colors).get_children():
#            art.append(artist)
        grid.plot(ax = axis, ec = ec, color = grid.colors)
        
    def colorise(self):
        
        # Проверка на наличие объявленных станций сетки
        if not bool(self.stations):
            raise ValueError('No stations is deployed')
        else:
            # Словарь для граничных ячеек станций - станция:её границы
            temp_dict = {}
            
            # Пополнение общего словаря границ
            for id, station in self.stations.items():
                temp_dict[id] = station.get_front()
                
            # Быстрый перевод словаря во фрейм данных
            dct_frame = pd.DataFrame({'station':chain(temp_dict.keys()),
                                           'frontier':chain(temp_dict.values())})
            # Расширение фрейма таким образом, что уникальными становятся значения посещенных ячеек, а не станций-посетителей (посетитель - список посещенных -> посетитель - посещенный)
            s = dct_frame['frontier']
            lens = s.str.len()
            dct_frame = pd.DataFrame({
                'is_front' : dct_frame['station'].values.repeat(lens),
                'frontier' : list(chain.from_iterable(s))
            })

            # Копия основного фрейма данных
            temp_grid = self.geo_grid.copy()
            
            # Подтягивание к основной сетке информации о посетителях ячеек
            temp_grid['visited_from'] = temp_grid.coords.map(self.visited_cells)

            # Подтягивание к основной сетке информации о граничных ячейках
            temp_grid = temp_grid.merge(dct_frame,
                                        left_on = 'coords',
                                        right_on = 'frontier',
                                        how = 'left',
                                        indicator = True)

            mp = temp_grid.visited_from[temp_grid.visited_from.notna()].unique()

            #self.geo_grid['tbc'] = list(zip(self.geo_grid.visited_from, self.geo_grid.is_front))

            colors = {}
            for station in mp:
                colors[station] = self.stations[station].color
                
            temp_grid.is_front[temp_grid.is_front.notna()] = True
            temp_grid.is_front.fillna(False, inplace = True)

            temp_grid['colors'] = temp_grid.visited_from.map(colors).fillna(self.color)
            temp_grid['colors'] = pd.Series(zip(temp_grid.is_front,
                                                temp_grid.colors)).apply(self.frontier_color)

            return temp_grid
            
    def get_known_territories(self):
        frontier = []
        
        for station in self.stations.values():
            for cell in station.queue.elements:
                frontier.append(cell)
            
        known_territory = [cell for cell in self.visited_cells.keys() if cell not in frontier]
        
        return known_territory, frontier
            
            
    def plot(self, axis, colors = [['lightcyan', 'bisque'], ['deepskyblue', 'bisque'], ['cyan', 'bisque']]):
        
#        self.grid()
        
        known_territory, frontier = self.get_known_territories()

        # known_territory
        self.geo_grid[self.geo_grid.coords.isin(known_territory)].plot(ax = ax,
                                                                       color = colors[0][0],
                                                                       ec = colors[0][1])

        # Frontier cells
        self.geo_grid[self.geo_grid.coords.isin(frontier)].plot(ax = ax,
                                                                color = colors[1][0],
                                                                ec = colors[1][1])

        # Unknown cells
        self.geo_grid[~self.geo_grid.coords.isin(list(self.visited_cells.keys()))].plot(ax = ax,
                                                                                        color = colors[2][0],
                                                                                        ec = colors[2][1])

    def plot_animation_stable(self,
                       axis,
                       colors = [['lightcyan', 'bisque'], ['deepskyblue', 'bisque'], ['cyan', 'bisque']]):
                       ##**kwargs):
        art = []
        known_territory, frontier = self.get_known_territories()

        # known_territory
        for artist in self.geo_grid[self.geo_grid.coords.isin(known_territory)].plot(ax = axis,
                                                                                     color = colors[0][0],
                                                                                     ec = colors[0][1]).get_children():
            art.append(artist)

        # Frontier cells
        for artist in self.geo_grid[self.geo_grid.coords.isin(frontier)].plot(ax = axis, 
                                                                              color = colors[1][0],
                                                                              ec = colors[1][1]).get_children():
            art.append(artist)


        # Other cells
        for artist in self.geo_grid[~self.geo_grid.coords.isin(list(self.visited_cells.keys()))].plot(ax = axis, 
                                                                                                      color = colors[2][0], 
                                                                                                      ec = colors[2][1]).get_children():
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

    