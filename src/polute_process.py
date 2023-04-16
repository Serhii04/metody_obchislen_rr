"""Module that provides some functionality for calculating
polute spreading

There will be next measures:
    time m,
    mass in mKg,
    length in meters.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def get_drain_characteristic_function(u_critical: float=0):
    def my_function(u: float):
        if u < u_critical:
            return 0

        return 1
        
    return my_function


class PoluteProcessInspector:
    """Class that provides functionality of predicting
    future state of my predefined process
    """
    # TODO: border_val and start_value not supported now
    def __init__(self, area_size: np.ndarray, partion: list,
                 k_1: float, k_2: float,
                 drain_characteristic, sourse: float
                 #  border_val: int=0, start_value: int=0,
                ):
        self.area_size = area_size
        self.partion = partion
        self.k_1 = k_1
        self.k_2 = k_2
        self.dx = area_size[0] / partion[0]
        self.dy = area_size[1] / partion[1]
        self.dt = 0.5 * pow(min(self.dx, self.dx), 2)
        self.drain_ch = drain_characteristic
        self.sourse = sourse

        self._build_table()

        # # TODO: that part might be in the solvation function
        # self._tables = list()
        # self._tables.append(self.current_table())

    def print_info(self)-> None:
        """Print all neseesary information for you
        to check the development of the class
        
        Returns:
            None
        """
        print(f"area_size: {self.area_size}")
        print(f"partion: {self.partion}")
        print(f"dx: {self.dx}")
        print(f"dy: {self.dy}")
        print(f"dt: {self.dt}")

    def _index_of(self, i: int, j: int) -> int:
        if i < 0 or j < 0 or i >= self.partion[0] or j >= self.partion[1]:
            raise ValueError(f"ERROR: indexes (i, j): ({i}, {j}) out of range, where range is ({self.partion})")
        
        return j * self.partion[0] + i
    
    def _double_index_of(self, index: int) -> list:
        if index < 0 or index > self.partion[0] * self.partion[1] - 1:
            raise ValueError(f"ERROR: index ({index}) out of range")
        
        j = int(index / self.partion[0])
        i = index - self.partion[0]* j

        return i, j

    def _get_start_table(self) -> np.ndarray:
        """Creates table for 0 time stamp and will be
        used when the next time stamp table will be calculated
        """
        temp_table = np.zeros(self.partion)

        return temp_table

    def _build_table(self) -> None:
        self.current_table = self._get_start_table()

    def set_value(self, x: int, y: int, value: float) -> None:
        self.current_table[x, y] = value

    def create_next_table_non_clear_scheme(self) -> None:
        A, b = self._create_diff_system()
        X = self._solve_equation(A, b)

        X = np.absolute(X)

        self.current_table = self.current_table + np.reshape(X, self.partion)

    def create_next_table_clear_scheme(self) -> None:
        """Creates table on the next time point
        where clear scheme is used
        """
        temp_table = self._get_start_table()
        
        for i in range(1, self.partion[0]-1):
            for j in range(1, self.partion[1]-1):
                temp_table[i, j] = (1 - self.k_1 * 2 * self.dt / (self.dx * self.dx) - self.k_2 * 2 * self.dt / (self.dy * self.dy)) * self.current_table[i, j]\
                    + self.k_1 * self.dt / (self.dx * self.dx) * (self.current_table[i+1, j] + self.current_table[i-1, j])\
                    + self.k_2 * self.dt / (self.dy * self.dy) * (self.current_table[i, j+1] + self.current_table[i, j-1])\
                    # - 1 / (2) * (self.current_table[i+1, j] * self.drain_ch(self.current_table[i+1, j]) - self.current_table[i-1, j] * self.drain_ch(self.current_table[i-1, j]))\
                    # - 1 / (2) * (self.current_table[i, j+1] * self.drain_ch(self.current_table[i, j+1]) - self.current_table[i, j-1] * self.drain_ch(self.current_table[i, j-1]))

        self.current_table = temp_table

    def print_curent_table(self) -> None:
        """Print curent table in the console
        """
        print(self.current_table)

    def plot_curent_table(self) -> None:
        """Plot curent table
        """
        self.plot_table(self.current_table)

    def plot_table(self, table: np.ndarray) -> None:
        """Plot gieven table
        """
        ax = plt.figure().add_subplot(projection='3d')
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Make data.
        X = np.arange(0, self.partion[0], 1)
        Y = np.arange(0, self.partion[1], 1)
        X, Y = np.meshgrid(X, Y)
        Z = table

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, linewidth=0)

        # Plot projections of the contours for each dimension.  By choosing offsets
        # that match the appropriate axes limits, the projected contours will sit on
        # the 'walls' of the graph.
        # ax.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
        # ax.contour(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
        # ax.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

        # ax.set(xlim=(0, 20), ylim=(0, 20), zlim=(-10, 10),
        #     xlabel='X', ylabel='Y', zlabel='Z')

        plt.show()

    def _create_diff_system(self, l_val: float=0.5) -> np.ndarray:
        """TODO: write comment
        """
        n = self.partion[0] * self.partion[1]
        
        # rename some long functions
        ct = self.current_table
        iof = self._index_of
        
        A = np.zeros((n, n))
        b = np.zeros((n))

        # fill lines corresponded to border points
        for i in range(self.partion[0]):
            for j in [0, self.partion[1] - 1]:
                A[iof(i=i, j=j), iof(i=i, j=j)] = 1
        
        for i in [0, self.partion[0] - 1]:
            for j in range(self.partion[1]):
                A[iof(i=i, j=j), iof(i=i, j=j)] = 1

        # Fill central elements
        for i in range(1, self.partion[0] - 1):
            for j in range(1, self.partion[1] - 1):
                cur_line = iof(i=i, j=j)
                
                b[cur_line] = l_val * self.dt * ((ct[i-1, j] - 2*ct[i, j] + ct[i+1, j])/(self.dx * self.dx)\
                    + (ct[i, j-1] - 2*ct[i, j] + ct[i, j+1])/(self.dy * self.dy))\
                    + ct[i, j] + ct[i, j] * self.drain_ch(ct[i, j]) * self.dt + self.sourse * self.dt

                A[cur_line, iof(i=i, j=j)] = 1 + self.drain_ch(ct[i, j]) * self.dt + 2 * (1 - l_val) * self.dt * (self.k_1/pow(self.dx, 2) + self.k_1/pow(self.dy, 2))
                A[cur_line, iof(i=i-1, j=j)] = (1 - l_val) * self.k_1 * self.dt / pow(self.dx, 2)
                A[cur_line, iof(i=i+1, j=j)] = A[cur_line, iof(i=i-1, j=j)]
                A[cur_line, iof(i=i, j=j-1)] = (1 - l_val) * self.k_2 * self.dt / pow(self.dy, 2)
                A[cur_line, iof(i=i, j=j+1)] = A[cur_line, iof(i=i, j=j-1)]

        return A, b

    def _solve_equation(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """If X that AX = b exists, then return X
        """
        # TODO: write my own algorithm
        X = np.linalg.solve(A, b)
        
        return X

    def _forvard_gaus(self, matrix: np.ndarray) -> np.ndarray:
        """ first step in gaus algorithm.
        Args:
            TODO: write the comment

        Returns:
            np.ndarray: matrix with zeros below diagonal line 
        """
        # TODO: rewrite old code
        if matrix.shape()[0] != matrix.shape()[1]:
            raise ValueError(f"Matrix must be square but {matrix.shape()} was gieven")

        temp_matrix = np.array(matrix, copy=True)
        places = [i for i in range(matrix)]

        for i in range(min(min(self.partion) + 1)):  # i - is column id, but will be used as diagonal index
            if temp_matrix[i][i] == 0:
                l = i + 1
                while temp_matrix[i][i] == 0 and l < len(temp_matrix):
                    if temp_matrix[l][i] == 1:
                        places[i], places[l] = places[l], places[i]
                        temp_matrix[i], temp_matrix[l] = temp_matrix[l], temp_matrix[i]

                        # for c in range(i, len(temp_matrix[0])):
                        #     temp_matrix[i][c] = (temp_matrix[l][c] + temp_matrix[i][c]) % 2

                    l += 1

            for l in range(i+1, len(temp_matrix)):
                if temp_matrix[l][i] == 1:
                    for c in range(i, len(temp_matrix[0])):
                        temp_matrix[l][c] = (temp_matrix[l][c] + temp_matrix[i][c]) % 2

        return temp_matrix, places


def predict(len: int=10, skip: int=10) -> None:
    area_size = np.array([1, 1])
    # partion = 4, 4
    partion = 20, 20
    k_1 = 10
    k_2 = 10
    u_cr = 12
    drain_characteristic = get_drain_characteristic_function(u_critical=u_cr)
    sourse = 5

    John = PoluteProcessInspector(area_size=area_size, partion=partion,
                                  k_1=k_1, k_2=k_2,
                                  drain_characteristic=drain_characteristic, sourse=sourse)
    
    John.print_info()
    John.set_value(x=5, y=5, value=sourse)
    John.set_value(x=8, y=8, value=sourse)
    # John.set_value(x=1, y=1, value=sourse)
    # John.set_value(x=2, y=2, value=sourse)

    John.plot_curent_table()
    
    for i in range(len):
        for s in range(skip):
            John.create_next_table_non_clear_scheme()

        John.plot_curent_table()

def main():
    predict(len=10, skip=10)

if __name__ == "__main__":
    main()


