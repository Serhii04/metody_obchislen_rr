"""Module that provides some functionality for calculating
polute spreading

There will be next measures:
    time m,
    mass in mKg,
    length in meters.
"""

from tqdm import tqdm
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def get_drain_characteristic_function(u_critical: float):
    def my_function(u: float):
        if u < u_critical:
            return 0

        return 1
        
    return my_function

def _gaus_forward(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A_cur = np.array(A, copy=True)
    b_cur = np.array(b, copy=True)
    
    n = len(A_cur)

    for j in range(n):
        # if diagonal element is zero
        if A_cur[j][j] == 0:
            big = 0
            k_row = j
        
            for k in range(j + 1, n):
                if abs(A_cur[k][j]) > big:
                    big = abs(A_cur[k][j])
                    k_row = k
            
            for l in range(j, n):
                A_cur[j][l], A_cur[k_row][l] = A_cur[k_row][l], A_cur[j][l]

            b_cur[j], b_cur[k_row] = b_cur[k_row], b_cur[j]

        pivot = A_cur[j][j]

        # error case
        if pivot == 0:
            raise ValueError("Given matrix is singular")

        # main part
        for i in range(j + 1, n):
            mult = A_cur[i][j] / pivot

            for l in range(j, n):
                A_cur[i][l] = A_cur[i][l] - mult * A_cur[j][l]
            
            b_cur[i] = b_cur[i] - mult * b_cur[j]
        
    return A_cur, b_cur

def _gaus_backward(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(A)
    X = np.zeros((n, 1))

    for i in range(n-1, -1, -1):
        sum = 0

        for j in range(i+1, n):
            sum += X[j] * A[i][j]
        
        X[i] = 1 / A[i][i] * (b[i] - sum)

    return X

def gaus(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve matrix equation
    """
    A, b = _gaus_forward(A=A, b=b)
    X = _gaus_backward(A=A, b=b)

    return X


class PoluteProcessInspector:
    """Class that provides functionality of predicting
    future state of my predefined process
    """
    # TODO: border_val and start_value not supported now
    def __init__(self, area_size: np.ndarray, partion: list,
                 k_1: float, k_2: float,
                 drain_characteristic, sourse: float,
                 #  border_val: int=0, start_value: int=0,
                 dt: float=None,
                ):
        self.area_size = area_size
        self.partion = partion
        self.k_1 = k_1
        self.k_2 = k_2
        self.dx = area_size[0] / partion[0]
        self.dy = area_size[1] / partion[1]
        if dt is None:
            self.dt = 0.1 * pow(min(self.dx, self.dy), 1/2)
        else:
            self.dt = dt
        self.drain_ch = drain_characteristic
        self.sourse = sourse

        self.sourse_points = []

        self._build_table()

        # # TODO: that part might be in the solvation function
        # self._tables = list()
        # self._tables.append(self.current_table())

    def append_sourse_point(self, x: int, y: int) -> None:
        self.sourse_points.append((x, y))

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
        print(f"sourse_points: {self.sourse_points}")

    def _index_of(self, c: int, l: int) -> int:
        if c < 0 or l < 0 or c >= self.partion[0] or l >= self.partion[1]:
            raise ValueError(f"ERROR: indexes (c, l): ({c}, {l}) out of range, where range is ({self.partion})")
        
        return l * self.partion[0] + c
    
    def _double_index_of(self, index: int) -> list:
        if index < 0 or index > self.partion[0] * self.partion[1] - 1:
            raise ValueError(f"ERROR: index ({index}) out of range")
        
        l = int(index / self.partion[0])
        c = index - self.partion[0]* l

        return c, l

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

    def print_table_bitvise(self, table):
        # print table
        print("A::::")
        for l in table:
            print("[", end="")
            for c in l:
                print(f"{c:0.0f}", end="")
            
            print("],")

    def sourse_action(self, X=None) -> None:
        if X is None:
            X = self.current_table

        for p in self.sourse_points:
            X[p[1], p[0]] += self.sourse * self.dt

    def create_next_table_non_clear_scheme(self) -> None:
        A, b = self._create_diff_system()
        # self.print_table_bitvise(table=A)
        X = self._solve_equation(A, b)

        X = np.reshape(X, self.partion)

        self.sourse_action(X)

        # self.print_table_bitvise(X)

        # for p in self.sourse_points:
        #     X[p[0], p[1]] += self.sourse * self.dt

        self.current_table = X

    def create_next_table_clear_scheme(self) -> None:
        """Creates table on the next time point
        where clear scheme is used
        """
        temp_table = self._get_start_table()
        
        for i in range(1, self.partion[0]-1):
            for j in range(1, self.partion[1]-1):
                temp_table[i, j] = (1 - self.k_1 * 2 * self.dt / pow(self.dx, 2) - self.k_2 * 2 * self.dt / pow(self.dy, 2)) * self.current_table[i, j]\
                    + self.k_1 * self.dt / pow(self.dx, 2) * (self.current_table[i+1, j] + self.current_table[i-1, j])\
                    + self.k_2 * self.dt / pow(self.dy, 2) * (self.current_table[i, j+1] + self.current_table[i, j-1])\
                    # - 1 / (2) * (self.current_table[i+1, j] * self.drain_ch(self.current_table[i+1, j]) - self.current_table[i-1, j] * self.drain_ch(self.current_table[i-1, j]))\
                    # - 1 / (2) * (self.current_table[i, j+1] * self.drain_ch(self.current_table[i, j+1]) - self.current_table[i, j-1] * self.drain_ch(self.current_table[i, j-1]))\
                    # + 0

        for p in self.sourse_points:
            temp_table[p[0], p[1]] += self.sourse * self.dt

        self.current_table = np.abs(temp_table)

    def print_curent_table(self) -> None:
        """Print curent table in the console
        """
        print("")
        for l in self.current_table:
            print(l)
        # print(self.current_table)

    def plot_curent_table(self) -> None:
        """Plot curent table
        """
        self.plot_table(self.current_table)

    def plot_table(self, table: np.ndarray) -> None:
        # self.plot_table_warm_map(table)
        # self.plot_table_lined_graph(table)
        self.plot_table_double(table)

    def plot_table_lined_graph(self, table: np.ndarray) -> None:
        """Plot gieven table
        """
        X = np.arange(0, self.partion[0], 1)
        Y = np.arange(0, self.partion[1], 1)
        X, Y = np.meshgrid(X, Y)
        Z = table
        
        ax = plt.figure(figsize=(7,6)).add_subplot(projection='3d')

        # Plot the 3D surface
        ax.plot_surface(X, Y, Z, edgecolor='royalblue', alpha=0.3)

        ax.set(xlim=(-2, self.partion[0] + 2), ylim=(-2, self.partion[1] + 2), zlim=(-2, 17),
            xlabel='X', ylabel='Y', zlabel='Z')

        plt.show()
    
    def plot_table_warm_map(self, table: np.ndarray) -> None:
        """Plot gieven table
        """
        plt.style.use('_mpl-gallery-nogrid')

        X = np.arange(0, self.partion[0], 1)
        Y = np.arange(0, self.partion[1], 1)
        X, Y = np.meshgrid(X, Y)
        Z = table
        
        # fig, ax = plt.subplots()

        plt.figure(figsize=(7,6))
        plt.contourf(X, Y, Z)
        plt.colorbar()

        plt.show()

    def plot_table_double(self, table: np.ndarray) -> None:
        """Plot gieven table
        """
        # plt.style.use('_mpl-gallery-nogrid')

        X = np.arange(0, self.partion[0], 1)
        Y = np.arange(0, self.partion[1], 1)
        X, Y = np.meshgrid(X, Y)
        Z = table

        fig = plt.figure(figsize=plt.figaspect(0.5))

        # First subplot
        ax = fig.add_subplot(1, 2, 1)

        # plot a 3D surface like in the example mplot3d/surface3d_demo
        surf = ax.contourf(X, Y, Z, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=10)

        # Second subplot
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        # ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        
        ax.plot_surface(X, Y, Z, edgecolor='royalblue', alpha=0.3)
        ax.set(xlim=(-2, self.partion[0] + 2), ylim=(-2, self.partion[1] + 2), zlim=(-2, 17),
            xlabel='X', ylabel='Y', zlabel='Z')
        
        plt.show()

    def _create_diff_system(self, l_val: float=0.7) -> np.ndarray:
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
                A[iof(c=i, l=j), iof(c=i, l=j)] = 1
        
        for i in [0, self.partion[0] - 1]:
            for j in range(self.partion[1]):
                A[iof(c=i, l=j), iof(c=i, l=j)] = 1

        for x in range(1, self.partion[0] - 1):
            for y in range(1, self.partion[1] - 1):
                cur_line = self.partion[1] * y + x

                b[cur_line] = l_val * (self.k_1 * (ct[x+1, y] - 2 * ct[x, y] + ct[x-1, y]) / pow(self.dx, 2)\
                    + self.k_2 * (ct[x, y+1] - 2 * ct[x, y] + ct[x, y-1]) / pow(self.dy, 2)) * self.dt\
                    + ct[x, y] + ct[x, y] * self.drain_ch(ct[x, y]) * self.dt
            
                A[cur_line, self.partion[1] * (y) + (x)] = 1\
                    + (1 - l_val) * (self.k_1/pow(self.dx, 2) + self.k_1/pow(self.dy, 2)) * 2 * self.dt\
                    + self.drain_ch(ct[x, y]) * self.dt
                A[cur_line, self.partion[1] * (y) + (x+1)] = self.k_1 * (l_val - 1) * self.dt / pow(self.dx, 2)
                A[cur_line, self.partion[1] * (y) + (x-1)] = self.k_1 * (l_val - 1) * self.dt / pow(self.dx, 2)
                
                A[cur_line, self.partion[1] * (y+1) + (x)] = self.k_2 * (l_val - 1) * self.dt / pow(self.dy, 2)
                A[cur_line, self.partion[1] * (y-1) + (x)] = self.k_2 * (l_val - 1) * self.dt / pow(self.dy, 2)

        # print("A")
        # for l in A:
        #     print("")
        #     for i in range(self.partion[0]):
        #         print(l[i * self.partion[0]: i * self.partion[0] + self.partion[0]])



        # Fill central elements
        # i is Oy partion, j is Ox partion
        # for i in range(1, self.partion[0] - 1):
        #     for j in range(1, self.partion[1] - 1):
        #         cur_line = iof(c=i, l=j)
                
        #         b[cur_line] = l_val * self.dt * (\
        #               (ct[i-1, j] - 2 * ct[i, j] + ct[i+1, j]) / (pow(self.dx, 2))\
        #             + (ct[i, j-1] - 2 * ct[i, j] + ct[i, j+1]) / (pow(self.dy, 2))\
        #             ) + ct[i, j] + ct[i, j] * self.drain_ch(ct[i, j]) * self.dt

        #         A[cur_line, iof(c=i, l=j)] = 1 + self.drain_ch(ct[i, j]) * self.dt\
        #               + 2 * (1 - l_val) * self.dt * (self.k_1 / pow(self.dx, 2) + self.k_1 / pow(self.dy, 2))
        #         A[cur_line, iof(c=i-1, l=j)] = - (1 - l_val) * self.k_1 * self.dt / pow(self.dx, 2)
        #         A[cur_line, iof(c=i+1, l=j)] = A[cur_line, iof(c=i-1, l=j)]
        #         A[cur_line, iof(c=i, l=j-1)] = - (1 - l_val) * self.k_2 * self.dt / pow(self.dy, 2)
        #         A[cur_line, iof(c=i, l=j+1)] = A[cur_line, iof(c=i, l=j-1)]

        return A, b

    def _solve_equation(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """If X that AX = b exists, then return X
        """
        X = np.linalg.solve(A, b)  # TODO: delete someones code
        # X = gaus(A=A, b=b)
        
        return X

    

    


def predict(len: int=10, skip: int=10) -> None:
    area_size = np.array([1000, 1000])  # works up to 10
    # partion = 20, 20
    partion = 30, 30
    k_1 = 10
    k_2 = 10
    dt = 1
    # dt = None
    u_cr = 12
    drain_characteristic = get_drain_characteristic_function(u_critical=u_cr)
    sourse = 5 / 60

    John = PoluteProcessInspector(area_size=area_size, partion=partion, k_1=k_1, k_2=k_2, dt=dt,
                                  drain_characteristic=drain_characteristic, sourse=sourse)
    
    John.append_sourse_point(x=15, y=15)
    John.append_sourse_point(x=11, y=11)
    # John.sourse_action()
    
    John.print_info()

    John.create_next_table_non_clear_scheme()
    John.plot_curent_table()
    # John.print_curent_table()
    
    for i in range(len):
        for s in tqdm(range(skip)):
            John.create_next_table_non_clear_scheme()

        John.plot_curent_table()
        # John.print_curent_table()

def main():
    predict(len=20, skip=600)

if __name__ == "__main__":
    main()


