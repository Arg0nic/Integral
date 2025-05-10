from math import *
from multiprocess import Process, Value, cpu_count
import random
import ctypes


class Definite_Integral: 
    def __init__(self, expr, left_bound:float|int, right_bound:float|int,
                  iter_num:int=1_000_000):
        
        self._batch_size = cpu_count()
        self._expr = expr
        self._left_bound = left_bound
        self._right_bound = right_bound
        self._iter_num = iter_num - iter_num%(iter_num//self._batch_size)+1

        self._length = abs(self._right_bound-self._left_bound)
        self._delta_x = self._length/self._iter_num


    @property
    def left_bound(self):
        return self._left_bound
    

    @left_bound.setter
    def left_bound(self, value:float|int):
        self._left_bound = value
        self._length = abs(self._right_bound-self._left_bound)
        self._delta_x = self._length/self._iter_num

    
    @property
    def right_bound(self):
        return self._left_bound
    

    @right_bound.setter
    def right_bound(self, value:float|int):
        self._right_bound = value
        self._length = abs(self._right_bound-self._left_bound)
        self._delta_x = self._length/self._iter_num


    @property
    def iter_num(self):
        return self._iter_num
    

    @iter_num.setter
    def iter_num(self, value:int):
        self._iter_num = value - value%(value//self._batch_size)+1
        self._delta_x = self._length/self._iter_num


    def calculate(self) -> float:

        summary = Value(ctypes.c_longdouble) 
     
        if self._left_bound < self._right_bound:
            '''
            if the left_bound is bigger than the right_bound -> 
            make the result negative by replacing the left_bound in lim_sum with the right_bound 
            & change the sign of the summary.value
            '''

            sign = 1
        
            processes = [Process(target=Definite_Integral._lim_sum, 
                            args=(i-self._iter_num//self._batch_size, i, self._expr, self._delta_x, self._left_bound, summary, )) 
                                for i in range(self._iter_num//self._batch_size+1, self._iter_num+1, self._iter_num//self._batch_size)]
            
        else:

            sign = -1

            processes = [Process(target=Definite_Integral._lim_sum, 
                            args=(i-self._iter_num//self._batch_size, i, self._expr, self._delta_x, self._right_bound, summary, )) 
                                for i in range(self._iter_num//self._batch_size+1, self._iter_num+1, self._iter_num//self._batch_size)]
        
        for process in processes:
            process.start()

        for process in processes:
            process.join()
        
        return summary.value * sign


    @staticmethod
    def _lim_sum(l:int, r:int, func, delta_x:float, left_bound:float|int, summary) -> None:
        s = sum((func(left_bound+(i-1)*delta_x) for i in range(l, r)))*delta_x
        summary.value += s


    
class Monte_Carlo:
    def __init__(self, expr, left_bound: float | int, right_bound: float | int,
                 iter_num: int = 1_000_000):
        
        self._batch_size = cpu_count()
        self._expr = expr
        self._left_bound = left_bound
        self._right_bound = right_bound
        self._iter_num = iter_num
        self._length = abs(self._right_bound - self._left_bound)

    @property
    def left_bound(self):
        return self._left_bound

    @left_bound.setter
    def left_bound(self, value: float | int):
        self._left_bound = value
        self._length = abs(self._right_bound - self._left_bound)

    @property
    def right_bound(self):
        return self._right_bound

    @right_bound.setter
    def right_bound(self, value: float | int):
        self._right_bound = value
        self._length = abs(self._right_bound - self._left_bound)

    @property
    def iter_num(self):
        return self._iter_num

    @iter_num.setter
    def iter_num(self, value: int):
        self._iter_num = value

    def calculate(self) -> float:

        total = Value(ctypes.c_double)

        '''
        if the left_bound is bigger than the right_bound -> 
        change the sign of the summary.value
        '''  
        sign = [-1, 1][self._left_bound < self._right_bound]
        
        per_proc = self._iter_num // self._batch_size
        processes = [Process(
            target=Monte_Carlo._accumulate,
            args=(self, per_proc, total)
        ) for _ in range(self._batch_size)]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        remainder = self._iter_num - per_proc * self._batch_size
        if remainder:
            self._accumulate(remainder, total)

        return (total.value / self._iter_num) * self._length * sign

    def _accumulate(self, count: int, total: Value) -> None:
        local_sum = 0.0
        for _ in range(count):
            x = random.uniform(self._left_bound, self._right_bound)
            local_sum += self._expr(x)
        total.value += local_sum

