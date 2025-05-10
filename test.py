from integral import Definite_Integral, Monte_Carlo
from sympy import sympify, symbols, lambdify
import typing
import time



def make_func(expr:str) -> typing.Callable:

        x = symbols('x') # Определяем символ

        expr = sympify(expr) # Преобразуем строку в символьное выражение

        function = lambdify(x, expr, "math") # Создаём функцию для численного вычисления
                
        return function



if __name__ == '__main__':

        user_input: str = 'x**2'

        expr = make_func(user_input)

        start_time = time.time()

        di_1 = Monte_Carlo(expr, -1, 2)
        print("Monte_Carlo:", di_1.calculate())

        print("--- %s seconds ---" % (time.time() - start_time))

        print()

        start_time = time.time()

        di_2 = Definite_Integral(expr, -1, 2)
        print("Definite_Integral:", di_2.calculate())

        print("--- %s seconds ---" % (time.time() - start_time))

        ################
        print("\nChanged bounds:\n")


        di_1.left_bound = 2
        di_1.right_bound = -1

        di_2.left_bound = 2
        di_2.right_bound = -1


        start_time = time.time()

        print("Monte_Carlo:", di_1.calculate())

        print("--- %s seconds ---" % (time.time() - start_time))

        print()

        start_time = time.time()

        print("Definite_Integral:", di_2.calculate())

        print("--- %s seconds ---" % (time.time() - start_time))



    


    






