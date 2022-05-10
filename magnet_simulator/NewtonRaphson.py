import numpy as np
import pandas as pd


class NewtonRaphson:
    """
    Newton-Raphson solver.
    """

    def __init__(self, f, x1: float, allowed_error: float, a: float, max_iterations: int = 10):
        self.f = f
        self.x1 = x1
        self.allowed_error = allowed_error
        self.a = a
        self.max_iterations = max_iterations

        self.iterations = pd.DataFrame(columns=['x', 'dx'])
        self.solution = None

    def CDA(self, x: float):
        """
        Returns an estimation of the derivative at a function f using the centred difference approximation:
            df/dx = (1/2•a) * (f(x + a) - f(x - a))

        :param x: The value of x we want to find the derivative at.

        :return: An estimation of the derivative of f at x.
        """
        return (1 / (2 * self.a)) * (self.f(x + self.a) - self.f(x - self.a))

    def delta_x(self, x):
        """
        Estimates an error in the guess for t using the Newton-Raphson method using the following equation:
            delta_x = -f(x) / f'(x)
        Where f'(x) is approximated using the central difference approximation using points x+a and x-a.

        :param x:

        :return:
        """
        return - self.f(x) / self.CDA(x)

    def solve(self):
        x = self.x1
        dx = self.delta_x(self.x1)

        self.iterations = pd.concat([self.iterations, pd.DataFrame({'x': self.x1, 'dx': dx}, index=[0])],
                                    ignore_index=True)

        while np.absolute(dx) >= self.allowed_error:
            x = x + dx
            dx = self.delta_x(x)

            self.iterations = pd.concat([self.iterations, pd.DataFrame({'x': x, 'dx': dx}, index=[0])],
                                        ignore_index=True)

            if len(self.iterations) > self.max_iterations:
                self.solution = np.nan
                break
            else:
                self.solution = x

        return x
