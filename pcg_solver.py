"""
Contains implementation of Conjugate Gradient Method solver.
For more information search:
https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
"""

import copy
from random import uniform
from typing import Tuple,List
from preconditioners import jacobi, get_preconditioner
import matplotlib.pyplot as plt
import numpy as np
from abc import ABCMeta, abstractmethod

class IterativeSolver(metaclass=ABCMeta):
    """Represents Iterative Solver interface."""

    def __init__(self, a_matrix, b_vec, x_vec, max_iter = 200,tol = 1e-5) -> None:
        """Initialize solver

        :param a_matrix: Matrix for which we search solution.
        :param b_vec: Vector for which we search solution.
        :param x_vec: result vector
        :param max_iter:

        """
        self.name = ''
        if not self._is_pos_def:
            raise TypeError('Provided matrix is not positively defined.')
        self.a_matrix = a_matrix
        self.b_vec = b_vec
        self.x_vec = x_vec
        self.max_iter = max_iter
        self.tolerance = tol
        self._finished_iter = 0
        self.residual_values = []   # type: list

    @property
    def finished_iter(self) -> int:
        """Return number of solver's iterations"""
        return self._finished_iter

    def _register_residual(self, conv: np.matrix) -> None:
        """Register residual value for particular iteration."""
        self.residual_values.append(np.linalg.norm(conv))

    def _is_pos_def(self) -> bool:
        """Check if matrix is positively defined using eigenvalues."""
        return np.all(np.linalg.eigvals(self.a_matrix) > 0)

    def get_convergence_profile(self) -> List:
        """Return convergence profile."""
        return self.residual_values

    def show_convergence_profile(self) -> None:
        """Show plot with convergence profile - normalised residual vector vs iteration."""
        y_es = self.get_convergence_profile()
        x_es = [i for i in range(len(y_es))]
        plt.title(self.name + ' method convergence profile')
        plt.ylabel('Convergence (residual norm)')
        plt.xlabel('Iterations')
        plt.plot(x_es, y_es, 'b--')
        plt.legend(['Total iter = ' + str(self.finished_iter)])
        plt.show()

    @staticmethod
    def compare_convergence_profiles(*args: 'IterativeSolver') -> None:
        """Show plot with multiple convergence profiles."""
        _to_print = []
        _legend = []
        plot_lines = ['-', '--', '*', '^']
        plot_colors = ['b', 'r', 'g', 'y']
        for ind, solver in enumerate(args):
            _y = solver.get_convergence_profile()
            _x = [i for i in range(len(_y))]
            try:
                line_color = plot_colors[ind] + plot_lines[ind]
            except IndexError:
                line_color = ''
            _to_print.append((_x, _y, line_color))
            _legend.append('{} iter = {}'.format(solver.name, solver.finished_iter))
        plt.title('Convergence profiles comparison')
        plt.ylabel('Convergence (residual norm)')
        plt.xlabel('Iterations')
        plt.plot(*[item for sublist in _to_print for item in sublist])
        plt.legend(_legend)
        plt.show()

    @abstractmethod
    def solve(self) -> Tuple[np.matrix, int,int]:
        """Solve system of linear equations."""
        raise NotImplementedError

class PreConditionedConjugateGradientSolver(IterativeSolver):
    """Implements Preconditioned Conjugate Gradient method to solve system of linear equations."""

    def __init__(self,preconditioner = None, *args, **kwargs) -> None:
        """Initialize PCG solver object, sets default pre-conditioner."""
        super(PreConditionedConjugateGradientSolver, self).__init__(*args, **kwargs)
        self.preconditioner = jacobi if not preconditioner else get_preconditioner(preconditioner)
        self.name = 'PCG {}'.format(self.preconditioner.__name__)

    def solve(self) -> Tuple[np.array, int,int]:
        """Solve system of linear equations."""
        i = 0
        x_vec = copy.deepcopy(self.x_vec)
        residual = self.b_vec - self.a_matrix * x_vec
        div = self.preconditioner(self.a_matrix, residual)
        delta_new = residual.dot(div)

        while i < self.max_iter and np.linalg.norm(residual) > self.tolerance:
            q_vec = self.a_matrix * div
            alpha = delta_new/(div.dot(q_vec))
            # numpy has some problems with casting when using += notation...
            x_vec = x_vec + alpha*div
            residual = residual - alpha*q_vec
            s_pre = self.preconditioner(self.a_matrix, residual)
            delta_old = delta_new
            delta_new = residual.dot(s_pre)
            beta = delta_new/delta_old
            div = s_pre +beta*div
            self._register_residual(residual)
            i += 1
        self._finished_iter = i     # pylint: disable=attribute-defined-outside-init
        if np.linalg.norm(residual) <= self.tolerance:
            flag = 0
        else:
            flag = 1
        return x_vec, flag,self._finished_iter
#
#
# if __name__ == "__main__":
#
#     matrix_size = 100
#     # patterns are: quadratic, rectangular, arrow, noise, curve
#     # pattern='qrana' means that testing matrix will be composition of all mentioned patterns
#     a_matrix = TestMatrices.get_random_test_matrix(matrix_size, pattern='q')
#     x_vec = np.vstack([1 for x in range(matrix_size)])
#     b_vec = np.vstack([uniform(0, 1) for x in range(matrix_size)])
#     pcg_solver = PreConditionedConjugateGradientSolver(None,a_matrix, b_vec, x_vec)
#     x,flag,it = pcg_solver.solve()
#     print(flag)