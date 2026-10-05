"""Run after pip install .: python tests/python_smoke.py"""
import unittest
import numpy as np
from mvn_moments import (
    Order, SamplingConfig, GenzConfig, MonteCarloSolver, HaltonSolver,
    GenzSolver, analytic, monte_carlo, quasi_monte_carlo,
    genz_quasi_monte_carlo,
)


class BindingsTest(unittest.TestCase):
    def test_array_layouts(self):
        config = SamplingConfig()
        config.samples = 20000
        for covariance in (np.eye(2), np.asfortranarray(np.eye(2)),
                           np.eye(4)[::2, ::2], np.eye(2, dtype=np.float32),
                           [[1, 0], [0, 1]]):
            result = monte_carlo([0, 0], covariance, [-1, -1], [1, 1], config=config, seed=7)
            self.assertAlmostEqual(result.zeroth, 0.4661, delta=0.02)
            self.assertEqual(result.second.shape, (2, 2))

    def test_orders(self):
        self.assertEqual(Order.ZEROTH | Order.SECOND, Order.ZEROTH_SECOND)
        self.assertEqual(Order.ALL & Order.FIRST, Order.FIRST)
        for function in (analytic, monte_carlo, quasi_monte_carlo,
                         genz_quasi_monte_carlo):
            result = function([0], [[1]], [-1], [1], order=Order.NONE)
            self.assertIsNone(result.zeroth)
            self.assertEqual(result.first.size, 0)
            self.assertEqual(result.second.size, 0)
            for order in (Order.ZEROTH, Order.FIRST, Order.SECOND, Order.FIRST_SECOND):
                result = function([0], [[1]], [-1], [1], order=order)
                self.assertEqual(result.zeroth is not None, bool(int(order) & 1))
                self.assertEqual(result.first.size != 0, bool(int(order) & 2))
                self.assertEqual(result.second.size != 0, bool(int(order) & 4))
        self.assertAlmostEqual(analytic([0], [[1]], [-1], [1]).zeroth, 0.6826894921370859)
        with self.assertRaises(ValueError):
            analytic([0, 0], np.eye(2), [-1, -1], [1, 1])

    def test_configs(self):
        for function, config_type in ((monte_carlo, SamplingConfig),
                                      (quasi_monte_carlo, SamplingConfig),
                                      (genz_quasi_monte_carlo, GenzConfig)):
            config = config_type()
            config.samples = 10000
            self.assertEqual(config.batch_size, 4096)
            config.batch_size = 17
            a = function([0], [[1]], [-1], [1], config=config, seed=7)
            b = function([0], [[1]], [-1], [1], config=config, seed=7)
            self.assertEqual(a.zeroth, b.zeroth)
            self.assertLessEqual(a.samples, 10000)
            self.assertGreater(a.accepted, 0)
            config.batch_size = 0
            with self.assertRaises(ValueError):
                function([0], [[1]], [-1], [1], config=config, seed=7)
            config.batch_size = 4096
            config.samples = 0
            with self.assertRaises(ValueError):
                function([0], [[1]], [-1], [1], config=config, seed=7)
            config.samples = 100
            if function is genz_quasi_monte_carlo:
                # No rejection: rare rectangles still produce estimates.
                result = function([0], [[1]], [30], [31], config=config, seed=7)
                self.assertAlmostEqual(result.zeroth / 4.90671392714793e-198, 1,
                                       delta=1e-12)
            else:
                with self.assertRaises(RuntimeError):
                    function([0], [[1]], [30], [31], config=config, seed=7)
        with self.assertRaises(TypeError):
            genz_quasi_monte_carlo([0], [[1]], [-1], [1], config=SamplingConfig())
        config = GenzConfig()
        self.assertEqual(config.shifts, 10)
        config.shifts = 1
        with self.assertRaises(ValueError):
            genz_quasi_monte_carlo([0], [[1]], [-1], [1], config=config)

    def test_genz_errors(self):
        inf = float("inf")
        result = genz_quasi_monte_carlo([0, 0], [[1, 0.5], [0.5, 1]], [0, 0],
                                        [inf, inf])
        self.assertAlmostEqual(result.zeroth, 1 / 3, delta=1e-6)
        self.assertGreater(result.zeroth_error, 0)
        self.assertEqual(result.first_error.shape, (2,))
        self.assertEqual(result.second_error.shape, (2, 2))
        result = genz_quasi_monte_carlo([0, 0], np.eye(2), [-1, -1], [1, 1],
                                        order=Order.FIRST)
        self.assertIsNone(result.zeroth_error)
        self.assertEqual(result.first_error.shape, (2,))
        self.assertEqual(result.second_error.size, 0)
        sampled = monte_carlo([0], [[1]], [-1], [1])
        self.assertIsNone(sampled.zeroth_error)
        self.assertEqual(sampled.first_error.size, 0)

    def test_solvers(self):
        # A solver allocates once and is reused in a loop. Its first call
        # matches a one-shot call with the same seed; later calls draw new
        # samples. Failures raise and leave the solver usable.
        problems = (([0, 0], [[1, 0.5], [0.5, 1]], [-1, -1], [1, 1]),
                    ([0.3], [[2]], [-1], [0.5]),
                    ([0, 0], [[2, 0.5], [0.5, 1]], [-1, -1], [1, 1]))
        for solver_type, function in ((MonteCarloSolver, monte_carlo),
                                      (HaltonSolver, quasi_monte_carlo),
                                      (GenzSolver, genz_quasi_monte_carlo)):
            solver = solver_type(2, seed=7)
            first = solver.compute(*problems[0])
            once = function(*problems[0], seed=7)
            self.assertEqual(first.zeroth, once.zeroth)
            np.testing.assert_array_equal(first.second, once.second)
            np.testing.assert_array_equal(solver.result.first, first.first)
            for problem in problems[1:]:
                result = solver.compute(*problem, order=Order.ZEROTH_FIRST)
                self.assertEqual(result.first.shape, (len(problem[0]),))
                self.assertEqual(result.second.size, 0)
            with self.assertRaises(ValueError):
                solver.compute([0], [[-1]], [-1], [1])
            with self.assertRaises(ValueError):
                solver.compute(np.zeros(3), np.eye(3), -np.ones(3), np.ones(3))
            self.assertIsNone(solver.result.zeroth)
            self.assertGreater(solver.compute(*problems[0]).zeroth, 0)
            with self.assertRaises(ValueError):
                solver_type(0)

    def test_validation_at_each_entry_point(self):
        for function in (analytic, monte_carlo, quasi_monte_carlo,
                         genz_quasi_monte_carlo):
            for mean, covariance, lower, upper in (
                ([], [], [], []),
                ([0], [[-1]], [-1], [1]),
                ([0], [[1]], [2], [1]),
                ([0], [[float('nan')]], [-1], [1]),
                ([0], [[1]], [-1, -1], [1]),
            ):
                with self.assertRaises(ValueError):
                    function(mean, covariance, lower, upper)


if __name__ == "__main__":
    unittest.main()
