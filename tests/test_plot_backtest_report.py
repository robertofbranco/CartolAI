import unittest

import matplotlib.pyplot as plt

from plot_backtest_report import use_integer_x_ticks


class PlotBacktestReportTest(unittest.TestCase):
    def test_use_integer_x_ticks_keeps_round_axis_labels_whole_numbers(self):
        fig, ax = plt.subplots()
        try:
            ax.plot([20, 21, 22, 23, 24], [1, 2, 3, 4, 5])

            use_integer_x_ticks(ax)
            fig.canvas.draw()

            self.assertTrue(
                all(float(tick).is_integer() for tick in ax.get_xticks()),
                ax.get_xticks(),
            )
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
