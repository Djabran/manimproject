from geometryobjects import *
from eq import *
import sympy
import math
from typing import Union


class LinearTransformation(SceneBase2d, MovingCameraScene):

    def setup(self):
        MovingCameraScene.setup(self)

    def construct(self):
        self.show_title("Linear Transformations")
        number_plane = NumberPlane()
        number_plane.apply_function(lambda x: x + 0.5)
        add(number_plane)
        wait()


class AV(SceneBase2d):
    def construct(self):
        def polar2c(p):
            return np.array([
                p[0] * np.cos(p[1]),
                p[0] * np.sin(p[1]),
                0
            ])

        def translatex(p, value=0.5):
            return np.array([p[0]**2, p[1], 0])

        def skewx(p, value=0.5):
            return np.array([p[0] + p[1], p[1], 0])

        def rotate(p, theta=0.5):
            A = np.matrix([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            A_ = A.dot(np.array((p[0], p[1])))
            return np.array((A_.item(0, 0), A_.item(0, 1), 0))

        grid = NumberPlane(
            x_line_frequency=PI / 4,
            y_line_frequency=PI / 4,
            x_min=-PI,
            x_max=PI,
            y_min=-PI,
            y_max=PI
        )
        # func = FunctionGraph(lambda x: 0.5 * np.sin(5 * x) + 2, x_min=-PI, x_max=PI)
        # grid.add(func)
        # add(grid)
        # grid.faded_lines[4:9].fade(1)
        # grid.faded_lines[12:].fade(1)
        # grid.background_lines[4:9].fade(1)
        # grid.background_lines[12:].fade(1)
        #play(Rotating(func, radians=PI, axis=UR, about_point=ORIGIN, run_time=2, rate_func=smooth))

        grid.generate_target()
        # grid.target.prepare_for_nonlinear_transform()
        grid.target.apply_function(lambda p: rotate(p))

        play(MoveToTarget(grid, run_time=4))
        wait(3)
