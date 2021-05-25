from geometryobjects import *
from eq import *
import sympy
import math
from typing import Union


def indexed_latex(name: Union[str, sympy.Symbol], *indices):
    if type(name) is sympy.Symbol:
        name = name.name
    s = name + r"_{\scriptscriptstyle{"
    for i in indices:
        if type(i) == int and len(indices) == 1:
            if i > 10:
                order = int(math.log(i, 10))
                for n in range(order, -1, -1):
                    s += str(i // 10 ** n)
                    i -= (i // 10 ** n * 10 ** n)
            else:
                s += str(i)
        else:
            s += str(i)
    s += "}}"
    return s


class MatrixMultiplication(SceneBase2d):

    def construct(self):
        _ = indexed_latex
        from sympy.abc import a, b
        m_A = Matrix(([_(a, 11), _(a, 12)],
                      [_(a, 21), _(a, 22)])).shift(3 * LEFT)
        add(m_A)
        add(Matrix(([_(b, 11), _(b, 12)],
                    [_(b, 21), _(b, 22)])).next_to(m_A, UR).shift(2 * RIGHT))
        C_tex = (r"\begin{{bmatrix}} {}{}+{}{} & {}{}+{}{} \\" +
                 r"{}{}+{}{} & {}{}+{}{} \end{{bmatrix}}").format(
            _(a, 11), _(b, 11), _(a, 12), _(b, 21),  _(a, 11), _(b, 12), _(a, 12), _(b, 22),
            _(a, 21), _(b, 11), _(a, 22), _(b, 21),  _(a, 21), _(b, 12), _(a, 22), _(b, 22))
        add(MathTex(C_tex).next_to(m_A, RIGHT))
        wait()


class InverseMatrix(SceneBase2d, MovingCameraScene):

    def setup(self):
        MovingCameraScene.setup(self)

    def construct(self):
        # p = Vec(1, 1)
        p = np.array((3, 2))
        theta = np.radians(30)
        c, s = np.cos(theta), np.sin(theta)
        A = np.array(((c, -s), (s, c)))
        p_prime = A.dot(p)
        #        self.camera_frame.scale(0.5)

        DOT_RADIUS = 0
        Vector.CONFIG["tip_length"] = 0.2
        Vector.CONFIG["stroke_width"] = 3
        m_v_p = Vector(p, color=BLUE)
        m_v_p_prime = Vector(p_prime, color=GREEN)
        add(m_v_p, m_v_p_prime)
        m_theta_arc = Arc(radius=1, start_angle=m_v_p.get_angle(), angle=theta)
        m_theta_label = MathTex(r"\theta")
        m_theta_label.next_to(m_theta_arc, DL, buff=-0.18)
        add(m_theta_arc, m_theta_label)
        m_p = LDot("p", color=BLUE, radius=DOT_RADIUS)
        m_p_prime = LDot("p'", color=GREEN, radius=DOT_RADIUS)
        m_p.move_to(p)
        m_p_prime.move_to(p_prime)
        m_axes = NumberPlane(axis_config={"number_scale_val": 0.5, "include_ticks": True})
        add(m_axes.get_coordinate_labels([1], [1]))
        add(m_axes)
        Matrix.CONFIG["left_bracket"] = "\\big("
        Matrix.CONFIG["right_bracket"] = "\\big)"
        m_A = Matrix([[r"\cos \theta", r"\sin \theta"], [r"-\sin \theta", r"\cos \theta"]])
        m_A.move_to(vec(config.frame_width / 2 - 2, - (config.frame_height / 2 - 2)))
        add(m_p, m_p_prime, m_A)
        Eq.show_operation = False
        Eq.animate = False
        wait()
        self.m_A = m_A
        self.m_p = m_p
        self.m_p_prime = m_p_prime
