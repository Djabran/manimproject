from geometryobjects import *
from eq import *


class LogBridge(SceneBase):

    def construct(self):

        self.show_title("Log Bridge")

        rects = [Rectangle(width=2, height=0.5).to_edge(DL)]
        C_n = 1.0
        C_n_tex = MathTex("C_1=").next_to(rects[-1], RIGHT).scale(0.7)
        eq = Eq(1).scale(0.7).next_to(C_n_tex, RIGHT, buff=0.1)
        play(ShowCreation(rects[0]), Write(C_n_tex), Write(eq))

        Eq.show_operation = False
        Eq.animate = False

        eq_C_n = Eq(1)

        for i in range(10):
            rects.append(rects[0].copy().next_to(rects[0], UP, buff=0).shift(RIGHT * C_n + UP * i / 2))
            C_n_tex = MathTex(f"C_{{{i+2}}}=").next_to(rects[-1], RIGHT).scale(0.7)
            eq_C_n, anim = eq_C_n + sympy.sympify(1) / sympy.sympify(i+2)
            eq_C_n.next_to(C_n_tex, RIGHT, buff=0.1).scale(0.4)
            play(ShowCreation(rects[-1]), Write(C_n_tex), Write(eq_C_n))
            C_n = C_n + 1 / (i+2)
        wait()
