from eq import *

#
# class GeometricSeries(SceneBase):
#
#     def construct(self):
#
#         Eq.new_line = False
#         Eq.show_operation = True
#         Eq.animate = True
#         Eq.run_time = 1.5
#
#         from abc_n import a, x
#
#         # eq = Eq(r"\sum_{n=0}^{n-1}", r"a^n = {1 - a^n \over 1 - a}")
#         # eq2 = eq * a
#         # add(eq)
#
#         eq1 = Eq(r"x + 2^x = 0")
#         add(eq1)
#         eq2 = eq1 - (x + 2**x)
#         wait()
#

class GeometricSeries(SceneBase):

    def construct(self):

        show_title("Geometric Series")

        Eq.new_line = True
        Eq.show_operation = True
        Eq.animate = True
        Eq.run_time = 1.5

        eq1, eq2, eq3, eq4, eq5 = load_equations("geometric-series.tex")
        eq1.shift(UP)
        eq1_copy = eq1.copy()

        from abc_n import a, n, A_n

        play(Write(eq1))
        eq6 = eq1.divide_by_expr(a, expressions={1: eq2})
        Eq.align = True
        Eq.new_line = False
        eq1.hide_operation()
        eq7 = eq1.subtract_expr(a**n, expressions={1: eq3}, new_line=False)
        eq8 = eq6.add_expr(-1/a, expressions={1: eq3})
        eq5 = Eq.remove_equal_expression(eq8, eq7)
        play(FadeIn(eq1_copy))
        eq10 = eq5.mult_by_expr(a, expressions={1: r"a A_n - a^{n+1}"})
        eq11 = eq10 + (-A_n + a**(n+1))
        eq12 = eq11 / (a - 1)
        eq13 = Eq.remove_equal_expression(eq1_copy, eq12)
        wait()
        console.print(":thumbs_up:")
