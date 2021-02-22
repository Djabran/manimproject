from geometryobjects import *


class PtolemaeusTheorem(SceneBase):

    def construct(self):
        # self.show_title("Ptolemaeus' Theorem")
        # self.camera.frame_center = vec(2, 0, 0)
        O = NullCircle()
        A = NullPoint(O, 0.0, "A", run_time=0.1)
        B = NullPoint(O, 0.3, "B", run_time=0.1)
        C = NullPoint(O, 0.6, "C", run_time=0.1)
        D = NullPoint(O, 0.8, "D", run_time=0.1)
        AB = Line(A.dot(), B.dot())
        BC = Line(B.dot(), C.dot())
        CD = Line(C.dot(), D.dot())
        AD = Line(A.dot(), D.dot())
        AC = Line(A.dot(), C.dot())
        BD = Line(B.dot(), D.dot())
        add(AB, BC, CD, AD, AC, BD)
        stroke_width, stroke_color = AB.get_stroke_width(), AB.get_stroke_color()
        eq = MathTex("AB", r"\cdot", "CD", "+", "AD", r"\cdot", "BC", "=", "BD", r"\cdot", "AC")
        math_color = eq.get_color()
        add(eq.to_edge(UP))
        wait()

        def hl(line, math_part, start, end):
            line.set_stroke(YELLOW, 7)
            eq.get_part_by_tex(math_part).set_color(YELLOW)
            start.set_color(YELLOW)
            end.set_color(YELLOW)
            wait()
            line.set_stroke(stroke_color, stroke_width)
            eq.get_part_by_tex(math_part).set_color(math_color)
            start.set_color(stroke_color)
            end.set_color(stroke_color)
            wait()

        hl(AB, "AB", A, B)
        hl(CD, "CD", C, D)
        hl(AD, "AD", A, D)
        hl(BC, "BC", B, C)
        hl(BD, "BD", B, D)
        hl(AC, "AC", A, C)



