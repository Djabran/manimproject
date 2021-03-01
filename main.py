import os
import re
import manim
import logging
from itertools import chain
from manim import *
from manim.__main__ import main, open_file_if_needed, open_media_file
from colors import *
from manim.constants import *
from glob import glob
#import monkey
from geometryobjects import *
from eq import *
from rich import print
import rich.traceback
# from PtolemaeusTheorem import *
from LogBridge import *

rich.traceback.install()

config.background_color = '#002b36'

logger.setLevel(logging.WARNING)


class ConeTest(SceneBase):

    def construct(self):
        self.set_camera_orientation(60*DEGREES, -75*DEGREES, distance=20, gamma=0*DEGREES)
        Axes3D(self)
        Cone(RIGHT, ORIGIN, radius=0.2)
        self.wait()


class ProjectiveGeometry3d(SceneBase):

    def construct(self):
        self.show_title("Planes")

        self.set_camera_orientation(60*DEGREES, -90*DEGREES, distance=20, gamma=0*DEGREES)

        Axes3D(self)

        color_map = ColorMap({
            r"\bar{P}": BLUE,
            "P": BLUE,
            "Q": NULL_POINT_COLOR_BRIGHT
        })

        P = LDot("P", pos=(1, 1, 1), color=BLUE, radius=0.05, draw_bases=True)

        textblock = TextBlock(color_map=color_map, remover=False)
        textblock += MText("We want to construct a plane through the origin")
        textblock += MText("perpendicular to a given point ")("$P$")

        # v = CurvedArrow(start_point=ORIGIN, end_point=np.array([1,1,1]), angle=0,
        #                     color=GOLD_D)
        # v.rotate(45 * DEGREES, (1,-1,0))
        # v.shift(np.array([0,0,0.5]))
        # self.add(v)

        v = Line(vec(0,0,0), vec(1,1,1))
        self.add(v)

        # self.add(LabeledVector(P.pos()))

        self.begin_ambient_camera_rotation(1)
        self.wait(TAU + 15 * DEGREES)


class LineIntersection(Scene):
    def construct(self):
        AXES_COLOR = GREY
        number_plane = NumberPlane(
            axis_config={
                "stroke_color": AXES_COLOR,
                "stroke_width": 1,
                "include_ticks": True,
                "include_tip": False,
                "line_to_number_buff": SMALL_BUFF,
                "label_direction": DR,
                "number_scale_val": 0.5,
            },
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_width": 1,
                "stroke_opacity": 0.3,
            })

        title = Tex("Intersection between two lines")
        self.wait(1)
        self.play(FadeOut(title))

        tex_a_0 = r"a_{\scriptscriptstyle 0}"
        tex_a_0_x = "{" + tex_a_0 + "}_x"
        tex_a_0_y = "{" + tex_a_0 + "}_y"
        tex_a_1 = r"a_{\scriptscriptstyle 1}"
        tex_a_1_x = "{" + tex_a_1 + "}_x"
        tex_a_1_y = "{" + tex_a_1 + "}_y"
        tex_b_0 = r"b_{\scriptscriptstyle 0}"
        tex_b_0_x = "{" + tex_b_0 + "}_x"
        tex_b_0_y = "{" + tex_b_0 + "}_y"
        tex_b_1 = r"b_{\scriptscriptstyle 1}"
        tex_b_0_x = "{" + tex_b_1 + "}_x"
        tex_b_0_y = "{" + tex_b_1 + "}_y"

        color_map = ColorMap({
            "a": BLUE,
            "b": GREEN,
        })

        text = TextBlock("Two lines ", "$a$", " and ", "$b$", " each given by a pair of points", color_map=color_map)
        text.last().move_to(3*UP)
        self.play(Write(text.last(), run_time=0.5))
        self.bring_to_front(text.last())
        coords_a_0 = (2, -2)
        coords_a_1 = (2, 2.5)
        coords_b_0 = (0, -3,)
        coords_b_1 = (4, 1)
        a_0 = LDot(f"{tex_a_0}{coords_a_0}", coords_a_0, color=BLUE)
        a_1 = LDot(f"{tex_a_1}{coords_a_1}", coords_a_1, color=BLUE)
        b_0 = LDot(f"{tex_b_0}{coords_b_0}", coords_b_0, color=GREEN)
        b_1 = LDot(f"{tex_b_1}{coords_b_1}", coords_b_1, color=GREEN)
        self.play(GrowFromCenter(a_0), FadeIn(a_0.label),
                  GrowFromCenter(a_1), FadeIn(a_1.label),
                  GrowFromCenter(b_0), FadeIn(b_0.label),
                  GrowFromCenter(b_1), FadeIn(b_1.label))
        a = LabeledLine("a", coords_a_0, coords_a_1, color=BLUE)
        b = LabeledLine("b", coords_b_0, coords_b_1, color=GREEN)
        # a.line().rotate(TAU / 8)
        label_np_y = Tex("1").move_to(np.array([0.2, 0.83, 0])).set_color(AXES_COLOR).scale(0.6)
        label_np_x = Tex("1").move_to(np.array([1.2, -0.17, 0])).set_color(AXES_COLOR).scale(0.6)

        coords = intersect(a.line(), b.line(), trace_curve=False)

        if is_coords(coords):
            P = LDot("P", coords)
            self.add(number_plane, label_np_x, label_np_y, a, b, P.group)
        else:
            assert is_number(coords)
            self.add(number_plane, label_np_x, label_np_y, a, b)
            print(f"Distance: {coords}")
        self.bring_to_front(text.last())

        a_0_x = coords_a_0[0]
        a_0_y = coords_a_0[1]
        a_1_x = coords_a_1[0]
        a_1_y = coords_a_1[1]
        b_0_x = coords_b_0[0]
        b_0_y = coords_b_0[1]
        b_1_x = coords_b_1[0]
        b_1_y = coords_b_1[1]
        a_ = (a_1_y - a_0_y) / (a_1_x - a_0_x) if a_1_x != a_0_x else None

        # text.append("$a'$", "$=$",
        #             rf"$\frac{{{{{tex_a_1}}}_y-{{{tex_a_0}}}_y}}{{{{{tex_a_1}}}_x - {{{tex_a_0}}}_x}}$",
        #             f"$={a_}$",
        #             pos=4*LEFT + 2*UP)

        text.append_tex(tex_a_0_y, "+", "a'", "(x-", tex_a_0_x, ")=",
                        tex_b_0_y, "+", "b'", "(x-", tex_b_0_x, ")",
                        color_map=color_map,
                        pos=3*LEFT + 2*UP)

        text.append_tex(tex_a_0_y, "+", "a'", "(x-", tex_a_0_x, ")=",
                        tex_b_0_y, "+", "b'", "(x-", tex_b_0_x, ")",
                        pos=3*LEFT + 2*UP)

        # text.append(f"${{{tex_a_0}}}_y$", "$-$", f"${{{tex_b_0}}}_y$", "$=$",
        #             "$b'$", "$(x-$", f"${{{tex_b_0}}}_x$", "$)-$", "$a'$", "$(x-$", f"${{{tex_a_0}}}_x$", "$)$",
        #             pos=3*LEFT + UP)

        # text.append_tex(tex_a_0_y, "-", tex_b_0_y, "=",
        #                 "x(", "b'", "-", "a'", ")-", tex_b_0_x, "b'", "+", tex_a_0_x, "a'",
        #                 pos=3*LEFT + UP)

        text.append_tex("x={", tex_a_0_y, "-", tex_b_0_y, "+",
                        tex_b_0_x, "b'", "-", tex_a_0_x, "a'", r" \over ", "b'", "-", "a'", "}",
                        pos=3*LEFT + UP)

        # text.append_tex(r"1", r"\over", r"7",
        #                 pos=3*LEFT)

        b_ = (b_1_y - b_0_y) / (b_1_x - b_0_x) if b_1_x != b_0_x else None

        # text.append("$b'$", "$=$",
        #             rf"$\frac{{{{{tex_b_1}}}_y - {{{tex_b_0}}}_y}}{{{{{tex_b_1}}}_x - {{{tex_b_0}}}_x}}$",
        #             f"$={b_}$",
        #             pos=4*LEFT + UP)

        text.append("$a$", "$=$", f"${{{tex_a_0}}}_y$", "$+$", "$a'$", "$(x-$", f"${{{tex_a_0}}}_x$", "$)$",
                    pos=4*LEFT + 2*DOWN)

        text.append("$b$", "$=$", f"${{{tex_b_0}}}_y$", "$+$", "$b'$", "$(x-$", f"${{{tex_b_0}}}_x$", "$)$",
                    pos=4*LEFT + 3*DOWN)


class ProjectiveGeometry(SceneBase):

    def construct(self):
        self.show_title("Projective Geometry 3D")
        axes = ThreeDAxes(x_min=-3.5, y_min=-3.5, x_max=5, y_max=3, z_min=-3.5, z_max=3.5,
                          num_axis_pieces=20,
                          x_axis_config={'label_direction': RIGHT, 'stroke_width': 1},
                          y_axis_config={'label_direction': UP, 'stroke_width': 1},
                          z_axis_config={'include_tip': True, 'stroke_width': 1, 'include_ticks': True})
        labels = axes.get_axis_labels()
        labels.scale(0.7)
        # self.add_fixed_in_frame_mobjects(labels)
        self.add_fixed_orientation_mobjects(labels)
        # axes.add(labels)
        self.add(axes)
        """
        By default camera is looking down from z-axis toward xy-plane
        phi = 0, theta = 0: x pointing down, y pointing right

        phi: rotation from z-axis toward xy-plane
        theta: rotation around z-axis clockwise

        To get an x-axis pointing right and a y-axis pointing up:
        phi = 0, theta = -90° 
        """
        # self.set_camera_orientation(-15 * DEGREES, -30 * DEGREES, distance=20, gamma=60*DEGREES)
        self.set_camera_orientation(60*DEGREES, -90*DEGREES, distance=20, gamma=0*DEGREES)
        null_sphere = NullSphere()

        null_point = NullPoint(null_sphere, (Proportion(0), Proportion(1, 5)), label="O")
        print(f"null_point: {null_point.get_x()},{null_point.get_y()},{null_point.get_z()}")
        self.add(null_sphere)

        null_plane = NullPlane(point=null_point, proportion=0)
        # self.begin_ambient_camera_rotation(0.1)
        # self.wait(20)

        P_x = r"P_{\scriptscriptstyle x}"
        Q_x = r"Q_{\scriptscriptstyle x}"
        P_y = r"P_{\scriptscriptstyle y}"
        Q_y = r"Q_{\scriptscriptstyle y}"
        P_z = r"P_{\scriptscriptstyle z}"
        Q_z = r"Q_{\scriptscriptstyle z}"

        color_map = ColorMap({
            r"\bar{P}": BLUE,
            "P": BLUE,
            "Q": NULL_POINT_COLOR_BRIGHT
        })

        textblock = TextBlock(color_map=color_map, remover=False)
        textblock += MText() + "We have a point " + "$P$" + " on the null sphere"
        textblock += MText() + "Let's find the plane through the origin parallel to the"
        textblock += MText() + "tangent plane of " + "$P$"
        textblock += MText() + "For each point " + "$Q$" + " on the tangent plane:"

        textblock += TeX() + r"\vec n \cdot (" + "Q" + "-" + "P" + ")=0"
        textblock += MText() + "For " + r"$\vec n$" + " we can choose " + "$P$"

        textblock.clear()

        # textblock += TeX() + tex_P_x + "(" + tex_Q_x + "-" + tex_P_x + ")+" +\
        #                      tex_P_y + "(" + tex_Q_y + "-" + tex_P_y + ")+" +\
        #                      tex_P_z + "(" + tex_Q_z + "-" + tex_P_z + ") = 0"
        # textblock += TeX() + tex_P_x + tex_Q_x + "-" + (tex_P_x + "^2") + "+" +\
        #     tex_P_y + tex_Q_y + "-" + (tex_P_y + "^2") +\
        #     "+" + tex_P_z + tex_Q_z + "-" + (tex_P_z + "^2") + "=0"

        textblock += TeX\
            (P_x)(Q_x)("+")(P_y)(Q_y)("+")(P_z)(Q_z)("=")\
            (P_x + "^2")("+")(P_y + "^2")("+")(P_z + "^2")

        textblock += TeX\
            (Q_x)("={")(P_x + "^2")("+")(P_y + "^2")("+")(P_z + "^2")\
            ("-")(P_y)(Q_y)("-")(P_z)(Q_z)(r"\over")(P_x)("}")

        textblock.clear()
        textblock += MText() + "We need two vectors wich are perpendicular to " + r"$\vec n$"
        v_1 = r"\vec v_{\scriptscriptstyle1}"
        v_1_x = r"\vec v_{\scriptscriptstyle1_{\scriptscriptstyle x}}"
        n_x = r"\vec n_{\scriptscriptstyle x}"
        v_1_y = r"\vec v_{\scriptscriptstyle1_{\scriptscriptstyle y}}"
        n_y = r"\vec n_{\scriptscriptstyle y}"
        v_1_z = r"\vec v_{\scriptscriptstyle1_{\scriptscriptstyle z}}"
        n_z = r"\vec n_{\scriptscriptstyle z}"

        textblock += TeX\
            (rf"{v_1} \cdot \vec n = 0")
        textblock += TeX\
            (v_1_x)(n_x)("+")(v_1_y)(n_y)("+")(v_1_z)(n_z)("=0")

        N = Vector(null_plane.n, tip_length=0.1, stroke_width=3)

        if null_plane.n[0] == 0 and null_plane.n[2] == 0:
            v_1 = normalize(np.cross(null_plane.n, vec(1, 0, 0)))
            V_1 = Vector(v_1, tip_length=0.1, stroke_width=3)
            v_2 = np.cross(null_plane.n, vec(0, 0, 1))
            V_2 = Vector(v_2, tip_length=0.1, stroke_width=3)
        elif null_plane.n[0] != 0 and null_plane.n[1] != 0:
            v_1 = normalize(np.cross(null_plane.n, vec(1, 0, 0)))
            v_2 = normalize(np.cross(null_plane.n, vec(0, 1, 0)))
            # v1_z = 0
            # v1_y = 0
            # nx, ny, nz = tuple(null_plane.n)
            # v1_x = (-v1_y * ny - v1_z * nz) / nx
            # v1_y = (-v1_x * nx - v1_z * nz) / ny
            # v_1 = normalize(vec(v1_x, v1_y, v1_z))
            # v_2 = normalize(np.cross(v_1, null_plane.n))
            V_1 = Vector(v_1, tip_length=0.1, stroke_width=3)
            # V_2 = Vector(v_2, tip_length=0.1, stroke_width=3)

        print(f"v_1: {v_1} v_2: {v_2}")
        self.bring_to_front(V_1)
        # self.add(N, V_1, V_2)
        self.add(N, V_1)
        self.wait()
        self.begin_ambient_camera_rotation(0.1)
        self.wait(10)


class ProjectiveGeometry1(SceneBase):

    def construct(self):
        self.show_title("Projective Geometry 2D")

        self.subscene_null_point_and_line()

        textblock = TextBlock("The same principle can be applied in 3D")
        textblock.append("where the we have planes instead of lines")
        textblock.append("and balls instead of circles")
        del textblock

        self.wait()

    def subscene_null_point_and_line(self):
        number_plane = NumberPlane(
            axis_config={
                "stroke_color": BLUE_E,
                "stroke_width": 2,
                "include_ticks": False,
                "include_tip": False,
                "line_to_number_buff": SMALL_BUFF,
                "label_direction": DR,
                "number_scale_val": 0.5,
            })

        null_circle = NullCircle(run_time=0.5)

        color_map = ColorMap({
            "O": NULL_POINT_COLOR,
            r"O^\circ": NULL_POINT_COLOR,
            "p": NULL_POINT_COLOR,
            "P": NULL_POINT_COLOR_BRIGHT,
            "g": GREEN,
            "P'": GREEN
        })

        textblock = TextBlock("Choose an arbitrary point ", "$P$", " on ", r"$O^\circ$", color_map=color_map)

        print("create null point")

        P = NullPoint(null_circle, 0, "P", color=NULL_POINT_COLOR_BRIGHT)

        print("rotate null point once counter-clockwise")

        def point_rotation_updater(obj, v=None):
            assert obj is P.dot()
            if v is None:
                v = P.value_tracker.get_value()
            # print(f"point_rotation_updater - {v}")
            obj.group.set_x(cos(v * TAU) * NULL_CIRCLE_RADIUS)
            obj.group.set_y(sin(v * TAU) * NULL_CIRCLE_RADIUS)

        P.animate(1.0, updater=point_rotation_updater, call_updater=False, run_time=3, rate_func=smooth)

        textblock.append("$P$", " is represented by a line ", "$p$", " through the origin")

        print("create null line")

        p = NullLine(0, label="p")
        self.bring_to_front(P)
        textblock.bring_to_front()
        self.play(FadeOut(p.label()))

        print("create and rotate null point and line")

        def line_rotation_updater(obj):
            assert obj is p.line()
            v = p.value_tracker.get_value()
            p.reset_rotation()
            if v != 1 and v != 0:
                p.rotate(v * TAU, about_point=ORIGIN)
            # print(f"line_rotation_updater - v: {v}")

        target_alpha_P = 1.065
        run_time_P = target_alpha_P * 4.5

        UpdateGroup((p, line_rotation_updater),
                    (P, lambda obj: point_rotation_updater(obj, p.value_tracker.get_value())),
                    end=target_alpha_P,
                    run_time=run_time_P)

        self.wait()

        print("create projection line g")

        textblock = TextBlock("$P$", " can be projected on a line ", "$g$", " not going through ", "$O$",
                              color_map=color_map)

        g = LabeledLine("g", (-config.frame_width / 2, -2.3), (config.frame_width / 2, -2.3), color=GREEN)

        textblock.append("to a point ", "$P'$", " := ", "$p$", r"$\cross$", "$g$")

        coords = intersect(p.line(), g.line(), trace_curve=False, tolerance=0.045, use_average=False)
        P_ = LDot("P'", color=GREEN)
        if is_coords(coords):
            P_.move_to(coords)
            self.play(GrowFromCenter(P_.dot()), FadeIn(P_.label()))

        def intersection_updater(obj):
            coords_P_ = intersect(p.line(), g.line(), tolerance=0.045, use_average=False)
            # print(f"angle p: {p.line().get_angle()}, angle g: {g.line().get_angle()}")
            if is_coords(coords_P_):
                obj.group.move_to(coords_P_)
            else:
                obj.group.set_x(0)
                obj.group.set_y(0)

        UpdateGroup((p, line_rotation_updater),
                    (P_, intersection_updater),
                    (P, lambda obj: point_rotation_updater(obj, p.value_tracker.get_value())),
                    run_time=4.5)

        textblock = TextBlock("When ", "$p$", " and ", "$g$", " are parallel, we project ", "$P$", " to ", "$O$",
                              color_map=color_map)
        UpdateGroup((p, line_rotation_updater),
                    (P_, intersection_updater),
                    (P, lambda obj: point_rotation_updater(obj, p.value_tracker.get_value())),
                    start=target_alpha_P, end=1.0,
                    func=p.value_tracker.set_value,
                    run_time=1.0)

        play(FadeOut(P_))
        play(FadeOut(P), FadeOut(p), FadeOut(g), FadeOut(null_circle))
        play(FadeOut(number_plane))


# class ProjectiveGeometry(GeometricSeries): pass
# class ProjectiveGeometry(ProjectiveGeometry3d): pass

class SnapshotScene(SceneBase):

    def construct(self):
        Eq.new_line = True
        Eq.animate = True
        Eq.align = True
        eq = Eq(r"1 \cdot 2 \cdot 3 = 4 \cdot 5 \cdot 6")
        eq1 = add(eq)
        eq2 = eq1 * 2

        wait()


def openexplorer():
    if get_current_scene():
        config.show_in_file_browser = True
        open_file_if_needed(get_current_scene().renderer.file_writer)
    else:
        print("[red]No current scene")


def playvid():
    if get_current_scene():
        filename = get_current_scene().__class__.__name__ + ".mp4"
        folder = os.path.join(config.media_dir, "videos", f"{config.pixel_height}p{config.frame_rate}")
        filepath = os.path.normpath(os.path.join(folder, filename))
        os.system(filepath)
        # config.show_in_file_browser = True
        # open_file_if_needed(get_current_scene().renderer.file_writer)
    else:
        print("[red]No current scene")


def mm(production_quality=False):

    if production_quality:
        q = QUALITIES["production_quality"]
    else:
        q = QUALITIES["low_quality"]
    for attr in ("pixel_width", "pixel_height", "frame_rate"):
        setattr(config, attr, q[attr])

    # SnapshotScene()
    # ProjectiveGeometry1()
    # PtolemaeusTheorem()
    config.save_last_frame = True
    LogBridge()
    get_current_scene().render()


if __name__ == '__main__':
    mm()
