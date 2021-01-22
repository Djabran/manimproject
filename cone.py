import re
import manim
import logging
from itertools import chain
from manim import *
from manim.__main__ import main
from colors import *
from manim.constants import *
from glob import glob
#import monkey
from geometryobjects import *
from eq import *

config.background_color = '#002b36'

logger.setLevel(logging.WARNING)


class ConeScene(SceneBase):

    def construct(self):
        self.set_camera_orientation(60*DEGREES, -75*DEGREES, distance=20, gamma=0*DEGREES)
        Axes3D()
        # Cone(tip=OUT, foot=vec(1,1,1), radius=1)
        color_map = ColorMap({
            "T": GREEN,
            "F": YELLOW,
            "r": RED,
            r"\overline{TF}": YELLOW
        })
        # [x.scale(0.5) for x in tb.texts]
        T_coords = OUT
        F_coords = vec(1, 1, 1)
        TF = Line(start=T_coords, end=F_coords, color=YELLOW, stroke_width=3)
        # v = Vector(direction=F_coords - T_coords, color=YELLOW, stroke_width=3,
        #            preserve_tip_size_when_scaling=False)
        dot = Dot(color=GREEN, radius=0.05).move_to(T_coords)
        T = VGroup(dot, Tex("T", color=GREEN).next_to(dot, RIGHT, buff=-0.01).scale(0.7))
        dot = Dot(color=YELLOW, radius=0.05).move_to(F_coords)
        F = VGroup(dot, Tex("F", color=YELLOW).next_to(dot, RIGHT, buff=-0.01).scale(0.7))
        C = Circle(arc_center=F_coords, radius=1)
        C.rotate(TAU / 4, vec(1, -1, 0), about_point=F_coords)
        P_coords = F_coords + vec(0, 0, 1)
        P = Sphere(fill_color=RED, stroke_color=RED, radius=0.05).move_to(P_coords)
        vt = ValueTracker(0)
        r = DashedLine(start=F, end=P.get_center(), color=RED, stroke_width=2)
        P_orig = P.copy()
        r_orig = r.copy()

        def updater_P(obj, index=0):
            assert obj is P
            console.print(f"value: {vt.get_value()}")
            obj.become(P_orig)
            obj.rotate(vt.get_value() * TAU, F_coords - T_coords, about_point=F_coords)

        def updater_r(obj, index=0):
            assert obj is r
            obj.become(r_orig)
            obj.rotate(vt.get_value() * TAU, F_coords - T_coords, about_point=F_coords)

        add(TF, r, C)
        self.add_fixed_orientation_mobjects(T, F)
        # add_fixed_orientation_mobjects(P)
        add(P)
        P.add_updater(updater_P, call_updater=True)
        r.add_updater(updater_r, call_updater=True)

        tb = TextBlock("We have a line defined by two points ", "$T$", " and ",  "$F$",
                       mobject_type=Tex, color_map=color_map, anim=None, scale=0.519, remover=False, run_time=0.2)
        tb.append("Looking for a parametric equation of a cone with the tip at ", "$T$", " and its axis going through ", "$F$")
        tb.append("A circle perpendicular to ", r"$\overline{TF}$",  " around ", "$F$", " has the given radius ", "$r$")
        self.add_fixed_in_frame_mobjects(*tb.texts)

        bring_to_front(T, F)
        play(vt.set_value, 1, run_time=3)
        Q = Sphere(fill_color=RED, stroke_color=RED, radius=0.05).move_to(F_coords)
        TP = Line(T_coords, P_coords, stroke_width=3, color='BLUE')
        TP_orig = TP.copy()

        def updater_TP(obj, index=0):
            assert obj is TP
            obj.become(TP_orig)
            obj.rotate(vt.get_value() * TAU, F_coords - T_coords, about_point=F_coords)

        TP.add_updater(updater_TP, call_updater=True)
        add(TP)
        vt.set_value(0)
        tb.fade_out()
        play(vt.set_value, 1, run_time=3)

        self.begin_ambient_camera_rotation(0.5)
        self.wait(1)
        self.stop_ambient_camera_rotation()
        self.begin_ambient_camera_rotation(-0.5)
        self.wait(1)
