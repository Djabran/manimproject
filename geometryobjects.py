from manimlib.imports import *
from manimlib.constants import BLACK, BLUE_A, BLUE_B, BLUE, BLUE_D, BLUE_E, DARK_BLUE, DARK_GREY, GREEN_A, GREEN_B,\
    GREEN, GREEN_D, GREEN_E, GREY, LIGHT_GREY, MAROON_A, MAROON_B, RED, RED_D, TEAL, WHITE, YELLOW

from math import *

from basicobjects import *
import sympy

NULL_CIRCLE_RADIUS = 2
NULL_POINT_COLOR = BLUE_D
NULL_POINT_COLOR_BRIGHT = BLUE_B
# NULL_SPHERE_RESOLUTION = (24, 36)
NULL_SPHERE_RESOLUTION = (12, 24)


def normalize(x):
    n = np.linalg.norm(x)
    return x / n if n else x


def is_coords(x):
    return type(x) is np.ndarray


def is_number(x):
    return type(x) is np.float64


def dxrange(in_val,end_val,step=1):
    return list(np.arange(in_val,end_val+step,step))


def get_points_from_curve(self, vmob, dx=0.005):
    coords = []
    for point in dxrange(0, 1, dx):
        dot = Dot(vmob.point_from_proportion(point))
        coords.append(dot.get_center())
    return coords


def intersect(a, b, trace_curve=False, tolerance=0.05, radius_error=0.2, use_average=True):
    """
    a,b are straight lines:

        if a and b are parallel, return the distance beetween both lines,
        else return the coords of the intersection
    """
    if not trace_curve and isinstance(a, Line) and isinstance(b, Line):
        a_0 = Vec2(a.points[0])
        a_1 = Vec2(a.points[3])
        b_0 = Vec2(b.points[0])
        b_1 = Vec2(b.points[3])

        if a_1.x == a_0.x:
            # a is vertical
            if b_1.x == b_0.x:
                # a and b are vertical, return distance
                return get_norm(a.get_center() - b.get_center())
            else:
                # only a is vertical, return intersection
                b_ = (b_1.y - b_0.y) / (b_1.x - b_0.x)
                y = b_0.y + b_ * (a_0.x - b_0.x)
                return np.array([a_0.x, y, 0])
        else:
            # a is not vertical
            if b_1.x == b_0.x:
                # only b is vertical, return intersection
                a_ = (a_1.y - a_0.y) / (a_1.x - a_0.x)
                y = a_0.y + a_ * (b_0.x - a_0.x)
                return np.array([b_0.x, y, 0])
            else:
                # none are vertical
                a_ = (a_1.y - a_0.y) / (a_1.x - a_0.x)
                b_ = (b_1.y - b_0.y) / (b_1.x - b_0.x)
                if a_ == b_:
                    # a and b are parallel, return distance
                    return get_norm(a.get_center() - b.get_center())
                else:
                    # a and b are not parallel and none is vertical, return intersection
                    x = (a_0.y - b_1.y + b_1.x * b_ - a_0.x * a_) / (b_ - a_)
                    y = a_0.y + a_ * (x - a_0.x)
                    return np.array([x, y, 0])
    else:
        coords_1 = get_points_from_curve(a)
        coords_2 = get_points_from_curve(b)
        intersections = []
        for coord_1 in coords_1:
            for coord_2 in coords_2:
                distance_between_points = get_norm(coord_1 - coord_2)
                if use_average:
                    coord_3 = (coord_2 - coord_1) / 2
                    average_point = coord_1 + coord_3
                else:
                    average_point = coord_2
                if len(intersections) > 0 and distance_between_points < tolerance:
                    last_intersection=intersections[-1]
                    distance_between_previus_point = get_norm(average_point - last_intersection)
                    if distance_between_previus_point > radius_error:
                        intersections.append(average_point)
                if len(intersections) == 0 and distance_between_points < tolerance:
                    intersections.append(average_point)

        if isinstance(a, Line) and isinstance(b, Line) and len(intersections) > 0:
            return intersections[0]
        else:
            return intersections if len(intersections) > 0 else None


def phi_of_vector(vector):
    xy = complex(*vector[:2])
    if xy == 0:
        return 0;
    a = ((vector[:1])**2 + (vector[1:2])**2)**(1/2)
    vector[0] = a
    vector[1] = vector[2]
    return np.angle(complex(*vector[:2]))

class LabeledVector(VGroup):

    def __init__(self):

# class LabeledVector(CurvedArrow):
#
#     def __init__(self, *args, stroke_width=2, color=WHITE, **kwargs):
#         super().__init__(start_point=ORIGIN, end_point=args[0], angle=0, num_components=2,
#                          stroke_width=stroke_width, color=color)
#         self.points[-1] = args[0]
#
    # def position_tip(self, tip, at_start=False):
    #     # Last two control points, defining both
    #     # the end, and the tangency direction
    #     if at_start:
    #         anchor = self.get_start()
    #         handle = self.get_first_handle()
    #     else:
    #         handle = self.get_last_handle()
    #         anchor = self.get_end()
    #     tip.rotate(angle_of_vector(handle - anchor) - PI - tip.get_angle())
    #
    #     angle = angle_of_vector(handle - anchor) + PI / 2
    #     a = np.array((np.cos(angle), np.sin(angle), 0))
    #     tip.rotate(-phi_of_vector(handle - anchor), a)
    #
    #     tip.shift(anchor - tip.get_tip_point())
    #     # self.rotate(-phi_of_vector(handle - anchor), a)
    #     return tip


class NullCircle(Circle):

    def __init__(self, scene=None, anim=None, run_time=1.0, **kwargs):
        super().__init__(radius=NULL_CIRCLE_RADIUS, color=NULL_POINT_COLOR, **kwargs)
        if scene is not None:
            if anim is None:
                anim = Write(self)
            scene.play(anim, run_time=run_time)


class LabeledDot(VGroup, SceneElement):

    label_offset = np.array([0.3, -0.3, 0])

    def __init__(self, *label_args, pos=(0, 0), radius=0.08, color=WHITE, dot_class=None, scene=None,
                 anims=None, draw_bases=False, **kwargs):
        if dot_class is None:
            dot_class = Sphere if len(pos) == 3 else Dot
            dot_args = {'stroke_width': 0, 'stroke_color': color}
        else:
            dot_args = {}
        dot = dot_class(radius=radius, color=color, **dot_args)
        dot.group = self
        if label_args:
            label = TexMobject(*label_args, color=color)
            label.group = self
            label.next_to(dot, direction=RIGHT, buff=0.1)
            VGroup.__init__(self, dot, label, **kwargs)
        else:
            VGroup.__init__(self, dot, **kwargs)

        if type(pos) == tuple:
            coords = np.array([pos[0], pos[1], pos[2] if len(pos) == 3 else 0])
        else:
            coords = pos
        self.move_to(coords)
        if scene is not None:
            if anims is None:
                if self.has_label():
                    scene.add_fixed_orientation_mobjects(self.label())
                    anims = GrowFromCenter(self.dot()), FadeIn(self.label())
                else:
                    anims = (GrowFromCenter(self.dot()),)
            if draw_bases:
                x, y, z = self.dot().get_x(), self.dot().get_y(), self.dot().get_z()
                scene.add(Line(vec(0, y, 0), vec(x, y, 0), stroke_width=1, color=DARK_GREY))
                scene.add(Line(vec(x, 0, 0), vec(x, y, 0), stroke_width=1, color=DARK_GREY))
                scene.add(Line(vec(x, y, 0), vec(x, y, z), stroke_width=1, color=DARK_GREY))
                scene.bring_to_front(self.dot())
            scene.play(*anims)

    def pos(self):
        return vec(self.get_x(), self.get_y(), self.get_z() if self.dim == 3 else 0)

    def dot(self):
        return self.submobjects[0]

    def label(self):
        return self.submobjects[1]

    def has_label(self):
        return len(self.submobjects) > 1

    def move_to(self, coords):
        if len(coords) == 2:
            x, y = tuple(coords)
        elif len(coords) == 3:
            x, y, z = tuple(coords)
            self.dot().set_z(z)
            if self.has_label():
                self.label().set_z(z)
        else:
            raise NotImplementedError("> 3D")
        self.dot().set_x(x)
        self.dot().set_y(y)
        if self.has_label():
            self.label().set_x(x + LabeledDot.label_offset[0])
            return self.label().set_y(y)
        else:
            return self

    def set_x(self, x, direction=ORIGIN) -> Mobject:
        self.dot().set_x(x)
        return self.label().set_x(x + LabeledDot.label_offset[0])

    def set_y(self, y, direction=ORIGIN):
        self.dot().set_y(y)
        return self.label().set_y(y + LabeledDot.label_offset[1])

    def add_updater(self, func, **kwargs):
        self.dot().add_updater(func, **kwargs)


class NullSphere(Sphere):

    def __init__(self):
        super().__init__(radius=2,
                         checkerboard_colors=None,
                         fill_color=NULL_POINT_COLOR,
                         fill_opacity=0.2,
                         stroke_color=NULL_POINT_COLOR,
                         stroke_opacity=0.1,
                         resolution=NULL_SPHERE_RESOLUTION
                         )


class NullPoint(LabeledDot):

    def __init__(self, circle_or_sphere, proportion=0, *label_args, color=NULL_POINT_COLOR, updater=None,
                 call_updater=False, anims=None, **kwargs):
        """
        for a sphere:

        p, t = proportion

        phi := p * TAU
        tau := t * TAU

        phi is the clockwise rotation between y and z-axis
        theta is the counter-clockwise rotation between x and y-axis

        theta is applied first.
        """
        self.updater = updater
        if isinstance(circle_or_sphere, Circle):
            if label_args:
                super().__init__(*label_args, color=color, **kwargs)
            else:
                super().__init__(**kwargs)
            self.value_tracker = ValueTracker(proportion)
            self.move_to(vec(NULL_CIRCLE_RADIUS * cos(proportion * TAU), NULL_CIRCLE_RADIUS * sin(proportion * TAU), 0))
        else:
            if label_args:
                super().__init__(*label_args, color=color, dot_class=Sphere, fill_color=color,
                                 stroke_color=color, checkerboard_colors=None, **kwargs)
            else:
                super().__init__(**kwargs, dot_class=Sphere,
                                 radius=0.04,
                                 fill_color=color,
                                 stroke_color=color, checkerboard_colors=None, stroke_width=0)
            self.value_tracker = ValueTracker(0)
            p, t = proportion
            phi = p.sym * sympy.pi * 2
            theta = sympy.pi * 2 * t.sym
            P_tau = vec(sympy.cos(theta), sympy.sin(theta), 0)
            rot_matrix = rotation_matrix(float(phi), -X_AXIS)
            P = P_tau.dot(rot_matrix)
            self.move_to(P * NULL_CIRCLE_RADIUS)

        if updater:
            self.add_updater(updater, call_updater=call_updater)
        # if scene is not None:
        #     if anims is None:
        #         if self.has_label():
        #             anims = GrowFromCenter(self.dot()), FadeIn(self.label())
        #         else:
        #             anims = (GrowFromCenter(self.dot()),)
        #     scene.play(*anims)

    def animate(self, scene, target_alpha=1.0, updater=None, call_updater=False, *args, **kwargs):
        if updater is not None:
            self.add_updater(updater, call_updater=call_updater)
        scene.play(self.value_tracker.increment_value, target_alpha, *args, **kwargs)
        self.dot().clear_updaters()

    def set_updater(self, updater, proportion=0):
        self.clear_updaters()
        self.value_tracker = ValueTracker(proportion)
        self.updater = updater
        self.add_updater(lambda obj: updater(obj))

    def set_x(self, x, direction=ORIGIN):
        self.dot().set_x(x)
        return self.label().set_x(x + LabeledDot.label_offset[0])

    def set_y(self, y, direction=ORIGIN):
        self.dot().set_y(y)
        return self.label().set_y(y + LabeledDot.label_offset[1])


class NullPlane(ParametricSurface, SceneElement):

    RADIUS = NULL_CIRCLE_RADIUS

    def __init__(self, point=None, proportion=0, scene=None, **kwargs):
        """
        The plane parallel to the tangent plane if the given point
        """
        self.point = point
        self.proportion = proportion
        self.n = None
        self.P = None
        ParametricSurface.__init__(self, self.func,
                                   fill_opacity=0.4,
                                   stroke_width=0,
                                   resolution=(20, 20),
                                   **kwargs)
        SceneElement.__init__(self, scene=scene)

    def func(self, u, v):
        """
        u,v: interval [0,1(
        """
        if u == 0 and v == 0:
            print("func: 0")
        if u == 1 and v == 1:
            print("func: 1")
        if self.point:
            if self.n is None:
                self.P = vec(self.point.get_x(), self.point.get_y(), self.point.get_z())
                self.n = normalize(self.P)

            if np.array_equal(self.n, vec(0, 1, 0)):
                return vec((u - 0.5) * NullPlane.RADIUS * 2, 0, (v - 0.5) * NullPlane.RADIUS * 2)

            return ORIGIN
            alpha = u * TAU
            beta = v * TAU
            # Torus
            # return vec(3 * np.cos(alpha) + np.cos(alpha) * np.cos(beta),
            #            3 * np.sin(alpha) + np.sin(alpha) * np.cos(beta),
            #            np.sin(beta))


            # n_x * x = 0
            # V_u = vec((u - 0.5) * 2 * NULL_CIRCLE_RADIUS, 0, 0)
            # V_v = vec(0, (v - 0.5) * 2 * NULL_CIRCLE_RADIUS, 0)
            # V_1 = np.cross(self.n, V_u)
            # V_2 = np.cross(self.n, V_v)
            # return ORIGIN + u * V_1 + v * V_2
            #
            # P = vec((u - 0.5) * NullPlane.RADIUS * 2, (v - 0.5) * NullPlane.RADIUS * 2, 0)
        else:
            return vec((u - 0.5) * NullPlane.RADIUS * 2, (v - 0.5) * NullPlane.RADIUS * 2, 0)


class LabeledLine(VGroup, SceneElement):

    def __init__(self, label, start=LEFT, end=RIGHT, color=BLUE, label_proportion=0.333, label_distance=0.3,
                 scene=None, anims=None, **kwargs):
        line = Line(np.array([start[0], start[1], 0]), np.array([end[0], end[1], 0]), color=color, **kwargs)
        line.group = self
        if label is not None:
            label = TexMobject(label, color=color)
            label.group = self
        VGroup.__init__(self, line, label,
                         scene=scene,
                         anims=[GrowFromCenter(self.line()), FadeIn(self.label())] if anims is None else anims)
        SceneElement.__init__(self, scene=scene, anims=anims)
        self.set_label_pos(label_proportion, label_distance)
        self.initial_line_state = self.line().copy()
        self.value_tracker = ValueTracker(0)
        self.line().clear_updaters()

    def animate(self, scene, target_alpha=1.0, *args, updater=None, func=None, **kwargs):
        if updater:
            self.add_updater(updater, index=0, call_updater=False)
        scene.play(self.value_tracker.increment_value if func is None else func, target_alpha, *args, **kwargs)
        self.line().clear_updaters()

    def add_updater(self, func, **kwargs):
        self.line().add_updater(func, **kwargs)

    def reset_rotation(self):
        self.line().become(self.initial_line_state)

    def rotate(self, v, about_point=ORIGIN):
        self.line().rotate(v, about_point)

    def line(self):
        return self.submobjects[0]

    def label(self):
        return self.submobjects[1]

    def set_label_pos(self, label_proportion=0.333, label_distance=0.3):
        label_direction = self.line().get_angle() - TAU / 4
        coords = self.line().point_from_proportion(label_proportion)
        coords += np.array([cos(label_direction) * label_distance, sin(label_direction) * label_distance, 0])
        self.label().move_to(coords)


class NullLine(LabeledLine):
    def __init__(self, proportion, color=NULL_POINT_COLOR, anims=None, label=None, scene=None, **kwargs):
        alpha = TAU * proportion
        x2 = cos(alpha)
        y2 = sin(alpha)
        # http://localhost:8888/notebooks/BoundedLine.ipynb

        y = FRAME_Y_RADIUS
        if y2 == 0:
            t = 0
            x = FRAME_X_RADIUS
        elif x2 != 0:
            t = y2/x2
            x = y / t

        if x2 != 0:
            if abs(x) >= FRAME_X_RADIUS:
                if x >= FRAME_X_RADIUS:
                    # hit right boundary
                    x = FRAME_X_RADIUS
                    y = t * x
                else:
                    # hit left boundary
                    x = -FRAME_X_RADIUS
                    y = t * x
        else:
            x = 0
            y = FRAME_Y_RADIUS
        end = np.array([x, y, 0])
        start = -end
        super().__init__(label, start, end, color=color, scene=scene, anims=anims, **kwargs)

