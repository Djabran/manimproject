import re
import sympy
from itertools import chain
import manim
from manim import *
from manim.constants import *
from colors import *
import rich

RT = ReplacementTransform
AG = AnimationGroup


def vec(*args):
    if len(args) == 2:
        return np.array(list(args) + [0])
    else:
        return np.array(list(args))


current_scene = None


class Vec:

    def __init__(self, *args, coord_name='x'):
        if len(args) == 1:
            assert len(args[0] > 1)
            args = args[0]
        assert len(args) > 1
        self.dimensions = len(args)
        if len(args) == 2:
            self.x = args[0]
            self.y = args[1]
        elif len(args) == 3:
            self.x = args[0]
            self.y = args[1]
            self.z = args[2]
        elif len(args) == 4:
            self.w = args[0]
            self.x = args[1]
            self.y = args[2]
            self.z = args[3]
        else:
            for n in range(len(args)):
                exec(f"self.{coord_name}_{n}={args[n]}")

    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        else:
            return 0

    def __str__(self):
        return f"{self.x:.2f},{self.y:.2f}"

    def __repr__(self):
        return f"{self.x:.3f},{self.y:.3f}"


class Proportion:

    def __init__(self, *args):
        if len(args) == 2:
            self.sym = sympy.sympify(f"{args[0]} / {args[1]}")
        elif len(args) == 1:
            self.sym = sympy.sympify(args[0])
        else:
            self.sym = sympy.sympify(0)


class SceneElement:

    def __init__(self, anims=None, **kwargs):
        if current_scene is not None:
            if anims is None:
                current_scene.add(self)
            else:
                try:
                    current_scene.play(*anims, **kwargs)
                except TypeError:
                    current_scene.play(anims, **kwargs)


class TeX:

    def __init__(self, s=None):
        self.s = [s] if s is not None else []

    def __add__(self, s):
        self.s += [s]
        return self

    def __call__(self, s):
        return self.__add__(s)


class MText(TeX):
    pass



class SceneBase2d(Scene):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        set_current_scene(self)

    def show_title(self, title):
        text_title = Tex(title)
        self.add(text_title)
        self.wait()
        self.play(FadeOut(text_title))


class SceneBase(ThreeDScene):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        set_current_scene(self)

    def show_title(self, title):
        text_title = Tex(title)
        self.add(text_title)
        self.wait()
        self.play(FadeOut(text_title))


class TextBlock(SceneElement):

    def __init__(self, *args, anim=Write, run_time=0.4, mobject_type=Tex,
                 color_map=None, remover=True, buff=1.2, scale=1.0, **kwargs):
        self.scale_factor = scale
        self.color_map = color_map
        self.remover = remover
        self.texts = []
        self.mobject_type = mobject_type
        self.anim = anim
        if len(args):
            self.append(*args, anim=anim, mobject_type=mobject_type, color_map=color_map,
                        run_time=run_time, buff=buff, **kwargs)

    def __del__(self):
        if self.remover:
            self.fade_out()

    def __add__(self, s):
        self.append(s)
        return self

    def last(self):
        return self.texts[-1]

    def use_tex(self, flag=True):
        self.mobject_type = MathTex if flag else TextMobject

    def append(self, *args, anim=None, pos=None, mobject_type=None, color_map=None, run_time=0.4, buff=0.5, **kwargs):
        assert args
        if type(args[0]) is TeX:
            args = tuple(args[0].s)
            mobject_type = MathTex
        elif type(args[0]) is MText:
            args = tuple(args[0].s)
            mobject_type = Tex
        if mobject_type:
            self.mobject_type = mobject_type
        text = self.mobject_type(*args, **kwargs)
        text.scale(self.scale_factor)
        self.texts.append(text)
        if color_map is not None:
            self.color_map = color_map
        if self.color_map is not None:
            text.set_color_by_tex_to_color_map(self.color_map.text if self.mobject_type is Tex else self.color_map)
        if pos is None:
            if len(self.texts) > 1:
                text.next_to(self.texts[-2], DOWN * self.scale_factor, buff=buff)
            else:
                text.to_edge(UP, buff=buff)
        else:
            text.move_to(np.array([pos[0], pos[1], 0]))
        anim = anim if anim is not None else self.anim
        if current_scene and anim:
            current_scene.add_fixed_in_frame_mobjects(text)
            if anim is not None:
                current_scene.play(anim(text), run_time=run_time)
        return text

    def append_tex(self, *args, anim=Write, pos=None, **kwargs) -> MathTex:
        return self.append(*args, mobject_type=MathTex, anim=anim, pos=pos, **kwargs)

    def append_text(self, *args, anim=Write, pos=None, **kwargs) -> MathTex:
        return self.append(*args, mobject_type=MathTex, anim=anim, pos=pos, **kwargs)

    def bring_to_front(self):
        for text in self.texts:
            current_scene.bring_to_front(text)

    def fade_out(self):
        if current_scene:
            anims = [FadeOut(x, remover=False) for x in self.texts]
            current_scene.play(*anims)
        self.clear()

    def clear(self):
        if current_scene:
            for text in self.texts:
                current_scene.remove(text)
        self.texts = []


class ColorMap(dict):

    def __init__(self, d):
        self.text = {}
        for k, v in d.items():
            self[k] = self.text[f"${k}$"] = v


class UpdateGroup:

    def __init__(self, anchor_updater_info, *updater_infos, func=None, start=None, end=1.0, run_time=1.0):
        anchor, anchor_updater = anchor_updater_info
        self.anchor = anchor
        if start is not None:
            anchor.value_tracker.set_value(start)
        self.updater_infos = updater_infos
        for updater_info in self.updater_infos:
            updater_info[0].add_updater(updater_info[1], call_updater=False)
        anchor.animate(end, func=func, updater=anchor_updater, run_time=run_time)

    def __del__(self):
        for updater_info in self.updater_infos:
            updater_info[0].clear_updaters()
        assert len(self.anchor.updaters) == 0


class Axes3D(ThreeDAxes):

    def __init__(self):
        super().__init__(x_min=-3.5, y_min=-3.5, x_max=5, y_max=3, z_min=-3.5, z_max=3.5,
                         num_axis_pieces=20,
                         x_axis_config={'stroke_width': 1,
                                        'tip_length': 0.2, 'stroke_color': DARK_GREY, 'color': DARK_GREY},
                         y_axis_config={'stroke_width': 1,
                                        'tip_length': 0.2, 'stroke_color': DARK_GREY, 'color': DARK_GREY},
                         z_axis_config={'include_tip': True, 'stroke_width': 1, 'include_ticks': True,
                                        'tip_length': 0.2, 'stroke_color': DARK_GREY, 'color': DARK_GREY})
        labels = VGroup(self.get_x_axis_label("x", edge=RIGHT, direction=UP),
                        self.get_y_axis_label("y", edge=UP, direction=RIGHT))
        labels.set_color(GREY)
        # currentScene.add_fixed_orientation_mobjects(labels)
        current_scene.add(labels)
        current_scene.add(self)


class Cone(ParametricSurface, SceneElement):

    def align_points_with_larger(self, larger_mobject):
        pass

    def __init__(self, tip=ORIGIN, foot=IN, radius=0.5, anims=None, **kwargs):
        self.radius = radius
        self.tip = tip
        self.height = np.linalg.norm(tip - foot)
        self.direction = foot - tip
        super().__init__(self.func,
                         v_min=0, v_max=TAU, u_min=-1, u_max=0,
                         resolution=(1, 36), checkerboard_colors=(BLUE_C, BLUE_B),
                         fill_color=BLUE_C,
                         stroke_color=WHITE,
                         stroke_width=1,
                         stroke_opacity=0.2,
                         **kwargs)
        from numpy import cos, sin, arctan
        tx, ty, tz = tuple(tip)
        fx, fy, fz = tuple(foot)
        dx = tx - fx
        dy = ty - fy
        dz = tz - fz
        alpha_xy = np.arccos(np.dot(self.direction, UP))
        gamma_xz = np.arccos(np.dot(self.direction, OUT))
        beta_yz = arctan(dz / dy) if dy != 0 else 0
        console.print(f"d={dx, dy, dz}")
        console.print(f"alpha_xy={alpha_xy / DEGREES}°, beta_yz={beta_yz / DEGREES}°, gamma_xz={gamma_xz / DEGREES}°")
        console.print(f"height={self.height}, direction={self.direction}")
        self.rotate(gamma_xz, DOWN, about_point=tip)
        self.rotate(alpha_xy, OUT, about_point=tip)
        #self.rotate(beta_yz, RIGHT)
#        self.rotate(TAU/4, UP)
        # self.shift(tip)

        SceneElement.__init__(self, anims=anims)

    def func(self, u, v):
        from numpy import cos, sin, arctan
        return vec(u * cos(v) * self.radius, u * sin(v) * self.radius, u * self.height) + self.tip


def set_current_scene(scene):
    global current_scene
    current_scene = scene


def get_current_scene():
    return current_scene


def play(*args, **kwargs):
    current_scene.play(*args, **kwargs)


def render():
    current_scene.render()


def wait(secs=1.0):
    current_scene.wait(secs)


def show_title(*args, **kwargs):
    current_scene.show_title(*args, **kwargs)


def add_fixed_orientation_mobjects(*args, **kwargs):
    current_scene.add_fixed_orientation_mobjects(*args, **kwargs)


def bring_to_front(*args, **kwargs):
    current_scene.bring_to_front(*args, **kwargs)


def add(*args, **kwargs):
    current_scene.add(*args, **kwargs)
    return args if len(args) > 1 else args[0]


def remove(*args, **kwargs):
    current_scene.remove(*args, **kwargs)


def part(obj: Mobject, tex, substring=True, case_sensitive=True):
    return obj.get_part_by_tex(tex, substring=substring, case_sensitive=case_sensitive)


def parts(obj: Mobject, *args, substring=True, case_sensitive=True):
    result = []
    for arg in args:
        if type(arg) is tuple:
            result.extend(obj.get_parts_by_tex(arg[0], substring=substring, case_sensitive=case_sensitive)[arg[1]])
        else:
            result.extend(obj.get_parts_by_tex(arg, substring=substring, case_sensitive=case_sensitive))
    if result:
        return result if len(result) > 1 else result[0]


def replremwri(eq1, eq2, replace_indices=[], remove_indices=[], write_indices=[]):
    play(*[RT(eq1[source_index], eq2[target_index]) for source_index, target_index in replace_indices.items()],
         *[FadeOut(eq1[n]) for n in remove_indices],
         *[Write(eq2[n]) for n in write_indices])


def playlist(*args_object_lists: [(Mobject, Mobject)], run_time=1.0):
    args = []
    for arg in args_object_lists:
        args += [RT(src, dst, run_time=run_time) for src, dst in arg]
    play(*args)


def match_tex(x, y):
    return x.tex_string.replace("{", "").replace("}", "") == y.tex_string.replace("{", "").replace("}", "")


def replace_and_add(eq1, eq2, fade_in_from=0.5 * UP):
    animations = []
    it1 = iter(eq1)
    x = next(it1)

    for y in eq2:
        if match_tex(x, y):
            console.print(f"replacing {x}")
            animations.append(RT(x, y))
            try:
                x = next(it1)
            except StopIteration:
                pass
        else:
            console.print(f"adding {y}")
            animations.append(FadeInFrom(y, fade_in_from))
    play(*animations)


def find_parts_from_tex(eq, tex, offset=0):
    match_start = -1
    match_string = ""
    for i in range(len(eq.submobjects)):
        p = eq[i]
        if match_start == -1:
            if tex.startswith(p.tex_string):
                if tex == p.tex_string:
                    if offset == 0:
                        console.print(f"find_parts_from_tex - '{tex}' -> {i}")
                        return eq[i:i + 1]
                    else:
                        offset -= 1
                else:
                    match_start = i
                    match_string = p.tex_string
        else:
            new_match_string = match_string + (p.tex_string if match_string.endswith("{")
                                               else " " + p.tex_string)
            if tex.startswith(new_match_string):
                if new_match_string == tex:
                    if offset == 0:
                        console.print(f"find_parts_from_tex - '{tex}' -> [{match_start}:{i+1}]")
                        return eq[match_start:i + 1]
                    else:
                        offset -= 1
                else:
                    match_string = new_match_string
            else:
                if tex.startswith(p.tex_string):
                    if tex == p.tex_string:
                        if offset == 0:
                            return eq[i:i+1]
                        else:
                            offset -= 1
                    else:
                        match_start = i
                        match_string = p.tex_string
                else:
                    match_start = -1
                    match_string = ""
    raise RuntimeError(f"parts for tex-substring '{tex}' not found")


def replace(eq1, eq2, tex_old, tex_new, offset_old=0, offset_new=0, animate=True, **kwargs):
    tex1 = " ".join(chain(*eq1.groups))
    tex1 = tex1.replace("{ ", "{")
    tex2 = " ".join(chain(*eq2.groups))
    tex2 = tex2.replace("{ ", "{")
    assert tex_old in tex1, f"texOld: {tex_old}tex1: {tex1}"

    subparts1 = find_parts_from_tex(eq1, tex_old, offset_old)
    if tex_new:
        subparts2 = find_parts_from_tex(eq2, tex_new, offset_new)

    if animate:
        if tex_old == tex_new:
            play(FadeOut(subparts1, run_time=0))
            p = subparts2.copy().move_to(subparts1)
            play(RT(p, subparts2, **kwargs))
        else:
            play(RT(subparts1, subparts2, **kwargs))
    else:
        if tex_new:
            anims = []
            if tex_old == tex_new:
                anims.append(FadeOut(subparts1, run_time=0.001))
                p = subparts2.copy().move_to(subparts1)
                anims.append(RT(p, subparts2, **kwargs))
            else:
                anims.append(RT(subparts1, subparts2, **kwargs))
            return anims
        else:
            return [FadeOut(subparts1, **kwargs)]


def substitute_parts(eq1, eq2, *tex_parts, additional_anims=[], **kwargs):
    texmap_old = {}
    texmap_new = {}
    for p in tex_parts:
        if type(p) is tuple:
            assert len(p) == 1 or len(p) == 2
            if len(p) == 2:
                texmap_old[p[0]] = texmap_new[p[0]] = texmap_old[p[1]] = texmap_new[p[1]] = 0
            else:
                texmap_old[p[0]] = 0
        else:
            texmap_old[p] = texmap_new[p] = 0
    console.print(f"eq1: {[str(i) + ': ' + x.tex_string for i, x in enumerate(eq1)]}")
    console.print(f"eq1: {[str(i) + ': ' + x.tex_string for i, x in enumerate(eq2)]}")
    anims = []
    for tex_part in tex_parts:
        if type(tex_part) is tuple:
            if len(tex_part) == 2:
                old, new = tex_part[0], tex_part[1]
            else:
                old = tex_part[0]
                new = None
        else:
            old = new = tex_part
        if " " in old:
            subparts = old.split(" ")
            for subpart in subparts:
                if subpart in texmap_old:
                    texmap_old[subpart] += 1
        if new and " " in new:
            subparts = new.split(" ")
            for subpart in subparts:
                if subpart in texmap_new:
                    texmap_new[subpart] += 1
        texmap_old[old] += 1
        if new:
            texmap_new[new] += 1
        anims.extend(replace(eq1, eq2, old, new, offset_old=texmap_old[old]-1, offset_new=texmap_new[new]-1 if new else 0, animate=False, **kwargs))
    play(*(anims + additional_anims))


def camera(*args, **kwargs):
    current_scene.set_camera_orientation(*args, **kwargs)

