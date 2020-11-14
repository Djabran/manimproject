from manimlib.imports import *
from manimlib.constants import *
from manimlib.constants import BLACK, BLUE_A, BLUE_B, BLUE, BLUE_D, BLUE_E, DARK_BLUE, DARK_GREY, GREEN_A, GREEN_B,\
    GREEN, GREEN_D, GREEN_E, GREY, LIGHT_GREY, MAROON_A, MAROON_B, RED, RED_D, TEAL, WHITE, YELLOW
import sympy


def vec(*args):
    return np.array(list(args))


class Vec2:

    def __init__(self, *args):
        if len(args) == 1:
            self.x = args[0][0]
            self.y = args[0][1]
        else:
            self.x = args[0]
            self.y = args[1]

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

    def __init__(self, scene=None, anims=None, **kwargs):
        if scene is not None:
            if anims is None:
                scene.add(self)
            else:
                try:
                    scene.play(*anims, **kwargs)
                except TypeError:
                    scene.play(anims, **kwargs)


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


class SceneBase(ThreeDScene):

    def show_title(self, title):
        text_title = TextMobject(title)
        self.add(text_title)
        self.wait()
        self.play(FadeOut(text_title))


class TextBlock(SceneElement):

    def __init__(self, *args, scene=None, anim=Write, run_time=0.4, mobject_type=TextMobject,
                 color_map=None, remover=True, buff=1.2, **kwargs):
        self.color_map = color_map
        self.scene = scene
        self.remover = remover
        self.texts = []
        self.mobject_type = mobject_type
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
        self.mobject_type = TexMobject if flag else TextMobject

    def append(self, *args, anim=Write, pos=None, mobject_type=None, color_map=None, run_time=0.4, buff=0.5, **kwargs):
        assert args
        if type(args[0]) is TeX:
            args = tuple(args[0].s)
            mobject_type = TexMobject
        elif type(args[0]) is MText:
            args = tuple(args[0].s)
            mobject_type = TextMobject
        if mobject_type:
            self.mobject_type = mobject_type
        text = self.mobject_type(*args, **kwargs)
        self.texts.append(text)
        if color_map is not None:
            self.color_map = color_map
        if self.color_map is not None:
            text.set_color_by_tex_to_color_map(self.color_map.text if mobject_type is TextMobject
                                               else self.color_map)
        if pos is None:
            if len(self.texts) > 1:
                text.next_to(self.texts[-2], DOWN, buff=buff)
            else:
                text.to_edge(UP, buff=buff)
        else:
            text.move_to(np.array([pos[0], pos[1], 0]))
        if self.scene:
            self.scene.add_fixed_in_frame_mobjects(text)
            self.scene.play(anim(text), run_time=run_time)
        return text

    def append_tex(self, *args, anim=Write, pos=None, **kwargs) -> TexMobject:
        return self.append(*args, mobject_type=TexMobject, anim=anim, pos=pos, **kwargs)

    def append_text(self, *args, anim=Write, pos=None, **kwargs) -> TextMobject:
        return self.append(*args, mobject_type=TextMobject, anim=anim, pos=pos, **kwargs)

    def bring_to_front(self, scene):
        for text in self.texts:
            scene.bring_to_front(text)

    def fade_out(self):
        if self.scene:
            anims = [FadeOut(x, remover=False) for x in self.texts]
            self.scene.play(*anims)
        self.clear()

    def clear(self):
        if self.scene:
            for text in self.texts:
                self.scene.remove(text)
        self.texts = []


class ColorMap(dict):

    def __init__(self, d):
        self.text = {}
        for k, v in d.items():
            self[k] = self.text[f"${k}$"] = v


class UpdateGroup:

    def __init__(self, scene, anchor_updater_info, *updater_infos, func=None, start=None, end=1.0, run_time=1.0):
        anchor, anchor_updater = anchor_updater_info
        self.anchor = anchor
        if start is not None:
            anchor.value_tracker.set_value(start)
        self.updater_infos = updater_infos
        for updater_info in self.updater_infos:
            updater_info[0].add_updater(updater_info[1], call_updater=False)
        anchor.animate(scene, end, func=func, updater=anchor_updater, run_time=run_time)

    def __del__(self):
        for updater_info in self.updater_infos:
            updater_info[0].clear_updaters()
        assert len(self.anchor.updaters) == 0


class Axes3D(ThreeDAxes):

    def __init__(self, scene):
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
        # scene.add_fixed_orientation_mobjects(labels)
        scene.add(labels)
        scene.add(self)
