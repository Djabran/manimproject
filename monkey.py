import os
import sys
import importlib
import inspect
from manim import logger
from pathlib import Path
import types
import manim


def get_scene_classes_from_module(module):
    from manim.scene.scene import Scene

    def is_child_scene(obj, m):
        if "Projective" in str(obj):
            print(f"{obj}: {type(obj)}: is scene: {isinstance(obj, Scene)}")
        return (inspect.isclass(obj)
                and issubclass(obj, Scene)
                and obj != Scene
                and obj.__module__.startswith(m.__name__))

    return [member[1] for member in inspect.getmembers(module, lambda x: is_child_scene(x, module))]


def get_module(file_name):
    if str(file_name) == "-":
        module = types.ModuleType("input_scenes")
        logger.info(
            "Enter the animation's code & end with an EOF (CTRL+D on Linux/Unix, CTRL+Z on Windows):"
        )
        code = sys.stdin.read()
        if not code.startswith("from manim import"):
            logger.warn(
                "Didn't find an import statement for Manim. Importing automatically..."
            )
            code = "from manim import *\n" + code
        logger.info("Rendering animation from typed code...")
        try:
            exec(code, module.__dict__)
            return module
        except Exception as e:
            logger.error(f"Failed to render scene: {str(e)}")
            sys.exit(2)
    else:
        if Path(file_name).exists():
            ext = file_name.suffix
            if ext != ".py":
                raise ValueError(f"{file_name} is not a valid Manim python script.")
            module_name = str(file_name).replace(os.sep, ".").split(".")[0]
            spec = importlib.util.spec_from_file_location(module_name, file_name)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            sys.path.insert(0, str(file_name.parent.absolute()))
            spec.loader.exec_module(module)
            return module
        else:
            raise FileNotFoundError(f"{file_name} not found")


manim.__main__.get_scene_classes_from_module = manim.utils.module_ops.get_scene_classes_from_module\
    = get_scene_classes_from_module
manim.__main__.get_module = manim.utils.module_ops.get_module = get_module
