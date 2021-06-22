import numpy as np
import sympy
from sympy.abc import x, y, theta, phi, alpha
from matplotlib import pyplot
import IPython.core.display as di  # Example: di.display_html('<h3>%s:</h3>' % str, raw=True)
from IPython.core.display import Video
from IPython.display import display
from IPython.display import HTML
import itertools
import collections
import functools
from typing import Tuple, Union


def show(*args: Tuple[Union[sympy.Symbol, str]]):
    """
    Displays the given symbols as latex
    """
    s = ""
    for arg in args:
        if type(arg) is str:
            s += arg
        else:
            s += sympy.latex(arg)
    # s = functools.reduce(lambda a, b: latex(a) + latex(b), args, "")
    di.display_latex(f"${s}$", raw=True)


def toggle_code():
    # This line will hide code by default when the notebook is exported as HTML
    # di.display_html('''
    # <script>
    #
    # jQuery(function() {
    #     jQuery(".input_area").toggle();
    #     jQuery(".prompt").toggle();
    #     });
    #
    # </script>''', raw=True)
    di.display_html('''
    <script>
    
    jQuery(function() {
        jQuery(".input").toggle();
        });
    
    </script>''', raw=True)


def display_toggle_code_button():
    #di.display_html('''<button onclick="jQuery('.input_area').toggle(); jQuery('.prompt').toggle();">Toggle code</button>''', raw=True)
    di.display_html('''<button onclick="jQuery('.input').toggle(); ">Toggle input</button>''', raw=True)

