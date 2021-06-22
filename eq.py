import re
import functools
from itertools import chain
from manim import *
from manim.constants import *
import sympy
from basicobjects import *
import myersdiff
from process_latex import process_sympy, remove_unnecessary_brackets, remove_unnecessary_brackets_tex
import sympy.core

re_latex_expr = re.compile(r"\s*([\w^{}\\\(\)]+)\s*")
#re_eq_term = re.compile(r"\(?[\w]+[^\(^\).\S]*")
re_eq_term = re.compile(r"\s*\(?{?[\w]+[.]*")


def find_matching_closer(s, closer):
    assert closer in ')}]'
    opener = '{' if closer == '}' else '[' if closer == ']' else '('
    stack = 1
    for i, c in enumerate(s):
        if c == opener:
            stack += 1
        elif c == closer:
            stack -= 1
            if stack == 0:
                return i
    return -1


def find_enclosed_keyword(s, keyword, closer):
    assert closer in ')}]'
    opener = '{' if closer == '}' else '[' if closer == ']' else '('
    stack = 0
    candidate = ""
    for i, c in enumerate(s):
        if c == opener:
            stack += 1
            candidate = ""
        elif c == closer:
            stack -= 1
            candidate = ""
        else:
            candidate += c
            if candidate.endswith(keyword) and stack == 1:
                return i - len(keyword) + 1
    return -1


def convert_frac_to_over(tex):
    n_frac = tex.find(r"\frac{")
    if n_frac >= 0:
        m = find_matching_closer(tex[n_frac + 6:], "}") + n_frac + 6
        assert m >= 0
        begin = tex[:n_frac + 6]
        outer_bracket_numerator = tex[n_frac + 6:m]
        end = tex[m+1:]
        numerator = convert_frac_to_over(outer_bracket_numerator).strip()
        assert end[0] == "{"
        m_end = find_matching_closer(tex[m + 2:], "}")
        outer_bracket_denom = tex[m + 2:m + 2 + m_end]
        end = tex[2 + m + m_end + 1:]
        denom = convert_frac_to_over(outer_bracket_denom).strip()
        return begin[:-6] + "{" + numerator + r" \over " + denom + "}" + convert_frac_to_over(end)
    else:
        return tex


def convert_over_to_frac(tex):
    n_over = find_enclosed_keyword(tex, r"\over", '}')
    if n_over >= 0:
        n = tex.find('{')
        assert 0 <= n < n_over
        m = find_matching_closer(tex[n+1:], "}") + n + 1
        assert m >= 0
        begin = tex[:n]
        outer_bracket = tex[n + 1:m]
        end = tex[m + 1:]
        numerator = convert_over_to_frac(outer_bracket[:n_over - n - 1]).strip()
        denom = convert_over_to_frac(outer_bracket[n_over - n + 4:]).strip()
        if end.find(r"\over") >= 0:
            return begin + r"\frac{" + numerator + "}{" + denom + "}" + convert_over_to_frac(end)
        else:
            return begin + r"\frac{" + numerator + "}{" + denom + "}" + end
    return tex


def add_multiply_symbol(tex):
    found_eq_term = False
    new_tex = ""
    while tex:
        match = re.match(re_eq_term, tex)
        if match:
            if found_eq_term:
                new_tex += r" \cdot"
                found_eq_term = False
            else:
                found_eq_term = True
        else:
            match = re.match(r"\s*=", tex)
            if match:
                found_eq_term = False
            else:
                match = re.match(r"\s*\W+\w*", tex)
                if match:
                    found_eq_term = False
                else:
                    match = re.match(r"\s+", tex)
        new_tex += match[0]
        tex = tex[len(match[0]):]
    return new_tex


def get_tex_from_expressions(expressions):
    """
    if an index and a tex string is given, the respective expression
    is replaced as is, otherwise the given expressions are converted
    :param expressions:
    :param index:
    :param tex_string:
    :return:
    """
    tex = ""
    for i, expr in enumerate(expressions):
        if i > 0:
            tex += "="
        if type(expr) is Eq:
            tex += str(expr)
        elif type(expr) is str:
            tex += expr
        else:
            new_tex = sympy.latex(expr)
            if r"\frac" in new_tex:
                new_tex = convert_frac_to_over(new_tex)
            new_tex = new_tex.replace("hdots", r"\hdots")
            new_tex = remove_unnecessary_brackets_tex(new_tex)
            tex += new_tex
    return tex


class Eq(MathTex):

    animate = True
    show_operation = True
    new_line = True
    align = True
    run_time = 1.0

    def __init__(self, *args):
        """
        :param args: if only one argument is given, the string is split into multiple submbojects.
                    This doesn't work for all TeX expressions, thus, if you don't want the string to be split,
                    provide multiple arguments.
        """
        self.operation = None
        if len(args) > 1:
            super().__init__(*args)
        else:
            if isinstance(args[0], str):
                super().__init__(*Eq.split_tex(args[0]))
            else:
                super().__init__(str(args[0]))

    def __str__(self):
        return " ".join([x.tex_string for x in self.submobjects])

    def is_identical(self, eq):
        return str(self) == str(eq)

    def get_expressions(self, expressions=None) -> sympy.Symbol:
        """
        :return: a list of the expressions between the equal signs as sympy symbols
        """
        tex = str(self)
        tex = tex.replace(r"\left(", "(").replace(r"\right)", ")")
        tex = add_multiply_symbol(tex)
        tex = convert_over_to_frac(tex)
        expressions_tex = tex.split("=")
        if expressions is None:
            expressions = {}
        for i, expr_tex in enumerate(expressions_tex):
            if i not in expressions:
                try:
                    expressions[i] = process_sympy(expr_tex)
                except Exception as ex:
                    console.print(f"ex: {ex}")
        return [expressions[i] for i in range(len(expressions))]

    def make_new_eq(self, func, new_line=False, align=None, expressions=None):
        expressions = self.get_expressions(expressions)
        new_expressions = []
        for expr in expressions:
            if type(expr) is Eq or type(expr) is str:
                new_expressions.append(expr)
            else:
                expr = func(expr)
                new_expressions.append(expr)
        new_eq = Eq(get_tex_from_expressions(new_expressions))
        if new_line:
            new_eq.next_to(self, DOWN)
            if align:
                if sum(map(lambda x: x.tex_string == "=", new_eq.submobjects)) == 1:
                    a = next(x for x in self.submobjects if x.tex_string == "=")
                    b = next(x for x in new_eq.submobjects if x.tex_string == "=")
                    diff = a.get_critical_point(ORIGIN) - b.get_critical_point(ORIGIN)
                    new_eq.shift(vec(diff[0], 0, 0))
        else:
            new_eq.move_to(self)
        return new_eq

    def transform(self, new_eq, op_tex=None, op_expr=None, new_line=None, animate=None, run_time=None):
        console.print(f"transform {self} -> {new_eq}")
        if new_line is None:
            new_line = Eq.new_line
        if animate is None:
            animate = Eq.animate
        anims = Eq.make_diff_anims(self, new_eq, new_line=new_line, run_time=run_time)
        if op_tex:
            op_expr_tex = sympy.latex(op_expr)
            found_plus_op = False
            if op_tex.endswith('+') and op_expr_tex.startswith('-'):
                op_tex = op_tex[:-1]
                found_plus_op = True
            if type(op_expr) is str and not found_plus_op:
                symtex = r"\left(" + op_expr_tex + r"\right)"
            anims = self.make_operation_anims(anims, op_tex, op_expr_tex, new_line=new_line, run_time=run_time)
        if animate:
            if op_tex:
                anims_start, anims_end = anims
                play(*anims_start)
                play(*anims_end)
            else:
                play(*anims)
            return new_eq
        else:
            return new_eq, anims

    def make_new_eq_anims(self, sym, func, new_line=None, op_tex=None, animate=None, run_time=None, align=None, expressions=None):
        if new_line is None:
            new_line = Eq.new_line
        new_eq = self.make_new_eq(func, new_line=new_line, expressions=expressions, align=align)
        return self.transform(new_eq, op_tex, sym, new_line=new_line,
                              animate=Eq.annimate if animate is None else animate,
                              run_time=Eq.run_time if run_time is None else run_time)

    def make_operation_anims(self, anims, *tex_parts, new_line=False, run_time=1.0):
        self.operation = MathTex(*tex_parts, color=BLUE)
        center = self.get_center()
        anim_start = FadeInFrom(self.operation.to_edge(RIGHT, 0.3).shift(vec(0, center[1] - self.operation.get_center()[1], 0)),
                                RIGHT, run_time=run_time)
        if new_line:
            return [anim_start], [AG(*anims, lag_ratio=0.1)]
        else:
            return [anim_start], [AG(*anims, FadeOut(self.operation, run_time=run_time * 0.7), lag_ratio=0.03)]

    def replace_expr(self, eq_or_tex, index=None, animate=True):
        """
        Replaces the given expression with the given tex string

        :param eq_or_tex: the tex string to expand to or another expression represented as equation (i.e. Eq)
                          without equal sign
        :param index: the zero-based index of the expression within the whole equation
                      (i.e. the expressions between the equal signs). If no index is given,
                      all expressions are replaced which are mathematically equal to the expression given
                      be the tex string
        :param animate: whether to play the animations immediately
        :return: the new equation or a 2-tuple of the new equation and the animations, if animate is False
        """
        if type(eq_or_tex) is Eq:
            tex = str(eq_or_tex)
            assert "=" not in tex
        else:
            tex = eq_or_tex
        expand_expressions = Eq(tex).get_expressions()
        assert len(expand_expressions) == 1
        expand_expression = expand_expressions[0]
        expressions = self.get_expressions()
        new_expressions = []
        expand_expression_simp = expand_expression.simplify()
        if index is None:
            for expression in expressions:
                if (expand_expression_simp - expression).simplify() == 0:
                    new_expressions.append(expand_expression)
                else:
                    new_expressions.append(expression)
        else:
            new_expressions = expressions
            new_expressions[index] = expand_expression
        eq = get_tex_from_expressions(new_expressions, index, tex).move_to(self)
        anims = Eq.make_diff_anims(self, eq, new_line=False)
        if animate:
            play(*anims)
            return eq
        else:
            return eq, anims

    def __mul__(self, sym):
        if type(sym) is str:
            sym = sympy.sympify(sym)
        return self.mult_by_expr(sym, show_operation=Eq.show_operation, new_line=Eq.new_line, animate=Eq.animate)

    def __truediv__(self, sym):
        if type(sym) is str:
            sym = sympy.sympify(sym)
        return self.divide_by_expr(sym, show_operation=Eq.show_operation, new_line=Eq.new_line, animate=Eq.animate)

    def __sub__(self, sym):
        if type(sym) is str:
            sym = sympy.sympify(sym)
        return self.subtract_expr(sym, show_operation=Eq.show_operation, new_line=Eq.new_line, animate=Eq.animate)

    def __add__(self, sym):
        assert type(sym) is not float, "no floats allowed in here"
        if type(sym) is str:
            sym = sympy.sympify(sym)
        return self.add_expr(sym, show_operation=Eq.show_operation, new_line=Eq.new_line, animate=Eq.animate)

    def hide_operation(self):
        if self.operation is not None:
            play(FadeOut(self.operation))

    def mult_by_expr(self, s, show_operation=None, new_line=None, animate=None, expressions=None, align=None, run_time=None):
        """
        Multiplies everything by s

        :param align:
        :param expressions: a dictionary of indices and expressions to be inserted as is
        :param animate:
        :param s: the symbol given as sympifiable string
        :param new_line: keep old equation and put the new equation below. If operations are shown,
                         they remain visible
        :param show_operation: animate indicator for the kind of operation we're doing
        :return: a 2-tuple with the new equation and the animations
                If the operations are shown, the animations itself is a 2-tuple of animations for start and end
        """
        return self.make_new_eq_anims(s, lambda expr: (expr * sympy.sympify(s)).simplify(),
                                      align=Eq.align if align is None else align,
                                      animate=Eq.animate if animate is None else animate,
                                      new_line=Eq.new_line if new_line is None else new_line,
                                      run_time=Eq.run_time if run_time is None else run_time,
                                      op_tex=r"|\cdot" if show_operation or (show_operation is None and Eq.show_operation) else None,
                                      expressions=expressions)

    def divide_by_expr(self, s, show_operation=None, new_line=None, animate=None, expressions=None, align=None, run_time=None):
        return self.make_new_eq_anims(s, lambda expr: (expr / sympy.sympify(s)).simplify(),
                                      align=Eq.align if align is None else align,
                                      animate=Eq.animate if animate is None else animate,
                                      new_line=Eq.new_line if new_line is None else new_line,
                                      run_time=Eq.run_time if run_time is None else run_time,
                                      op_tex=r"|:" if show_operation or (show_operation is None and Eq.show_operation) else None,
                                      expressions=expressions)

    def add_expr(self, s, show_operation=None, new_line=None, animate=None, expressions=None, align=None, run_time=None):
        return self.make_new_eq_anims(s, lambda expr: (expr + sympy.sympify(s)).simplify(),
                                      align=Eq.align if align is None else align,
                                      animate=Eq.animate if animate is None else animate,
                                      new_line=Eq.new_line if new_line is None else new_line,
                                      run_time=Eq.run_time if run_time is None else run_time,
                                      op_tex=r"|+" if show_operation or (show_operation is None and Eq.show_operation) else None,
                                      expressions=expressions)

    def subtract_expr(self, s, show_operation=None, new_line=None, animate=None, expressions=None, align=None, run_time=None):
        return self.make_new_eq_anims(s, lambda expr: (expr - sympy.sympify(s)).simplify(),
                                      align=Eq.align if align is None else align,
                                      animate=Eq.animate if animate is None else animate,
                                      new_line=Eq.new_line if new_line is None else new_line,
                                      run_time=Eq.run_time if run_time is None else run_time,
                                      op_tex=r"|-" if show_operation or (show_operation is None and Eq.show_operation)else None,
                                      expressions=expressions)

    def get_expression_index(self, n):
        expr_index = 0
        for i, mobj in enumerate(self.submobjects):
            if i == n:
                return expr_index
            elif mobj.tex_string == "=":
                expr_index += 1

    @staticmethod
    def make_diff_anims(eq1, eq2, new_line=False, run_time=1.0):
        diff = myersdiff.diff(eq1, eq2, lambda a, b: match_tex(a, b))
        anims = []
        added_exprs = {}
        added_indices = {}
        removed_indices = {}
        removed_exprs = {}

        def store_added(n, expr_index):
            if expr_index in added_exprs:
                added_exprs[expr_index][eq2[n].tex_string] = n
            else:
                added_exprs[expr_index] = {eq2[n].tex_string: n}
            if expr_index not in added_indices:
                added_indices[expr_index] = [n]
            else:
                added_indices[expr_index].append(n)

        def store_removed(n, expr_index):
            if expr_index in removed_indices:
                removed_indices[expr_index].append(n)
            else:
                removed_indices[expr_index] = [n]
            if expr_index in removed_exprs:
                removed_exprs[expr_index][eq1[n].tex_string] = n
            else:
                removed_exprs[expr_index] = {eq1[n].tex_string: n}


        if len(str(eq1).split("=")) == 2 and len(str(eq2).split("=")) == 2:
            for d, n in diff:
                if d == 1:
                    expr_index = eq2.get_expression_index(n)
                    store_added(n, expr_index)
                elif d == -1:
                    expr_index = eq1.get_expression_index(n)
                    store_removed(n, expr_index)
                else:
                    assert d == 0
                    n, m = n
                    ei1 = eq1.get_expression_index(n)
                    ei2 = eq2.get_expression_index(m)
                    if ei1 != ei2:
                        store_removed(n, ei1)
                        store_added(m, ei2)


        def is_removed(expr_or_n, expr_index):
            if type(expr_or_n) == str:
                expr = expr_or_n
                return expr_index in removed_exprs and expr in removed_exprs[expr_index]
            else:
                n = expr_or_n
                return removed_indices and expr_index in removed_indices and n in removed_indices[expr_index]

        def is_added(expr_or_n, expr_index):
            if type(expr_or_n) == str:
                expr = expr_or_n
                return expr_index in added_exprs and expr in added_exprs[expr_index]
            else:
                n = expr_or_n
                return added_indices and expr_index in added_indices and n in added_indices[expr_index]

        def get_added_index(expr):
            if 0 in added_exprs and expr in added_exprs[0]:
                return added_exprs[0][expr]
            elif 1 in added_exprs and expr in added_exprs[1]:
                return added_exprs[1][expr]

        def get_removed_index(expr):
            if 0 in removed_exprs and expr in removed_exprs[0]:
                return removed_exprs[0][expr]
            elif 1 in removed_exprs and expr in removed_exprs[1]:
                return removed_exprs[1][expr]

        swapped_from_removed = []
        swapped_from_added = []

        for d, n in diff:
            if d == 0:
                x, y = n
                console.print(f"= {eq1[x].tex_string}, {eq2[y].tex_string}")
                if (is_removed(x, 0) and is_added(eq1[x].tex_string, 1)) or (is_removed(x, 1) and is_added(eq1[x].tex_string, 0)):
                    console.print(f"swapping (=) eq1[{x}] and eq2[{y}]: {eq1[x].tex_string}")
                    anims.append(RT(eq1[x], eq2[y], path_arc=np.pi if is_removed(x, 0) else -np.pi, run_time=run_time))
                else:
                    if new_line:
                        anims.append(TransformFromCopy(eq1[x], eq2[y], run_time=run_time))
                    else:
                        anims.append(RT(eq1[x], eq2[y], run_time=run_time))
            elif d == -1:
                console.print(f"-  {eq1[n].tex_string}")
                if n not in swapped_from_added:
                    if (is_removed(n, 0) and is_added(eq1[n].tex_string, 1)) or (is_removed(n, 1) and is_added(eq1[n].tex_string, 0)):
                        m = get_added_index(eq1[n].tex_string)
                        console.print(f"swapping eq1[{n}] and eq2[{m}]: {eq1[n].tex_string} <-> {eq2[m].tex_string}")
                        anims.append(RT(eq1[n], eq2[m], path_arc=np.pi if is_removed(n, 0) else -np.pi, run_time=run_time))
                        swapped_from_removed.append(m)
                    elif not new_line:
                        anims.append(FadeOut(eq1[n], run_time=run_time))
                else:
                    console.print("passing, swapped from added")
            else:
                assert d == 1
                if n not in swapped_from_removed:
                    if (is_added(n, 0) and is_removed(eq2[n].tex_string, 1)) or (is_added(n, 1) and is_removed(eq2[n].tex_string, 0)):
                        console.print(f"swapping eq2[{n}]: {eq2[n].tex_string}")
                        m = get_removed_index(eq2[n].tex_string)
                        anims.append(RT(eq1[m], eq2[n], path_arc=np.pi if is_added(n, 0) else -np.pi, run_time=run_time))
                        swapped_from_added.append(m)
                    else:
                        console.print(f"+  {eq2[n].tex_string}")
                        anims.append(FadeIn(eq2[n], run_time=run_time))
                else:
                    console.print(f"+  {eq2[n].tex_string}")
                    console.print("passing, swapped from removed")
                    # anims.append(FadeIn(eq2[n], run_time=run_time))

        return anims

    @staticmethod
    def split_tex(tex):
        def next_tex_part(s):
            match = re.match(re_latex_expr, s)
            if match:
                if match[1].startswith(r"\frac"):
                    return s, ""
                p = match[1]
                s = s[len(match[0]):]
            elif len(s):
                p = s[0]
                s = s[1:]
            else:
                p = None
            return p, s

        tex_parts = []
        while tex:
            tex_part, tex = next_tex_part(tex)
            tex_parts.append(tex_part)

        console.print(tex_parts)
        return tex_parts

    @staticmethod
    def remove_equal_expression(*equations):
        expr_map = {}
        for eq in equations:
            expressions = str(eq).split('=')
            for i, expr in enumerate(expressions):
                expr = expr.strip()
                if expr in expr_map:
                    expr_map[expr] = [expr_map[expr], (eq, i)]
                else:
                    expr_map[expr] = eq, i
        new_eq_tex = None
        first = True
        rects = []
        equal_exprs = []
        for expr, pos in expr_map.items():
            if type(pos) == tuple:
                if not new_eq_tex:
                    new_eq_tex = expr
                else:
                    new_eq_tex += "=" + expr
            else:
                for eq, i in pos:
                    equal_expr = find_parts_from_tex(eq, str(expr))
                    rects.append(SurroundingRectangle(equal_expr))
                    equal_exprs.append(equal_expr)

        play(*[ShowCreation(rect) for rect in rects])
        equal_signs = []
        for eq in equations[1:]:
            equal_signs.extend(parts(eq, "="))
        # play(*[FadeOut(rect) for rect in rects], *[FadeOut(expr) for expr in equal_exprs],
        #      *[FadeOut(equal_sign) for equal_sign in equal_signs])
        assert "=" in new_eq_tex
        new_eq = Eq(new_eq_tex).align_to(equations[0], UP)
        keepers_anims = [RT(part(equations[0], "="), part(new_eq, "="), run_time=3.0)]
        for expr, pos in expr_map.items():
            if type(pos) == tuple:
                keepers_anims.append(RT(find_parts_from_tex(pos[0], expr), find_parts_from_tex(new_eq, expr), run_time=3.0))
        # play(*keepers_anims)
        play(*[FadeOut(rect) for rect in rects], *[FadeOut(expr) for expr in equal_exprs],
             *[FadeOut(equal_sign) for equal_sign in equal_signs], *keepers_anims)
        return new_eq

    def align_points_with_larger(self, larger_mobject):
        pass


def load_equations(filename):
    with open(filename, "r") as f:
        s = f.read()
    lines = s.split("\n")
    eqname = None
    equations = []
    for line in lines:
        if eqname and line.strip():
            exec(eqname + "=Eq(*line.strip().split(" "))")
            equations.append(eval(eqname))
            eqname = None
        match = re.match(r"\s*\\begin{equation}\\label{eq:(\w*)}", line)
        if match:
            eqname = match.group(1)
    return tuple(equations) if len(equations) > 1 else equations[0] if equations else None


if __name__ == '__main__':
    result = add_multiply_symbol("1=A_n (a - 1)")
    assert result == r'1=A_n \cdot (a - 1)', "result: " + result
    result = add_multiply_symbol(r"{A_n \over a}")
    assert result == r"{A_n \over a}", "result: " + result
    result = add_multiply_symbol("A_n (a - 1)")
    assert result == r"A_n \cdot (a - 1)", "result: " + result
    result = add_multiply_symbol("a^{n + 1} - 1=A_n (a - 1)")
    assert result == r'a^{n + 1} - 1=A_n \cdot (a - 1)', "result: " + result

    Eq.animate = False
    Eq.show_operation = False
    Eq.new_line = True
    result, anims = Eq(r"A_n - a^n = {A_n \over a} - {1 \over a}") * 'a'
    assert result.is_identical(Eq('a \\left(A_n - a^n\\right) = A_n - 1')), f"result: {result}"
    result, anims = result.replace_expr(r"a A_n - a^{n+1}", animate=False)
    assert result.is_identical(Eq(r"A_n a - a^{n + 1} = A_n - 1")), f"result: {result}"

    result = remove_unnecessary_brackets(sympy.Symbol("A_{n}"))
    assert str(result) == "A_n", "result: " + str(result)
    result = remove_unnecessary_brackets(sympy.Symbol("A_{n} - a^n"))
    assert str(result) == "A_n - a**n", "result: " + str(result)
    result = remove_unnecessary_brackets(sympy.Symbol("(1 + A_{n})*B_m"))
    assert str(result) == "B_m*(A_n + 1)", "result: " + str(result)

    result = convert_over_to_frac(r"{1 \over 2}{3 \over 4}")
    assert result == r"\frac{1}{2}\frac{3}{4}", "result: " + result
    result = convert_over_to_frac(r"{1 \over 2}")
    assert result == r"\frac{1}{2}", "result: " + result
    result = convert_over_to_frac(r"{{1\over 2} \over 2}")
    assert result == r"\frac{\frac{1}{2}}{2}", "s: " + result
    result = convert_over_to_frac(r"{1 \over {1\over 2}}")
    assert result == r"\frac{1}{\frac{1}{2}}", "s: " + result
    result = convert_over_to_frac(r"{{1\over 2} \over {3\over 4}}")
    assert result == r"\frac{\frac{1}{2}}{\frac{3}{4}}", "s: " + result
    result = convert_over_to_frac(r"{{1xx\over 2y} \over { 3aaaaa  \over bbbbb }}")
    assert result == r"\frac{\frac{1xx}{2y}}{\frac{3aaaaa}{bbbbb}}", "s: " + result
    result = convert_over_to_frac(r"<< {{1xx\over 2y} \over { 3aaaaa  \over bbbbb }} >>>")
    assert result == r"<< \frac{\frac{1xx}{2y}}{\frac{3aaaaa}{bbbbb}} >>>", "s: " + result
