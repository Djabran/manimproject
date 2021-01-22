from manim import *


def diff(a, b, matchfunc=None):
    trace = shortest_edit(a, b, matchfunc)
    trace.reverse()
    x, y = len(a), len(b)
    moves = []
    for d, v in enumerate(trace):
        d = len(trace) - d - 1
        k = x - y
        if k == -d or (k != d and v[k - 1] < v[k + 1]):
            prev_k = k + 1
        else:
            prev_k = k - 1
        prev_x = v[prev_k]
        prev_y = prev_x - prev_k

        while x > prev_x and y > prev_y:
            moves.append((x - 1, y - 1, x, y))
            x -= 1
            y -= 1
        if d > 0:
            moves.append((prev_x, prev_y, x, y))
        x, y = prev_x, prev_y

    result = []
    for move in moves:
        # console.print(move)
        prev_x, prev_y, x, y = move
        if x == prev_x:
            result.append((1, prev_y))  # index of part inserted from b
        elif y == prev_y:
            result.append((-1, prev_x))  # index of part removed from a
        else:
            result.append((0, (prev_x, prev_y)))  # indices of same parts in a and b
    result.reverse()
    return result


def shortest_edit(a, b, matchfunc=None):
    n, m = len(a), len(b)
    # console.print(f"n: {n}, m: {m}")
    max_ = n + m
    v = np.zeros(2 * max_ + 1, dtype=np.int16)
    v[1] = 0
    trace = []
    for d in range(max_ + 1):
        # console.print(f"d={d}: {v}")
        trace.append(v.copy())
        for k in range(-d, d+1, 2):
            if k == -d or (k != d and v[k - 1] < v[k + 1]):
                x = v[k + 1]
            else:
                x = v[k - 1] + 1
            y = x - k
            # console.print(f"  k: {k}, x,y: {x}, {y}")
            while x < n and y < m and ((not matchfunc and a[x] == b[y]) or (matchfunc and matchfunc(a[x], b[y]))):
                # console.print(f"   ++x, ++y (a[{x}]=b[{y}]={a[x]}")
                x += 1
                y += 1
            v[k] = x
            # console.print(f"    v[{k}]: {x}")
            if x >= n and y >= m:
                # console.print(f"trace {d} (k={k}): {trace[d]}")
                return trace
