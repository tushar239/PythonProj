def fun(s):
    if isinstance(s, str) and s.isdigit():
        return int(s)
    elif isinstance(s, int):
        return int(s)
    else:
        raise TypeError("input value is expected as digit")
