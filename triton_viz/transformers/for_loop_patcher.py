import ast


def _visit_For(self, node: ast.For):  # type: ignore[override]
    """
    for i in R:
        ...
    ==>
    for i in _triton_viz_loop_patcher.hooks.loop_iter_wrapper(iter_callable, args, kwargs, lineno, range_type):
        ...
    where _triton_viz_loop_patcher.hooks.loop_iter_wrapper returns a _LoopIter object.
    """
    self.generic_visit(node)

    # Detect range type
    range_type = "unknown"
    if isinstance(node.iter, ast.Call):
        func = node.iter.func
        if isinstance(func, ast.Name) and func.id == "range":
            range_type = "python_range"
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "tl"
        ):
            if func.attr == "range":
                range_type = "tl_range"
            elif func.attr == "static_range":
                range_type = "tl_static_range"

    if isinstance(node.iter, ast.Call):
        iter_callable = node.iter.func
        iter_args = ast.Tuple(elts=node.iter.args, ctx=ast.Load())
        kw_keys = []
        kw_vals = []
        for kw in node.iter.keywords:
            if kw.arg is None:  # skip **kwargs for simplicity
                continue
            kw_keys.append(ast.Constant(value=kw.arg))
            kw_vals.append(kw.value)
        iter_kwargs = ast.Dict(keys=kw_keys, values=kw_vals)
    else:
        iter_callable = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=node.iter,
        )
        iter_args = ast.Tuple(elts=[], ctx=ast.Load())
        iter_kwargs = ast.Dict(keys=[], values=[])

    new_iter = ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="_triton_viz_loop_patcher", ctx=ast.Load()),
                attr="hooks",
                ctx=ast.Load(),
            ),
            attr="loop_iter_wrapper",
            ctx=ast.Load(),
        ),
        args=[
            iter_callable,
            iter_args,
            iter_kwargs,
            ast.Constant(value=node.lineno),
            ast.Constant(value=range_type),
        ],
        keywords=[],
    )

    new_for = ast.For(
        target=node.target,
        iter=new_iter,
        body=node.body,
        orelse=node.orelse,
        type_comment=node.type_comment,
    )
    return ast.fix_missing_locations(new_for)
