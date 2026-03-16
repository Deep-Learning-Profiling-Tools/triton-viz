import ast


def _visit_If(self, node: ast.If):  # type: ignore[override]
    """
    Transform if/elif/else to inject branch tracking hooks.

    if cond:           _triton_viz_if_patcher.pre_if(cond, LINENO)
        body1    ==>   try:
    else:                  if _triton_viz_if_patcher.eval_condition(LINENO):
        body2                  body1
                           else:
                               _triton_viz_if_patcher.flip_condition(LINENO)
                               body2
                       finally:
                           _triton_viz_if_patcher.post_if(LINENO)

    elif chains work automatically because generic_visit transforms the inner
    If node (which sits in node.orelse) first.
    """
    # Handle nested if statements (including elif chains) first
    self.generic_visit(node)

    lineno_const = ast.Constant(value=node.lineno)
    patcher_name = ast.Name(id="_triton_viz_if_patcher", ctx=ast.Load())

    # _triton_viz_if_patcher.pre_if(cond, LINENO)
    pre_if_call = ast.Expr(
        value=ast.Call(
            func=ast.Attribute(
                value=patcher_name,
                attr="pre_if",
                ctx=ast.Load(),
            ),
            args=[node.test, lineno_const],
            keywords=[],
        )
    )

    # _triton_viz_if_patcher.eval_condition(LINENO)
    eval_condition_call = ast.Call(
        func=ast.Attribute(
            value=patcher_name,
            attr="eval_condition",
            ctx=ast.Load(),
        ),
        args=[lineno_const],
        keywords=[],
    )

    # _triton_viz_if_patcher.flip_condition(LINENO)
    flip_condition_stmt = ast.Expr(
        value=ast.Call(
            func=ast.Attribute(
                value=patcher_name,
                attr="flip_condition",
                ctx=ast.Load(),
            ),
            args=[lineno_const],
            keywords=[],
        )
    )

    # _triton_viz_if_patcher.post_if(LINENO)
    post_if_call = ast.Expr(
        value=ast.Call(
            func=ast.Attribute(
                value=patcher_name,
                attr="post_if",
                ctx=ast.Load(),
            ),
            args=[lineno_const],
            keywords=[],
        )
    )

    # Build the else body: flip_condition + original else body
    if node.orelse:
        new_orelse = [flip_condition_stmt] + node.orelse
    else:
        new_orelse = [flip_condition_stmt, ast.Pass()]

    # New if: if eval_condition(LINENO): body else: flip + orelse
    new_if = ast.If(
        test=eval_condition_call,
        body=node.body,
        orelse=new_orelse,
    )

    # Wrap in try/finally to guarantee post_if runs
    try_node = ast.Try(
        body=[new_if],
        handlers=[],
        orelse=[],
        finalbody=[post_if_call],
    )

    # Return list: [pre_if_call, try_node]
    # NodeTransformer splices lists into parent body
    result = [pre_if_call, try_node]
    for n in result:
        ast.fix_missing_locations(n)
    return result
