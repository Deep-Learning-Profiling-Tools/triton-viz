import ast


class StoreCallTransformer(ast.NodeTransformer):
    """
    A targeted AST transformer to rewrite:
    - `nl.store(x[...], ...)` -> `nl.masked_store(x, slice_obj, ...)`
    - `nl.load(x[...], ...)` -> `nl.masked_load(x, slice_obj, ...)`
    - `nl.load_transpose2d(x[...], ...)` -> `nl.load_transpose2d(x, slice_obj, ...)`
    """

    def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.AST]:
        """
        Intercept and transform expression statements.
        """
        # We only care about expressions that are function calls.
        if not isinstance(node.value, ast.Call):
            return self.generic_visit(node)

        new_call = self._rewrite_nl_slice_call(node.value)
        if new_call is None:
            return self.generic_visit(node)
        return ast.Expr(value=new_call)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """
        strip out decorators i.e.
        @triton_viz.trace('tracer')
        def nki_kernel(): ...

        or else the NKI interpreter will only see a NKITrace object and not nki_kernel
        """
        node.decorator_list = []
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """
        Handle assignment statements where the value is an nl.load or nl.store call.
        """
        if not isinstance(node.value, ast.Call):
            return self.generic_visit(node)

        new_call = self._rewrite_nl_slice_call(node.value)
        if new_call is None:
            return self.generic_visit(node)
        return ast.Assign(
            targets=node.targets,
            value=new_call,
            type_comment=getattr(node, "type_comment", None),
        )

    def _rewrite_nl_slice_call(self, call_node: ast.Call) -> ast.Call | None:
        if not (
            isinstance(call_node.func, ast.Attribute)
            and isinstance(call_node.func.value, ast.Name)
            and call_node.func.value.id == "nl"
            and call_node.func.attr in ("store", "load", "load_transpose2d")
        ):
            return None

        if not call_node.args or not isinstance(call_node.args[0], ast.Subscript):
            return None

        subscript_node = call_node.args[0]
        sliced_object = subscript_node.value
        slice_content = subscript_node.slice
        remaining_args = call_node.args[1:]
        func_name = (
            "masked_" + call_node.func.attr
            if call_node.func.attr in ("store", "load")
            else "load_transpose2d"
        )
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="nl", ctx=ast.Load()), attr=func_name, ctx=ast.Load()
            ),
            args=[
                sliced_object,
                self._create_slice_value_node(slice_content),
                *remaining_args,
            ],
            keywords=call_node.keywords,
        )

    def _create_slice_value_node(self, node: ast.AST) -> ast.AST:
        """
        (This helper function is unchanged and correct)
        Recursively transforms a slice's AST content into a constructible object.
        """
        match node:
            case ast.Slice(lower, upper, step):
                return ast.Call(
                    func=ast.Name(id="slice", ctx=ast.Load()),
                    args=[
                        lower or ast.Constant(value=None),
                        upper or ast.Constant(value=None),
                        step or ast.Constant(value=None),
                    ],
                    keywords=[],
                )
            case ast.Tuple(elts):
                return ast.Tuple(
                    elts=[self._create_slice_value_node(e) for e in elts],
                    ctx=ast.Load(),
                )
            case _:
                return node


def transform_code(source_code: str) -> str:
    """
    Applies the StoreCallTransformer to a string of Python code.
    """
    tree = ast.parse(source_code)
    transformer = StoreCallTransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)
