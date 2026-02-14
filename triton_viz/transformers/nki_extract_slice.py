import ast


class NKITransformer(ast.NodeTransformer):
    """
    An AST transformer for NKI.
    Transformations:
    - `nl.<func>(x[...], ...)` -> `nl.<func>(x, slice_obj, ...)`
       where <func> is one of {load, store, load_transpose2d}
    - strips out kernel decorators
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
        Strip out decorators i.e.
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
        """Convert func(x[keys], ...) into func(x, keys, ...)."""
        if not (
            isinstance(call_node.func, ast.Attribute)
            and isinstance(call_node.func.value, ast.Name)
            and call_node.func.value.id == "nl"
            and call_node.func.attr in ("store", "load", "load_transpose2d")
        ):
            return None

        if not call_node.args or not isinstance(call_node.args[0], ast.Subscript):
            return None

        slice_node = call_node.args[0]  # for func(x[keys], ...), slice_node = x[keys]
        return ast.Call(
            func=call_node.func,
            args=[
                slice_node.value,
                self._create_slice_value_node(slice_node.slice),
                *call_node.args[1:],
            ],
            keywords=call_node.keywords,
        )

    def _create_slice_value_node(self, node: ast.AST) -> ast.AST:
        """
        Transforms a slice's AST content into a constructible object.
        Turns something like [2:4:-1, 0, arange(3)+3, 2:] into
        (slice(2,4,-1), 0, arange(3)+3, slice(2,None,None)).
        """

        def process_key(key: ast.AST):
            if isinstance(
                key, ast.Slice
            ):  # start:stop:step -> slice(start, stop, step)
                return ast.Call(
                    func=ast.Name(id="slice", ctx=ast.Load()),
                    args=[
                        key.lower or ast.Constant(value=None),
                        key.upper or ast.Constant(value=None),
                        key.step or ast.Constant(value=None),
                    ],
                    keywords=[],
                )
            return key

        if isinstance(node, ast.Tuple):
            # the indexing keys for something like x[0, :, ...], node = ast.tuple(0, :, ...)
            return ast.Tuple(elts=[process_key(e) for e in node.elts], ctx=ast.Load())
        return process_key(node)  # something like x[index], node = AST of <index>


def transform_code(source_code: str) -> str:
    """
    Applies the NKITransformer to a string of Python code.
    """
    tree = ast.parse(source_code)
    transformer = NKITransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)
