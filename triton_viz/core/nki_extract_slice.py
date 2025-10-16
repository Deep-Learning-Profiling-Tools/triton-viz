import ast

class StoreCallTransformer(ast.NodeTransformer):
    """
    A targeted AST transformer to rewrite `nl.store(x[...], ...)` calls
    into `masked_store(x, slice_obj, ...)` and `nl.load(x[...])` calls
    into `masked_load(x, slice_obj)` with a preceding assignment.
    """

    # MODIFICATION 1: We now visit the 'Expr' statement wrapper.
    def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.AST]:
        """
        Intercept and transform expression statements.
        """
        # We only care about expressions that are function calls.
        if not isinstance(node.value, ast.Call):
            return self.generic_visit(node)

        call_node = node.value

        # --- The logic from the old visit_Call starts here ---

        # 1. IDENTIFY: Is this the specific call we want to transform?
        if not (
            isinstance(call_node.func, ast.Attribute)
            and isinstance(call_node.func.value, ast.Name)
            and call_node.func.value.id == 'nl'
            and call_node.func.attr in ('store', 'load')
        ):
            return self.generic_visit(node)

        # 2. VALIDATE: Does the call have the expected structure?
        if not call_node.args or not isinstance(call_node.args[0], ast.Subscript):
            return self.generic_visit(node)

        # 3. DECONSTRUCT
        subscript_node = call_node.args[0]
        sliced_object = subscript_node.value
        slice_content = subscript_node.slice
        remaining_args = call_node.args[1:]

        # 4. REBUILD
        # Convert to nl.masked_load or nl.masked_store
        func_name = 'masked_' + call_node.func.attr

        new_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='nl', ctx=ast.Load()),
                attr=func_name,
                ctx=ast.Load()
            ),
            args=[
                sliced_object,
                self._create_slice_value_node(slice_content),
                *remaining_args
            ],
            keywords=call_node.keywords
        )

        # 5. REPLACE
        return ast.Expr(value=new_call)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """
        Handle assignment statements where the value is an nl.load or nl.store call.
        """
        # Check if the value being assigned is a function call
        if not isinstance(node.value, ast.Call):
            return self.generic_visit(node)

        call_node = node.value

        # 1. IDENTIFY: Is this the specific call we want to transform?
        if not (
            isinstance(call_node.func, ast.Attribute)
            and isinstance(call_node.func.value, ast.Name)
            and call_node.func.value.id == 'nl'
            and call_node.func.attr in ('store', 'load')
        ):
            return self.generic_visit(node)

        # 2. VALIDATE: Does the call have the expected structure?
        if not call_node.args or not isinstance(call_node.args[0], ast.Subscript):
            return self.generic_visit(node)

        # 3. DECONSTRUCT
        subscript_node = call_node.args[0]
        sliced_object = subscript_node.value
        slice_content = subscript_node.slice
        remaining_args = call_node.args[1:]

        # 4. REBUILD
        # Convert to nl.masked_load or nl.masked_store
        func_name = 'masked_' + call_node.func.attr

        new_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='nl', ctx=ast.Load()),
                attr=func_name,
                ctx=ast.Load()
            ),
            args=[
                sliced_object,
                self._create_slice_value_node(slice_content),
                *remaining_args
            ],
            keywords=call_node.keywords
        )

        # 5. REPLACE: Create a new assignment with the transformed call
        return ast.Assign(
            targets=node.targets,
            value=new_call,
            type_comment=getattr(node, 'type_comment', None)
        )

    def _create_slice_value_node(self, node: ast.AST) -> ast.AST:
        """
        (This helper function is unchanged and correct)
        Recursively transforms a slice's AST content into a constructible object.
        """
        match node:
            case ast.Slice(lower, upper, step):
                return ast.Call(
                    func=ast.Name(id='slice', ctx=ast.Load()),
                    args=[
                        lower or ast.Constant(value=None),
                        upper or ast.Constant(value=None),
                        step or ast.Constant(value=None)
                    ],
                    keywords=[]
                )
            case ast.Tuple(elts):
                return ast.Tuple(
                    elts=[self._create_slice_value_node(e) for e in elts],
                    ctx=ast.Load()
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

# --- Input Code ---
source_code = """
import numpy as np

# This load call should also be transformed
sbuf_value = nl.load(x[2:4, None, ..., nl.arange(128)[None, :]], mask=mask)

# This is the call we want to transform
nl.store(x[2:4, None, ..., nl.arange(128)[None, :]], sbuf_value, mask=mask)


# This call should be ignored
other_lib.store(y[10:])

# This call should also be ignored
nl.some_other_func(z[:5])
"""

if __name__ == '__main__':
    # --- Transformation ---
    transformed_code = transform_code(source_code)

    # --- Output ---
    print("--- Original Code ---")
    print(source_code)
    print("\n--- Transformed Code ---")
    print(transformed_code)
