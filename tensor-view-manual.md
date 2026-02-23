# Tensor View Manual

## Visible Dimensions Semantics

- tokens are space-separated: `A B C D`
- axis labels are uppercase when visible and lowercase when hidden
- `1` inserts a singleton dimension
- coalesced groups are written as one token: `BC`, `DA`, `ABC`
- coalesced groups are atomic for visibility:
  - valid: `BC` or `bc`
  - invalid: `Bc`, `bC`
- every original axis must appear exactly once across tokens (plus any number of `1` tokens)

## Meaning of Case

- uppercase token: visible axis/group
- lowercase token: hidden axis/group (sliced to one index)
- hidden tokens stay in the string so we preserve full-tensor context for outlines

## Outline Shape Semantics

- outline shape is derived from the token sequence (ignoring case)
- examples:
  - `ADG B C 1 E F 1 H I` and `adg B C 1 E F 1 H I` share the same outline shape
  - that outline is different from raw `A B C D E F G H I`

## Slider Semantics

- one index slider is rendered per hidden token (not per axis)
- example: `dc AB` renders a slider labeled `dc`

## Tensor Slice Preview Semantics

- preview slices hidden axes by index
- visible tokens define output axis order
- coalesced visible tokens imply reshape dimensions from grouped products
- inserted `1` tokens appear in reshape output shape

## Examples

1. base case
- `(A, B, C, D)` -> `A B C D`

2. pure permutation
- `(A, B, C, D).permute(3, 2, 1, 0)` -> `D C B A`

3. permute + coalesce
- `(A, B, C, D).permute(3, 2, 1, 0).view(D, CB, A)` -> `D CB A`

4. hide one dimension
- from `A B C`, show only `B C` -> `a B C`

5. hide multiple dimensions
- from `A B C D`, show only `C D` -> `a b C D`

6. all hidden
- from `A B C` -> `a b c`

7. insert singleton
- `(N, C, H, W) -> (N, C, 1, H, W)` -> `N C 1 H W`

8. insert singleton + permutation
- `(N, C, H, W).view(N, C, 1, H, W).permute(2, 0, 1, 3, 4)` -> `1 N C H W`

9. coalesced visibility is all-or-nothing
- `(A, B, C).view(A, BC)` -> `A BC` or `A bc`
- invalid: `A bC`, `A Bc`

10. two coalesced groups
- `(A, B, C, D).view(AB, CD)` -> `AB CD`
- hide one group -> `ab CD`
- hide both -> `ab cd`

11. permute then coalesce, hidden first group
- `(A, B, C, D).permute(2, 0, 3, 1).view(CA, DB)`
- with `CA` hidden -> `ca DB`

12. mixed single + coalesced + single
- `(A, B, C, D).view(A, BC, D)` -> `A BC D`
- hide middle only -> `A bc D`

13. coalesced + singleton
- `(A, B, C).view(AB, 1, C)` -> `AB 1 C`
- hide `AB` and `C` -> `ab 1 c`

14. degenerate full coalesce
- `(A, B, C).view(ABC)` -> `ABC`
- hidden -> `abc`

15. larger regrouping after permutation
- `(A, B, C, D, E).permute(4, 1, 3, 0, 2).view(EB, DA, C)` -> `EB DA C`
- hide first group only -> `eb DA C`

16. corrected complex case
- `(A, B, C, D, E).permute(4, 2, 0, 3, 1).view(EC, A, DB)`
- valid forms include `EC A DB` and `ec A db`
- mixed-case grouped tokens are invalid
