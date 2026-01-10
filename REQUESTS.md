# Requests

1. Commit: `997c76374bac2211ad94048b93bd49d7b6fca722`
   Prompt: "Add indirect load example."
   Summary: Added the indirect load example to the repo.

2. Commit: `8bff32ebfc5d96ebf9192319432301a1cf9283b5`
   Prompt: "Adjust indirect load example defaults."
   Summary: Tuned the indirect load example defaults to the requested behavior.

3. Commits: `ae5a12aabbdfe9b6a0ab2e57952692d11ddfebc2`, `261a16997d3a19410248e83c08b4c20a237fc5a8`
   Prompt: "Can you add the program ID slider on a sidebar to the left? Can you also remove the window that has the drag color by value, drag cubes, showcode, value histogram, and so on? I want the color by value, show code, and value histogram buttons on the left sidebar as well. Can you also remove the show code button on the top right?"
   Summary: Moved operation controls into the sidebar and removed the top-right Show Code toggle.

4. Commit: `05fe0e8c0dac776a05122823d437d4252ee03a8c`
   Prompt: "The black screen containing the 3D tensors view is too small. Can you make it take up a much larger part of the screen? Like the only part that shouldn't be black should be the left sidebar containing the program ID sliders, the code (show it by default), the color by value button, and the value histogram button."
   Summary: Expanded the detailed view layout so the 3D canvas dominates the screen alongside the sidebar.

5. Commits: `71f4e9aaf93bec11437d136431215f33667b4045`, `50b513d035b1424460e65baa072649e4d45b4f3d`, `7e2d57fc8b48bba2ca49b4349a390099e03a9917`, `89a1e09fff0cc980dd908b6cc28d30db96e6e386`, `7d4d8a3aaf8156759808adc284957c07126e89fc`, `5427f68c4908b2a2bbc15799036c57541a813645`, `050d8d4623a46f0239ba07bf76268b2fe7eef883`, `3cb31af6cc0ab6e7c9b2e4ea2a14e961289ee0d5`, `c574d3e703e117793fed0df64794b7161945d4d1`
   Prompt: "Remove the Show Code button (keep on always). If there's only one program ID along an axis, make the slider read-only. Rename Program Id 0/1/2 to X/Y/Z. Add a colorbar for Color by Value. Show hover index/value. Make Store UI like Load but orange selection. Make the code view fill empty sidebar space. Add a splitter. Remove the Program (X, Y, Z) label."
   Summary: Implemented slider renaming/locking, load/store colorbars and hover details, store orange styling, code panel expansion, sidebar resizer, and removed the program title label.

6. Commits: `fb6720ceb491c4fb189addc595d1b133d9471b87`, `ea32deba3204f9af676e51778655c5691587ec82`, `58c15453ab24cea578e8b6aa679e509020cfb794`, `317b8fd74df3caca192b9d20a360aa3cf29f477d`, `6dc3e367a022bf85905875d90ca90a839dd123d9`, `4f35dfe1d64530db03aff993527aae05ee0c4acd`
   Prompt: "Issues: code window goes away on PID change; sidebar splitter doesn't work; store animation crashes; Value Histogram should be a toggle and remove Close button; read-only sliders should be shaded; remove the flow arrow and Current Operation panel."
   Summary: Kept code panel visible on PID changes, fixed resizer visibility, resolved the store label crash, wired histogram toggling, styled disabled sliders, and removed flow arrow UI.

7. Commits: `92eaa226088d18f1c56091918712b54ce7903add`, `3316a778d9036a0937e14ab5df39302e0d3def0c`
   Prompt: "More issues: widen splitter range to 0-100%, remove the leftover highlighted cube from flow arrows, and fix remaining store issues."
   Summary: Expanded the resizer range and removed per-frame highlight artifacts from load/store animations.

8. Commit: `c2789f3254e5f6e123d77537b11856feb612e76b`
   Prompt: "Value Histogram window: add labels for bins and activation dropdown, show all activation options, auto-refresh on input changes (remove refresh button)."
   Summary: Added labels/options and auto-refresh behavior for histogram controls.

9. Commit: `5352ec0de6dd5b43d8476ca2dac78c3cfce15f6a`
   Prompt: "PID sliders start at All; add a sidebar button to visualize all program IDs at once."
   Summary: Defaulted PID filters to 0 and added an all-programs overlay mode.

10. Commits: `8662f09b0699dff9cc9528f8fb77a99ecbaf996a`, `6895cf4f4e5b36572f7a1df1c56b56fe149d78ce`, `c6c88cbf613f0c7ad638856d337c9db6ab975e94`
    Prompt: "On PID slider change: keep the current visualization, do not refresh code view, remove per-frame highlight flashes, and pin the colorbar (simplify its title)."
    Summary: Preserved the active op/code on PID changes, removed per-frame flashes, and pinned/simplified value legends.

11. Commits: `6fa8d68d4c36e90c3f9d44010c6504f8c6e43a2b`, `775135c1358cf6ac9d08c86d61de9d4cc393a7a4`, `112d55fd798893e9402f5fb108585a19e9dc3f29`, `e40824d7daa9af94ce65c6a183f13cf791fde8c4`
    Prompt: "Idle CPU/GPU usage is high; all-programs overlay is incorrect with overlaps; move hover window to bottom-right; make PID sliders more compact."
    Summary: Rendered load/store on demand, fixed overlapping program overlays, moved hover panels, and compacted PID slider layout.

12. Commits: `1d83499a4c444a0c060a814ae04358ac0dadc992`, `6e088e8d4476c6cc300e20cab342bf78ff12e08a`, `206532750a420e9a8a0be5a6fcb2a3c90bca57f1`, `c50fa2eaef5b6d3c84883e710a073d8dad210628`, `f0002d7eb18b0609f053442b4f4e20a22ea38f53`
    Prompt: "Store hover selects the wrong cell; preserve camera across PID changes; add Code View label/blurb; add PID slider ticks; remove the All option."
    Summary: Fixed store raycasting, preserved camera state, added the code label, added slider ticks, and removed the All option.

13. Commits: `39036099b54096de0154aa3b9c77e0edaa0f91c2`, `0c5148bb5fc0d785aa068bd952dde1daa8c8f74c`, `031f7922b468a9505e89ac235b5667352cddca9f`, `296fa35bc4c0bf1df04009c71274d4d6f8bdcf38`
    Prompt: "Load/Store visualizations are not visible; remove sidebar Z slice/Actions; Dot hover panel should be bottom-right; Color by Value should color A/B matrices and show colorbars."
    Summary: Fixed load/store render init, removed the extra sidebar sections, and extended Dot coloring (A/B/C) with legends and stable scaling.

14. Commit: `8b899d54169e5cd25c7ca570da43c956072127fa`
    Prompt: "Dot colorbars are all the same; use different colormaps for each activation."
    Summary: Gave A/B/C distinct color gradients for both meshes and legends.

15. Commit: `7e3841dc3a3f9b61fa4425a6ca84e07104788f22`
    Prompt: "Operation Controls menu buttons should stay enabled for Load/Store, and only the relevant ones for Dot."
    Summary: Ensured load/store always register op-control handlers so the buttons remain clickable.

16. Commit: `646148acd3fdae2a370839528a1bf0a9027a9418`
    Prompt: "Make camera rotations more snappy (remove momentum)."
    Summary: Disabled OrbitControls damping for immediate camera response.

17. Commit: `02cd32f6a724a4d1d514688a5268a82a8d293b63`
    Prompt: "Add labels on the black screen that show tensor sizes on each axis."
    Summary: Added axis size labels for tensor views across load/store and dot.

18. Commit: `3c7cca62895f6904e57a8bed4f7425dfcf26f328`
    Prompt: "Tensor size labels should be numeric-only and placed like dimension labels, not corner labels."
    Summary: Replaced axis labels with numeric dimension annotations along the tensor edges.

19. Commit: `2c17ae71324a038d107abe19be6da1dcb984c007`
    Prompt: "Dot -> Load console error: destroyLegend is not defined."
    Summary: Fixed matmul cleanup to call the correct legend teardown helper.

20. Commit: `7566c1ca1e53723ed32a9c846612b4c011fcb623`
    Prompt: "Web UI startup time is slow; investigate."
    Summary: Avoided recompute on index load by caching launch snapshots and deferring heavy data work to the API.
