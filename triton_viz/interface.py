import gradio as gr
import triton_viz
import tempfile
from .analysis import analyze_records
from .tooltip import create_tooltip
import pandas as pd


def launch(share=True):
    cache = {}
    analysis_data = analyze_records()
    program_records, tt, failures = triton_viz.collect_grid()
    m = [0, 0, 0]
    size = [0, 0]
    for k in program_records.keys():
        m[0] = max(k[0] + 1, m[0])
        m[1] = max(k[1] + 1, m[1])
        m[2] = max(k[2] + 1, m[2])
    w, h = triton_viz.draw_record(program_records[(0, 0, 0)], tt, "tmp.png")
    size[0] = w
    size[1] = h
    height = 600 * size[1] / size[0]
    with gr.Blocks(
        css=".gradio-container button {overflow: auto} img.with-caption {height: %fpx !important; } .thumbnails { display: none; }  "
        % height
    ) as demo:
        with gr.Row():
            with gr.Column(scale=3, min_width=500):
                img = gr.Gallery(
                    height=500,
                    min_width=500,
                    show_label=False,
                    selected_index=0,
                    preview=True,
                    object_fit="cover",
                )
            with gr.Column(scale=1):
                s1 = gr.Slider(0, m[0] - 1, value=0, step=1, label="Program Id 0")
                s2 = gr.Slider(0, m[1] - 1, value=0, step=1, label="Program Id 1")
                s3 = gr.Slider(0, m[2] - 1, value=0, step=1, label="Program Id 2")
                b1 = gr.Button("Precompute")
                gr.Markdown("## Analysis")
                df = pd.DataFrame(analysis_data, columns=["Metric", "Value"])
                analysis_with_tooltip = create_tooltip(df)
                gr.HTML(analysis_with_tooltip)
                if failures:
                    gr.Markdown(
                        show_label=False,
                        value="## Invalid memory access in "
                        + "\n * "
                        + "\n* ".join(list(map(str, failures.keys()))),
                    )

        def cache_block(idx):
            name = tempfile.NamedTemporaryFile(suffix=".png")
            w, h = triton_viz.draw_record(program_records[idx], tt, name.name)
            size[0] = w
            size[1] = h
            cache[idx] = (name, len(cache))

        def update(inp):
            a = inp[s1]
            b = inp[s2]
            c = inp[s3]
            idx = (a, b, c)

            if idx not in cache:
                cache_block(idx)
                return gr.Gallery(
                    value=[(cache[k][0].name, str(k)) for k in cache.keys()],
                    selected_index=cache[idx][1],
                    height=700,
                ), gr.Slider()
            # * size[1]/size[0]
            return gr.Gallery(selected_index=cache[idx][1]), gr.Slider()

        def precompute(inp):
            a = inp[s1]
            b = inp[s2]
            c = inp[s3]
            idx = (a, b, c)
            for i in range(m[0]):
                for j in range(m[1]):
                    for k in range(m[2]):
                        if (i, j, k) not in cache:
                            cache_block((i, j, k))
            return gr.Gallery(
                value=[(cache[k][0].name, str(k)) for k in cache.keys()],
                selected_index=cache[idx][1],
            )

        s1.change(update, inputs={s1, s2, s3}, outputs=[img, b1], show_progress=False)
        s2.change(update, inputs={s1, s2, s3}, outputs=[img, b1], show_progress=False)
        s3.change(update, inputs={s1, s2, s3}, outputs=[img, b1], show_progress=False)
        b1.click(precompute, inputs={s1, s2, s3}, outputs=img, show_progress=True)
        demo.load(update, inputs={s1, s2, s3}, outputs=[img, b1])

    demo.launch(share=share, debug=False, height=800, quiet=True, show_api=False)
    return failures
