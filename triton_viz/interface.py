import gradio as gr
import triton_viz
import tempfile



def launch():
    cache = {}
    program_records, tt = triton_viz.collect_grid()
    m = [0, 0, 0]
    for k in program_records.keys():
        m[0] = max(k[0], m[0])
        m[1] = max(k[1], m[1])
        m[2] = max(k[2], m[2])
    with gr.Blocks(css=".gradio-container button {overflow: auto}") as demo:
        with gr.Column():
            img = gr.Image(height=800, width=500)
            with gr.Row():
                s1 = gr.Slider(0, m[0], value=0, step=1, label="Program Id 0")
                s2 = gr.Slider(0, m[1], value=0, step=1, label="Program Id 1")
                s3 = gr.Slider(0, m[2], value=0, step=1, label="Program Id 2")
        
        def update(inp):
            a = inp[s1]
            b = inp[s2]
            c = inp[s3]
            idx = (a, b, c)

            if idx not in cache:
                name = tempfile.NamedTemporaryFile(suffix=".svg")
                triton_viz.draw_record(program_records[idx], tt, name.name)
                cache[idx] = name
            return cache[idx].name

        s1.change(update, inputs={s1, s2, s3}, outputs=img, show_progress=False)
        s2.change(update, inputs={s1, s2, s3}, outputs=img, show_progress=False)
        s3.change(update, inputs={s1, s2, s3}, outputs=img, show_progress=False)
        demo.load(update, inputs={s1, s2, s3}, outputs=img)

    demo.launch(share=True, debug=True)
