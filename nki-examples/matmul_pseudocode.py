from triton_viz.core.trace import launches
import triton_viz, pickle
with open('nki_mm_launches.pkl', 'rb') as f:
    launches += pickle.load(f)
    triton_viz.launch()
