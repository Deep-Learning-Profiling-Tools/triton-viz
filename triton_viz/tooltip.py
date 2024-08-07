tooltip_descriptions = {
    "Grid Size": "Indicates the grid size used in the kernel launch.",
    "BinaryOp": "Shows the total number of binary operations performed in this kernel.",
    "MakeRange": "Shows the total number of arange operations performed in this kernel.",
    "Load": "Shows the total number of load operations performed in this kernel.",
    "Store": "Shows the total number of store operations performed in this kernel.",
    "ExpandDims": "Shows the total number of expand_dims operations performed in this kernel.",
    "Dot": "Shows the total number of dot operations performed in this kernel.",
    "Reduce": "Shows the total number of reduce operations performed in this kernel.",
    "Total number of bytes loaded": "Shows the total number of bytes loaded (mask=True). Note: On GPUs, this metric does not equate to the total number of bytes loaded from global memory (DRAM), as some data accesses may be handled through GPU caches.",
    "Masked Load Ratio": "Ratio of total number of bytes loaded (mask=True)/total number of bytes loaded (mask=True) + (mask=False).",
    "Total number of bytes stored": "Shows the total number of bytes stored (mask=True).",
    "Masked Store Ratio": "Ratio of total number of bytes stored (mask=True)/total number of bytes stored (mask=True) + (mask=False).",
}

def get_tooltip_data(df):
    """Return the tooltip data in a format suitable for JSON serialization."""
    return df.to_dict()

def create_tooltip(df):
    styles = """
    <style>
        .dataframe {
            width: 100%;
            border-collapse: collapse;
            margin: auto;
        }
        .dataframe th {
            color: white !important;
            background-image: linear-gradient(to right, #434343, #757575); !important;
        }
        .dataframe td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .tooltip {
            position: relative;
            display: block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 100%;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 100;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        .gradio_container, .gradio_container * {
            overflow: visible !important;
        }
    </style>
    """

    html = styles + '<table class="dataframe"><thead><tr>'
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"

    for index, row in df.iterrows():
        tooltip_text = tooltip_descriptions.get(
            row["Metric"], "No description available."
        )
        html += f'<tr><td class="tooltip">{row["Metric"]}<span class="tooltiptext">{tooltip_text}</span></td>'
        html += f'<td>{row["Value"]}</td></tr>'

    html += "</tbody></table>"

    return html
