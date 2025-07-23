const infoContent = [
    {
        title: "Visualization Tool Overview",
        content: `
            <p>This tool allows you to explore and visualize tensor operations.</p>
            <ul>
                <li>Use the sliders to filter grid blocks along different dimensions.</li>
                <li>Click on a grid block to view detailed information about its operations.</li>
                <li>The Z-axis slider controls the current Z-level being displayed.</li>
            </ul>

            <h3>Need a bit of a tutorial?</h3>
            <p>If you're new to Triton or need to brush up on your skills, check out Triton Puzzles. It's a collection of Triton problems designed to get you up to speed:</p>
            <p><a href="https://github.com/srush/Triton-Puzzles" target="_blank" rel="noopener noreferrer" style="color: #4a90e2; text-decoration: none; font-weight: bold;">Triton Puzzles on GitHub</a></p>
            <p>These puzzles provide hands-on experience with Triton concepts, which can help you better understand the visualizations in this tool.</p>
        `
    },
    {
        title: "Using the Sliders",
        content: `
            <p>The sliders on the right side of the screen allow you to filter the grid blocks:</p>
            <ul>
                <li>Program Id 0: Filters blocks along the X-axis</li>
                <li>Program Id 1: Filters blocks along the Y-axis</li>
                <li>Program Id 2: Filters blocks along the Z-axis</li>
            </ul>
            <p>Set a slider to -1 to show all blocks along that dimension.</p>
        `
    },
    {
        title: "Detailed View",
        content: `
            <p>Clicking on a grid block will open a detailed view of the operations for that block.</p>
            <p>In the detailed view, you can:</p>
            <ul>
                <li>See the specific operations performed in the block</li>
                <li>View visualizations of matrix multiplications, loads, and stores</li>
                <li>Navigate between different operations using the tabs at the top</li>
            </ul>

        `
    },
    {
        title: "Tensor Slicing in Triton",
        content: `
            <p>Triton operates on slices of tensors rather than entire tensors. This is a key concept in understanding how Triton kernels process data efficiently.</p>

            <h3>Global Tensor vs Slice Tensor</h3>
            <ul>
                <li><strong>Global Tensor:</strong> This is the full tensor created by PyTorch. It represents the entire data structure you're working with.</li>
                <li><strong>Slice Tensor:</strong> This is a portion of the global tensor that Triton operates on within its kernels.</li>
            </ul>

            <h3>Loading Tensor Slices</h3>
            <p>Inside Triton kernels, <code>tl.load</code> is used to load slices of the global tensor. For example:</p>
            <pre><code>a = tl.load(a_ptrs, mask=offs_k[None, :])</code></pre>
            <p>This operation loads only the necessary slice of data, optimizing memory access and computation.</p>

            <h3>Key Points</h3>
            <ul>
                <li>PyTorch creates and manages the global tensor.</li>
                <li>Triton kernels work with slices of this global tensor for efficiency.</li>
                <li><code>tl.load</code> and <code>tl.store</code> operations in Triton handle the slicing automatically.</li>
                <li>This approach allows for parallel processing of different parts of the tensor.</li>
            </ul>

            <img src="/static/images/tensor-slicing-diagram.png" alt="Tensor Slicing Diagram" style="width: 100%; max-width: 1200px; height: auto; margin-top: 30px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <p style="font-style: italic; margin-top: 10px;">Diagram showing how a global PyTorch tensor is sliced for processing in Triton kernels.</p>
        `
    },
    {
        title: "Memory Hierarchy in AI Accelerators and Triton Operations",
        content: `
            <p>Understanding the memory hierarchy in AI accelerators is crucial for optimizing Triton kernels. While this example uses a GPU, it's important to note that Triton is designed for various AI accelerators, not just GPUs.</p>

            <h3>Memory Types in AI Accelerators</h3>
            <ul>
                <li><strong>Global Memory (GMEM):</strong> The magenta region in the image represents global memory, often referred to as DRAM, HBM, or device memory. This is typically the largest but slowest memory on the accelerator.</li>
                <li><strong>Shared Memory (SMEM)*:</strong> The red region* in the image contains shared memory. Note that not the entire red area is shared memory, as there are other on-die components as well.</li>
                <li><strong>Other Memory Types:</strong> Many accelerators also include additional faster memory types like L2 cache and registers, which are not the focus of this explanation but play important roles in performance.</li>
            </ul>

            <h3>Memory Access in Triton</h3>
            <p>Triton provides mechanisms to efficiently move data between these memory types across various accelerators:</p>
            <ul>
                <li><code>tl.load</code>: Generally used to load a specified portion of the global tensor into shared memory*. This operation can significantly speed up subsequent computations.</li>
                <li><code>tl.store</code>: Used to write processed data back to global memory.</li>
            </ul>

            <p><em>* It's important to note that Triton doesn't always transfer tensor slices to shared memory. This operation is performed only when it makes sense for performance optimization.</em></p>

            <h3>Key Points</h3>
            <ul>
                <li>Triton is designed for various AI accelerators, not just GPUs. Most AI accelerators have equivalent concepts to GMEM and SMEM.</li>
                <li>The RTX 4090 GPU shown in the image is just one example of an AI accelerator architecture.</li>
                <li>Shared memory (or its equivalent) is typically on-chip and much faster than global memory.</li>
                <li>Efficient use of shared memory can significantly boost kernel performance across different accelerators.</li>
                <li>Triton's <code>tl.load</code> and <code>tl.store</code> operations abstract the complexity of memory transfers, making code portable across different accelerator types.</li>
                <li>The decision to use shared memory is based on the specific requirements of the computation being performed and the characteristics of the target accelerator.</li>
            </ul>

            <img src="/static/images/accelerator-memory-hierarchy.png" alt="AI Accelerator Memory Hierarchy" style="width: 100%; max-width: 1200px; height: auto; margin-top: 30px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <p style="font-style: italic; margin-top: 10px;">Diagram showing the memory hierarchy of an RTX 4090 GPU as an example of AI accelerator architecture. Global memory (magenta) and on-chip memory including shared memory* (red) are highlighted. Note that this structure is similar across many AI accelerators.</p>
        `
    }
];

let currentPage = 0;

export function createInfoPopup() {
    const infoPopup = document.createElement('div');
    infoPopup.style.display = 'none';
    infoPopup.style.position = 'fixed';
    infoPopup.style.top = '5%';
    infoPopup.style.left = '5%';
    infoPopup.style.width = '85%';
    infoPopup.style.height = '75%';
    infoPopup.style.backgroundColor = '#1a1a1a';
    infoPopup.style.color = '#fff';
    infoPopup.style.padding = '30px';
    infoPopup.style.borderRadius = '15px';
    infoPopup.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.5)';
    infoPopup.style.zIndex = '1000';
    infoPopup.style.overflow = 'auto';

    const closeButton = createButton('×', () => { infoPopup.style.display = 'none'; });
    closeButton.style.position = 'absolute';
    closeButton.style.top = '20px';
    closeButton.style.right = '20px';
    closeButton.style.fontSize = '36px';
    closeButton.style.color = '#fff';

    const content = document.createElement('div');
    content.id = 'info-content';
    content.style.fontSize = '18px';
    content.style.lineHeight = '1.6';

    const navigation = document.createElement('div');
    navigation.style.display = 'flex';
    navigation.style.justifyContent = 'space-between';
    navigation.style.marginTop = '30px';

    const prevButton = createNavButton('&#8592; Previous', () => navigatePage(-1));
    const nextButton = createNavButton('Next &#8594;', () => navigatePage(1));

    navigation.appendChild(prevButton);
    navigation.appendChild(nextButton);

    infoPopup.appendChild(closeButton);
    infoPopup.appendChild(content);
    infoPopup.appendChild(navigation);

    document.body.appendChild(infoPopup);

    updateContent();

    return infoPopup;
}

function createButton(text, onClick) {
    const button = document.createElement('button');
    button.innerHTML = text;
    button.style.border = 'none';
    button.style.background = 'none';
    button.style.cursor = 'pointer';
    button.style.color = '#fff';
    button.onclick = onClick;
    return button;
}

function createNavButton(text, onClick) {
    const button = createButton(text, onClick);
    button.style.fontSize = '20px';
    button.style.padding = '15px 25px';
    button.style.backgroundColor = '#333';
    button.style.borderRadius = '8px';
    button.style.transition = 'background-color 0.3s';
    button.onmouseover = () => { button.style.backgroundColor = '#444'; };
    button.onmouseout = () => { button.style.backgroundColor = '#333'; };
    return button;
}

function navigatePage(direction) {
    currentPage += direction;
    if (currentPage < 0) currentPage = infoContent.length - 1;
    if (currentPage >= infoContent.length) currentPage = 0;
    updateContent();
}

function updateContent() {
    const contentDiv = document.getElementById('info-content');
    const pageContent = infoContent[currentPage];
    contentDiv.innerHTML = `
        <h1 style="color: #fff; font-size: 36px; margin-bottom: 20px;">${pageContent.title}</h1>
        ${pageContent.content}
    `;

    // Ensure all text in the content is white and styled appropriately
    contentDiv.querySelectorAll('p, li').forEach(element => {
        element.style.color = '#fff';
        element.style.marginBottom = '15px';
    });

    contentDiv.querySelectorAll('ul').forEach(element => {
        element.style.paddingLeft = '30px';
    });
}

export function showInfoPopup(infoPopup) {
    infoPopup.style.display = 'block';
}
