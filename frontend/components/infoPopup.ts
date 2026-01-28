type InfoPage = {
    title: string;
    content: string;
};

const infoPages: InfoPage[] = [
    {
        title: 'exploring programs',
        content: `
            <p>use the program id sliders to filter the 3d grid before launching a detail view.</p>
            <p>click any block to open the tensor visualization pane that shows loads, stores, and dot products.</p>
        `
    },
    {
        title: 'tensor view tour',
        content: `
            <p>the detailed canvas renders the global tensor beside its current slice so you can spot patterns.</p>
            <p>drag, hover, and select cubes to see the sampled values and use the control buttons for histograms or coloring.</p>
        `
    },
    {
        title: 'useful toggles',
        content: `
            <p>color by value and the histogram buttons are enabled once you open a block.</p>
            <p>refer to the code view panel for the source around the selected operation.</p>
        `
    }
];

let currentPage = 0;

function createNavButton(label: string, onClick: () => void): HTMLButtonElement {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.innerHTML = label;
    btn.style.border = 'none';
    btn.style.background = '#222';
    btn.style.color = '#fff';
    btn.style.padding = '10px 16px';
    btn.style.borderRadius = '6px';
    btn.style.cursor = 'pointer';
    btn.addEventListener('click', onClick);
    return btn;
}

function updatePage(popup: HTMLElement): void {
    const region = popup.querySelector('[data-info-content]') as HTMLElement | null;
    if (!region) return;
    const page = infoPages[currentPage] || infoPages[0];
    if (!page) return;
    region.innerHTML = `
        <h2 style="margin-top:0;color:#f9fafb;">${page.title}</h2>
        ${page.content}
    `;
}

export function createInfoPopup(): HTMLDivElement {
    const popup = document.createElement('div');
    popup.setAttribute('data-info-popup', 'true');
    Object.assign(popup.style, {
        position: 'fixed',
        top: '6%',
        left: '6%',
        right: '6%',
        bottom: '6%',
        background: 'rgba(5,5,10,0.95)',
        color: '#f1f5f9',
        padding: '20px',
        borderRadius: '14px',
        boxShadow: '0 12px 30px rgba(0,0,0,0.6)',
        overflowY: 'auto',
        zIndex: '1200',
        display: 'none',
        fontFamily: 'Inter, system-ui, sans-serif'
    });
    const close = document.createElement('button');
    close.type = 'button';
    close.textContent = '×';
    Object.assign(close.style, {
        position: 'absolute',
        top: '14px',
        right: '18px',
        background: 'none',
        border: 'none',
        color: '#f8fafc',
        fontSize: '26px',
        cursor: 'pointer'
    });
    close.addEventListener('click', () => {
        popup.style.display = 'none';
    });
    const content = document.createElement('div');
    content.setAttribute('data-info-content', 'true');
    const navigation = document.createElement('div');
    navigation.style.display = 'flex';
    navigation.style.justifyContent = 'space-between';
    navigation.style.marginTop = '28px';
    const prev = createNavButton('← previous', () => {
        currentPage = (currentPage - 1 + infoPages.length) % infoPages.length;
        updatePage(popup);
    });
    const next = createNavButton('next →', () => {
        currentPage = (currentPage + 1) % infoPages.length;
        updatePage(popup);
    });
    navigation.appendChild(prev);
    navigation.appendChild(next);
    popup.appendChild(close);
    popup.appendChild(content);
    popup.appendChild(navigation);
    document.body.appendChild(popup);
    currentPage = 0;
    updatePage(popup);
    return popup;
}

export function showInfoPopup(popup: HTMLElement | null): void {
    if (!popup) return;
    popup.style.display = 'block';
    popup.scrollTop = 0;
}
