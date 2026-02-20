import { showToast } from '../main';

export interface ToolbarState {
  getProcessedBytes: () => Uint8Array | null;
  getFilename: () => string;
}

export function createToolbar(
  container: HTMLElement,
  state: ToolbarState,
) {
  container.innerHTML = `
    <button class="toolbar-btn toolbar-copy" type="button">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
      </svg>
      Copy
    </button>
    <button class="toolbar-btn toolbar-download" type="button">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="7 10 12 15 17 10"/>
        <line x1="12" y1="15" x2="12" y2="3"/>
      </svg>
      Download
    </button>
  `;

  const copyBtn = container.querySelector<HTMLButtonElement>('.toolbar-copy')!;
  const downloadBtn = container.querySelector<HTMLButtonElement>('.toolbar-download')!;

  copyBtn.addEventListener('click', async () => {
    const bytes = state.getProcessedBytes();
    if (!bytes) return;
    try {
      const blob = new Blob([bytes as BlobPart], { type: 'image/png' });
      await navigator.clipboard.write([
        new ClipboardItem({ 'image/png': blob }),
      ]);
      showToast('Copied to clipboard');
    } catch {
      showToast('Failed to copy');
    }
  });

  downloadBtn.addEventListener('click', () => {
    const bytes = state.getProcessedBytes();
    if (!bytes) return;
    const blob = new Blob([bytes as BlobPart], { type: 'image/png' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = state.getFilename();
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
}
