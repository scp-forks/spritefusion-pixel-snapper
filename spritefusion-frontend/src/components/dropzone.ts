const ACCEPTED_TYPES = ['image/png', 'image/jpeg'];
const MAX_SIZE = 10 * 1024 * 1024; // 10MB

export interface DropzoneCallbacks {
  onFile: (file: File) => void;
}

export function createDropzone(
  container: HTMLElement,
  callbacks: DropzoneCallbacks,
): { collapse: () => void; expand: () => void } {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/png,image/jpeg';
  input.classList.add('dropzone-input');

  const label = document.createElement('div');
  label.classList.add('dropzone-label');
  label.innerHTML = `
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
    <span class="dropzone-text">Drop a PNG or JPEG here, or click to browse</span>
    <span class="dropzone-hint">Max 10 MB</span>
  `;

  const changeStrip = document.createElement('div');
  changeStrip.classList.add('dropzone-strip', 'hidden');
  changeStrip.textContent = 'Change image';

  container.classList.add('dropzone');
  container.appendChild(input);
  container.appendChild(label);
  container.appendChild(changeStrip);

  function validate(file: File): string | null {
    if (!ACCEPTED_TYPES.includes(file.type)) return 'Only PNG and JPEG files are supported.';
    if (file.size > MAX_SIZE) return 'File must be under 10 MB.';
    return null;
  }

  function handleFile(file: File) {
    const err = validate(file);
    if (err) {
      container.classList.add('dropzone-error');
      label.querySelector('.dropzone-text')!.textContent = err;
      setTimeout(() => {
        container.classList.remove('dropzone-error');
        label.querySelector('.dropzone-text')!.textContent =
          'Drop a PNG or JPEG here, or click to browse';
      }, 2000);
      return;
    }
    callbacks.onFile(file);
  }

  container.addEventListener('dragover', (e) => {
    e.preventDefault();
    container.classList.add('dropzone-active');
  });

  container.addEventListener('dragleave', () => {
    container.classList.remove('dropzone-active');
  });

  container.addEventListener('drop', (e) => {
    e.preventDefault();
    container.classList.remove('dropzone-active');
    const file = e.dataTransfer?.files[0];
    if (file) handleFile(file);
  });

  container.addEventListener('click', (e) => {
    if ((e.target as HTMLElement).closest('.dropzone-strip')) {
      input.click();
      return;
    }
    if (!container.classList.contains('dropzone-collapsed')) {
      input.click();
    }
  });

  input.addEventListener('change', () => {
    const file = input.files?.[0];
    if (file) handleFile(file);
    input.value = '';
  });

  return {
    collapse() {
      container.classList.add('dropzone-collapsed');
      label.classList.add('hidden');
      changeStrip.classList.remove('hidden');
    },
    expand() {
      container.classList.remove('dropzone-collapsed');
      label.classList.remove('hidden');
      changeStrip.classList.add('hidden');
    },
  };
}
