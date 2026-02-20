export function bytesToObjectUrl(bytes: Uint8Array): string {
  const blob = new Blob([bytes as BlobPart], { type: 'image/png' });
  return URL.createObjectURL(blob);
}

export function revokeObjectUrl(url: string | null): void {
  if (url) URL.revokeObjectURL(url);
}

export function fileToUint8Array(file: File): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(new Uint8Array(reader.result as ArrayBuffer));
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(file);
  });
}

export function getImageDimensions(
  url: string,
): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => reject(new Error('Failed to load image'));
    img.src = url;
  });
}

export function countUniqueColors(url: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      const colors = new Set<number>();
      for (let i = 0; i < data.length; i += 4) {
        colors.add((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]);
      }
      resolve(colors.size);
    };
    img.onerror = () => reject(new Error('Failed to load image'));
    img.src = url;
  });
}

export function getOutputFilename(originalName: string): string {
  const dot = originalName.lastIndexOf('.');
  const base = dot > 0 ? originalName.substring(0, dot) : originalName;
  return `${base}-snapped.png`;
}
