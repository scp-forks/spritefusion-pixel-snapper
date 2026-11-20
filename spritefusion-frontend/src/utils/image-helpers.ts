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

export interface ImageAnalysis {
  /** Distinct RGB colours over every pixel. */
  uniqueColors: number;
  /** Pixels neither fully opaque nor fully transparent. */
  partialAlphaPixels: number;
}

export function analyzeImage(url: string): Promise<ImageAnalysis> {
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
      let partialAlphaPixels = 0;
      for (let i = 0; i < data.length; i += 4) {
        colors.add((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]);
        const a = data[i + 3];
        if (a !== 0 && a !== 255) partialAlphaPixels++;
      }
      resolve({ uniqueColors: colors.size, partialAlphaPixels });
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
