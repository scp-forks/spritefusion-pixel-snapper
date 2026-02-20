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

export function getOutputFilename(originalName: string): string {
  const dot = originalName.lastIndexOf('.');
  const base = dot > 0 ? originalName.substring(0, dot) : originalName;
  return `${base}-snapped.png`;
}
