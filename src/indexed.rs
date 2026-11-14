//! A quantized image as a palette plus one label per pixel.
//!
//! The six quantizers return finished `RgbaImage`s. Everything built on top of them
//! (cell voting, un-mixing, the sieve) needs labels instead, and those are recovered
//! here from the quantizer's output rather than by rewriting the quantizers.


use crate::{PixelSnapperError, Result};
use image::{Rgba, RgbaImage};
use std::collections::HashMap;

/// Label of a pixel that takes no part in any stage: its source alpha is 0.
pub const TRANSPARENT: u16 = u16::MAX;

pub struct IndexedImage<'a> {
    /// The original image: alpha, the hidden RGB of transparent pixels, and the source
    /// colours that cleanup and scoring measure against.
    pub source: &'a RgbaImage,
    /// Palette in order of first appearance in raster order. Empty when nothing is opaque.
    pub palette: Vec<[u8; 3]>,
    /// Palette index per pixel (row-major), or [`TRANSPARENT`].
    pub labels: Vec<u16>,
}

impl IndexedImage<'_> {
    pub fn width(&self) -> usize {
        self.source.width() as usize
    }

    #[allow(dead_code)] // first non-test caller arrives with the Smooth cleanup
    pub fn height(&self) -> usize {
        self.source.height() as usize
    }
}

/// Recover the label map of `quantized`, which must be `source` after palette
/// reduction: same size, same alpha. Pixels whose **source** alpha is 0 are transparent.
pub fn index_image<'a>(source: &'a RgbaImage, quantized: &RgbaImage) -> Result<IndexedImage<'a>> {
    if source.dimensions() != quantized.dimensions() {
        return Err(PixelSnapperError::ProcessingError(format!(
            "Cannot index a {}x{} image against a {}x{} source",
            quantized.width(),
            quantized.height(),
            source.width(),
            source.height()
        )));
    }

    let mut lookup: HashMap<[u8; 3], u16> = HashMap::new();
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut labels = Vec::with_capacity(source.as_raw().len() / 4);

    for (src, q) in source.pixels().zip(quantized.pixels()) {
        if src[3] == 0 {
            labels.push(TRANSPARENT);
            continue;
        }
        let rgb = [q[0], q[1], q[2]];
        let label = match lookup.get(&rgb) {
            Some(&l) => l,
            None => {
                // u16::MAX is reserved for TRANSPARENT.
                if palette.len() >= TRANSPARENT as usize {
                    return Err(PixelSnapperError::ProcessingError(format!(
                        "Image has more than {} colours after palette reduction",
                        TRANSPARENT
                    )));
                }
                let l = palette.len() as u16;
                palette.push(rgb);
                lookup.insert(rgb, l);
                l
            }
        };
        labels.push(label);
    }

    Ok(IndexedImage { source, palette, labels })
}

/// Paint the label map back. Opaque pixels get their palette colour and keep their
/// source alpha; transparent pixels are copied from the source, hidden RGB included.
pub fn render(ix: &IndexedImage) -> RgbaImage {
    let mut out = RgbaImage::new(ix.source.width(), ix.source.height());
    for ((dst, src), &label) in out.pixels_mut().zip(ix.source.pixels()).zip(&ix.labels) {
        *dst = if label == TRANSPARENT {
            *src
        } else {
            let c = ix.palette[label as usize];
            Rgba([c[0], c[1], c[2], src[3]])
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32, px: &[[u8; 4]]) -> RgbaImage {
        assert_eq!(px.len() as u32, w * h);
        let mut out = RgbaImage::new(w, h);
        for (p, v) in out.pixels_mut().zip(px) {
            *p = Rgba(*v);
        }
        out
    }

    #[test]
    fn round_trip_keeps_hidden_rgb_and_partial_alpha() {
        let source = img(3, 2, &[
            [10, 10, 10, 255], [200, 0, 0, 128], [9, 8, 7, 0],
            [12, 11, 10, 255], [1, 2, 3, 0], [190, 5, 5, 255],
        ]);
        // What a quantizer would return: two colours, alpha and transparent pixels untouched.
        let quantized = img(3, 2, &[
            [11, 10, 10, 255], [195, 2, 2, 128], [9, 8, 7, 0],
            [11, 10, 10, 255], [1, 2, 3, 0], [195, 2, 2, 255],
        ]);
        let ix = index_image(&source, &quantized).unwrap();
        assert_eq!(ix.palette, vec![[11, 10, 10], [195, 2, 2]]);
        assert_eq!(ix.labels, vec![0, 1, TRANSPARENT, 0, TRANSPARENT, 1]);
        assert_eq!((ix.width(), ix.height()), (3, 2));
        assert_eq!(render(&ix), quantized);
    }

    #[test]
    fn palette_order_is_first_appearance() {
        let source = img(4, 1, &[[0, 0, 0, 255]; 4]);
        let quantized = img(4, 1, &[[5, 5, 5, 255], [1, 1, 1, 255], [5, 5, 5, 255], [3, 3, 3, 255]]);
        let ix = index_image(&source, &quantized).unwrap();
        assert_eq!(ix.palette, vec![[5, 5, 5], [1, 1, 1], [3, 3, 3]]);
    }

    #[test]
    fn all_transparent_image_has_an_empty_palette_and_renders_unchanged() {
        let source = img(2, 2, &[[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 0]]);
        let ix = index_image(&source, &source).unwrap();
        assert!(ix.palette.is_empty());
        assert!(ix.labels.iter().all(|&l| l == TRANSPARENT));
        assert_eq!(render(&ix), source);
    }

    #[test]
    fn size_mismatch_is_an_error() {
        let a = img(2, 1, &[[0, 0, 0, 255]; 2]);
        let b = img(1, 2, &[[0, 0, 0, 255]; 2]);
        assert!(index_image(&a, &b).is_err());
    }

    #[test]
    fn too_many_colours_is_an_error_not_a_wrapped_label() {
        // 65 536 distinct colours: one more than the label type can address.
        let (w, h) = (256u32, 256u32);
        let mut source = RgbaImage::new(w, h);
        for (x, y, p) in source.enumerate_pixels_mut() {
            *p = Rgba([x as u8, y as u8, 0, 255]);
        }
        assert!(index_image(&source, &source).is_err());

        // One fewer fits.
        source.put_pixel(255, 255, Rgba([0, 0, 0, 255]));
        let ix = index_image(&source, &source).unwrap();
        assert_eq!(ix.palette.len(), 65_535);
        assert_eq!(render(&ix), source);
    }
}
