//! 4-connected components of a label map, and the speckle statistics built on them.

use image::RgbaImage;
use std::collections::HashMap;

/// Label value that marks a pixel as "not part of any region" (fully transparent).
pub const NO_LABEL: u32 = u32::MAX;

/// Connected components of `labels` (row-major, `w * h` entries) under 4-connectivity.
/// Pixels labelled [`NO_LABEL`] belong to no component.
pub struct Components {
    /// Component id per pixel, or [`NO_LABEL`]. (Read by the sieve from Phase 3 on.)
    #[allow(dead_code)]
    pub id: Vec<u32>,
    /// Pixel count per component id.
    pub area: Vec<u32>,
}

pub fn connected_components(labels: &[u32], w: usize, h: usize) -> Components {
    assert_eq!(labels.len(), w * h, "label map does not match its dimensions");

    // Union-find over pixel indices, path-halving.
    let mut parent: Vec<u32> = (0..labels.len() as u32).collect();
    fn find(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            let grand = parent[parent[i as usize] as usize];
            parent[i as usize] = grand;
            i = grand;
        }
        i
    }

    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let l = labels[i];
            if l == NO_LABEL {
                continue;
            }
            if x > 0 && labels[i - 1] == l {
                let (a, b) = (find(&mut parent, i as u32), find(&mut parent, (i - 1) as u32));
                if a != b {
                    parent[a.max(b) as usize] = a.min(b);
                }
            }
            if y > 0 && labels[i - w] == l {
                let (a, b) = (find(&mut parent, i as u32), find(&mut parent, (i - w) as u32));
                if a != b {
                    parent[a.max(b) as usize] = a.min(b);
                }
            }
        }
    }

    // Roots are always the smallest pixel index of their component, so numbering
    // components in raster order of their roots is deterministic.
    let mut id = vec![NO_LABEL; labels.len()];
    let mut area: Vec<u32> = Vec::new();
    for i in 0..labels.len() {
        if labels[i] == NO_LABEL {
            continue;
        }
        let root = find(&mut parent, i as u32) as usize;
        if id[root] == NO_LABEL {
            id[root] = area.len() as u32;
            area.push(0);
        }
        id[i] = id[root];
        area[id[i] as usize] += 1;
    }

    Components { id, area }
}

/// Distinct RGB values among pixels with `alpha > 0`.
///
/// Results of this tool have at most a few dozen colours and long runs of each, so a
/// short list checked last-hit-first beats hashing every pixel; images with thousands
/// of colours fall back to a set.
pub fn count_unique_colors(img: &RgbaImage) -> usize {
    const LIST_LIMIT: usize = 256;
    let mut list: Vec<[u8; 3]> = Vec::new();
    let mut last: Option<[u8; 3]> = None;
    let mut pixels = img.pixels();
    for p in pixels.by_ref() {
        if p[3] == 0 {
            continue;
        }
        let rgb = [p[0], p[1], p[2]];
        if last == Some(rgb) {
            continue;
        }
        last = Some(rgb);
        if !list.contains(&rgb) {
            list.push(rgb);
            if list.len() > LIST_LIMIT {
                break;
            }
        }
    }
    if list.len() <= LIST_LIMIT {
        return list.len();
    }
    let mut set: std::collections::HashSet<[u8; 3]> = list.into_iter().collect();
    for p in pixels {
        if p[3] != 0 {
            set.insert([p[0], p[1], p[2]]);
        }
    }
    set.len()
}

/// Speckle statistics of a finished image.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageStats {
    pub width: u32,
    pub height: u32,
    /// Distinct RGB values among pixels with `alpha > 0`.
    pub unique_colors: usize,
    /// 4-connected same-colour regions (alpha ignored, fully transparent pixels excluded).
    pub regions: usize,
    /// Regions smaller than 4 px.
    pub regions_lt4: usize,
    /// Regions smaller than 16 px.
    pub regions_lt16: usize,
}

pub fn image_stats(img: &RgbaImage) -> ImageStats {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut index: HashMap<[u8; 3], u32> = HashMap::new();
    let labels: Vec<u32> = img
        .pixels()
        .map(|p| {
            if p[3] == 0 {
                NO_LABEL
            } else {
                let next = index.len() as u32;
                *index.entry([p[0], p[1], p[2]]).or_insert(next)
            }
        })
        .collect();

    let comps = connected_components(&labels, w, h);
    ImageStats {
        width: img.width(),
        height: img.height(),
        unique_colors: index.len(),
        regions: comps.area.len(),
        regions_lt4: comps.area.iter().filter(|&&a| a < 4).count(),
        regions_lt16: comps.area.iter().filter(|&&a| a < 16).count(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn components_are_four_connected() {
        // 1 . 1
        // . 1 .      diagonal neighbours must not join
        // 1 . 1
        let labels = [1, 0, 1, 0, 1, 0, 1, 0, 1];
        let c = connected_components(&labels, 3, 3);
        assert_eq!(c.area.len(), 9);
        assert!(c.area.iter().all(|&a| a == 1));
    }

    #[test]
    fn components_merge_across_a_u_shape() {
        // 1 0 1
        // 1 0 1      the two arms only meet on the last row
        // 1 1 1
        let labels = [1, 0, 1, 1, 0, 1, 1, 1, 1];
        let c = connected_components(&labels, 3, 3);
        assert_eq!(c.area.len(), 2);
        assert_eq!(c.id[0], c.id[2]);
        assert_eq!(c.area[c.id[0] as usize], 7);
        assert_eq!(c.area[c.id[1] as usize], 2);
    }

    #[test]
    fn unlabelled_pixels_split_regions_and_are_not_counted() {
        let labels = [5, NO_LABEL, 5];
        let c = connected_components(&labels, 3, 1);
        assert_eq!(c.area, vec![1, 1]);
        assert_eq!(c.id[1], NO_LABEL);
    }

    #[test]
    fn unique_colours_are_counted_the_same_on_both_paths() {
        // Few colours: the list path. Transparent pixels and their hidden RGB do not count.
        let mut img = RgbaImage::from_pixel(8, 8, Rgba([1, 2, 3, 255]));
        img.put_pixel(0, 0, Rgba([9, 9, 9, 255]));
        img.put_pixel(7, 7, Rgba([9, 9, 9, 128]));
        img.put_pixel(3, 3, Rgba([50, 60, 70, 0]));
        assert_eq!(count_unique_colors(&img), 2);
        assert_eq!(count_unique_colors(&RgbaImage::from_pixel(4, 4, Rgba([5, 5, 5, 0]))), 0);

        // Many colours: the set path, crossing the list limit mid-image, with repeats.
        let mut big = RgbaImage::new(64, 64);
        for (x, y, p) in big.enumerate_pixels_mut() {
            *p = Rgba([(x * 4) as u8, (y % 20) as u8, 0, 255]);
        }
        assert_eq!(count_unique_colors(&big), 64 * 20);
        assert_eq!(image_stats(&big).unique_colors, 64 * 20);
    }

    #[test]
    fn stats_ignore_transparent_pixels() {
        let mut img = RgbaImage::from_pixel(4, 4, Rgba([10, 20, 30, 255]));
        img.put_pixel(1, 1, Rgba([200, 0, 0, 255])); // one speck
        img.put_pixel(3, 3, Rgba([1, 2, 3, 0])); // transparent, hidden RGB differs
        let s = image_stats(&img);
        assert_eq!((s.width, s.height), (4, 4));
        assert_eq!(s.unique_colors, 2);
        assert_eq!(s.regions, 2);
        assert_eq!(s.regions_lt4, 1);
        assert_eq!(s.regions_lt16, 2);
    }
}
