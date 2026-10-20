//! Canvas RGBA → 28x28 MNIST-style input. Integer arithmetic throughout.
//!
//! 1. Grayscale: `g = (299R + 587G + 114B + 500) / 1000`. Alpha is ignored;
//!    expects light ink on a dark background (as MNIST).
//! 2. Bounding box of pixels with `g > INK_THRESHOLD`, expanded to a square.
//! 3. Box-filter the square down to 20x20. Cell `(ox, oy)` averages source
//!    `[x0, x1) × [y0, y1)` with `x0 = left + ox·side/20`, rounding half up.
//!    Pixels outside the canvas count as 0.
//! 4. Paste into 28x28 so the center of mass lands on (14, 14), clipped to
//!    the frame.
//!
//! A blank canvas returns an all-zero image.

use crate::image::{Image, IMAGE_LEN, IMAGE_SIDE};

pub const INK_THRESHOLD: u8 = 32;
pub const BOX_SIDE: usize = 20;

pub fn grayscale(rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
    assert_eq!(
        rgba.len(),
        width * height * 4,
        "rgba length must be width*height*4"
    );
    rgba.chunks_exact(4)
        .map(|p| ((299 * p[0] as u32 + 587 * p[1] as u32 + 114 * p[2] as u32 + 500) / 1000) as u8)
        .collect()
}

pub fn preprocess(rgba: &[u8], width: u32, height: u32) -> Image {
    let (w, h) = (width as usize, height as usize);
    let gray = grayscale(rgba, w, h);

    let (mut x_min, mut y_min, mut x_max, mut y_max) = (w, h, 0, 0);
    for y in 0..h {
        for x in 0..w {
            if gray[y * w + x] > INK_THRESHOLD {
                x_min = x_min.min(x);
                y_min = y_min.min(y);
                x_max = x_max.max(x);
                y_max = y_max.max(y);
            }
        }
    }
    if x_min > x_max {
        return Image::new(vec![0; IMAGE_LEN]);
    }

    // Square box centered on the ink, at least BOX_SIDE so every cell has ≥1 pixel.
    let bw = x_max - x_min + 1;
    let bh = y_max - y_min + 1;
    let side = bw.max(bh).max(BOX_SIDE) as i64;
    let left = (x_min + x_max + 1) as i64 / 2 - side / 2;
    let top = (y_min + y_max + 1) as i64 / 2 - side / 2;

    let sample = |x: i64, y: i64| -> u64 {
        if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
            0
        } else {
            gray[y as usize * w + x as usize] as u64
        }
    };

    let n = BOX_SIDE as i64;
    let mut small = [0u8; BOX_SIDE * BOX_SIDE];
    for oy in 0..n {
        let y0 = top + oy * side / n;
        let y1 = top + (oy + 1) * side / n;
        for ox in 0..n {
            let x0 = left + ox * side / n;
            let x1 = left + (ox + 1) * side / n;
            let mut sum = 0u64;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += sample(x, y);
                }
            }
            let count = ((x1 - x0) * (y1 - y0)) as u64;
            small[(oy * n + ox) as usize] = ((sum + count / 2) / count) as u8;
        }
    }

    // Center of mass of the 20x20 block, rounded to the nearest pixel.
    let (mut mass, mut mx, mut my) = (0u64, 0u64, 0u64);
    for y in 0..BOX_SIDE {
        for x in 0..BOX_SIDE {
            let v = small[y * BOX_SIDE + x] as u64;
            mass += v;
            mx += v * x as u64;
            my += v * y as u64;
        }
    }
    let half = IMAGE_SIDE as i64 / 2;
    let (cx, cy) = if mass == 0 {
        (BOX_SIDE as i64 / 2, BOX_SIDE as i64 / 2)
    } else {
        (
            ((mx + mass / 2) / mass) as i64,
            ((my + mass / 2) / mass) as i64,
        )
    };
    let (dx, dy) = (half - cx, half - cy);

    let mut out = vec![0u8; IMAGE_LEN];
    for y in 0..BOX_SIDE as i64 {
        for x in 0..BOX_SIDE as i64 {
            let (tx, ty) = (x + dx, y + dy);
            if (0..IMAGE_SIDE as i64).contains(&tx) && (0..IMAGE_SIDE as i64).contains(&ty) {
                out[ty as usize * IMAGE_SIDE + tx as usize] = small[(y * n + x) as usize];
            }
        }
    }
    Image::new(out)
}
