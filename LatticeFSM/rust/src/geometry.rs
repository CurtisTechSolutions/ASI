//! The matrix as a cube: its central node, its shells, and the order that visits it from the middle outward.
//!
//! A matrix of `S` states over `A` symbols is an `S × A × S` block of cells. Its **central node** is the cell
//! `(S / 2, A / 2, S / 2)` - `(6, g, 6)` on the default 13 × 13 × 13 machine. A cell's **shell** is its Chebyshev
//! distance from the centre: shell 0 is the central node, shell 1 the 26 cells around it, out to shell 6, the 866
//! cells of the surface. [`center_out`] lists every cell from the middle outward, shell by shell and in
//! `(s, a, t)` order within a shell: the order the compressed code is laid out and rebuilt in (`compress.rs`).
//! The same as `../latticefsm/geometry.py`.

/// The central node of an `(S, A, S)` matrix.
pub fn center(shape: (usize, usize, usize)) -> (usize, usize, usize) {
    (shape.0 / 2, shape.1 / 2, shape.2 / 2)
}

/// How many shells out from the central node the cell `(s, a, t)` lies.
pub fn shell(shape: (usize, usize, usize), s: usize, a: usize, t: usize) -> usize {
    let (cs, ca, ct) = center(shape);
    s.abs_diff(cs).max(a.abs_diff(ca)).max(t.abs_diff(ct))
}

/// How many shells the matrix has, the central node's included: 7 for 13 × 13 × 13.
pub fn shells(shape: (usize, usize, usize)) -> usize {
    let (cs, ca, ct) = center(shape);
    1 + cs
        .max(shape.0 - 1 - cs)
        .max(ca)
        .max(shape.1 - 1 - ca)
        .max(ct)
        .max(shape.2 - 1 - ct)
}

/// Every cell's matrix offset (`(s · A + a) · S + t`), from the central node outward.
pub fn center_out(shape: (usize, usize, usize)) -> Vec<usize> {
    let (s_n, a_n, t_n) = shape;
    let mut cells = Vec::with_capacity(s_n * a_n * t_n);
    for s in 0..s_n {
        for a in 0..a_n {
            for t in 0..t_n {
                cells.push((shell(shape, s, a, t), s, a, t));
            }
        }
    }
    cells.sort();
    cells.into_iter().map(|(_, s, a, t)| (s * a_n + a) * t_n + t).collect()
}

/// How many cells each shell holds, from the centre out: `[1, 26, 98, 218, 386, 602, 866]` for 13 × 13 × 13.
pub fn shell_sizes(shape: (usize, usize, usize)) -> Vec<usize> {
    let mut sizes = vec![0; shells(shape)];
    for s in 0..shape.0 {
        for a in 0..shape.1 {
            for t in 0..shape.2 {
                sizes[shell(shape, s, a, t)] += 1;
            }
        }
    }
    sizes
}
