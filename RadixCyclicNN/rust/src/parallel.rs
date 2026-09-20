//! The worker pool every fan-out goes through.
//!
//! The Go port spawns one goroutine per text, because a goroutine costs a few
//! kilobytes of stack and a scheduler entry.  An OS thread does not, so this
//! port fans out over a pool the size of the machine instead and hands each
//! thread a contiguous block of the work.  `workers = 1` runs everything on the
//! calling thread, which is what the single-threaded half of the benchmark
//! compares against Go's `--workers 1`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

/// The pool size for a `workers` setting: 0 means "the machine".
pub fn effective_workers(workers: usize) -> usize {
    if workers > 0 {
        return workers;
    }
    thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Runs `f(i)` for every `i` in `0..n` and waits.  Indices are taken off a
/// shared cursor, so an uneven text costs one thread rather than the pass.
pub fn parallel_for<F>(n: usize, workers: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    if n == 0 {
        return;
    }
    let pool = effective_workers(workers).min(n);
    if pool <= 1 {
        for i in 0..n {
            f(i);
        }
        return;
    }
    let next = AtomicUsize::new(0);
    let f = &f;
    let next = &next;
    thread::scope(|scope| {
        for _ in 0..pool {
            scope.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                f(i);
            });
        }
    });
}

/// Fills `out` in parallel: `f(i, &mut out[i])` for every entry, each thread
/// owning a contiguous block of the slice (so nothing is shared mutably).
pub fn parallel_fill<T, F>(out: &mut [T], workers: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut T) + Sync,
{
    let n = out.len();
    if n == 0 {
        return;
    }
    let pool = effective_workers(workers).min(n);
    if pool <= 1 {
        for (i, slot) in out.iter_mut().enumerate() {
            f(i, slot);
        }
        return;
    }
    let block = n.div_ceil(pool);
    let f = &f;
    thread::scope(|scope| {
        for (b, chunk) in out.chunks_mut(block).enumerate() {
            let base = b * block;
            scope.spawn(move || {
                for (i, slot) in chunk.iter_mut().enumerate() {
                    f(base + i, slot);
                }
            });
        }
    });
}

/// Splits `0..n` into contiguous ranges, one per worker, and calls
/// `f(lo, hi)` for each - the shape the weight and cost recomputes want.
pub fn parallel_ranges<F>(n: usize, workers: usize, f: F)
where
    F: Fn(usize, usize) + Sync,
{
    if n == 0 {
        return;
    }
    let pool = effective_workers(workers);
    if pool <= 1 {
        f(0, n);
        return;
    }
    let block = n.div_ceil(pool).max(256);
    let f = &f;
    thread::scope(|scope| {
        let mut lo = 0;
        while lo < n {
            let hi = (lo + block).min(n);
            scope.spawn(move || f(lo, hi));
            lo = hi;
        }
    });
}

/// A raw view of a slice that scoped threads may share **only** when each
/// thread writes indices no other thread touches.
///
/// The two recomputes need it: they fan out over the nodes, and every edge
/// belongs to exactly one node (`edge_parent[e] == p` for every edge listed
/// under `p`, an invariant `Graph::check_invariants` proves), so two threads
/// working on different nodes can never write the same edge.  Rust cannot see
/// that from the types, and the alternative - an atomic per edge, or a
/// per-thread buffer merged afterwards - would pay for it on every read the
/// search makes.
///
/// # Safety
/// The caller must guarantee that no two threads write the same index, that no
/// index is written while another thread reads it, and that the slice outlives
/// every thread that holds the view.
pub(crate) struct Disjoint<'a, T> {
    ptr: *mut T,
    len: usize,
    _life: std::marker::PhantomData<&'a mut [T]>,
}

// SAFETY: the type carries no way to read or write without an `unsafe` block,
// and the contract above puts the burden of disjointness on the caller.
unsafe impl<T: Send> Send for Disjoint<'_, T> {}
unsafe impl<T: Send> Sync for Disjoint<'_, T> {}

impl<'a, T> Disjoint<'a, T> {
    pub(crate) fn new(slice: &'a mut [T]) -> Disjoint<'a, T> {
        Disjoint {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _life: std::marker::PhantomData,
        }
    }

    /// Writes one index.
    ///
    /// # Safety
    /// No other thread may touch `i` at the same time.
    #[inline]
    pub(crate) unsafe fn set(&self, i: usize, value: T) {
        debug_assert!(i < self.len);
        std::ptr::write(self.ptr.add(i), value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicI64;

    #[test]
    fn every_index_is_handled_exactly_once() {
        for workers in [1, 2, 3, 8] {
            let seen: Vec<AtomicI64> = (0..1000).map(|_| AtomicI64::new(0)).collect();
            parallel_for(seen.len(), workers, |i| {
                seen[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(seen.iter().all(|c| c.load(Ordering::Relaxed) == 1));

            let mut out = vec![0usize; 1000];
            parallel_fill(&mut out, workers, |i, slot| *slot = i * 2);
            assert!(out.iter().enumerate().all(|(i, &v)| v == i * 2));

            let covered: Vec<AtomicI64> = (0..1000).map(|_| AtomicI64::new(0)).collect();
            parallel_ranges(covered.len(), workers, |lo, hi| {
                for c in &covered[lo..hi] {
                    c.fetch_add(1, Ordering::Relaxed);
                }
            });
            assert!(covered.iter().all(|c| c.load(Ordering::Relaxed) == 1));
        }
    }
}
