//! The worker pool every fan-out goes through.
//!
//! The Go port spawns one goroutine per text, because a goroutine costs a few
//! kilobytes of stack and a scheduler entry.  An OS thread does not, so this
//! port uses up to the available CPU count, including the calling thread,
//! and distributes work dynamically.  `workers = 1` runs everything on the
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
    if pool <= 1 || ACTIVE.with(|active| active.get()) {
        for i in 0..n {
            f(i);
        }
        return;
    }
    let next = AtomicUsize::new(0);
    let f = &f;
    let next = &next;
    let work = || {
        let _active = Active::enter();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break;
            }
            f(i);
        }
    };
    thread::scope(|scope| {
        for _ in 1..pool {
            scope.spawn(work);
        }
        work();
    });
}

/// Fills `out` in order, with disjoint chunks claimed dynamically. Workers
/// finish a chunk before taking another, so long texts do not strand a core.
/// The caller participates and nested passes stay on their current worker.
pub fn parallel_fill<T, F>(out: &mut [T], workers: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut T) + Sync,
{
    let pool = effective_workers(workers).min(out.len());
    if pool <= 1 || ACTIVE.with(|active| active.get()) {
        for (i, slot) in out.iter_mut().enumerate() {
            f(i, slot);
        }
        return;
    }
    let block = out.len().div_ceil(pool.saturating_mul(8));
    let queue = std::sync::Mutex::new(out.chunks_mut(block).enumerate());
    let work = || {
        let _active = Active::enter();
        loop {
            // Move a disjoint mutable slice out of the queue, releasing the
            // lock before the expensive work. No unsafe sharing is needed.
            let next = queue.lock().unwrap_or_else(|e| e.into_inner()).next();
            let Some((b, chunk)) = next else { break };
            for (i, slot) in chunk.iter_mut().enumerate() {
                f(b * block + i, slot);
            }
        }
    };
    thread::scope(|scope| {
        for _ in 1..pool {
            scope.spawn(work);
        }
        work();
    });
}

/// Ordered map: completion order never changes the order of returned values.
pub fn parallel_map<T: Sync, R: Send, F: Fn(&T) -> R + Sync>(input: &[T], workers: usize, f: F) -> Vec<R> {
    let mut out: Vec<Option<R>> = (0..input.len()).map(|_| None).collect();
    parallel_fill(&mut out, workers, |i, slot| *slot = Some(f(&input[i])));
    out.into_iter().map(Option::unwrap).collect()
}

thread_local! {
    static ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct Active(bool);
impl Active {
    fn enter() -> Self {
        Self(ACTIVE.with(|active| active.replace(true)))
    }
}
impl Drop for Active {
    fn drop(&mut self) {
        ACTIVE.with(|active| active.set(self.0));
    }
}

/// Disjoint ranges claimed dynamically; all workers may participate even
/// when the graph has fewer than 256 nodes per available core.
pub fn parallel_ranges<F>(n: usize, workers: usize, f: F)
where
    F: Fn(usize, usize) + Sync,
{
    if n == 0 {
        return;
    }
    let pool = effective_workers(workers).min(n);
    if pool <= 1 || ACTIVE.with(|active| active.get()) {
        f(0, n);
        return;
    }
    let block = n.div_ceil(pool.saturating_mul(8));
    parallel_for(n.div_ceil(block), pool, |i| {
        let lo = i * block;
        f(lo, (lo + block).min(n));
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

#[cfg(test)]
mod scheduling_tests {
    use super::*;
    use std::sync::{Barrier, Mutex};

    #[test]
    fn empty_and_uneven_work_keeps_input_order() {
        for workers in [1, 2, 7, 32] {
            for n in [0, 1, 3, 259] {
                let input: Vec<usize> = (0..n).collect();
                let got = parallel_map(&input, workers, |&i| {
                    for _ in 0..i % 11 {
                        std::thread::yield_now();
                    }
                    i * i
                });
                assert_eq!(got, input.iter().map(|i| i * i).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn all_requested_workers_can_run_and_nested_work_stays_with_them() {
        let gate = Barrier::new(4);
        let threads = Mutex::new(std::collections::HashSet::new());
        let mut out = [0; 4];
        parallel_fill(&mut out, 4, |i, value| {
            let id = thread::current().id();
            threads.lock().unwrap().insert(id);
            gate.wait(); // proves four workers can be active simultaneously
            let mut nested = [0; 19];
            parallel_fill(&mut nested, 8, |j, slot| {
                assert_eq!(thread::current().id(), id);
                *slot = j;
            });
            *value = i + nested.iter().sum::<usize>();
        });
        assert_eq!(threads.into_inner().unwrap().len(), 4);
        assert_eq!(out, [171, 172, 173, 174]);
    }

    #[test]
    fn a_panicking_worker_propagates_and_restores_the_caller() {
        assert!(std::panic::catch_unwind(|| {
            parallel_fill(&mut [0; 8], 4, |i, _| {
                if i == 3 {
                    panic!("worker failed");
                }
            });
        })
        .is_err());
        assert!(!ACTIVE.with(|active| active.get()));
        assert_eq!(parallel_map(&[1, 2, 3], 4, |n| n * 2), [2, 4, 6]);
    }
}
