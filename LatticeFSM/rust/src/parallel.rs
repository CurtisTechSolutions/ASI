//! Scoped CPU work, using every available logical core by default.
//! `LATTICEFSM_WORKERS=1` keeps work on the caller; a positive value limits
//! parallelism. Nested operations reuse their outer worker rather than spawning.

use std::thread;

/// Worker limit, read once per process. Zero or an unset variable means auto.
pub fn effective_workers(workers: usize) -> usize {
    if workers > 0 {
        return workers;
    }
    static WORKERS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *WORKERS.get_or_init(|| {
        std::env::var("LATTICEFSM_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
    })
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
