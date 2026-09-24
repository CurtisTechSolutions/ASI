//! Python's `heapq`, element for element.
//!
//! A priority queue only has to agree with another on the order it pops in,
//! and wherever the keys are unique any correct heap will do.  But the phase
//! beam (`radixnet/phasesearch.py`) keeps its finished paths in a `heapq` list
//! and afterwards reads that list *as a list* - in array order - before a
//! stable sort by cost, so two paths of equal cost come back in whatever order
//! `heapq` happened to leave them in.  To return the same paths in the same
//! order, this is CPython's `heappush` / `heappop` with its exact sift
//! routines (`_siftdown`, `_siftup`), comparing with a caller's `<`.

/// Pushes `item`, keeping the heap invariant (`heapq.heappush`).
pub fn heappush<T>(heap: &mut Vec<T>, item: T, less: &impl Fn(&T, &T) -> bool) {
    heap.push(item);
    let last = heap.len() - 1;
    siftdown(heap, 0, last, less);
}

/// Pops the smallest item (`heapq.heappop`).
pub fn heappop<T>(heap: &mut Vec<T>, less: &impl Fn(&T, &T) -> bool) -> Option<T> {
    let last = heap.pop()?;
    if heap.is_empty() {
        return Some(last);
    }
    let top = std::mem::replace(&mut heap[0], last);
    siftup(heap, 0, less);
    Some(top)
}

/// `_siftdown`: moves the item at `pos` up towards `start` while it is
/// smaller than its parent.
fn siftdown<T>(heap: &mut [T], start: usize, mut pos: usize, less: &impl Fn(&T, &T) -> bool) {
    while pos > start {
        let parent = (pos - 1) >> 1;
        if less(&heap[pos], &heap[parent]) {
            heap.swap(pos, parent);
            pos = parent;
            continue;
        }
        break;
    }
}

/// `_siftup`: bubbles the smaller child up until a leaf, then sifts the
/// displaced item back down - CPython's variant, which makes fewer comparisons
/// and a different (but equally valid) layout than the textbook one.
fn siftup<T>(heap: &mut [T], mut pos: usize, less: &impl Fn(&T, &T) -> bool) {
    let end = heap.len();
    let start = pos;
    let mut child = 2 * pos + 1;
    while child < end {
        let right = child + 1;
        if right < end && !less(&heap[child], &heap[right]) {
            child = right;
        }
        heap.swap(pos, child);
        pos = child;
        child = 2 * pos + 1;
    }
    siftdown(heap, start, pos, less);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layout_is_cpythons() {
        // import heapq; h = []
        // for x in [5, 3, 8, 1, 9, 2, 7]: heapq.heappush(h, x)
        // heapq.heappop(h); print(h)  ->  [2, 3, 7, 5, 9, 8]
        let less = |a: &i32, b: &i32| a < b;
        let mut h = Vec::new();
        for x in [5, 3, 8, 1, 9, 2, 7] {
            heappush(&mut h, x, &less);
        }
        assert_eq!(h, vec![1, 3, 2, 5, 9, 8, 7]);
        assert_eq!(heappop(&mut h, &less), Some(1));
        assert_eq!(h, vec![2, 3, 7, 5, 9, 8]);
        let mut sorted = Vec::new();
        while let Some(x) = heappop(&mut h, &less) {
            sorted.push(x);
        }
        assert_eq!(sorted, vec![2, 3, 5, 7, 8, 9]);
    }
}
