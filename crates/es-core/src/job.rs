//! Job system and deterministic partitioning (P14).
//!
//! Deterministic scheduling is one of the eight conditions of the execution contract (§3.4):
//! static partitioning, no branching derived from device properties, and reduction results
//! merged in chunk-index order rather than completion order. See
//! `docs/design/job-partitioning.md`.

use std::ops::Range;

/// Splits `0..n_items` into at most `n_workers` contiguous chunks.
///
/// A pure function of `(n_items, n_workers)`: no device query, no load balancing, no work
/// stealing. The first `n_items % n_workers` chunks take one extra item. Empty chunks are not
/// emitted, so the result has `min(n_items, n_workers)` entries. `n_workers == 0` is treated
/// as 1.
pub fn partition_static(n_items: usize, n_workers: usize) -> Vec<Range<usize>> {
    let workers = n_workers.max(1);
    let base = n_items / workers;
    let rem = n_items % workers;
    let mut ranges = Vec::with_capacity(workers.min(n_items));
    let mut start = 0;
    for i in 0..workers {
        let end = start + base + usize::from(i < rem);
        if end > start {
            ranges.push(start..end);
        }
        start = end;
    }
    ranges
}

/// Runs chunked work on scoped threads.
///
/// The worker count is always given explicitly; deriving it from the machine would make the
/// partition — and therefore any non-associative reduction over it — machine dependent (§3.4).
#[derive(Clone, Copy, Debug)]
pub struct JobSystem {
    workers: usize,
}

impl JobSystem {
    /// `workers` is clamped to at least 1.
    pub fn new(workers: usize) -> Self {
        Self {
            workers: workers.max(1),
        }
    }

    pub const fn workers(&self) -> usize {
        self.workers
    }

    /// Applies `f(chunk_index, range, chunk)` to every chunk of `items` in parallel.
    pub fn par_for_each_chunk<T, F>(&self, items: &mut [T], f: F)
    where
        T: Send,
        F: Fn(usize, Range<usize>, &mut [T]) + Sync,
    {
        let ranges = partition_static(items.len(), self.workers);
        let mut rest = items;
        let mut chunks = Vec::with_capacity(ranges.len());
        for range in ranges {
            let (chunk, tail) = rest.split_at_mut(range.len());
            chunks.push((range, chunk));
            rest = tail;
        }
        let f = &f;
        std::thread::scope(|scope| {
            for (index, (range, chunk)) in chunks.into_iter().enumerate() {
                scope.spawn(move || f(index, range, chunk));
            }
        });
    }

    /// Maps each chunk of `items` in parallel, then folds the per-chunk results with `merge`
    /// in **chunk-index order**, never in completion order.
    ///
    /// `merge` is any closure: `es-core` knows nothing about floating point. Across different
    /// worker counts the result is stable exactly when `merge` is associative (§18.4) — that is
    /// what `es-math`'s `DeterministicAcc` will provide for float sums.
    ///
    /// Returns `None` for an empty input.
    pub fn par_reduce<T, R, F, M>(&self, items: &[T], map: F, merge: M) -> Option<R>
    where
        T: Sync,
        R: Send,
        F: Fn(&[T]) -> R + Sync,
        M: Fn(R, R) -> R,
    {
        let ranges = partition_static(items.len(), self.workers);
        let mut results: Vec<Option<R>> = ranges.iter().map(|_| None).collect();
        let map = &map;
        std::thread::scope(|scope| {
            for (slot, range) in results.iter_mut().zip(&ranges) {
                let chunk = &items[range.clone()];
                scope.spawn(move || *slot = Some(map(chunk)));
            }
        });
        results.into_iter().flatten().reduce(merge)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn partition_is_a_pure_function_of_its_arguments() {
        assert_eq!(partition_static(10, 3), vec![0..4, 4..7, 7..10]);
        assert_eq!(partition_static(3, 10), vec![0..1, 1..2, 2..3]);
        assert_eq!(partition_static(0, 4), Vec::<Range<usize>>::new());
        assert_eq!(partition_static(5, 0), vec![0..5]);
        assert_eq!(partition_static(10, 3), partition_static(10, 3));
    }

    proptest! {
        #[test]
        fn partition_covers_every_item_exactly_once(n_items in 0usize..500, n_workers in 0usize..17) {
            let ranges = partition_static(n_items, n_workers);
            let mut next = 0;
            for range in &ranges {
                prop_assert_eq!(range.start, next);
                prop_assert!(range.end > range.start);
                next = range.end;
            }
            prop_assert_eq!(next, n_items);
            prop_assert!(ranges.len() <= n_workers.max(1).min(n_items.max(1)));
        }
    }

    #[test]
    fn par_for_each_chunk_visits_every_element_once() {
        for workers in 1..=8 {
            let mut items: Vec<usize> = (0..37).collect();
            JobSystem::new(workers).par_for_each_chunk(&mut items, |_, range, chunk| {
                for (offset, item) in chunk.iter_mut().enumerate() {
                    assert_eq!(*item, range.start + offset);
                    *item += 100;
                }
            });
            assert_eq!(items, (0..37).map(|i| i + 100).collect::<Vec<_>>());
        }
    }

    #[test]
    fn par_reduce_is_independent_of_worker_count_and_scheduling() {
        let items: Vec<i64> = (0..1000).collect();
        let expected: i64 = items.iter().sum();
        for _ in 0..20 {
            for workers in 1..=8 {
                let sum = JobSystem::new(workers).par_reduce(
                    &items,
                    |chunk| chunk.iter().sum::<i64>(),
                    |a, b| a + b,
                );
                assert_eq!(sum, Some(expected));
            }
        }
        assert_eq!(
            JobSystem::new(4).par_reduce(&[] as &[i64], <[i64]>::len, |a, b| a + b),
            None
        );
    }

    #[test]
    fn par_reduce_merges_in_chunk_index_order_not_completion_order() {
        // A deliberately non-commutative merge: the answer is only stable if chunks are folded
        // by index. The first chunk sleeps longest, so completion order is reversed.
        let items: Vec<usize> = (0..8).collect();
        let job = JobSystem::new(4);
        for _ in 0..20 {
            let joined = job
                .par_reduce(
                    &items,
                    |chunk| {
                        std::thread::sleep(std::time::Duration::from_millis(
                            20 - 5 * chunk[0] as u64 / 2,
                        ));
                        chunk.iter().map(ToString::to_string).collect::<String>()
                    },
                    |a, b| a + &b,
                )
                .unwrap();
            assert_eq!(joined, "01234567");
        }
    }
}
