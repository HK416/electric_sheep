//! Generational pool (P17).
//!
//! Dense storage with stable `Copy` handles. A handle to a removed slot never resolves again,
//! even after the slot is reused, so a stale id is a `None` instead of silently reading a
//! different object.

use std::fmt;
use std::marker::PhantomData;

/// Handle into a [`Pool`]. `Copy + Ord` for any `T`.
pub struct Handle<T> {
    index: u32,
    generation: u32,
    // `fn() -> T` keeps `Handle<T>` `Send`/`Sync` regardless of `T`.
    marker: PhantomData<fn() -> T>,
}

impl<T> Handle<T> {
    pub const fn index(self) -> u32 {
        self.index
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }

    const fn key(self) -> (u32, u32) {
        (self.index, self.generation)
    }
}

// Derived impls would demand `T: Clone`/`T: Ord` and so on; a handle is just two integers.
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Handle<T> {}
impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}
impl<T> Eq for Handle<T> {}
impl<T> PartialOrd for Handle<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<T> Ord for Handle<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key().cmp(&other.key())
    }
}
impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}
impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Handle({}v{})", self.index, self.generation)
    }
}

#[derive(Debug)]
struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

/// Slot storage with generational handles.
#[derive(Debug)]
pub struct Pool<T> {
    slots: Vec<Slot<T>>,
    /// Free slot indices, used last-in first-out so reuse order is deterministic (§3.4).
    free: Vec<u32>,
    len: usize,
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Pool<T> {
    pub const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            len: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free: Vec::new(),
            len: 0,
        }
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// # Panics
    /// If the pool exceeds `u32::MAX` slots.
    pub fn insert(&mut self, value: T) -> Handle<T> {
        self.len += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.value = Some(value);
            return Handle {
                index,
                generation: slot.generation,
                marker: PhantomData,
            };
        }
        let index = u32::try_from(self.slots.len()).expect("pool slot overflow");
        self.slots.push(Slot {
            generation: 0,
            value: Some(value),
        });
        Handle {
            index,
            generation: 0,
            marker: PhantomData,
        }
    }

    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        self.slot(handle)?.value.as_ref()
    }

    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.value.as_mut()
    }

    pub fn contains(&self, handle: Handle<T>) -> bool {
        self.get(handle).is_some()
    }

    /// Frees the slot and invalidates every handle to it.
    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        let value = slot.value.take()?;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(handle.index);
        self.len -= 1;
        Some(value)
    }

    fn slot(&self, handle: Handle<T>) -> Option<&Slot<T>> {
        self.slots
            .get(handle.index as usize)
            .filter(|slot| slot.generation == handle.generation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_remove() {
        let mut pool = Pool::new();
        let a = pool.insert("a".to_owned());
        let b = pool.insert("b".to_owned());
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.get(a).unwrap(), "a");
        pool.get_mut(b).unwrap().push('!');
        assert_eq!(pool.get(b).unwrap(), "b!");

        assert_eq!(pool.remove(a).unwrap(), "a");
        assert_eq!(pool.len(), 1);
        assert!(!pool.contains(a));
        assert_eq!(pool.get(a), None);
        assert_eq!(pool.get_mut(a), None);
        assert_eq!(pool.remove(a), None);
    }

    #[test]
    fn stale_handles_never_resolve_after_slot_reuse() {
        let mut pool = Pool::new();
        let old = pool.insert(1u32);
        pool.remove(old);
        let new = pool.insert(2u32);
        assert_eq!(new.index(), old.index());
        assert_eq!(new.generation(), old.generation() + 1);
        assert_eq!(pool.get(old), None);
        assert_eq!(pool.get(new), Some(&2));
        assert_ne!(old, new);
    }

    #[test]
    fn handles_are_copy_and_ordered_for_non_ord_payloads() {
        struct NotOrd(#[allow(dead_code)] f32);
        let mut pool = Pool::new();
        let a = pool.insert(NotOrd(1.0));
        let b = pool.insert(NotOrd(2.0));
        let copies = [a, b, a];
        assert!(a < b);
        assert_eq!(copies[0], copies[2]);
        assert!(!pool.is_empty() && pool.len() == 2);
    }
}
