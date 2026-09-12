//! Archetype ECS with struct-of-arrays storage (P13).
//!
//! Entities with the same component set share an archetype, and each component type is one
//! contiguous `Vec` inside it. Iteration order is deterministic (§3.4): archetypes in creation
//! order, rows in the order the entity entered the archetype. `despawn` swap-removes, which
//! permutes rows — the order stays a pure function of the operation sequence, which is what
//! replay needs.
//!
//! Deliberately minimal: no systems, no schedules, no events, no change detection, no
//! component removal. Those belong to `es-env` (layer 9) if they are ever needed at all.

use std::any::{Any, TypeId};
use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;

/// Marker for anything that can be stored in the [`World`]. Blanket implemented; not an
/// extension point (INV-17).
pub trait Component: 'static + Send + Sync {}
impl<T: 'static + Send + Sync> Component for T {}

/// Handle to an entity: a slot index plus the generation of that slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Entity {
    index: u32,
    generation: u32,
}

impl Entity {
    pub const fn index(self) -> u32 {
        self.index
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }
}

// ---------------------------------------------------------------------------------------
// Columns
// ---------------------------------------------------------------------------------------

trait Column: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn swap_remove(&mut self, row: usize);
    /// Moves `row` out of `self` (swap-remove) and pushes it onto `dst`, which must hold the
    /// same component type.
    fn migrate(&mut self, row: usize, dst: &mut dyn Column);
    /// An empty column of the same component type.
    fn empty_clone(&self) -> Box<dyn Column>;
}

struct TypedColumn<T>(Vec<T>);

impl<T: Component> Column for TypedColumn<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn swap_remove(&mut self, row: usize) {
        self.0.swap_remove(row);
    }

    fn migrate(&mut self, row: usize, dst: &mut dyn Column) {
        let value = self.0.swap_remove(row);
        dst.as_any_mut()
            .downcast_mut::<Self>()
            .expect("migrate target column has a different component type")
            .0
            .push(value);
    }

    fn empty_clone(&self) -> Box<dyn Column> {
        Box::new(Self(Vec::new()))
    }
}

// ---------------------------------------------------------------------------------------
// Archetype
// ---------------------------------------------------------------------------------------

/// Storage for all entities sharing one component set. Opaque; only queries reach inside.
pub struct Archetype {
    /// Sorted, so the set is a canonical key.
    types: Vec<TypeId>,
    entities: Vec<Entity>,
    columns: BTreeMap<TypeId, Box<dyn Column>>,
}

impl Archetype {
    fn contains_type(&self, type_id: TypeId) -> bool {
        self.columns.contains_key(&type_id)
    }

    fn column<T: Component>(&self) -> Option<&Vec<T>> {
        let column = self.columns.get(&TypeId::of::<T>())?;
        Some(&column.as_any().downcast_ref::<TypedColumn<T>>()?.0)
    }

    fn column_mut<T: Component>(&mut self) -> Option<&mut Vec<T>> {
        let column = self.columns.get_mut(&TypeId::of::<T>())?;
        Some(&mut column.as_any_mut().downcast_mut::<TypedColumn<T>>()?.0)
    }

    /// Start of this archetype's column for `T`. Query plumbing only.
    ///
    /// # Panics
    /// If the archetype has no column for `T`.
    #[doc(hidden)]
    pub fn column_ptr<T: Component>(&mut self) -> *mut T {
        self.column_mut::<T>()
            .expect("queried archetype has no column for this type")
            .as_mut_ptr()
    }
}

impl fmt::Debug for Archetype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Archetype")
            .field("components", &self.types.len())
            .field("entities", &self.entities.len())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Location {
    archetype: usize,
    row: usize,
}

#[derive(Debug)]
struct EntityMeta {
    generation: u32,
    location: Option<Location>,
}

/// Entity storage.
pub struct World {
    entities: Vec<EntityMeta>,
    /// Free slot indices, popped last-in first-out so reuse is deterministic.
    free: Vec<u32>,
    /// Creation order; also the query iteration order.
    archetypes: Vec<Archetype>,
    /// Component set -> archetype index. `BTreeMap`, never `HashMap` (§3.4).
    index: BTreeMap<Vec<TypeId>, usize>,
    len: usize,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    pub fn new() -> Self {
        let empty = Archetype {
            types: Vec::new(),
            entities: Vec::new(),
            columns: BTreeMap::new(),
        };
        Self {
            entities: Vec::new(),
            free: Vec::new(),
            archetypes: vec![empty],
            index: BTreeMap::from([(Vec::new(), 0)]),
            len: 0,
        }
    }

    /// Live entity count.
    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn archetype_count(&self) -> usize {
        self.archetypes.len()
    }

    pub fn contains(&self, entity: Entity) -> bool {
        self.location(entity).is_some()
    }

    /// Creates an entity with no components.
    ///
    /// # Panics
    /// If more than `u32::MAX` entity slots are needed.
    pub fn spawn(&mut self) -> Entity {
        let index = if let Some(index) = self.free.pop() {
            index
        } else {
            let index = u32::try_from(self.entities.len()).expect("entity slot overflow");
            self.entities.push(EntityMeta {
                generation: 0,
                location: None,
            });
            index
        };
        let generation = self.entities[index as usize].generation;
        let entity = Entity { index, generation };
        let row = self.archetypes[0].entities.len();
        self.archetypes[0].entities.push(entity);
        self.entities[index as usize].location = Some(Location { archetype: 0, row });
        self.len += 1;
        entity
    }

    /// Removes an entity and every component it holds. Returns `false` for a stale handle.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        let Some(location) = self.location(entity) else {
            return false;
        };
        let archetype = &mut self.archetypes[location.archetype];
        for column in archetype.columns.values_mut() {
            column.swap_remove(location.row);
        }
        archetype.entities.swap_remove(location.row);
        let moved = archetype.entities.get(location.row).copied();
        if let Some(moved) = moved {
            self.entities[moved.index as usize].location = Some(location);
        }
        let meta = &mut self.entities[entity.index as usize];
        meta.location = None;
        meta.generation = meta.generation.wrapping_add(1);
        self.free.push(entity.index);
        self.len -= 1;
        true
    }

    /// Adds or replaces the `T` of `entity`. Returns `false` for a stale handle.
    pub fn insert<T: Component>(&mut self, entity: Entity, value: T) -> bool {
        let Some(location) = self.location(entity) else {
            return false;
        };
        let type_id = TypeId::of::<T>();
        let src = location.archetype;

        if self.archetypes[src].contains_type(type_id) {
            self.archetypes[src]
                .column_mut::<T>()
                .expect("column exists")[location.row] = value;
            return true;
        }

        let dst = self.archetype_with::<T>(src);
        let moved = {
            let (src_archetype, dst_archetype) = two_mut(&mut self.archetypes, src, dst);
            for (type_id, column) in &mut src_archetype.columns {
                let target = dst_archetype
                    .columns
                    .get_mut(type_id)
                    .expect("column exists");
                column.migrate(location.row, target.as_mut());
            }
            src_archetype.entities.swap_remove(location.row);
            dst_archetype.entities.push(entity);
            dst_archetype
                .column_mut::<T>()
                .expect("column exists")
                .push(value);
            src_archetype.entities.get(location.row).copied()
        };

        if let Some(moved) = moved {
            self.entities[moved.index as usize].location = Some(location);
        }
        let row = self.archetypes[dst].entities.len() - 1;
        self.entities[entity.index as usize].location = Some(Location {
            archetype: dst,
            row,
        });
        true
    }

    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        let location = self.location(entity)?;
        self.archetypes[location.archetype]
            .column::<T>()?
            .get(location.row)
    }

    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        let location = self.location(entity)?;
        self.archetypes[location.archetype]
            .column_mut::<T>()?
            .get_mut(location.row)
    }

    /// Iterates every entity holding all of `Q`'s component types, in deterministic order.
    ///
    /// ```
    /// # use es_core::ecs::World;
    /// struct Pos(f32);
    /// struct Vel(f32);
    /// let mut world = World::new();
    /// let e = world.spawn();
    /// world.insert(e, Pos(0.0));
    /// world.insert(e, Vel(2.0));
    /// for (_entity, (pos, vel)) in world.query::<(&mut Pos, &Vel)>() {
    ///     pos.0 += vel.0;
    /// }
    /// assert_eq!(world.get::<Pos>(e).unwrap().0, 2.0);
    /// ```
    ///
    /// # Panics
    /// If `Q` names the same component type twice (that would alias).
    pub fn query<Q: Query>(&mut self) -> QueryIter<'_, Q> {
        let mut types = Q::type_ids();
        let count = types.len();
        types.sort_unstable();
        types.dedup();
        assert_eq!(
            types.len(),
            count,
            "a query may not name the same component type twice"
        );
        QueryIter {
            world: std::ptr::from_mut(self),
            types,
            archetype: 0,
            row: 0,
            len: 0,
            ptrs: None,
            entities: std::ptr::null(),
            marker: PhantomData,
        }
    }

    fn location(&self, entity: Entity) -> Option<Location> {
        let meta = self.entities.get(entity.index as usize)?;
        if meta.generation != entity.generation {
            return None;
        }
        meta.location
    }

    /// Index of the archetype holding `src`'s components plus `T`, creating it if new.
    fn archetype_with<T: Component>(&mut self, src: usize) -> usize {
        let mut types = self.archetypes[src].types.clone();
        types.push(TypeId::of::<T>());
        types.sort_unstable();
        if let Some(&existing) = self.index.get(&types) {
            return existing;
        }
        let mut columns: BTreeMap<TypeId, Box<dyn Column>> = self.archetypes[src]
            .columns
            .iter()
            .map(|(type_id, column)| (*type_id, column.empty_clone()))
            .collect();
        columns.insert(TypeId::of::<T>(), Box::new(TypedColumn::<T>(Vec::new())));
        let index = self.archetypes.len();
        self.archetypes.push(Archetype {
            types: types.clone(),
            entities: Vec::new(),
            columns,
        });
        self.index.insert(types, index);
        index
    }
}

impl fmt::Debug for World {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("World")
            .field("entities", &self.len)
            .field("archetypes", &self.archetypes.len())
            .finish_non_exhaustive()
    }
}

fn two_mut(archetypes: &mut [Archetype], a: usize, b: usize) -> (&mut Archetype, &mut Archetype) {
    assert_ne!(a, b, "source and destination archetype must differ");
    if a < b {
        let (left, right) = archetypes.split_at_mut(b);
        (&mut left[a], &mut right[0])
    } else {
        let (left, right) = archetypes.split_at_mut(a);
        (&mut right[0], &mut left[b])
    }
}

// ---------------------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------------------

/// One element of a query: `&T` or `&mut T`.
///
/// # Safety
/// `ptr` must return the start of the archetype's column for this component type, and `item`
/// may only be called with a row inside that column.
pub unsafe trait QueryParam {
    type Item<'a>;
    type Ptr: Copy;

    fn type_id() -> TypeId;

    #[doc(hidden)]
    fn ptr(archetype: &mut Archetype) -> Self::Ptr;

    /// # Safety
    /// `row` must be in bounds for the column `ptr` came from, and no other live reference may
    /// alias that row.
    #[doc(hidden)]
    unsafe fn item<'a>(ptr: Self::Ptr, row: usize) -> Self::Item<'a>;
}

// SAFETY: `ptr` is the column start for `T`; `item` offsets by an in-bounds row.
unsafe impl<T: Component> QueryParam for &T {
    type Item<'a> = &'a T;
    type Ptr = *const T;

    fn type_id() -> TypeId {
        TypeId::of::<T>()
    }

    fn ptr(archetype: &mut Archetype) -> Self::Ptr {
        archetype.column_ptr::<T>().cast_const()
    }

    unsafe fn item<'a>(ptr: Self::Ptr, row: usize) -> Self::Item<'a> {
        // SAFETY: guaranteed by the caller of `item`.
        unsafe { &*ptr.add(row) }
    }
}

// SAFETY: as above; the caller of `item` guarantees each row is handed out at most once.
unsafe impl<T: Component> QueryParam for &mut T {
    type Item<'a> = &'a mut T;
    type Ptr = *mut T;

    fn type_id() -> TypeId {
        TypeId::of::<T>()
    }

    fn ptr(archetype: &mut Archetype) -> Self::Ptr {
        archetype.column_ptr::<T>()
    }

    unsafe fn item<'a>(ptr: Self::Ptr, row: usize) -> Self::Item<'a> {
        // SAFETY: guaranteed by the caller of `item`.
        unsafe { &mut *ptr.add(row) }
    }
}

/// A component pattern: a [`QueryParam`] or a tuple of up to four of them.
///
/// # Safety
/// Implementations must report every component type they fetch in `type_ids`, so that
/// [`World::query`] can reject duplicates before handing out aliasing `&mut`.
pub unsafe trait Query {
    type Item<'a>;
    type Ptrs: Copy;

    fn type_ids() -> Vec<TypeId>;

    #[doc(hidden)]
    fn ptrs(archetype: &mut Archetype) -> Self::Ptrs;

    /// # Safety
    /// `row` must be in bounds for the archetype `ptrs` came from, and each row may be fetched
    /// at most once while the returned items are alive.
    #[doc(hidden)]
    unsafe fn item<'a>(ptrs: Self::Ptrs, row: usize) -> Self::Item<'a>;
}

macro_rules! impl_query {
    ($($param:ident),+) => {
        // SAFETY: every fetched type is reported by `type_ids`, so `World::query` rejects
        // duplicates, and the parameters therefore point into distinct columns.
        unsafe impl<$($param: QueryParam),+> Query for ($($param,)+) {
            type Item<'a> = ($(<$param as QueryParam>::Item<'a>,)+);
            type Ptrs = ($(<$param as QueryParam>::Ptr,)+);

            fn type_ids() -> Vec<TypeId> {
                vec![$(<$param as QueryParam>::type_id()),+]
            }

            fn ptrs(archetype: &mut Archetype) -> Self::Ptrs {
                ($(<$param as QueryParam>::ptr(archetype),)+)
            }

            unsafe fn item<'a>(ptrs: Self::Ptrs, row: usize) -> Self::Item<'a> {
                #[allow(non_snake_case)]
                let ($($param,)+) = ptrs;
                // SAFETY: guaranteed by the caller of `item`.
                unsafe { ($(<$param as QueryParam>::item($param, row),)+) }
            }
        }
    };
}

impl_query!(A);
impl_query!(A, B);
impl_query!(A, B, C);
impl_query!(A, B, C, D);

macro_rules! impl_query_single {
    ($lt:lifetime, $ty:ty) => {
        // SAFETY: a single parameter reports its own type and fetches only its own column.
        unsafe impl<$lt, T: Component> Query for $ty {
            type Item<'a> = <$ty as QueryParam>::Item<'a>;
            type Ptrs = <$ty as QueryParam>::Ptr;

            fn type_ids() -> Vec<TypeId> {
                vec![<$ty as QueryParam>::type_id()]
            }

            fn ptrs(archetype: &mut Archetype) -> Self::Ptrs {
                <$ty as QueryParam>::ptr(archetype)
            }

            unsafe fn item<'a>(ptrs: Self::Ptrs, row: usize) -> Self::Item<'a> {
                // SAFETY: guaranteed by the caller of `item`.
                unsafe { <$ty as QueryParam>::item(ptrs, row) }
            }
        }
    };
}

impl_query_single!('q, &'q T);
impl_query_single!('q, &'q mut T);

/// Iterator returned by [`World::query`], yielding `(Entity, Q::Item)`.
pub struct QueryIter<'w, Q: Query> {
    world: *mut World,
    /// Sorted, deduplicated component types the archetype must hold.
    types: Vec<TypeId>,
    archetype: usize,
    row: usize,
    len: usize,
    ptrs: Option<Q::Ptrs>,
    entities: *const Entity,
    marker: PhantomData<&'w mut World>,
}

impl<'w, Q: Query> Iterator for QueryIter<'w, Q> {
    type Item = (Entity, Q::Item<'w>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                let row = self.row;
                self.row += 1;
                let ptrs = self.ptrs.expect("column pointers are set whenever len > 0");
                // SAFETY: `row < len`, the length of the archetype the pointers came from. The
                // row counter only moves forward, so a `&mut` item is handed out at most once
                // per row, and the iterator holds the world's exclusive borrow for `'w`, so
                // nothing else can touch these columns meanwhile.
                let item = unsafe { Q::item(ptrs, row) };
                // SAFETY: same bounds; `entities` is the archetype's entity column.
                let entity = unsafe { *self.entities.add(row) };
                return Some((entity, item));
            }

            // SAFETY: the iterator borrows the world exclusively for `'w`, and no item
            // reference is created from this borrow.
            let world = unsafe { &mut *self.world };
            let index = self.archetype;
            if index >= world.archetypes.len() {
                return None;
            }
            self.archetype += 1;
            let archetype = &mut world.archetypes[index];
            if archetype.entities.is_empty()
                || !self
                    .types
                    .iter()
                    .all(|type_id| archetype.contains_type(*type_id))
            {
                continue;
            }
            self.len = archetype.entities.len();
            self.entities = archetype.entities.as_ptr();
            self.ptrs = Some(Q::ptrs(archetype));
            self.row = 0;
        }
    }
}

impl<Q: Query> fmt::Debug for QueryIter<'_, Q> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueryIter")
            .field("archetype", &self.archetype)
            .field("row", &self.row)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Pos(i32);
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Vel(i32);
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Tag;

    #[test]
    fn spawn_insert_get_despawn() {
        let mut world = World::new();
        let e = world.spawn();
        assert!(world.contains(e));
        assert_eq!(world.len(), 1);
        assert_eq!(world.get::<Pos>(e), None);

        assert!(world.insert(e, Pos(1)));
        assert!(world.insert(e, Vel(2)));
        assert_eq!(world.get::<Pos>(e), Some(&Pos(1)));
        world.get_mut::<Pos>(e).unwrap().0 = 5;
        assert_eq!(world.get::<Pos>(e), Some(&Pos(5)));

        // Replacing in place does not create an archetype.
        let archetypes = world.archetype_count();
        assert!(world.insert(e, Pos(7)));
        assert_eq!(world.archetype_count(), archetypes);
        assert_eq!(world.get::<Pos>(e), Some(&Pos(7)));

        assert!(world.despawn(e));
        assert!(!world.contains(e));
        assert!(world.is_empty());
        // Stale handle.
        assert_eq!(world.get::<Pos>(e), None);
        assert!(!world.insert(e, Pos(0)));
        assert!(!world.despawn(e));
    }

    #[test]
    fn migration_preserves_values_and_fixes_up_swapped_rows() {
        let mut world = World::new();
        let a = world.spawn();
        let b = world.spawn();
        let c = world.spawn();
        for (e, v) in [(a, 1), (b, 2), (c, 3)] {
            world.insert(e, Pos(v));
        }
        // Moving `a` out of the Pos archetype swap-removes it and moves `c` into its row.
        world.insert(a, Vel(10));
        assert_eq!(world.get::<Pos>(a), Some(&Pos(1)));
        assert_eq!(world.get::<Pos>(b), Some(&Pos(2)));
        assert_eq!(world.get::<Pos>(c), Some(&Pos(3)));
        assert_eq!(world.get::<Vel>(a), Some(&Vel(10)));
        assert_eq!(world.get::<Vel>(b), None);

        world.despawn(b);
        assert_eq!(world.get::<Pos>(c), Some(&Pos(3)));
        assert_eq!(world.len(), 2);
    }

    #[test]
    fn stale_handles_do_not_alias_reused_slots() {
        let mut world = World::new();
        let old = world.spawn();
        world.insert(old, Pos(1));
        world.despawn(old);
        let new = world.spawn();
        assert_eq!(new.index(), old.index());
        assert_ne!(new.generation(), old.generation());
        assert_eq!(world.get::<Pos>(old), None);
        assert!(world.contains(new));
    }

    #[test]
    fn query_visits_every_matching_archetype_in_creation_order() {
        let mut world = World::new();
        let a = world.spawn();
        world.insert(a, Pos(1));
        world.insert(a, Vel(1));
        let b = world.spawn();
        world.insert(b, Pos(2)); // Pos-only archetype, does not match
        let c = world.spawn();
        world.insert(c, Pos(3));
        world.insert(c, Vel(3));
        world.insert(c, Tag); // third archetype

        let seen: Vec<(Entity, i32, i32)> = world
            .query::<(&Pos, &Vel)>()
            .map(|(e, (pos, vel))| (e, pos.0, vel.0))
            .collect();
        assert_eq!(seen, vec![(a, 1, 1), (c, 3, 3)]);

        for (_, (pos, vel)) in world.query::<(&mut Pos, &Vel)>() {
            pos.0 += vel.0;
        }
        assert_eq!(world.get::<Pos>(a), Some(&Pos(2)));
        assert_eq!(world.get::<Pos>(c), Some(&Pos(6)));
        assert_eq!(world.get::<Pos>(b), Some(&Pos(2)));

        let single: Vec<i32> = world.query::<&Pos>().map(|(_, pos)| pos.0).collect();
        assert_eq!(single, vec![2, 2, 6]);
    }

    #[test]
    #[should_panic(expected = "same component type twice")]
    fn duplicate_query_types_panic() {
        let mut world = World::new();
        let _ = world.query::<(&mut Pos, &mut Pos)>().count();
    }

    #[derive(Debug, Clone, Copy)]
    enum Op {
        Spawn,
        InsertPos(u8),
        InsertVel(u8),
        Despawn(u8),
    }

    fn apply(ops: &[Op]) -> Vec<(Entity, i32, i32)> {
        let mut world = World::new();
        let mut spawned: Vec<Entity> = Vec::new();
        for op in ops {
            match *op {
                Op::Spawn => spawned.push(world.spawn()),
                Op::InsertPos(i) => {
                    if let Some(&e) = pick(&spawned, i) {
                        world.insert(e, Pos(i32::from(i)));
                    }
                }
                Op::InsertVel(i) => {
                    if let Some(&e) = pick(&spawned, i) {
                        world.insert(e, Vel(i32::from(i)));
                    }
                }
                Op::Despawn(i) => {
                    if let Some(&e) = pick(&spawned, i) {
                        world.despawn(e);
                    }
                }
            }
        }
        world
            .query::<(&Pos, &Vel)>()
            .map(|(e, (p, v))| (e, p.0, v.0))
            .collect()
    }

    fn pick(entities: &[Entity], i: u8) -> Option<&Entity> {
        if entities.is_empty() {
            None
        } else {
            entities.get(usize::from(i) % entities.len())
        }
    }

    fn op_strategy() -> impl Strategy<Value = Op> {
        prop_oneof![
            Just(Op::Spawn),
            any::<u8>().prop_map(Op::InsertPos),
            any::<u8>().prop_map(Op::InsertVel),
            any::<u8>().prop_map(Op::Despawn),
        ]
    }

    proptest! {
        #[test]
        fn iteration_order_is_a_pure_function_of_the_operation_sequence(
            ops in proptest::collection::vec(op_strategy(), 0..64)
        ) {
            prop_assert_eq!(apply(&ops), apply(&ops));
        }
    }
}
