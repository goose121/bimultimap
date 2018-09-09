use ndarray::prelude::*;
use ndarray::Array2;
use std::collections::hash_map::RandomState;
use std::collections::HashSet;
use std::fmt::{self, Debug};
use std::hash::BuildHasher;
use std::hash::Hash;
use std::hash::Hasher;

/// `BiMultiMap` is an implementation of a bidirectional multimap,
/// where each key can be associated with multiple values, and each
/// value with multiple keys.
/// 
/// It is implemented like a regular hashmap
/// using separate chaining, except that the buckets are in a 2D
/// array. A given pair `(k, v)` is put into the bucket
/// `buckets[k.hash()][v.hash()]`.
///
/// # Note
///
/// Looking up the values for a given key is probably faster than
/// looking up the keys for a given value, due to the caching issues
/// inherent to striding over the array.
pub struct BiMultiMap<K, V, S = RandomState>
where
    K: Hash + Eq,
    V: Hash + Eq,
{
    buckets: Array2<HashSet<(K, V)>>,
    builder: S,
}

impl<K, V> BiMultiMap<K, V, RandomState>
where
    K: Hash + Eq,
    V: Hash + Eq,
{
    /// Creates a new `BiMultiMap` with the default hasher which uses
    /// a bucket array with `keys` rows and `vals` columns.
    pub fn new(keys: usize, vals: usize) -> BiMultiMap<K, V, RandomState> {
        BiMultiMap::with_hasher(keys, vals, Default::default())
    }
}

impl<K, V, S> BiMultiMap<K, V, S>
where
    K: Hash + Eq,
    V: Hash + Eq,
    S: BuildHasher,
{
    /// Hashes a value with a new hasher and returns the result.
    fn hash<T: Hash>(&self, val: &T) -> u64 {
        let mut hasher = self.builder.build_hasher();
        val.hash(&mut hasher);
        hasher.finish()
    }

    /// Gets an immutable reference to the bucket which should contain
    /// this key-value pair.
    fn bucket(&self, key: &K, val: &V) -> &HashSet<(K, V)> {
        let key_hash = self.hash(key);
        let val_hash = self.hash(val);

        let (key_ind, val_ind) = {
            let dims = self.buckets.shape();
            ((key_hash as usize) % dims[0], (val_hash as usize) % dims[1])
        };

        &self.buckets[[key_ind, val_ind]]
    }

    /// Gets a mutable reference to the bucket which should contain
    /// this key-value pair.
    fn bucket_mut(&mut self, key: &K, val: &V) -> &mut HashSet<(K, V)> {
        let key_hash = self.hash(key);
        let val_hash = self.hash(val);

        let (key_ind, val_ind) = {
            let dims = self.buckets.shape();
            ((key_hash as usize) % dims[0], (val_hash as usize) % dims[1])
        };

        &mut self.buckets[[key_ind, val_ind]]
    }

    /// Creates a new `BiMultiMap` with the specified hasher which uses
    /// a bucket array with `keys` rows and `vals` columns.
    pub fn with_hasher(keys: usize, vals: usize, builder: S) -> BiMultiMap<K, V, S> {
        BiMultiMap {
            buckets: Array2::default([keys, vals]),
            builder,
        }
    }

    /// Inserts a relation between a key and a value into the map. If
    /// this relation was not present, `true` is returned. If this
    /// relation was present, `false` is returned.
    pub fn insert(&mut self, key: K, val: V) -> bool {
        self.bucket_mut(&key, &val).insert((key, val))
    }

    /// Removes a relation between a key and a value. Returns `true`
    /// if that relation was present.
    pub fn remove(&mut self, entry: &(K, V)) -> bool {
        self.bucket_mut(&entry.0, &entry.1).remove(entry)
    }

    /// Returns an iterator over all values associated with a given key.
    ///
    /// # Example
    ///
    /// ```
    /// # use bimultimap::BiMultiMap;
    /// # use std::collections::HashSet;
    /// let mut map = BiMultiMap::new(10, 10);
    /// map.insert(10, 10);
    /// map.insert(12, 32);
    /// map.insert(10, 3389283);
    ///
    /// let ten_set: HashSet<_> = [10, 3389283].into_iter().collect();
    /// assert_eq!(map.key_iter(&10).collect::<HashSet<_>>(), ten_set);
    ///
    /// let twelve_set: HashSet<_> = [32].into_iter().collect();
    /// assert_eq!(map.key_iter(&12).collect::<HashSet<_>>(), twelve_set);
    /// ```
    pub fn key_iter<'a, 'b>(&'a self, key: &'b K) -> impl Iterator<Item = &'a V> + 'b
    where
        'a: 'b,
    {
        let key_hash = self.hash(key);
        let key_ind = {
            let dims = self.buckets.shape();
            (key_hash as usize) % dims[0]
        };

        let key_buckets: ArrayView<'a, _, _> = self.buckets.subview(Axis(0), key_ind);
        key_buckets.into_iter().flat_map(move |bucket| {
            bucket
                .iter()
                .filter_map(move |(k, v)| if k == key { Some(v) } else { None })
        })
    }

    /// Returns an iterator over all keys associated with a given value.
    ///
    /// # Example
    ///
    /// ```
    /// # use bimultimap::BiMultiMap;
    /// # use std::collections::HashSet;
    /// let mut map = BiMultiMap::new(10, 10);
    /// map.insert(10, 10);
    /// map.insert(12, 10);
    /// map.insert(9, 3);
    ///
    /// let ten_set: HashSet<_> = [10, 12].into_iter().collect();
    /// assert_eq!(map.val_iter(&10).collect::<HashSet<_>>(), ten_set);
    ///
    /// let three_set: HashSet<_> = [9].into_iter().collect();
    /// assert_eq!(map.val_iter(&3).collect::<HashSet<_>>(), three_set);
    /// ```
    pub fn val_iter<'a, 'b>(&'a self, val: &'b V) -> impl Iterator<Item = &'a K> + 'b
    where
        'a: 'b,
    {
        let val_hash = self.hash(val);
        let val_ind = {
            let dims = self.buckets.shape();
            (val_hash as usize) % dims[1]
        };

        let val_buckets: ArrayView<'a, _, _> = self.buckets.subview(Axis(1), val_ind);
        val_buckets.into_iter().flat_map(move |bucket| {
            bucket
                .iter()
                .filter_map(move |(k, v)| if v == val { Some(k) } else { None })
        })
    }

    /// Returns an iterator over all key-value pairs in the map.
    pub fn iter(&self) -> impl Iterator<Item = &(K, V)> {
        self.buckets.iter().flat_map(HashSet::iter)
    }
}

impl<K, V, S> Debug for BiMultiMap<K, V, S>
where
    K: Hash + Eq + Debug,
    V: Hash + Eq + Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut d = f.debug_map();
        for &(ref k, ref v) in self.iter() {
            d.entry(k, v);
        }
        d.finish()
    }
}
