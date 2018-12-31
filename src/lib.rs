/*
   Copyright 2018 DarkOtter

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

use std::cmp::{max, min};

extern crate indexed_bitvec;
extern crate indexed_bitvec_core;
use indexed_bitvec::{Bits, IndexedBits};

fn must_have_or_bug<T>(opt: Option<T>) -> T {
    opt.expect(
        "If this happens there is a bug in the PackedIntegers implementation",
    )
}

#[derive(Clone, Debug)]
pub struct PackedIntegers {
    index: IndexedBits<Box<[u8]>>,
    data: Box<[u64]>,
    len: usize,
}

#[cfg(feature = "implement_heapsize")]
impl heapsize::HeapSizeOf for PackedIntegers {
    fn heap_size_of_children(&self) -> usize {
        self.index.heap_size_of_children() + self.data.heap_size_of_children() +
            self.len.heap_size_of_children()
    }
}

#[derive(Clone, Debug)]
struct BuildIndex {
    data: Vec<u8>,
    used_bits: u64,
}

impl BuildIndex {
    fn new() -> Self {
        BuildIndex {
            data: Vec::new(),
            used_bits: 0,
        }
    }

    fn with_capacity(i: usize) -> Self {
        BuildIndex {
            data: Vec::with_capacity((i + 8) / 8),
            used_bits: 0,
        }
    }

    fn into_indexed_bits(self) -> IndexedBits<Box<[u8]>> {
        IndexedBits::build_index(must_have_or_bug(
            Bits::from(self.data.into_boxed_slice(), self.used_bits),
        ))
    }

    fn push_one_bit(&mut self) {
        let within_byte = self.used_bits % 8;
        self.used_bits += 1;
        if within_byte == 0 {
            self.data.push(0x80);
        } else {
            let byte = self.data.len() - 1;
            self.data[byte] |= 0x80u8 >> within_byte;
        }
    }

    fn push_zero_bits(&mut self, n_zeros: usize) {
        if n_zeros == 0 {
            return;
        }
        self.used_bits += n_zeros as u64;
        let need_bytes = (self.used_bits + 7) / 8;
        let add_bytes = need_bytes as usize - self.data.len();
        for _ in 0..add_bytes {
            self.data.push(0)
        }
    }
}

fn find_bit_width(chunk: &[u64]) -> usize {
    let leading_zeros = chunk.iter().map(|x| x.leading_zeros()).min().unwrap_or(0);
    max(1, 64 - leading_zeros as usize)
}

fn compressed_length(n: usize, n_bits: usize) -> usize {
    ((n * n_bits) + 63) / 64
}

fn pack_lsbs(chunk: &[u64], n_bits: usize, into: &mut [u64]) {
    debug_assert!(chunk.len() > 0);
    debug_assert!(chunk.len() <= 64);
    debug_assert!(n_bits > 0);
    debug_assert!(n_bits <= 64);
    debug_assert!(into.len() == compressed_length(chunk.len(), n_bits));

    let leading_zeros = (64 - n_bits) as u32;
    let mut into = into.iter_mut();
    let mut ready_bits = 0;
    let mut building_part = 0u64;

    for &item in chunk.iter() {
        building_part |= (item << leading_zeros) >> ready_bits;
        ready_bits += n_bits;
        if ready_bits >= 64 {
            *(must_have_or_bug(into.next())) = building_part;
            ready_bits -= 64;
            building_part = 0;

            if ready_bits > 0 {
                building_part |= item << (64 - ready_bits)
            }
        }
    }

    if ready_bits > 0 {
        *(must_have_or_bug(into.next())) = building_part;
    }
}

fn unpack_lsbs(from: &[u64], n_bits: usize, chunk: &mut [u64]) {
    debug_assert!(chunk.len() > 0);
    debug_assert!(chunk.len() <= 64);
    debug_assert!(n_bits > 0);
    debug_assert!(n_bits <= 64);
    debug_assert!(from.len() == compressed_length(chunk.len(), n_bits));

    let n_bits = n_bits as u32;
    let leading_zeros = 64 - n_bits;
    let mut chunk = chunk.iter_mut();
    let mut ready_bits = 0;
    let mut reading_part = 0u64;

    'outer: for &item in from.iter() {
        if ready_bits > 0 {
            debug_assert!(n_bits > ready_bits);
            let overflow = n_bits - ready_bits;
            reading_part |= item >> ready_bits;
            *(must_have_or_bug(chunk.next())) = reading_part >> leading_zeros;
            reading_part = item << overflow;
            ready_bits = 64 - overflow;
        } else {
            reading_part = item;
            ready_bits = 64;
        }

        loop {
            if ready_bits < n_bits {
                continue 'outer;
            };
            match chunk.next() {
                None => break 'outer,
                Some(write_to) => {
                    *write_to = reading_part >> leading_zeros;
                    if n_bits < 64 {
                        reading_part <<= n_bits;
                    } else {
                        reading_part = 0;
                    }
                    ready_bits -= n_bits;
                }
            }
        }
    }
}

impl PackedIntegers {
    /// The number of integers packed into the array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Build a packed array of integers from a vector.
    ///
    /// This re-uses the space the vector is allocated in while
    /// building the packed array, though it may involve a re-allocation
    /// of the space to reduce the size at the end (it should never
    /// increase this allocation, though the additional size of the separate
    /// index may mean more space is used overall in some cases).
    ///
    /// ```
    /// use packed_integers::*;
    /// let input_data: Vec<u64> = vec![5, 138, 10, 90, 242, 312, 541, 48];
    /// let packed = PackedIntegers::from_vec(input_data.clone());
    /// assert_eq!(input_data.len(), packed.len());
    /// for i in 0..input_data.len() {
    ///     assert_eq!(Some(input_data[i]), packed.get(i));
    /// }
    /// ```
    pub fn from_vec(mut data: Vec<u64>) -> Self {
        let total_elements = data.len();
        let mut index = BuildIndex::with_capacity(total_elements);

        fn build_initially_with_buffer(
            data: &mut Vec<u64>,
            index: &mut BuildIndex,
        ) -> (usize, usize) {
            let total_elements = data.len();
            let mut buffer = [0u64; 64];
            let mut read_from = 0;
            let mut write_at = 0;

            while read_from < total_elements && write_at + 64 > read_from {
                let chunk_start = read_from;
                let chunk_end = min(total_elements, chunk_start + 64);

                let (chunk_bit_width, chunk_output_length) = {
                    let chunk = &data[chunk_start..chunk_end];
                    let bit_width = find_bit_width(chunk);
                    let output_length = compressed_length(chunk.len(), bit_width);
                    pack_lsbs(chunk, bit_width, &mut buffer[..output_length]);
                    (bit_width, output_length)
                };

                debug_assert!(chunk_bit_width > 0);
                debug_assert!(chunk_bit_width <= 64);
                debug_assert!(chunk_output_length > 0);
                debug_assert!(chunk_output_length <= 64);

                index.push_one_bit();
                index.push_zero_bits(chunk_bit_width - 1);

                (&mut data[write_at..write_at + chunk_output_length])
                    .copy_from_slice(&buffer[..chunk_output_length]);

                read_from = chunk_end;
                write_at += chunk_output_length;
            }

            (read_from, write_at)
        }

        let (mut read_from, mut write_at) = build_initially_with_buffer(&mut data, &mut index);

        while read_from < total_elements {
            debug_assert!(write_at + 64 <= read_from);

            let chunk_start = read_from;

            let (writing_part, reading_part) = data.as_mut_slice().split_at_mut(chunk_start);
            let chunk_len = min(reading_part.len(), 64);

            let chunk = &reading_part[..chunk_len];
            let chunk_bit_width = find_bit_width(chunk);
            let chunk_output_length = compressed_length(chunk.len(), chunk_bit_width);
            pack_lsbs(
                chunk,
                chunk_bit_width,
                &mut writing_part[write_at..write_at + chunk_output_length],
            );

            debug_assert!(chunk_bit_width > 0);
            debug_assert!(chunk_bit_width <= 64);
            debug_assert!(chunk_output_length > 0);
            debug_assert!(chunk_output_length <= 64);

            index.push_one_bit();
            index.push_zero_bits(chunk_bit_width - 1);

            read_from = chunk_start + chunk_len;
            write_at += chunk_output_length;

            debug_assert!(write_at + 64 <= read_from);
        }

        index.push_one_bit();
        data.truncate(write_at);

        PackedIntegers {
            index: index.into_indexed_bits(),
            data: data.into_boxed_slice(),
            len: total_elements,
        }
    }

    pub fn from_iter<I, T>(iter: I) -> Self
    where
        T: Into<u64>,
        I: IntoIterator<Item = T>,
    {
        let iter = iter.into_iter();
        let size_hint = match iter.size_hint() {
            (min, None) => min,
            (_, Some(max)) => max,
        };
        let mut index = if size_hint >= 64 {
            BuildIndex::with_capacity(size_hint)
        } else {
            BuildIndex::new()
        };
        let mut data = if size_hint >= 64 {
            Vec::with_capacity(64)
        } else {
            Vec::new()
        };

        let mut buffer = [0u64; 64];
        let mut in_buffer = 0;
        let mut total_elements = 0;

        {
            let mut write_chunk = |chunk: &[u64]| {
                debug_assert!(chunk.len() > 0);
                debug_assert!(chunk.len() <= 64);

                let bit_width = find_bit_width(chunk);
                debug_assert!(bit_width > 0);
                debug_assert!(bit_width <= 64);

                let output_size = ((bit_width * chunk.len()) + 63) / 64;
                debug_assert!(output_size > 0);
                debug_assert!(output_size <= 64);
                debug_assert!(output_size <= chunk.len());

                data.reserve(output_size);
                let write_at = data.len();
                for _ in 0..output_size {
                    data.push(0)
                }

                let output_size = compressed_length(chunk.len(), bit_width);
                {
                    let write_into = &mut data[write_at..];
                    pack_lsbs(chunk, bit_width, write_into)
                };
                data.truncate(write_at + output_size);

                index.push_one_bit();
                index.push_zero_bits(bit_width - 1);
            };

            for item in iter {
                total_elements += 1;
                buffer[in_buffer] = item.into();
                in_buffer += 1;

                if in_buffer == buffer.len() {
                    write_chunk(&buffer);
                    in_buffer = 0;
                }
            }

            if in_buffer > 0 {
                write_chunk(&buffer[..in_buffer])
            }
        }

        index.push_one_bit();

        PackedIntegers {
            index: index.into_indexed_bits(),
            data: data.into_boxed_slice(),
            len: total_elements,
        }
    }

    /// Get the integer at a specific index.
    ///
    /// This operation should be fast, and does not involve
    /// unpacking many others of the packed integers or using
    /// a lot of memory for the unpacking.
    ///
    /// Returns `None` for out-of-bounds.
    ///
    /// ```
    /// use rand::prelude::*;
    /// use packed_integers::*;
    /// let mut rng = rand::thread_rng();
    /// let len = rng.gen_range(137, 549);
    /// let unpacked: Vec<u64> = (0..len).map(|_| rng.gen()).collect();
    /// let packed = PackedIntegers::from_iter(unpacked.iter().cloned());
    /// for i in 0..len {
    ///     assert_eq!(Some(unpacked[i]), packed.get(i));
    /// }
    /// assert_eq!(None, packed.get(len));
    /// assert_eq!(None, packed.get(len + 5));
    /// ```
    pub fn get(&self, idx: usize) -> Option<u64> {
        if idx >= self.len {
            return None;
        }

        let idx_of_block = (idx / 64) as u64;
        let idx_in_block = idx % 64;

        let block_start = must_have_or_bug(self.index.select_ones(idx_of_block)) as usize;
        let block_fake_end = must_have_or_bug(self.index.select_ones(idx_of_block + 1)) as usize;
        let bit_width = block_fake_end - block_start;
        debug_assert!(bit_width > 0);
        debug_assert!(bit_width <= 64);

        let bit_idx_in_block = idx_in_block * bit_width;

        let whole_words_offset = bit_idx_in_block / 64;
        let in_word_offset = bit_idx_in_block % 64;

        let first_part = {
            (self.data[block_start + whole_words_offset] << in_word_offset) >> (64 - bit_width)
        };

        if in_word_offset + bit_width <= 64 {
            Some(first_part)
        } else {
            Some(
                first_part |
                    (self.data[block_start + whole_words_offset + 1] >>
                         (128 - (in_word_offset + bit_width))),
            )
        }
    }
}

trait NextChunkOfInts {
    fn take_ints(&mut self, upto_n: usize) -> Option<&[u64]>;
}

#[derive(Clone, Debug)]
struct BorrowData<'a>(&'a [u64]);

impl<'a> NextChunkOfInts for BorrowData<'a> {
    fn take_ints(&mut self, upto_n: usize) -> Option<&[u64]> {
        debug_assert!(upto_n > 0);
        debug_assert!(upto_n <= 64);
        let l = min(upto_n, self.0.len());
        if l == 0 {
            return None;
        };
        let (ret, remaining) = self.0.split_at(l);
        self.0 = remaining;
        Some(ret)
    }
}

#[derive(Clone, Debug)]
struct ConsumeData {
    data: Box<[u64]>,
    consume_from: usize,
}

impl NextChunkOfInts for ConsumeData {
    fn take_ints(&mut self, upto_n: usize) -> Option<&[u64]> {
        debug_assert!(upto_n > 0);
        debug_assert!(upto_n <= 64);
        let start = self.consume_from;
        let end = min(start + upto_n, self.data.len());
        if end <= start {
            return None;
        };
        self.consume_from = end;
        Some(&self.data[start..end])
    }
}

struct GenericIterator<I, D> {
    index: I,
    data: D,
    chunk: [u64; 64],
    remaining_ints: usize,
    in_chunk_idx: usize,
    last_chunk_marker: usize,
}

impl<I: Iterator<Item = u64>, D: NextChunkOfInts> GenericIterator<I, D> {
    fn new(index: I, data: D, len: usize) -> Self {
        let mut res = GenericIterator {
            index,
            data,
            chunk: [0; 64],
            remaining_ints: len,
            in_chunk_idx: 0,
            last_chunk_marker: 0,
        };

        res.last_chunk_marker = must_have_or_bug(res.index.next()) as usize;
        debug_assert_eq!(0, res.last_chunk_marker);
        res.in_chunk_idx = res.chunk.len();

        res
    }

    fn prepare_chunk(&mut self) {
        if self.remaining_ints > 0 && self.in_chunk_idx >= self.chunk.len() {
            let next_chunk_marker = must_have_or_bug(self.index.next()) as usize;
            let chunk_bit_width = next_chunk_marker - self.last_chunk_marker;
            let chunk_output_size = min(self.remaining_ints, self.chunk.len());

            self.last_chunk_marker = next_chunk_marker;
            self.in_chunk_idx = self.chunk.len() - chunk_output_size;

            unpack_lsbs(
                must_have_or_bug(self.data.take_ints(chunk_bit_width)),
                chunk_bit_width,
                &mut self.chunk[self.in_chunk_idx..],
            );
        }
    }
}

impl<I: Iterator<Item = u64>, D: NextChunkOfInts> Iterator for GenericIterator<I, D> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.remaining_ints <= 0 {
            return None;
        }

        self.prepare_chunk();

        let ret = self.chunk[self.in_chunk_idx];
        self.in_chunk_idx += 1;
        self.remaining_ints -= 1;
        Some(ret)
    }

    fn for_each<F: FnMut(u64)>(mut self, f: F) {
        let mut f = f;
        let mut run_chunk = |s: &mut Self| {
            (&s.chunk[s.in_chunk_idx..]).into_iter().for_each(|&x| f(x));
            s.remaining_ints -= s.chunk.len() - s.in_chunk_idx;
            s.in_chunk_idx = s.chunk.len();
        };

        if self.in_chunk_idx < self.chunk.len() {
            run_chunk(&mut self);
        }

        while self.remaining_ints > 0 {
            self.prepare_chunk();
            run_chunk(&mut self);
        }
    }
}

pub struct Iter<'a>(GenericIterator<indexed_bitvec_core::bits::SetBitIndexIterator<&'a [u8]>, BorrowData<'a>>);

impl<'a> Iterator for Iter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        self.0.next()
    }

    fn for_each<F: FnMut(u64)>(self, f: F) {
        self.0.for_each(f)
    }
}

impl PackedIntegers {
    fn iter<'a>(&'a self) -> Iter<'a> {
        Iter(GenericIterator::new(
            self.index.bits().into_iter_set_bits(),
            BorrowData(&self.data[..]),
            self.len,
        ))
    }
}

impl<'a> IntoIterator for &'a PackedIntegers {
    type IntoIter = Iter<'a>;

    type Item = u64;

    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

pub struct IntoIter(GenericIterator<indexed_bitvec_core::bits::SetBitIndexIterator<Box<[u8]>>, ConsumeData>);

impl Iterator for IntoIter {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        self.0.next()
    }

    fn for_each<F: FnMut(u64)>(self, f: F) {
        self.0.for_each(f)
    }
}

impl IntoIterator for PackedIntegers {
    type IntoIter = IntoIter;

    type Item = u64;

    fn into_iter(self) -> IntoIter {
        IntoIter(GenericIterator::new(
            self.index.decompose().into_iter_set_bits(),
            ConsumeData {
                data: self.data,
                consume_from: 0,
            },
            self.len,
        ))
    }
}

use std::cmp::{Ordering, Ord};

impl std::cmp::Ord for PackedIntegers {
    fn cmp(&self, other: &Self) -> Ordering {
        let mut self_iter = self.iter();
        let mut other_iter = other.iter();

        loop {
            let self_int = match self_iter.next() {
                Some(i) => i,
                None => {
                    match other_iter.next() {
                        None => return Ordering::Equal,
                        Some(_) => return Ordering::Less,
                    }
                }
            };
            let other_int = match other_iter.next() {
                Some(i) => i,
                None => return Ordering::Greater,
            };
            match self_int.cmp(&other_int) {
                Ordering::Equal => continue,
                c => return c,
            }
        }
    }
}

impl std::cmp::PartialOrd for PackedIntegers {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Eq for PackedIntegers {}

impl std::cmp::PartialEq for PackedIntegers {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.cmp(other) == Ordering::Equal
    }
}

#[cfg(test)]
impl PackedIntegers {
    fn compute_len(&self) -> (usize, usize) {
        if self.data.len() == 0 {
            return (0, 0);
        }

        let init_chunks = self.index.rank_ones(self.data.len() as u64).unwrap() - 1;
        let last_chunk_width = self.index.select_ones(init_chunks + 1).unwrap() -
            self.index.select_ones(init_chunks).unwrap();
        let last_chunk_width = last_chunk_width as usize;
        let last_chunk_data_len = self.data.len() -
            self.index.select_ones(init_chunks).unwrap() as usize;

        let max_last_chunk_len = (last_chunk_data_len * 64) / last_chunk_width;
        let min_last_chunk_len = (((last_chunk_data_len - 1) * 64 + 1) + (last_chunk_width - 1)) /
            last_chunk_width;
        let init_chunks_len = init_chunks as usize * 64;
        (
            init_chunks_len + min_last_chunk_len,
            init_chunks_len + max_last_chunk_len,
        )
    }

    fn invariant(&self) -> Result<(), String> {
        match self.index.bits().get(0) {
            None => Err("Index is empty")?,
            Some(false) => Err("First index bit is not set")?,
            Some(true) => (),
        };

        if self.index.bits().used_bits() <= self.data.len() as u64 {
            Err("Index is not long enough")?;
        } else if self.index.bits().used_bits() > self.data.len() as u64 + 64 {
            Err("Index is longer than it should be")?;
        }

        let count_in_data_range = self.index.rank_ones(self.data.len() as u64).unwrap();
        let count_overall = self.index.count_ones();

        if count_overall != count_in_data_range + 1 {
            Err("Expected exactly one but after the data range")?;
        }

        let mut iter = self.index.bits().into_iter_set_bits();
        let mut last_set_index = iter.next().unwrap();

        debug_assert!(last_set_index == 0);
        for set_index in iter {
            if set_index > last_set_index + 64 {
                Err(format!("Gap too large in index at {}", set_index))?
            }
            last_set_index = set_index;
        }

        let (low, high) = self.compute_len();

        if self.len() < low || self.len() > high {
            Err(format!(
                "Length does not match computing length: actual {}, expected between {} and {}",
                self.len(),
                low, high,
            ))?;
        }

        Ok(())
    }
}

#[cfg(test)]
extern crate rand;
#[cfg(test)]
#[macro_use]
extern crate proptest;

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use proptest::collection::SizeRange;
    use proptest::collection::vec as gen_vec;

    prop_compose! {
        fn gen_data(len: impl Into<SizeRange>)
            (data in gen_vec(any::<u64>(), len))
            (chunk_info in gen_vec((0..=64u32, 0..64usize), (data.len() + 63) / 64),
             data in Just(data)
            )
            -> Vec<u64> {
                let mut data = data;

                let iter = 
                data
                    .chunks_mut(64)
                    .zip(chunk_info.into_iter());
                for (chunk, (bits_used, high_idx)) in iter {
                    if bits_used < 64 {
                        let mask =
                            if bits_used == 0 { 0 } else {
                                u64::max_value() >> (64 - bits_used)
                            };
                        for x in chunk.iter_mut() {
                            *x &= mask
                        }
                    }
                    if bits_used > 0 && chunk.len() > 0 {
                        let bit = 1u64 << (bits_used - 1);
                        chunk[high_idx % chunk.len()] |= bit;
                    }
                }

                data
            }
    }

    proptest! {
        #[test]
        fn from_vec_matches_invariants(data in gen_data(0..1000)) {
            let packed = PackedIntegers::from_vec(data);
            prop_assert_eq!(Ok(()), packed.invariant());
        }

        #[test]
        fn from_vec_data_not_larger(data in gen_data(0..1000)) {
            let data_len = data.len();
            let packed = PackedIntegers::from_vec(data);
            prop_assert!(packed.data.len() <= data_len);
        }

        #[test]
        fn from_iter_matches_invariants(data in gen_data(0..1000)) {
            let packed = PackedIntegers::from_iter(data.into_iter());
            prop_assert_eq!(Ok(()), packed.invariant());
        }

        #[test]
        fn from_iter_from_vec_agreement(data in gen_data(0..1000)) {
            let from_iter = PackedIntegers::from_iter(data.iter().cloned());
            let from_vec = PackedIntegers::from_vec(data);
            prop_assert_eq!(from_iter, from_vec);
        }
    }

    fn set_bits(bitvec: &mut Vec<u8>, at_indexes: Vec<u64>) {
        for set_at in at_indexes {
            let byte = bitvec.get_mut((set_at / 8) as usize).unwrap();
            let bit = 0x80 >> (set_at % 8);
            *byte = *byte | bit
        }
    }

    fn ensure_appropriate_set_bits(mut index_bytes: Vec<u8>) -> Vec<u8> {
        set_bits(&mut index_bytes, vec![0]);

        let index_bits_len = index_bytes.len() as u64 * 8;
        let index_bits = Bits::from(index_bytes, index_bits_len).unwrap();

        let mut break_runs_at = Vec::with_capacity(64);

        {
            let mut iter = index_bits.iter_set_bits();
            let mut last_set_index = iter.next().unwrap();
            debug_assert_eq!(0, last_set_index);
            for set_index in iter {
                if set_index > last_set_index + 64 {
                    break_runs_at.push(last_set_index + 64);
                }
                last_set_index = set_index;
            }
        }

        let (mut index_bytes, _) = index_bits.decompose();
        set_bits(&mut index_bytes, break_runs_at);

        index_bytes
    }

    prop_compose! {
        // TODO: Direct packed data generator
        fn gen_packed(len: impl Into<SizeRange>)
            (data in gen_vec(any::<u64>(), len))
            (index in gen_vec(any::<u8>(), ((data.len() + 7) / 8) + 8),
             len_helper in 0..64usize,
             data in Just(data))
             -> PackedIntegers {

                let index = ensure_appropriate_set_bits(index);

                let index_used_bits = {
                    let set_bits_in_data_range =
                        Bits::from(&index[..], data.len() as u64)
                        .unwrap()
                        .count_ones();
                    let first_set_bit_index_after_data_range =
                        Bits::from(&index[..], index.len() as u64 * 8)
                        .unwrap()
                        .select_ones(set_bits_in_data_range)
                        .unwrap();
                    first_set_bit_index_after_data_range + 1
                };

                let index =
                    IndexedBits::build_index(
                        Bits::from(index.into_boxed_slice(), index_used_bits)
                            .unwrap());

                let mut res =
                PackedIntegers {
                    index,
                    data: data.into_boxed_slice(),
                    // Temporary length
                    len: 0,
                };

                let (low_len, high_len) = res.compute_len();
                if low_len == high_len {
                    res.len = low_len
                } else {
                    res.len = low_len + (len_helper % (high_len - low_len));
                }

                res
            }
    }

    proptest! {
        #[test]
        fn packed_gen_matches_invariants(packed in gen_packed(0..1000)) {
            prop_assert_eq!(Ok(()), packed.invariant());
        }

        #[test]
        fn from_vec_iter_roundtrip(data in gen_data(0..1000)) {
            let packed = PackedIntegers::from_vec(data.clone());
            let unpacked: Vec<_> = packed.iter().collect();
            prop_assert_eq!(data, unpacked);
        }
    }


    /*

    #[test]
    fn iter_basic_tests() {
        fn check(data: Vec<u64>) {
            let packed = PackedIntegers::from_iter(data.iter().cloned());
            assert_eq!(data.len(), packed.len());
            let unpacked: Vec<_> = packed.iter().collect();
            assert_eq!(data, unpacked);
        }

        check(vec![]);
        check(vec![0]);
        check(vec![u64::max_value()]);
        check(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        check(vec![42; 424]);
    }

    proptest! {
        #[test]
        fn pack_iter_round_trip(data in gen_data(0..1000)) {
            let packed = PackedIntegers::from_iter(data.iter().cloned());
            prop_assert_eq!(data.len(), packed.len());
            let unpacked: Vec<_> = packed.iter().collect();
            prop_assert_eq!(data, unpacked);
        }

        #[test]
        fn iter_pack_round_trip(packed in gen_packed(0..1000)) {
            let unpacked: Vec<_> = packed.iter().collect();
            let repacked = PackedIntegers::from_iter(unpacked.iter().cloned());
            prop_assert_eq!(packed, repacked);
        }
    }

    proptest! {
        #[test]
        fn from_iter_is_repeatable(data in gen_data(0..1000)) {
            let build_a = PackedIntegers::from_iter(data.iter().cloned());
            let build_b = PackedIntegers::from_iter(data.iter().cloned());

            prop_assert!(build_a.structural_eq(&build_b));
        }

        #[test]
        fn from_iter_from_vec_agreement(data in gen_data(0..1000)) {
            let build_a = PackedIntegers::from_iter(data.iter().cloned());
            let build_b = PackedIntegers::from_vec(data);

            prop_assert!(build_a.structural_eq(&build_b));
        }

        #[test]
        fn iter_get_agreement(packed in gen_packed(0..1000)) {
            let unpacked_a : Vec<_> = packed.iter().collect();
            let unpacked_b : Vec<_> = (0..packed.len()).map(|idx| packed.get(idx).unwrap()).collect();
            prop_assert_eq!(unpacked_a, unpacked_b);
        }

        #[test]
        fn iter_next_for_each_agreement(packed in gen_packed(0..1000)) {
            let mut unpacked_a = Vec::with_capacity(packed.len());
            {
                let mut iter = packed.iter();
                loop {
                    match iter.next() {
                        None => break,
                        Some(x) => unpacked_a.push(x),
                    }
                }
            };

            let mut unpacked_b = Vec::with_capacity(packed.len());
            packed.iter().for_each(|x| unpacked_b.push(x));

            prop_assert_eq!(unpacked_a, unpacked_b);
        }
    }
    
     */
}
