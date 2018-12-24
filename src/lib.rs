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
        "If this happens there is a bug in the PackedIntegers implementation")
}

#[derive(Clone, Debug)]
pub struct PackedIntegers {
    index: IndexedBits<Box<[u8]>>,
    data: Box<[u64]>,
    len: usize,
}

impl PackedIntegers {
    pub fn len(&self) -> usize {
        self.len
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
        IndexedBits::build_index(
            must_have_or_bug(Bits::from(self.data.into_boxed_slice(), self.used_bits)))
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

fn pack_lsbs(chunk: &[u64], n_bits: usize, into: &mut [u64]) -> usize {
    debug_assert!(chunk.len() > 0);
    debug_assert!(chunk.len() <= 64);
    debug_assert!(n_bits > 0);
    debug_assert!(n_bits <= 64);
    debug_assert!(into.len() >= ((chunk.len() * n_bits) + 63) / 64);

    let leading_zeros = (64 - n_bits) as u32;
    let mut idx = 0;
    let mut ready_bits = 0;
    let mut building_part = 0u64;

    for &item in chunk.iter() {
        building_part |= (item << leading_zeros) >> ready_bits;
        ready_bits += n_bits;
        if ready_bits >= 64 {
            into[idx] = building_part;
            idx += 1;
            ready_bits -= 64;
            building_part = 0;

            if ready_bits > 0 {
                building_part |= item << (64 - ready_bits)
            }
        }
    }

    if ready_bits > 0 {
        into[idx] = building_part;
        idx += 1;
    }

    idx
}

impl PackedIntegers {
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
                    (bit_width, pack_lsbs(chunk, bit_width, &mut buffer))
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
            let chunk_output_length =
                pack_lsbs(chunk, chunk_bit_width, &mut writing_part[write_at..]);

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

                let output_size = {
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
    /// Returns `None` for out-of-bounds.
    ///
    /// ```
    /// use rand::prelude::*;
    /// use packed_integers::*;
    /// let unpacked: Vec<u64> = (0..139).map(|_| rand::random()).collect();
    /// let packed = PackedIntegers::from_iter(unpacked.iter().cloned());
    /// assert_eq!(Some(unpacked[7]), packed.get(7));
    /// assert_eq!(Some(unpacked[112]), packed.get(112));
    /// assert_eq!(Some(unpacked[138]), packed.get(138));
    /// assert_eq!(None, packed.get(139));
    /// assert_eq!(None, packed.get(150));
    /// ```
    pub fn get(&self, idx: usize) -> Option<u64> {
        if idx >= self.len {
            return None;
        }

        let idx_of_block = (idx / 64) as u64;
        let idx_in_block = idx % 64;

        let block_start =
            must_have_or_bug(self.index.select_ones(idx_of_block)) as usize;
        let block_fake_end =
            must_have_or_bug(self.index.select_ones(idx_of_block + 1)) as usize;
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

impl std::cmp::PartialEq for PackedIntegers {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
            && self.index.bits() == other.index.bits()
            && self.data == other.data
    }
}

impl std::cmp::Eq for PackedIntegers {}

pub struct PackedIntegersIterator<'a> {
    index: indexed_bitvec_core::bits::SetBitIndexIterator<&'a [u8]>,
    data: std::slice::Iter<'a, u64>,
    current_data: u64,
    chunk_mark: u64,
    chunk_remain: usize,
    total_remain: usize,
    chunk_bits: u32,
    current_bits: u32,
}

impl PackedIntegers {
    pub fn iter<'a>(&'a self) -> PackedIntegersIterator<'a> {
        let mut index_iter = self.index.bits().into_iter_set_bits();
        let first_bit_idx = must_have_or_bug(index_iter.next());

        PackedIntegersIterator {
            index: index_iter,
            data: self.data.iter(),
            chunk_bits: 0,
            chunk_mark: first_bit_idx,
            chunk_remain: 0,
            current_data: 0,
            current_bits: 0,
            total_remain: self.len
        }
    }
}

impl<'a> Iterator for PackedIntegersIterator<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.total_remain == 0 {
            return None
        }

        if self.chunk_remain == 0 {
            debug_assert!(self.current_bits == 0);
            let next_chunk_mark = must_have_or_bug(self.index.next());
            self.chunk_bits = (next_chunk_mark - self.chunk_mark) as u32;
            self.chunk_mark = next_chunk_mark;
            self.chunk_remain = 64;
        }

        debug_assert!(self.total_remain > 0);
        debug_assert!(self.chunk_remain > 0);
        debug_assert!(self.chunk_bits > 0);
        debug_assert!(self.chunk_bits <= 64);

        let mask = u64::max_value() >> (64 - self.chunk_bits);

        if self.current_bits == 0 {
            self.current_data = *must_have_or_bug(self.data.next());
            self.current_bits = 64;
        }

        debug_assert!(self.current_bits > 0);

        let ret =
            if self.current_bits >= self.chunk_bits {
                let ret = {
                    let shift = self.current_bits - self.chunk_bits;
                    (self.current_data >> shift) & mask
                };
                self.current_bits -= self.chunk_bits;
                ret
            } else {
                let next_data = *must_have_or_bug(self.data.next());
                let used_extra_bits = self.chunk_bits - self.current_bits;
                let ret = {
                    let from_current = self.current_data << used_extra_bits;
                    let from_next = next_data >> (64 - used_extra_bits);
                    (from_current | from_next) & mask
                };
                self.current_data = next_data;
                self.current_bits = 64 - used_extra_bits;
                ret
            };
        self.chunk_remain -= 1;
        self.total_remain -= 1;
        Some(ret)
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

    prop_compose! {
        // TODO: Direct packed data generator
        fn gen_packed(len: impl Into<SizeRange>)
            (input_data in gen_data(len))
             -> PackedIntegers {
                PackedIntegers::from_vec(input_data)
            }
    }

    #[test]
    fn iter_basic_tests() {
        fn check(data: Vec<u64>) {
            let packed = PackedIntegers::from_iter(data.iter().cloned());
            assert_eq!(data.len(), packed.len());
            let unpacked : Vec<_> = packed.iter().collect();
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

    #[test]
    fn pack_lsbs_ready_increment_bug() {
        // pack_lsbs had a bug where it was always adding 64 new bits every time
        // this would result in a panic (at least in dev build)
        let chunk = &[0u64; 2];
        let mut output = [0u64; 1];
        assert_eq!(1, pack_lsbs(chunk, 1, &mut output));
    }

    proptest! {
        #[test]
        fn from_iter_is_repeatable(data in gen_data(0..1000)) {
            let build_a = PackedIntegers::from_iter(data.iter().cloned());
            let build_b = PackedIntegers::from_iter(data.iter().cloned());

            prop_assert_eq!(true, build_a.eq(&build_b));
        }

        #[test]
        fn from_iter_from_vec_agreement(data in gen_data(0..1000)) {
            let build_a = PackedIntegers::from_iter(data.iter().cloned());
            let build_b = PackedIntegers::from_vec(data);

            prop_assert_eq!(build_a, build_b);
        }

        #[test]
        fn iter_get_agreement(packed in gen_packed(0..1000)) {
            let unpacked_a : Vec<_> = packed.iter().collect();
            let unpacked_b : Vec<_> = (0..packed.len()).map(|idx| packed.get(idx).unwrap()).collect();
            prop_assert_eq!(unpacked_a, unpacked_b);
        }
    }
}
