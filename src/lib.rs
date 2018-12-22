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
use indexed_bitvec::{Bits, IndexedBits};

#[derive(Clone, Debug)]
pub struct PackedIntegers {
    index: IndexedBits<Box<[u8]>>,
    data: Box<[u64]>,
    len: usize,
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
            Bits::from(self.data.into_boxed_slice(), self.used_bits).expect(
                "We should have correctly kept track of the used bits, if not it is a bug",
            ),
        )
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
        ready_bits += 64;
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
                data.reserve(output_size);
                let write_at = data.len();
                for _ in 0..output_size {
                    data.push(0)
                }

                let output_size = pack_lsbs(chunk, bit_width, &mut data[write_at..]);
                data.truncate(write_at + output_size);

                index.push_one_bit();
                index.push_zero_bits(bit_width - 1);
            };

            for item in iter {
                total_elements += 1;
                buffer[in_buffer] = item.into();
                in_buffer += 1;

                if in_buffer == 64 {
                    write_chunk(&buffer)
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

    pub fn get(&self, idx: usize) -> Option<u64> {
        if idx >= self.len {
            return None;
        }

        let idx_of_block = (idx / 64) as u64;
        let idx_in_block = idx % 64;

        let block_start = self.index.select_ones(idx_of_block).expect(
            "If we don't have as many bits set as we expect then there was a bug",
        ) as usize;
        let block_fake_end = self.index.select_ones(idx_of_block + 1).expect(
            "If we don't have as many bits set as we expect then there was a bug",
        ) as usize - block_start;
        let bit_width = block_fake_end - block_start;

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
    use core::ops::Range;

    prop_compose! {
        fn gen_chunk(len: usize)
            (bits_used in 0..=64u32,
             data in prop::collection::vec(any::<u64>(), len),
             high_idx in 0..(max(len, 1)),
             len in Just(len))
             -> Vec<u64> {
                let mut data = data;
                if bits_used < 64 {
                    let mask = u64::max_value() >> (64 - bits_used);
                    for x in data.iter_mut() {
                        *x &= mask
                    }
                }
                if bits_used > 0 && len > 0 {
                    let bit = 1u64 << (bits_used - 1);
                    data[high_idx] |= bit;
                }

                data
            }
    }

    prop_compose! {
        fn gen_data(len: impl Into<SizeRange>)
            (len in <Range<usize>>::from(len.into()))
            (whole_chunks in prop::collection::vec(gen_chunk(64), len / 64),
             partial_chunk in gen_chunk(len % 64))
             -> Vec<u64> {
                let mut chunks = whole_chunks;
                chunks.push(partial_chunk);
                let mut res = Vec::with_capacity(chunks.iter().map(|c| c.len()).sum());
                for chunk in chunks {
                    for item in chunk {
                        res.push(item)
                    }
                }
                res
            }
    }
}
