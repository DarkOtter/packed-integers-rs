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
            data: Vec::with_capacity(i / 8),
            used_bits: 0,
        }
    }

    fn into_bits(self) -> Bits<Box<[u8]>> {
        Bits::from(self.data.into_boxed_slice(), self.used_bits).expect(
            "We should have correctly kept track of the used bits, if not it is a bug",
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

fn pack_lsbs(chunk: &[u64], into: &mut [u64]) {
    debug_assert!(chunk.len() <= 64);
    debug_assert!(into.len() > 0);
    debug_assert!(into.len() <= 64);

    let leading_zeros = (64 - into.len()) as u32;
    let mut idx = 0;
    let mut ready_bits = 0;
    let mut building_part = 0u64;

    {
        let mut add_item = |item: u64| {
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
        };

        for &item in chunk.iter() {
            add_item(item);
        }

        if chunk.len() < 64 {
            for _ in 0..(64 - chunk.len()) {
                add_item(0);
            }
        }
    }

    debug_assert_eq!(ready_bits, 0);
    debug_assert_eq!(building_part, 0);
    debug_assert_eq!(idx, into.len());
}

impl PackedIntegers {
    pub fn from_vec(mut data: Vec<u64>) -> Self {
        let total_elements = data.len();
        let mut index = BuildIndex::with_capacity(total_elements / 2);
        let mut buffer = [0u64; 64];
        let mut write_at = 0;

        {
            let data_len = data.len();
            let mut write_chunk = |chunk_start, chunk_end| {
                debug_assert!(write_at <= chunk_start);
                debug_assert!(chunk_start <= chunk_end);

                let chunk_output_length = {
                    let chunk = &data[chunk_start..chunk_end];
                    let chunk_output_length = find_bit_width(chunk);
                    pack_lsbs(chunk, &mut buffer[..chunk_output_length]);
                    chunk_output_length
                };

                debug_assert!(chunk_output_length > 0);
                debug_assert!(chunk_output_length <= 64);

                index.push_one_bit();
                index.push_zero_bits(chunk_output_length - 1);

                let write_back_length = min(chunk_end - chunk_start, chunk_output_length);

                (&mut data[write_at..write_at + write_back_length])
                    .copy_from_slice(&buffer[..write_back_length]);
                write_at += write_back_length;
            };

            let n_whole_chunks = data_len / 64;
            for i in 0..n_whole_chunks {
                let chunk_start = i * 64;
                write_chunk(chunk_start, chunk_start + 64);
            }

            let last_whole_chunk_end = n_whole_chunks * 64;
            if last_whole_chunk_end < data_len {
                write_chunk(last_whole_chunk_end, data_len);
            }
        };

        index.push_one_bit();
        data.truncate(write_at);

        PackedIntegers {
            index: IndexedBits::build_index(index.into_bits()),
            data: data.into_boxed_slice(),
            len: total_elements,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
