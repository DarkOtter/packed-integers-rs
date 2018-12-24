#!/bin/sh
set -e
cargo test --verbose --no-default-features
cargo test --verbose --features "implement_heapsize"
