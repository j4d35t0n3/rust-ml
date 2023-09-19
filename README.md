# rust-ml
Rust ML test using tch-rs

# Init
Install rust: https://rustup.rs

Clone libtorch and unzip into the libtorch folder: https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip

Set your $LIBTORCH environmental variable to the path of the libtorch folder

# Build and run
## Dev
cargo run -- your_file.jpg

## Release
cargo run -r -- your_file.jpg
