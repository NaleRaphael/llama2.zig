# llama2.zig
This's just a work to reimplement [llama2.c][1] in Zig, also a toy project for
me to explore Zig.

This repo would be more like a direct implementation to keep things simple and
easy to understand (at least for myself).

If you are looking for a stable & fast implementations, please consider checking
out [cgbur/llama2.zig][2] and [clebert/llama2.zig][3]!

## Requirements
- zig: 0.12.0

## Build
```bash
# XXX: Currently the build have to look up `ztracy` even if it's dependency for
# development only, so you have to fetch the submodule once.
# $ git submodule update --init --recursive

$ zig build -Doptimize=ReleaseFast
```

## Usage
Almost all arguments in [llama2.c][1] are supported except those ones related
to `chat` mode:
```bash
# For stories15M, remember to download the model and tokenizer first:
# $ wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin -P models
# $ wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin -P models

$ ./zig-out/bin/run models/stories15M.bin \
    -z models/tokenizer.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"
```
(if you want to compare the output with llama2.c, remember to specify an
identical seed)

## Tests
To run tests, it currently requires installing `PyTorch` to load checkpoint for
checking whether weights are correctly mapped.

```bash
# Remember to download the model `stories15M.pt` (PyTorch model) first:
# wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt -P models

$ zig test tests.zig
```

## Profiling
> [!NOTE]  
> Currently `ztracy` supports only zig up to 0.11.0. If you want to try it,
> please checkout to branch `dev/zig-0.11.0`.

If you want to profile the code, please fetch the submodules:
```bash
$ git submodule update --init --recursive
```

Then build the code with [`tracy`][4] enabled:
```bash
$ zig build -Doptimize=ReleaseFast -Duse_tracy=true
```

For further details, please checkout [docs/INSTALL.md](./docs/INSTALL.md).


[1]: https://github.com/karpathy/llama2.c
[2]: https://github.com/cgbur/llama2.zig
[3]: https://github.com/clebert/llama2.zig
[4]: https://github.com/wolfpld/tracy
