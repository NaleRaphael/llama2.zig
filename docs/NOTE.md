## TODOs
- Support int8 & q80 inference?
- Use `@Vector` for SIMD


### Header
- Header in hex (using command `$ xxd -l 32 stories15M.bin`)
    ```hex
    00000000: 2001 0000 0003 0000 0600 0000 0600 0000
    00000010: 0600 0000 007d 0000 0001 0000          
    ```
- Header definitions:
    ```python
    # In `legacy_export()`
    # byte order: little endian (default)
    header = struct.pack(
        'iiiiiii',
        p.dim,          # 288   (0x120)
        hidden_dim,     # 768   (0x300)
        p.n_layers,     # 6     (0x6)
        p.n_heads,      # 6     (0x6)
        n_kv_heads,     # 6     (0x6)
        p.vocab_size,   # 32000 (0x7d00)
        p.max_seq_len,  # 256   (0x100)
    )
    ```

### Tokenizer file
```python
# llama2.c/tokenizer.py
with open(tokenizer_bin, 'wb') as f:
    f.write(struct.pack("I", max_token_length))
    for bytes, score in zip(tokens, scores):
        f.write(struct.pack("fI", score, len(bytes)))
        f.write(bytes)

# First five tokens:
# score: 0.0, len: 5, bytes: b'<unk>'
# score: 0.0, len: 5, bytes: b'\n<s>\n'  # postprocessed, see `Tokenizer.export()`
# score: 0.0, len: 6, bytes: b'\n</s>\n' # postprocessed
# score: 0.0, len: 6, bytes: b'<0x00>'
# score: 0.0, len: 6, bytes: b'<0x01>'
```

```raw
# tokenizer.bin (`$ xxd -l 32 tokenizer.bin`)
          <-  1  ->
00000000: 1b00 0000 0000 0000 0500 0000 3c75 6e6b  ............<unk
00000010: 3e00 0000 0005 0000 000a 3c73 3e0a 0000  >.........<s>...

# 1b00 0000: max_token_length (i32) = 0x0000001b = 27
# 0000 0000: "score of tokens[0]" (f32) = 0.0
# 0500 0000: "length of tokens[0]" (i32) = 0x00000005 = 5
# 3c75 6e6b 3e00: "tokens[0]" (byte) = [3c, 75, 6e, 6b, 3e]
#            = ['<', 'u', 'n', 'k', '>']

Note that we would only read "length of tokens[n]" bytes for "tokens[n]"
```

### C vs zig
- `fread` vs `read`?
```c
// To read an int (i32)
fread(&t->max_token_length, sizeof(int), 1, file)
```

```zig
// Allocate a buffer first (4 bytes for i32)
var buf_x32: [4]u8 = undefined;
var buffered_file = std.io.bufferedReader(file.reader());
var nb_read = try buffered_file.read(&buf_x32);

// Convert bytes to single i32 (endianness should be considered)
var value: i32 = std.mem.readIntSliceLittle(u32, &buf_x32);
```

