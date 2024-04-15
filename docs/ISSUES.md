## Conditional compilation
In zig 0.11, it seems we cannot [use a local repository as a dependency][1]
since field `url` is required. That is, given settings below:
```
# in build.zig.zon
.{
    # ...
    .dependencies = .{
        .ztracy = .{
            .path = "./third_party/zig-gamedev/libs/ztracy",
        },
    },
}
```

compiler would raise this error:
```raw
error: dependency is missing 'url' field
   .ztracy = .{
              ^
```

To workaround this issue:
- In `build.zig`, `ztracy` is imported only when the flag `use_tracy` is true.
- In `run.zig`, check with the build option before importing `ztracy`, and wrap
  any function calls with the same check.

[1]: https://zig.news/fuzhouch/use-git-submodule-local-path-to-manage-dependencies-24ig

