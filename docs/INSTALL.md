## Tools for development
### [tracy][gh_tracy]
- [ztracy][gh_ztracy]: Zig bindings for tracy client.
    - Fetch the submodule and build with flag `-Duse_tracy=true`.
- Bulid `tracy` server (it's recommended to checkout section "2.3 Building the
    server" in [tracy manuals][pdf_tracy_manuals]):
    ```bash
    $ git clone https://github.com/wolfpld/tracy
    $ cd tracy
    # We are using ztracy 0.12 which works for tracy 0.11
    $ git checkout v0.11.0

    # Specify source and binary directory, and pass compile options
    # - If you are not using Wayland, you might need add flag `-DLEGACY=ON`
    # - If you ran into any error and required modifying options, remember to
    #   delete folder "profiler/build" and rerun this command again.
    $ cmake -B profiler/build -S profiler -DCMAKE_BUILD_TYPE=Release

    $ cmake --build profiler/build --config Release --parallel
    ```
    The build might fail because of missing/outdated libraries, consider my
    case to see whether it helps:
    ```bash
    # OS: Ubuntu 20.04 (docker image: nvidia/cudagl:11.4.2-devel-ubuntu20.04)

    # Here are the missing libraries I need to install:
    # (It really depends. You can check with the error message while building tracy.)
    $ sudo apt install libtbb-dev libfreetype6-dev libdbus-glib-1-dev libwayland-dev wayland-protocols

    # Also, I cannot build in non-legacy mode (same error message as it's mentioned
    # in tracy issues#582), so `glfw` is required.
    $ sudo apt install libglfw3-dev

    # `capstone` will be downloaded by CPM if it doesn't exist, so we don't need
    # to build it manually as it's done for ztracy 0.11.

    # Rerun the build command to see whether it works.
    ```

[gh_tracy]: https://github.com/wolfpld/tracy
[gh_ztracy]: https://github.com/zig-gamedev/zig-gamedev/tree/main/libs/ztracy
[pdf_tracy_manuals]: https://github.com/wolfpld/tracy/releases/download/v0.10/tracy.pdf

