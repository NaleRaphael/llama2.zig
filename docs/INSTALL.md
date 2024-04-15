## Tools for development
### [tracy][gh_tracy]
- [ztracy][gh_ztracy]: Zig bindings for tracy client.
    - Run `scripts/setup_dev_deps.sh` to clone it.
- Bulid `tracy` server (it's recommended to checkout section "2.3 Building the
    server" in [tracy manuals][pdf_tracy_manuals]):
    ```bash
    $ git clone https://github.com/wolfpld/tracy
    $ cd tracy
    $ git checkout v0.10
    $ cd profiler/build/unix
    $ make -j8 release
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

    # And the version of `capstone` on Ubuntu 20.04 (4.0.1+really+3.0.5-1build1) is
    # outdated for tracy v0.10 (similar to tracy issue#484). Since I don't want to
    # manipulate the system-side packages, here is my solution:
    # 1. Clone libcapstone to local (inside tracy folder)
    $ cd tracy
    $ mkdir third_party && cd third_party
    $ git clone https://github.com/libcapstone/libcapstone
    # (9a486f5 is the latest commit for me at this moment, you can pick a stable tag)
    $ cd libcapstone && git checkout 9a486f5

    # 2. Build libcapstone and install artifacts to `build/dist`
    $ mkdir -p build && cd build
    $ cmake \
        -DCMAKE_INSTALL_PREFIX=dist \
        -DCAPSTONE_BUILD_TESTS=OFF \
        -DCAPSTONE_INSTALL=ON \
        ..
    $ make -j8
    $ cmake -P cmake_install.cmake

    # 3. Go back to the folder of tracy server
    $ cd ../../profiler/build/unix

    # 4. In `legacy.mk`, replace dependency `capstone` with path of `capstone.pc`, e.g.,
    # ```
    # LIBCAPSTONE := ../../../third_party/libcapstone/build/dist/lib/pkgconfig/capstone.pc
    # INCLUDES := $(shell pkg-config --cflags glfw3 freetype2 $(LIBCAPSTONE)) -I../../../imgui
    # LIBS := $(shell pkg-config --libs glfw3 freetype2 $(LIBCAPSTONE)) -lpthread -ldl
    # ```

    # 5. Build tracy server
    $ make -j8 LEGACY=1

    # 6. Note that we need to set LD_LIBRARY_PATH while running the executable
    # since shared libraries of `libcapstone` are not installed in default paths.
    $ DIR_LIBCAPSTONE_DIST=`realpath ../../../third_party/libcapstone/build/dist`
    $ LD_LIBRARY_PATH="${DIR_LIBCAPSTONE_DIST}/lib" ./Tracy-release
    ```

[gh_tracy]: https://github.com/wolfpld/tracy
[gh_ztracy]: https://github.com/zig-gamedev/zig-gamedev/tree/main/libs/ztracy
[pdf_tracy_manuals]: https://github.com/wolfpld/tracy/releases/download/v0.10/tracy.pdf

