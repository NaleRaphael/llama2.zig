const std = @import("std");
const Allocator = std.mem.Allocator;

const ParseError = error{
    InvalidInputFormat,
};

/// Parse text to an array of floats. In input text, an array should be enclosed
/// by square brackets. Leading and trailing spaces will be ignored.
pub fn parseArray(
    text: *const []const u8,
    delimiter: *const u8,
    allocator: Allocator,
) !std.ArrayList(f32) {
    var result = std.ArrayList(f32).init(allocator);

    var il: usize = 0; // index of the first character of a word
    var ir: usize = 0; // index of the last character of a word
    var found: bool = false;

    // find where to start & stop iterating, and also check whether format is valid
    var istart: usize = blk: {
        for (text.*, 0..) |c, i| {
            if (c == '[') {
                break :blk i + 1;
            }
        }
        break :blk text.len;
    };

    var iend: usize = blk: {
        for (text.*[istart..], istart..) |c, i| {
            if (c == ']') {
                break :blk i + 1;
            }
        }
        break :blk text.len;
    };

    // std.debug.print("input text: {s}\n", .{text});
    if (istart == text.len or istart >= iend) {
        return ParseError.InvalidInputFormat;
    }

    for (istart..iend) |i| {
        const c = text.*[i];

        if (c == ' ' and c != delimiter.*) continue;

        if (c == ']' or c == delimiter.*) {
            ir = i;

            if (found and ir > il) {
                const val: f32 = std.fmt.parseFloat(f32, text.*[il..ir]) catch |err| {
                    std.debug.print("Invalid text to parse: '{s}'\n", .{text.*[il..ir]});
                    return err;
                };
                try result.append(val);
                found = false;
            }
        } else if (!found) {
            il = i;
            found = true;
        }
    }

    return result;
}

// Check whether 2 values are close enough (like `numpy.isclose()`). Note that
// NaN is considered not to be the same as any value.
pub fn isClose(comptime T: type, a: T, b: T, rtol: T, atol: T) bool {
    std.debug.assert(@typeInfo(T) == .Float);
    std.debug.assert(atol > 0);
    std.debug.assert(rtol > 0);

    const delta = a - b;
    const delta_abs = if (delta > 0) delta else -delta;
    const b_abs = if (b > 0) b else -b;
    const result: bool = delta_abs <= atol + rtol * b_abs;
    return result;
}

pub fn assertArrayEqual(comptime T: type, left: []const T, right: []const T) !void {
    std.testing.expect(left.len == right.len) catch |err| {
        std.log.err("Length mismatched, left: {d}, right: {d}", .{ left.len, right.len });
        return err;
    };

    for (left, right, 0..) |l, r, i| {
        std.testing.expect(isClose(f32, l, r, 1e-5, 1e-8)) catch |err| {
            std.log.err("Found different value in {d}-th element, left: {d}, right: {d}", .{ i, l, r });
            return err;
        };
    }
}
