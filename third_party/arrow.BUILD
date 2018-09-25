# Description:
#   Apache Arrow library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE.txt"])

load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

flatbuffer_cc_library(
    name = "file_fbs",
    srcs = [
        "format/File.fbs",
    ],
    includes = [
        ":schema_fbs",
    ]
)

flatbuffer_cc_library(
    name = "message_fbs",
    srcs = [
        "format/Message.fbs",
    ],
    includes = [
        ":schema_fbs",
        ":tensor_fbs",
    ]
)

flatbuffer_cc_library(
    name = "schema_fbs",
    srcs = [
        "format/Schema.fbs",
    ],
)

flatbuffer_cc_library(
    name = "tensor_fbs",
    srcs = [
        "format/Tensor.fbs",
    ],
    includes = [
        ":schema_fbs",
    ]
)

cc_library(
    name = "arrow",
    srcs = glob([
        "cpp/src/arrow/*.cc",
        "cpp/src/arrow/*.h",
        "cpp/src/arrow/io/*.cc",
        "cpp/src/arrow/io/*.h",
        "cpp/src/arrow/ipc/*.cc",
        "cpp/src/arrow/ipc/*.h",
        "cpp/src/arrow/util/*.cc",
        "cpp/src/arrow/util/*.h",
    ],
    exclude=[
        "cpp/src/arrow/**/*-test.cc",
        "cpp/src/arrow/**/*benchmark*.cc",
        "cpp/src/arrow/**/*hdfs*",
        "cpp/src/arrow/util/compression_zstd.*",
        "cpp/src/arrow/util/compression_lz4.*",
        "cpp/src/arrow/util/compression_brotli.*",
        "cpp/src/arrow/ipc/feather.*",
        "cpp/src/arrow/ipc/json*",
        #"cpp/src/arrow/ipc/message.cc",
        #"cpp/src/arrow/ipc/writer.cc",
        #"cpp/src/arrow/ipc/reader.cc",
        #"cpp/src/arrow/ipc/metadata-internal.*",
        "cpp/src/arrow/ipc/stream-to-file.cc",
        "cpp/src/arrow/ipc/file-to-stream.cc",       
    ]),
    hdrs = [
    ],
    data = [
        "format/File.fbs",
        "format/Message.fbs",
        "format/Schema.fbs",
        "@flatbuffers",
        "@flatbuffers//:flatc",
    ],
    defines = [
        "ARROW_WITH_SNAPPY",
    ],
    includes = [
        "cpp/src",
    ],
    copts = [
    ],
    deps = [
        #"format/File.fbs",
        #"format/Message.fbs",
        #"format/Schema.fbs",
        #"@flatbuffers",
        #"@flatbuffers//:flatc",
        #"@flatbuffers//:flatc_library",
        ":file_fbs",
        ":message_fbs",
        ":schema_fbs",
        ":tensor_fbs",
        #":arrow_format",
        "@snappy",
    ],
)

