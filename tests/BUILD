
cc_test(
    name = "cpp_bark_test",
    srcs = [
        "cpp_bark_test.cc",
    ],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "@bark_project//modules/geometry",
        "@bark_project//modules/models/dynamic:dynamic",
        "@gtest//:main",
        "@boost//:geometry"
    ],
)