# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

android_library(
    name = "category",
    srcs = ["Category.java"],
    deps = [
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "detection",
    srcs = ["Detection.java"],
    deps = [
        ":category",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "classification_entry",
    srcs = ["ClassificationEntry.java"],
    deps = [
        ":category",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "classifications",
    srcs = ["Classifications.java"],
    deps = [
        ":classification_entry",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "landmark",
    srcs = ["Landmark.java"],
    deps = [
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)
