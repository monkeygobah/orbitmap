/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/vision/gesture_recognizer/gesture_recognizer.h"

#include <memory>
#include <type_traits>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/image_preprocessing.h"
#include "mediapipe/tasks/cc/components/processors/proto/classifier_options.pb.h"
#include "mediapipe/tasks/cc/core/base_task_api.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/core/vision_task_api_factory.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/proto/gesture_recognizer_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/proto/hand_gesture_recognizer_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_detector/proto/hand_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace gesture_recognizer {

namespace {

using GestureRecognizerGraphOptionsProto = ::mediapipe::tasks::vision::
    gesture_recognizer::proto::GestureRecognizerGraphOptions;

using ::mediapipe::tasks::components::containers::GestureRecognitionResult;

constexpr char kHandGestureSubgraphTypeName[] =
    "mediapipe.tasks.vision.gesture_recognizer.GestureRecognizerGraph";

constexpr char kImageTag[] = "IMAGE";
constexpr char kImageInStreamName[] = "image_in";
constexpr char kImageOutStreamName[] = "image_out";
constexpr char kHandGesturesTag[] = "HAND_GESTURES";
constexpr char kHandGesturesStreamName[] = "hand_gestures";
constexpr char kHandednessTag[] = "HANDEDNESS";
constexpr char kHandednessStreamName[] = "handedness";
constexpr char kHandLandmarksTag[] = "LANDMARKS";
constexpr char kHandLandmarksStreamName[] = "landmarks";
constexpr char kHandWorldLandmarksTag[] = "WORLD_LANDMARKS";
constexpr char kHandWorldLandmarksStreamName[] = "world_landmarks";
constexpr int kMicroSecondsPerMilliSecond = 1000;

// Creates a MediaPipe graph config that contains a subgraph node of
// "mediapipe.tasks.vision.GestureRecognizerGraph". If the task is running
// in the live stream mode, a "FlowLimiterCalculator" will be added to limit the
// number of frames in flight.
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<GestureRecognizerGraphOptionsProto> options,
    bool enable_flow_limiting) {
  api2::builder::Graph graph;
  auto& subgraph = graph.AddNode(kHandGestureSubgraphTypeName);
  subgraph.GetOptions<GestureRecognizerGraphOptionsProto>().Swap(options.get());
  graph.In(kImageTag).SetName(kImageInStreamName);
  subgraph.Out(kHandGesturesTag).SetName(kHandGesturesStreamName) >>
      graph.Out(kHandGesturesTag);
  subgraph.Out(kHandednessTag).SetName(kHandednessStreamName) >>
      graph.Out(kHandednessTag);
  subgraph.Out(kHandLandmarksTag).SetName(kHandLandmarksStreamName) >>
      graph.Out(kHandLandmarksTag);
  subgraph.Out(kHandWorldLandmarksTag).SetName(kHandWorldLandmarksStreamName) >>
      graph.Out(kHandWorldLandmarksTag);
  subgraph.Out(kImageTag).SetName(kImageOutStreamName) >> graph.Out(kImageTag);
  if (enable_flow_limiting) {
    return tasks::core::AddFlowLimiterCalculator(graph, subgraph, {kImageTag},
                                                 kHandGesturesTag);
  }
  graph.In(kImageTag) >> subgraph.In(kImageTag);
  return graph.GetConfig();
}

// Converts the user-facing GestureRecognizerOptions struct to the internal
// GestureRecognizerGraphOptions proto.
std::unique_ptr<GestureRecognizerGraphOptionsProto>
ConvertGestureRecognizerGraphOptionsProto(GestureRecognizerOptions* options) {
  auto options_proto = std::make_unique<GestureRecognizerGraphOptionsProto>();

  bool use_stream_mode = options->running_mode != core::RunningMode::IMAGE;

  // TODO remove these workarounds for base options of subgraphs.
  // Configure hand detector options.
  auto base_options_proto_for_hand_detector =
      std::make_unique<tasks::core::proto::BaseOptions>(
          tasks::core::ConvertBaseOptionsToProto(
              &(options->base_options_for_hand_detector)));
  base_options_proto_for_hand_detector->set_use_stream_mode(use_stream_mode);
  auto* hand_detector_graph_options =
      options_proto->mutable_hand_landmarker_graph_options()
          ->mutable_hand_detector_graph_options();
  hand_detector_graph_options->mutable_base_options()->Swap(
      base_options_proto_for_hand_detector.get());
  hand_detector_graph_options->set_num_hands(options->num_hands);
  hand_detector_graph_options->set_min_detection_confidence(
      options->min_hand_detection_confidence);

  // Configure hand landmark detector options.
  auto base_options_proto_for_hand_landmarker =
      std::make_unique<tasks::core::proto::BaseOptions>(
          tasks::core::ConvertBaseOptionsToProto(
              &(options->base_options_for_hand_landmarker)));
  base_options_proto_for_hand_landmarker->set_use_stream_mode(use_stream_mode);
  auto* hand_landmarks_detector_graph_options =
      options_proto->mutable_hand_landmarker_graph_options()
          ->mutable_hand_landmarks_detector_graph_options();
  hand_landmarks_detector_graph_options->mutable_base_options()->Swap(
      base_options_proto_for_hand_landmarker.get());
  hand_landmarks_detector_graph_options->set_min_detection_confidence(
      options->min_hand_presence_confidence);

  auto* hand_landmarker_graph_options =
      options_proto->mutable_hand_landmarker_graph_options();
  hand_landmarker_graph_options->set_min_tracking_confidence(
      options->min_tracking_confidence);

  // Configure hand gesture recognizer options.
  auto base_options_proto_for_gesture_recognizer =
      std::make_unique<tasks::core::proto::BaseOptions>(
          tasks::core::ConvertBaseOptionsToProto(
              &(options->base_options_for_gesture_recognizer)));
  base_options_proto_for_gesture_recognizer->set_use_stream_mode(
      use_stream_mode);
  auto* hand_gesture_recognizer_graph_options =
      options_proto->mutable_hand_gesture_recognizer_graph_options();
  hand_gesture_recognizer_graph_options->mutable_base_options()->Swap(
      base_options_proto_for_gesture_recognizer.get());
  if (options->min_gesture_confidence >= 0) {
    hand_gesture_recognizer_graph_options->mutable_classifier_options()
        ->set_score_threshold(options->min_gesture_confidence);
  }
  return options_proto;
}

}  // namespace

absl::StatusOr<std::unique_ptr<GestureRecognizer>> GestureRecognizer::Create(
    std::unique_ptr<GestureRecognizerOptions> options) {
  auto options_proto = ConvertGestureRecognizerGraphOptionsProto(options.get());
  tasks::core::PacketsCallback packets_callback = nullptr;
  if (options->result_callback) {
    auto result_callback = options->result_callback;
    packets_callback = [=](absl::StatusOr<tasks::core::PacketMap>
                               status_or_packets) {
      if (!status_or_packets.ok()) {
        Image image;
        result_callback(status_or_packets.status(), image,
                        Timestamp::Unset().Value());
        return;
      }
      if (status_or_packets.value()[kImageOutStreamName].IsEmpty()) {
        return;
      }
      Packet gesture_packet =
          status_or_packets.value()[kHandGesturesStreamName];
      Packet handedness_packet =
          status_or_packets.value()[kHandednessStreamName];
      Packet hand_landmarks_packet =
          status_or_packets.value()[kHandLandmarksStreamName];
      Packet hand_world_landmarks_packet =
          status_or_packets.value()[kHandWorldLandmarksStreamName];
      Packet image_packet = status_or_packets.value()[kImageOutStreamName];
      result_callback(
          {{gesture_packet.Get<std::vector<ClassificationList>>(),
            handedness_packet.Get<std::vector<ClassificationList>>(),
            hand_landmarks_packet.Get<std::vector<NormalizedLandmarkList>>(),
            hand_world_landmarks_packet.Get<std::vector<LandmarkList>>()}},
          image_packet.Get<Image>(),
          gesture_packet.Timestamp().Value() / kMicroSecondsPerMilliSecond);
    };
  }
  return core::VisionTaskApiFactory::Create<GestureRecognizer,
                                            GestureRecognizerGraphOptionsProto>(
      CreateGraphConfig(
          std::move(options_proto),
          options->running_mode == core::RunningMode::LIVE_STREAM),
      std::move(options->base_options.op_resolver), options->running_mode,
      std::move(packets_callback));
}

absl::StatusOr<GestureRecognitionResult> GestureRecognizer::Recognize(
    mediapipe::Image image) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "GPU input images are currently not supported.",
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  ASSIGN_OR_RETURN(auto output_packets,
                   ProcessImageData({{kImageInStreamName,
                                      MakePacket<Image>(std::move(image))}}));
  return {
      {/* gestures= */ {output_packets[kHandGesturesStreamName]
                            .Get<std::vector<ClassificationList>>()},
       /* handedness= */
       {output_packets[kHandednessStreamName]
            .Get<std::vector<mediapipe::ClassificationList>>()},
       /* hand_landmarks= */
       {output_packets[kHandLandmarksStreamName]
            .Get<std::vector<mediapipe::NormalizedLandmarkList>>()},
       /* hand_world_landmarks */
       {output_packets[kHandWorldLandmarksStreamName]
            .Get<std::vector<mediapipe::LandmarkList>>()}},
  };
}

absl::StatusOr<GestureRecognitionResult> GestureRecognizer::RecognizeForVideo(
    mediapipe::Image image, int64 timestamp_ms) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  ASSIGN_OR_RETURN(
      auto output_packets,
      ProcessVideoData(
          {{kImageInStreamName,
            MakePacket<Image>(std::move(image))
                .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}}));
  return {
      {/* gestures= */ {output_packets[kHandGesturesStreamName]
                            .Get<std::vector<ClassificationList>>()},
       /* handedness= */
       {output_packets[kHandednessStreamName]
            .Get<std::vector<mediapipe::ClassificationList>>()},
       /* hand_landmarks= */
       {output_packets[kHandLandmarksStreamName]
            .Get<std::vector<mediapipe::NormalizedLandmarkList>>()},
       /* hand_world_landmarks */
       {output_packets[kHandWorldLandmarksStreamName]
            .Get<std::vector<mediapipe::LandmarkList>>()}},
  };
}

absl::Status GestureRecognizer::RecognizeAsync(mediapipe::Image image,
                                               int64 timestamp_ms) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  return SendLiveStreamData(
      {{kImageInStreamName,
        MakePacket<Image>(std::move(image))
            .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}});
}

}  // namespace gesture_recognizer
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
