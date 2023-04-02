// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// A utility to extract iris depth from a single image of face using the graph
// mediapipe/graphs/iris_tracking/iris_depth_cpu.pbtxt.
#include <cstdlib>
#include <memory>


#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
// #ifndef ABSL_FLAGS_DECLARE_H_
// #define ABSL_FLAGS_DECLARE_H_
#include "absl/flags/declare.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
// #include "mediapipe/framework/port/image_file_properties_calculator.h"

constexpr char kInputStream[] = "input_image_bytes";
constexpr char kOutputImageStream[] = "output_image";
constexpr char kLeftIrisDepthMmStream[] = "left_iris_depth_mm";
constexpr char kRightIrisDepthMmStream[] = "right_iris_depth_mm";
constexpr char kWindowName[] = "MediaPipe";
constexpr char kCalculatorGraphConfigFile[] =
    "mediapipe/graphs/iris_tracking/iris_depth_cpu.pbtxt";
constexpr float kMicrosPerSecond = 1e6;

ABSL_FLAG(std::string, input_image_path, "",
          "Full path of image to load. "
          "If not provided, nothing will run.");
ABSL_FLAG(std::string, output_image_path, "",
          "Full path of where to save image result (.jpg only). "
          "If not provided, show result in a window.");

ABSL_FLAG(std::string, focal_length, "",
          "Full path of where to save image result (.jpg only). "
          "If not provided, show result in a window.");

ABSL_FLAG(std::string, mm35_focal_length, "",
          "Full path of where to save image result (.jpg only). "
          "If not provided, show result in a window.");

// ABSL_DECLARE_FLAG(std::string, focal_length);
// ABSL_DECLARE_FLAG(double, mm35_focal_length);




namespace {




absl::StatusOr<std::string> ReadFileToString(const std::string& file_path) {
  std::string contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(file_path, &contents));

  return contents;

}

absl::Status ProcessImage(std::unique_ptr<mediapipe::CalculatorGraph> graph) {
  std::string fl = absl::GetFlag(FLAGS_focal_length);
  std::string fl_35 = absl::GetFlag(FLAGS_mm35_focal_length);

  // std::cout << typeid(absl::GetFlag(FLAGS_focal_length)).name() << std::endl;

  // int fl_int = std::stoi(fl);
  // int fl_save = 60;
  // std::cout << typeid(fl_save).name() << std::endl;

  // float fl_35_int = std::stof(fl_35);


  LOG(INFO) << "Load the image.";
  ASSIGN_OR_RETURN(const std::string raw_image,
                   ReadFileToString(absl::GetFlag(FLAGS_input_image_path)));

  LOG(INFO) << "Start running the calculator graph.";
  LOG(INFO) << "1.";

  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller output_image_poller,
                   graph->AddOutputStreamPoller(kOutputImageStream));  
  LOG(INFO) << "2.";

  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller left_iris_depth_poller,
                   graph->AddOutputStreamPoller(kLeftIrisDepthMmStream));
  LOG(INFO) << "3.";

  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller right_iris_depth_poller,
                   graph->AddOutputStreamPoller(kRightIrisDepthMmStream));
  LOG(INFO) << "4.";

  MP_RETURN_IF_ERROR(graph->StartRun({}));
  LOG(INFO) << "5.";

  // Send image packet into the graph.
  const size_t fake_timestamp_us = (double)cv::getTickCount() /
                                   (double)cv::getTickFrequency() *
                                   kMicrosPerSecond;
  LOG(INFO) << "6.";

  MP_RETURN_IF_ERROR(graph->AddPacketToInputStream(
      kInputStream, mediapipe::MakePacket<std::string>(raw_image).At(
                        mediapipe::Timestamp(fake_timestamp_us))));
  LOG(INFO) << "7.";

  // Get the graph result packets, or stop if that fails.
  mediapipe::Packet left_iris_depth_packet;
  LOG(INFO) << "8.";

  if (!left_iris_depth_poller.Next(&left_iris_depth_packet)) {
    return absl::UnknownError(
        "Failed to get packet from output stream 'left_iris_depth_mm'.");
  }
  const auto& left_iris_depth_mm = left_iris_depth_packet.Get<float>();
  const int left_iris_depth_cm = std::round(left_iris_depth_mm / 10);
  std::cout << "Left Iris Depth: " << left_iris_depth_cm << " cm." << std::endl;

  mediapipe::Packet right_iris_depth_packet;
  if (!right_iris_depth_poller.Next(&right_iris_depth_packet)) {
    return absl::UnknownError(
        "Failed to get packet from output stream 'right_iris_depth_mm'.");
  }
  const auto& right_iris_depth_mm = right_iris_depth_packet.Get<float>();
  const int right_iris_depth_cm = std::round(right_iris_depth_mm / 10);
  std::cout << "Right Iris Depth: " << right_iris_depth_cm << " cm."
            << std::endl;

  mediapipe::Packet output_image_packet;
  if (!output_image_poller.Next(&output_image_packet)) {
    return absl::UnknownError(
        "Failed to get packet from output stream 'output_image'.");
  }
  auto& output_frame = output_image_packet.Get<mediapipe::ImageFrame>();

  // Convert back to opencv for display or saving.
  cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
  cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
  const bool save_image = !absl::GetFlag(FLAGS_output_image_path).empty();
  if (save_image) {
    LOG(INFO) << "Saving image to file...";
    cv::imwrite(absl::GetFlag(FLAGS_output_image_path), output_frame_mat);
  } else {
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
    cv::imshow(kWindowName, output_frame_mat);
    // Press any key to exit.
    cv::waitKey(0);
  }

  LOG(INFO) << "Shutting down.";
  MP_RETURN_IF_ERROR(graph->CloseInputStream(kInputStream));
  return graph->WaitUntilDone();
}

absl::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  LOG(INFO) << "made it 5"; 

  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      kCalculatorGraphConfigFile, &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  std::unique_ptr<mediapipe::CalculatorGraph> graph =
      absl::make_unique<mediapipe::CalculatorGraph>();
  MP_RETURN_IF_ERROR(graph->Initialize(config));

  const bool load_image = !absl::GetFlag(FLAGS_input_image_path).empty();
  if (load_image) {
    return ProcessImage(std::move(graph));
  } else {
    return absl::InvalidArgumentError("Missing image file.");
  }
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  absl::Status run_status = RunMPPGraph();


  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
