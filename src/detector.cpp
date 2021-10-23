/** Copyright 2021 Sakshi Kakde, Siddharth Telang, Anubhav Paras */

#include <detector.hpp>
#include <memory>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

typedef HumanDetector HD;


HumanDetector::HumanDetector() {}


// HD::HumanDetector(std::shared_ptr<Model<DetectionOutput, Image>> model,
//                   std::shared_ptr<FrameTransformation> robotFrame) {
//     this->model = model;
//     this->robotFrame = robotFrame;
// }

HumanDetector::HumanDetector(AbstractSVMModel* model,
                  FrameTransformation* robotFrame) {
    this->model = model;
    this->robotFrame = robotFrame;
}

HumanDetector::~HumanDetector() {
}

std::vector<Coord2D> HumanDetector::getCentroids(
                                      const Rectangles& boundingBoxes) {
  std::vector<Coord2D> centroids;
  for (const cv::Rect& box : boundingBoxes) {
    Coord2D centroid;
    centroid.x = (box.tl().x + box.br().x)/2.0;
    centroid.y = (box.tl().y + box.br().y)/2.0;
    centroids.push_back(centroid);
  }
  return centroids;
}

std::vector<Coord3D> HumanDetector::getRobotFrameCoordinates(
                                const Rectangles& boundingBoxes) {
  std::vector<Coord2D> centroids = this->getCentroids(boundingBoxes);
  std::vector<Coord3D> robotFrameCoordinates;
  for (const Coord2D& centroid : centroids) {
    Coord3D robotFrameCoord = this->robotFrame->getRobotFrame(centroid);
    robotFrameCoordinates.push_back(robotFrameCoord);
  }
  return robotFrameCoordinates;
}

void HumanDetector::displayOutput(const cv::Mat &image,
                       const DetectionOutput &predictionOutput, bool isTestMode) {
  Rectangles boundingBoxes = predictionOutput.getData().first;
  std::vector<double> confidenceScores = predictionOutput.getData().second;
  int i = 0;
  for (cv::Rect &box : boundingBoxes) {
    // draw bounding box
    cv::rectangle(image, box.tl(), box.br(), cv::Scalar(0, 255, 0), 2);

    std::string text = "ID: " + std::to_string(i + 1) + " | Score: "
                        + std::to_string(confidenceScores[i]);

    // put ID and confidence score
    cv::putText(image, text, cv::Point(box.x, box.y + 50),
                cv::FONT_HERSHEY_SIMPLEX,
                1, cv::Scalar(0, 255, 0));

    i++;
  }
  if (!isTestMode) {
    cv::imshow("Detected Humans", image);
    cv::waitKey(600);
  }
}

std::vector<Coord3D> HumanDetector::detect(const cv::Mat &inputData, 
                                            bool isTestMode) {
  std::cout << "Detecting objects" << std::endl;
  DetectionOutput predictionOutput = this->model->predict(inputData);

  // null check should be there
  Rectangles boundingBoxes = predictionOutput.getData().first;
  if (boundingBoxes.size() < 1) {
    std::cout << "No humans detected for the given image." << std::endl;
    return {};
  }

  // get coordinates in robot frame:
  // TODO(Anubhav, Sakshi)
  std::vector<Coord3D> coordinates =
                          this->getRobotFrameCoordinates(boundingBoxes);

  // draw bounding boxes for each detected human and set the id
  this->displayOutput(inputData, predictionOutput, isTestMode);

  return coordinates;
}



DetectorImpl::DetectorImpl() {}

DetectorImpl::DetectorImpl(AbstractSVMModel* model,
                           FrameTransformation* robotFrame) {
  this->model = model;
  this->robotFrame = robotFrame;
}

std::vector<Coord2D> DetectorImpl::getCentroids(
                    const Rectangles& boundingBoxes) {
  std::vector<Coord2D> centroids;
  for (const cv::Rect& box : boundingBoxes) {
    Coord2D centroid;
    centroid.x = (box.tl().x + box.br().x)/2.0;
    centroid.y = (box.tl().y + box.br().y)/2.0;
    centroids.push_back(centroid);
  }
  return centroids;
}

std::vector<Coord3D> DetectorImpl::getRobotFrameCoordinates(
                                const Rectangles& boundingBoxes) {
  std::vector<Coord2D> centroids = this->getCentroids(boundingBoxes);
  std::vector<Coord3D> robotFrameCoordinates;
  for (const Coord2D& centroid : centroids) {
    Coord3D robotFrameCoord = this->robotFrame->getRobotFrame(centroid);
    robotFrameCoordinates.push_back(robotFrameCoord);
  }
  return robotFrameCoordinates;
}

void DetectorImpl::displayOutput(const cv::Mat &image,
                       const DetectionOutput &predictionOutput,
                       bool isTestMode) {
  Rectangles boundingBoxes = predictionOutput.getData().first;
  std::vector<double> confidenceScores = predictionOutput.getData().second;
  int i = 0;
  for (cv::Rect &box : boundingBoxes) {
    // draw bounding box
    cv::rectangle(image, box.tl(), box.br(), cv::Scalar(0, 255, 0), 2);

    std::string text = "ID: " + std::to_string(i + 1) + " | Score: "
                        + std::to_string(confidenceScores[i]);

    // put ID and confidence score
    cv::putText(image, text, cv::Point(box.x, box.y + 50),
                cv::FONT_HERSHEY_SIMPLEX,
                1, cv::Scalar(0, 255, 0));

    i++;
  }

  if (!isTestMode) {
    cv::imshow("Detected Humans", image);
    cv::waitKey(600);
  }
}


std::vector<Coord3D> DetectorImpl::detect(const cv::Mat& inputData,
                                          bool isTestMode) {
  DetectionOutput predictionOutput = this->model->predict(inputData);

  Rectangles boundingBoxes = predictionOutput.getData().first;
  if (boundingBoxes.size() < 1) {
    std::cout << "No humans detected for the given image." << std::endl;
    return {};
  }

  Coord2D centroid;
  Coord3D robotFrameCoord = this->robotFrame->getRobotFrame(centroid);
  std::vector<Coord3D> coordinates;

  // draw bounding boxes for each detected human and set the id
  this->displayOutput(inputData, predictionOutput, isTestMode);

  return coordinates;
}



