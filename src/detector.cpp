/** Copyright 2021 Sakshi Kakde, Siddharth Telang, Anubhav Paras */

#include <detector.hpp>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

typedef HumanDetector HD;

HD::HumanDetector() {}


HD::HumanDetector(Model<DetectionOutput, Image>* model,
                  FrameTransformation* robotFrame) {
  this->model = model;
  this->robotFrame = robotFrame;
}


HD::~HumanDetector() {
  if (this->model != nullptr) {
    delete this->model;
    this->model = nullptr;
  }
  if (this->robotFrame != nullptr) {
    delete this->robotFrame;
    this->robotFrame = nullptr;
  }
}

std::vector<Coord2D> HD::getCentroids(const Rectangles& boundingBoxes) {
  std::vector<Coord2D> centroids;
  for (const cv::Rect& box : boundingBoxes) {
    Coord2D centroid;
    centroid.x = (box.tl().x + box.br().x)/2.0;
    centroid.y = (box.tl().y + box.br().y)/2.0;
    centroids.push_back(centroid);
  }
  return centroids;
}

std::vector<Coord3D> HD::getRobotFrameCoordinates(
                                const Rectangles& boundingBoxes) {
  std::vector<Coord2D> centroids = this->getCentroids(boundingBoxes);
  std::vector<Coord3D> robotFrameCoordinates;
  for (const Coord2D& centroid : centroids) {
    Coord3D robotFrameCoord = this->robotFrame->getRobotFrame(centroid);
    robotFrameCoordinates.push_back(robotFrameCoord);
  }
  return robotFrameCoordinates;
}

void HD::displayOutput(const cv::Mat &image,
                       const DetectionOutput &predictionOutput) {
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

  cv::imshow("Detected Humans", image);
  cv::waitKey(100);
}

std::vector<Coord3D> HD::detect(const cv::Mat &inputData) {
  std::cout << "Detecting objects" << std::endl;
  std::vector<Coord3D> coordinates;

  return coordinates;
}


