#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#define CERES_FOUND 1
#include <opencv2/sfm.hpp>

#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace cv::xfeatures2d;
using namespace cv;
using namespace std;

/* count the of feature matches */
int countMatches(vector<KeyPoint> kp) {
  int pointCount = 0;
  for (int i = 0; i < kp.size(); i++) {
    if (kp[i].pt.x > 0 && kp[i].pt.y > 0)
      pointCount += 1;
  }
  return pointCount;
}

void showFeatureMatches(vector<vector<KeyPoint>> keypoints, vector<Mat> frames, vector<DMatch> good_matches) {
  Mat img, img_matches;
  vector<vector<Point2f>> points(2);
  // convert keypoints to points
  KeyPoint::convert(keypoints[0], points[0], vector<int>());
  KeyPoint::convert(keypoints.back(), points[1], vector<int>());
  // clone the last frame
  img = frames.back().clone();
  // add lines connecting matching features
  for (int i = 0; i < keypoints[0].size(); i++) {
    if ((points[0][i].x > 0 && points[0][i].y > 0) && (points[1][i].x > 0 && points[1][i].y > 0)) {
      line(img, points[1][i], points[0][i], Scalar(200, 200, 200), 1);
    }
  }
  // show keypoints
  drawKeypoints(img, keypoints[0], img, Scalar::all(-1));
  drawKeypoints(img, keypoints.back(), img, Scalar::all(-1));
  hconcat(frames[0],img,img);
  imshow("Matches (1)", img);
  waitKey(0);
  return;
}

/* compute the rigid body transformation between the frames */
Mat getMotion(vector<vector<KeyPoint>> keypoints, Mat intrinsic, float scale) {
  Mat mask, extrinsicMat;
  float data[] = {0, 0, 0, 1};
  Mat E, T, rotation, translation;
  Mat magic = Mat(1, 4, CV_32F, data); // bottom row of transformation
  vector<Point2f> points0, points1;
  // convert keypoints to points
  KeyPoint::convert(keypoints[0], points0, vector<int>());
  KeyPoint::convert(keypoints[2], points1, vector<int>());
  // get the rotation and translation
  E = findEssentialMat(points0, points1, intrinsic, RANSAC, 0.999, 1.0, mask);
  recoverPose(E, points0, points1, intrinsic, rotation, translation, mask);
  translation = translation * scale;  // compute the translation up to scale
  hconcat(rotation, translation, extrinsicMat); // form the extrinsic matrix
  extrinsicMat.convertTo(extrinsicMat, CV_32F); // convert extrinsic matrix to float
  vconcat(extrinsicMat, magic, T); // concatenate bottom row to form rigib body transformation
  return T;
}

/* convert homogeneous coodrdinates to cartesian coordinates */
void hom2cart(Mat points0, Mat points1, vector<Point3f> &X0,  vector<Point3f> &X1) {
  float x, y, z, s;
  for (int i = 0; i < points0.cols; i++) {
    s = points0.at<float>(3,i);  // scale factor
    x = points0.at<float>(0,i) / s;
    y = points0.at<float>(1,i) / s;
    z = points0.at<float>(2,i) / s;
    X0.push_back(Point3f(x, y, z));
    s = points1.at<float>(3,i);  // scale factor
    x = points1.at<float>(0,i) / s;
    y = points1.at<float>(1,i) / s;
    z = points1.at<float>(2,i) / s;
    X1.push_back(Point3f(x, y, z));
  }
  return;
}

/* compute distance */
float normPoints(Point3f a, Point3f b) {
  float dx, dy, dz, norm;
  // take the difference of coordinates
  dx = a.x - b.x;
  dy = a.y - b.y;
  dz = a.z - b.z;
  // find the distance between the points
  norm = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
  return norm;
}

/* compute projection matrix from essential matrix */
Mat getProjMat(vector<Point2f> points0, vector<Point2f> points1, Mat intrinsic) {
  Mat E, R, t;
  Mat mask, projMat;

  E = findEssentialMat(points0, points1, intrinsic, RANSAC, 0.999, 1.0, mask);
  recoverPose(E, points0, points1, intrinsic, R, t, mask);
  hconcat(R, t, projMat);
  return projMat;
}

Mat points2Mat(vector<Point2f> points) {
  Mat matPoints(2, 1, CV_32F);
  for (int i = 0; i < points.size(); i++) {
    matPoints.at<float>(0,0) = points[i].x;
    matPoints.at<float>(1,0) = points[i].y;
  }
  return matPoints;
}

/* compute the relative scale given three successive frames */
float getScale(vector<vector<KeyPoint>> keypoints, Mat intrinsic, Mat &initialProj) {
  vector<Point3f> X0, X1;
  vector<Mat> projMat(3);
  float scale;
  float num, denom, count;
  vector<vector<Point2f>> points(3);
  Mat pnts3D0(4,points[0].size(),CV_32F), pnts3D1(4,points[1].size(),CV_32F);
  // convert keypoints to points
  KeyPoint::convert(keypoints[0], points[0], vector<int>());
  KeyPoint::convert(keypoints[1], points[1], vector<int>());
  KeyPoint::convert(keypoints[2], points[2], vector<int>());
  for (int i = 0; i < points[0].size(); i++) {
    float x0 = points[0][i].x;
    float y0 = points[0][i].y;
    float x1 = points[1][i].x;
    float y1 = points[1][i].y;
    float x2 = points[2][i].x;
    float y2 = points[2][i].y;
    if ((x0 == 0 && y0 == 0) || (x1 == 0 && y1 == 0) || (x2 == 0 && y2 == 0)) {
      points[0].erase(points[0].begin() + i);
      points[1].erase(points[1].begin() + i);
      points[2].erase(points[2].begin() + i);
      i--;
    }
  }
  // compute the projection matrices
  projMat[0] = initialProj.clone();
  projMat[1] = getProjMat(points[0], points[1], intrinsic);
  projMat[2] = getProjMat(points[1], points[2], intrinsic);
  // compute the 3d homogeneous points
  triangulatePoints(projMat[0], projMat[1], points[0], points[1], pnts3D0);
  triangulatePoints(projMat[1], projMat[2], points[1], points[2], pnts3D1);
  // convert the homoegeneous points to cartension
  hom2cart(pnts3D0, pnts3D1, X0, X1);
  count = 0;
  scale = 0;
  // compute the scale
  for (int i = 0; i < X0.size(); i++) {
    for (int j = 0; j < X1.size(); j++) {
      if (i != j) {
        num = normPoints(X0[i], X0[j]); // compute distance between world points
        denom = normPoints(X1[i], X1[j]); // compute distance between world ponints
        if ((num / denom) < pow(3,38)) {
          scale += (num / denom); // compute scale
          count += 1; // for averaging
        }
      }
    }
  }
  scale = scale / count; // compute the average scale
  initialProj = projMat[2].clone();
  return scale;
}

/*
  sort new keypoints so that they are in the same position (in array)
  as the corresponding keypoint from the first frame
*/
void sortPoints(vector<DMatch> matches, vector<KeyPoint> keypoints, vector<KeyPoint> &kp2Sorted) {
  int kp1Index, kp2Index;
  for (int i = 0; i < matches.size(); i++) {
    kp1Index = matches[i].queryIdx; // index of ith match in first frame
    kp2Index = matches[i].trainIdx; // index of ith match in second frame
    kp2Sorted[kp1Index] = keypoints[kp2Index]; // insert ith match (of second frame)
  }
  return;
}

/* find features to track */
void getFeatures(Mat frame, Mat &descriptor, vector<KeyPoint> &keypoints) {
  int minHessian = 400;
  Mat frameGray;
  Ptr<SURF> detector = SURF::create(minHessian);
  cvtColor(frame, frameGray, COLOR_BGR2GRAY);
  detector->detectAndCompute(frameGray, noArray(), keypoints, descriptor);
  // find SURF features
  detector->detect(frameGray, keypoints);
  return;
}

/* match features between two frames */
void matchFrames(Mat newFrame, Mat descriptor, vector<KeyPoint> &keypoints, int minPoints, int &kpCount) {
  // get features for new frame
  Mat newDescriptors;
  vector<DMatch> good_matches;
  const float ratio_thresh = 0.4f;
  vector<std::vector<DMatch>> knn_matches;
  vector<KeyPoint> newKeypoints, kp2Sorted(descriptor.rows, KeyPoint());
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
  // get features for new frame
  getFeatures(newFrame, newDescriptors, newKeypoints);
  // not enough features to track
  if (newKeypoints.size() < minPoints) {
    kpCount = 0;
    keypoints = vector<KeyPoint>();
    return;
  }
  // match features between initial features and new features
  matcher->knnMatch(descriptor, newDescriptors, knn_matches, 2);
  //-- Filter matches using the Lowe's ratio test
  for (size_t i = 0; i < knn_matches.size(); i++) {
      if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
          good_matches.push_back(knn_matches[i][0]);
      }
  }
  // align the new features with the corresponding features from first frame
  sortPoints(good_matches, newKeypoints, kp2Sorted);
  kpCount = countMatches(kp2Sorted);
  // add new sorted keypoints into array of keypoints
  keypoints = kp2Sorted;
}

float getWorldUnits(float value, float scale) {
  float conv = value * scale;
  return conv;
}

/* set third frame as first frame, clear other frames */
vector<Mat> cleanFrames(vector<Mat> frames) {
  frames[0] = frames[2].clone();
  frames[1] = Mat::zeros(frames[0].size(), frames[0].type());
  frames[2] = Mat::zeros(frames[0].size(), frames[0].type());
  return frames;
}

/* get two frames through which features from an earlier frame are tracked */
void getFrames(vector<Mat> &frames, VideoCapture capture, float scale, int count) {
  Mat currFrame;
  for (int i = 1; i < count+1; i++) {
    capture.read(currFrame); // get next frame
    if (currFrame.empty()) {
      frames.clear(); // end of video stream
      return;
    }
    // resize frame
    resize(currFrame, frames[i], Size(), scale, scale, INTER_AREA);
  }
  return;
}

/* reset variables */
void cleanUp(vector<Mat> &frames, Mat &currFrame, Mat &desc, vector<vector<KeyPoint>> &kps) {
  desc.release(); // empty descriptor
  currFrame.release(); // empty frame
  // resize keypoints and reset 1st element
  kps[0] = vector<KeyPoint>();
  kps[1] = vector<KeyPoint>();
  kps[2] = vector<KeyPoint>();
  // empty frames
  frames[2] = Mat::zeros(frames[0].size(), frames[0].type());
  return;
}

/* perform monocular visual odometry */
void getPoseFromVideo(string videofile, Mat intrinsic, float scale) {
  int kpCount[3], minPoints = 8, frames2Fetch = 2;
  float relativeScale;
  vector<vector<KeyPoint>> keypoints(3);
  VideoCapture capture(videofile);
  Mat descriptor, currFrame, currentT;
  Mat C = Mat::eye(4, 4, CV_32F);
  Mat T = Mat::eye(4, 4, CV_32F);
  Mat initialProj = Mat::eye(3,4, CV_32F);
  vector<Mat> frames(3);

  capture.read(frames[0]); // get the first frame from webcam
  resize(frames[0], frames[0], Size(), scale, scale, INTER_AREA);
  // loop through the video stream
  while (true) {
    // find features in first frame to track throughout next frames
    getFeatures(frames[0], descriptor, keypoints[0]);
    getFrames(frames, capture, scale, frames2Fetch);
    if (frames.size() == 3 && keypoints[0].size() > minPoints*10) {
      // track features through two additional frames
      matchFrames(frames[1], descriptor, keypoints[1], minPoints, kpCount[1]);
      matchFrames(frames[2], descriptor, keypoints[2], minPoints, kpCount[2]);
    }
    if (frames.size() == 3 && kpCount[1] > minPoints && kpCount[2] > minPoints) {
      // get translation up to scale
      relativeScale = getScale(keypoints, intrinsic, initialProj);
      currentT = getMotion(keypoints, intrinsic, relativeScale);
      //showFeatureMatches(keypoints, frames, vector<DMatch>());
      frames2Fetch = 2;
      frames[0] = frames[2].clone(); // make last frame, the referance frame
      frames[1] = Mat::zeros(frames[0].size(), frames[0].type()); // empty frame
    }
    else {
      if (frames.size() < 3)
        return;
      currentT = T; // repeate same motion (or lack of)
      frames2Fetch = 1;
      frames[0] = frames[1].clone(); // make the second frame the referance
      frames[1] = frames[2].clone();// shift frames
    }
    T = T * currentT; // compute rigib body transformation
    C = C * T; // compute camera pose
    //cout << C << endl;
    cout << C.at<float>(0,3) << ", " << C.at<float>(1,3) << ", " << C.at<float>(2,3) << ";" << endl;
    cleanUp(frames, currFrame, descriptor, keypoints); // reinitialize variables
  }
}

int main(int argc, char **argv) {
  Mat intrinsic;
  string videofile = "sources/adamapt2.mp4";
  float Fx, Fy, Cx, Cy, scaleX, scaleY, scale;
  float sensorWidth = 5.14, sensorHeight = 3.50;
  scale = 1.0;
  // scale to go from pixel to world units
  scaleX = (sensorWidth/scale) / 1920;
  scaleY = (sensorHeight/scale) / 1080;
  // intrinsic properties in world units (mm)
  Fx = getWorldUnits(3273.77625415137, scaleX);
  Fy = getWorldUnits(103406.475098688, scaleY);
  Cx = getWorldUnits(743.535235278513, scaleX);
  Cy = getWorldUnits(655.5546525875903, scaleY);
  // intrinsic matrix
  intrinsic = (Mat1f(3, 3) << Fx, 0, Cx, 0, Fy, Cy, 0, 0, 1);
  getPoseFromVideo(videofile, intrinsic, scale);
}
