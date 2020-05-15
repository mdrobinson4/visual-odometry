#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>

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

void getFeatures(Mat frame, Mat &descriptor, vector<KeyPoint> &keypoints) {
  int minHessian = 400;
  Mat frameGray;
  Ptr<SURF> detector = SURF::create(minHessian);

  cvtColor(frame, frameGray, COLOR_BGR2GRAY);
  detector->detectAndCompute(frameGray, noArray(), keypoints, descriptor);
  // find SURF features
  detector->detect(frameGray, keypoints);
  // convert vector of keypoints to vector of points
  return;
}

/* count the of feature matches */
int countMatches(vector<KeyPoint> kp) {
  int pointCount = 0;
  for (int i = 0; i < kp.size(); i++) {
    if (kp[i].pt.x > 0 && kp[i].pt.y > 0)
      pointCount += 1;
  }
  return pointCount;
}

void sortPoints(vector<DMatch> matches, vector<KeyPoint> keypoints, vector<KeyPoint> &kp2Sorted) {
  // initialize keypoints
  int kp1Index, kp2Index;
  for (int i = 0; i < matches.size(); i++) {
    kp1Index = matches[i].queryIdx;
    kp2Index = matches[i].trainIdx;
    kp2Sorted[kp1Index] = keypoints[kp2Index];
  }
  return;
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
  imshow("Matches (1)", img);
  imshow("Reference (1)", frames[0]);
  //-- Draw matches
  if (good_matches.size() > 0) {
    drawMatches(frames[0], keypoints[0], frames.back(), keypoints.back(), good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Show detected matches
    imshow("Matches (2)", img_matches );
  }
  return;
}

bool checkPoint(Mat points, int col) {
  bool flag;
  flag = false;
  if ((points.at<int>(0,col) > 0) && (points.at<int>(1,col) > 0)) {
    if ((points.at<int>(2,col) > 0) && (points.at<int>(3,col) > 0)) {
      flag = true;
    }
  }
  return flag;
}

/* convert matrix of 3d points to vector of point3f */
void mat2Points(Mat points0, Mat points1, vector<Point3d> &X0,  vector<Point3d> &X1) {
  int x, y, z, s;
  for (int i = 0; i < points0.cols; i++) {
    if (checkPoint(points0, i) && checkPoint(points1, i)) {
      s = points0.at<int>(3,i);  // scale factor
      x = points0.at<int>(0,i) / s;
      y = points0.at<int>(1,i) / s;
      z = points0.at<int>(2,i) / s;
      X0.push_back(Point3f(x, y, z));
      s = points1.at<int>(3,i);  // scale factor
      x = points1.at<int>(0,i) / s;
      y = points1.at<int>(1,i) / s;
      z = points1.at<int>(2,i) / s;
      X1.push_back(Point3f(x, y, z));
    }
  }
  return;
}

Mat constructExtrinsic(Mat R, Mat t) {
  Mat extrinsic;
  hconcat(R, t, extrinsic);
  return extrinsic;
}

double normPoints(Point3d a, Point3d b) {
  double dx, dy, dz, norm;
  dx = a.x - b.x;
  dy = a.y - b.y;
  dz = a.z - b.z;
  norm = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
  return norm;
}

double computeScale(vector<Point3d> X0, vector<Point3D> X1) {
  int count;
  double num, denom, scale;
  count = 0;
  scale = 0;
  for (int i = 0; i < X0.size(); i++) {
    for (int j = 0; j < X1.size(); j++) {
      if (i != j) {
        num = normPoints(X0[i], X0[j]);
        denom = normPoints(X1[i], X1[j]);
        scale += (num / denom);
        count += 1;
      }
    }
  }
  scale = scale / count;
  return scale;
}

float getScale(vector<vector<Point2f>> points, Mat intrinsic, Mat proj) {
  double scale = 0;
  vector<Point3d> X0, X1;
  Mat newProj, extrinsic, E, R, t, mask;
  Mat pnts3D0(4,points[0].size(),CV_64F), pnts3D1(4,points[1].size(),CV_64F);
  for (int i = 0; i < 2; i++) {
    // get the essential matrix
    E = findEssentialMat(points[i], points[i+1], intrinsic, RANSAC, 0.999, 1.0, mask);
    // get the pose from the essential matrix
    recoverPose(E, points[i], points[i+1], intrinsic, R, t, mask);
    // get the extrinsic matrix
    extrinsic = constructExtrinsic(R, t);
    // get the projection matrix: 3d -> 2d
    newProj = intrinsic * extrinsic;
    triangulatePoints(proj, newProj, points[i], points[i+1], pnts3D0);
    E.release();
    R.release();
    t.release();
    mask.release();
  }
  mat2Points(pnts3D0, pnts3D1, X0, X1);
  scale = computeScale(X0, X1);
  return scale;
}

/* compute the translation between the first and last frames */
void getMotion(vector<Mat> frames, vector<vector<KeyPoint>> keypoints, int pointCount, Mat intrinsics, Mat &proj) {
  Mat mask;
  double thresh;
  vector<Mat> E(2), t(2), R(2), extrinsic(2), newProj(2);
  int middleIndex;
  vector<vector<Point2f>> points(3);
  vector<vector<Point3d>> points3d(2);
  if (keypoints.size() > 6) {
    thresh = 1000;
    // get the index of the middle keypoints
    middleIndex = (keypoints.size() + 1) / 2;
    // convert keypoints to points
    KeyPoint::convert(keypoints[0], points[0], vector<int>());
    KeyPoint::convert(keypoints[middleIndex], points[1], vector<int>());
    KeyPoint::convert(keypoints.back(), points[2], vector<int>());
    float scale = getScale(points, intrinsics, proj);
  }
}

void matchFrames(vector<Mat> frames, Mat descriptor, vector<vector<KeyPoint>> &keypoints, int &pointCount) {
  // get features for new frame
  Mat newDescriptors;
  vector<DMatch> good_matches;
  const float ratio_thresh = 0.7f;
  vector<std::vector<DMatch>> knn_matches;
  vector<KeyPoint> newKeypoints, kp2Sorted(keypoints[0].size(), KeyPoint());
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
  // get features for new frame
  getFeatures(frames.back(), newDescriptors, newKeypoints);
  // match features
  matcher->knnMatch(descriptor, newDescriptors, knn_matches, 2);
  //-- Filter matches using the Lowe's ratio test
  for (size_t i = 0; i < knn_matches.size(); i++) {
      if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
          good_matches.push_back(knn_matches[i][0]);
      }
  }
  // sort the feature matches
  sortPoints(good_matches, newKeypoints, kp2Sorted);
  keypoints.push_back(kp2Sorted);
  pointCount = countMatches(keypoints.back());
  // show the matching features
  showFeatureMatches(keypoints, frames, vector<DMatch>());
}

int main(int argc, char **argv) {
  int pointCount = 0;
  vector<Mat> frames(1);
  float f, cx, cy;
  VideoCapture capture("sources/record_0003_0212_Trim.mp4");
  Mat descriptor, currFrame, mask, proj;
  vector<vector<KeyPoint>> keypoints(1);

  // intrinsic parameters
  f = 693.80737981046035;
  cx = 319.5;
  cy = 2395;
  Mat intrinsics = (Mat1d(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);
  // attempt to open source video file (or webcam)
  if (!capture.isOpened()){
    //error in opening the video input
    cerr << "Unable to open file!" << endl;
    return 0;
  }
  // loop through the frames
  proj = Mat::eye(3,4, CV_64F);
  capture >> frames[0];
  while (true) {
    mask = Mat::zeros(frames[0].size(), frames[0].type());
    // get the features to track
    getFeatures(frames[0], descriptor, keypoints[0]);
    // get the # of features found
    pointCount = keypoints[0].size();
    // continue getting matching features untill not enough overlap
    // if not enough overlap -> start over w/ new reference frame
    while (pointCount > 200) {
      // get the next frame
      capture >> currFrame;
      frames.push_back(currFrame);
      // find matching features
      matchFrames(frames, descriptor, keypoints, pointCount);
    }
    getMotion(frames, keypoints, pointCount, intrinsics, proj);
    // set reference frame to be the last frame we got
    frames[0] = frames.back().clone();
    mask.release();
    frames.resize(1);
    keypoints.clear();
    keypoints.resize(1);
    currFrame.release();
  }
}
