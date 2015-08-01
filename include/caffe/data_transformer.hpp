#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class DataTransformer {
 public:
  explicit DataTransformer(const TransformationParameter& param, Phase phase);
  virtual ~DataTransformer() {}

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);
  void Transform_nv(const Datum& datum, Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_label_blob, int cnt); //image and label
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Datum.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<Datum> & datum_vector,
                Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Mat.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<cv::Mat> & mat_vector,
                Blob<Dtype>* transformed_blob);
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   */
  void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the same transformation defined in the data layer's
   * transform_param block to all the num images in a input_blob.
   *
   * @param input_blob
   *    A Blob containing the data to be transformed. It applies the same
   *    transformation to all the num images in the blob.
   * @param transformed_blob
   *    This is destination blob, it will contain as many images as the
   *    input blob. It can be part of top blob's data.
   */
  void Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);

  struct AugmentSelection
  {
      bool flip;
      float degree;
      Size crop;
      float scale;
  };

  void generateLabelMap(Dtype*, Mat img_aug, vector<vector<Dtype> > bboxlist_aug, int cnt);
  void visualize_bboxlist(Mat& img, Mat& img_aug, vector<vector<Dtype> >& bboxlist, vector<vector<Dtype> >& bboxlist_aug, Dtype* transformed_label, AugmentSelection, int cnt);

  bool augmentation_flip(Mat& img, Mat& img_aug, vector<vector<Dtype> > bboxlist, vector<vector<Dtype> >&);
  float augmentation_rotate(Mat& img_src, Mat& img_aug, vector<vector<Dtype> > bboxlist, vector<vector<Dtype> >&);
  float augmentation_scale(Mat& img, Mat& img_temp, vector<vector<Dtype> > bboxlist, vector<vector<Dtype> >&);
  Size augmentation_crop(Mat& img_temp, Mat& img_aug, vector<vector<Dtype> > bboxlist, vector<vector<Dtype> >&);

  AugmentSelection as;

 protected:
   /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  virtual int Rand(int n);

  void Transform(const Datum& datum, Dtype* transformed_data);
  void Transform_nv(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label, int cnt);
  
  // Tranformation parameters
  TransformationParameter param_;

  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
};



}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_

