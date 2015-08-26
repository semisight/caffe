#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
using namespace cv;

#include <string>
#include <sstream>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype> DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param, Phase phase) : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
  LOG(INFO) << "DataTransformer constructor done.";
}

template<typename Dtype> void DataTransformer<Dtype>::Transform(const Datum& datum, Dtype* transformed_data) {
  //LOG(INFO) << "Function 1 is used";
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype> void DataTransformer<Dtype>::Transform(const Datum& datum, Blob<Dtype>* transformed_blob) {
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  const int crop_size = param_.crop_size();

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  Transform(datum, transformed_data);
}

template<typename Dtype> void DataTransformer<Dtype>::Transform_nv(const Datum& datum, Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label, int cnt) {
  //std::cout << "Function 2 is used"; std::cout.flush();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int im_channels = transformed_data->channels();
  const int im_height = transformed_data->height();
  const int im_width = transformed_data->width();
  const int im_num = transformed_data->num();

  //const int lb_channels = transformed_label->channels();
  //const int lb_height = transformed_label->height();
  //const int lb_width = transformed_label->width();
  const int lb_num = transformed_label->num();

  CHECK_EQ(datum_channels, 4);
  CHECK_EQ(im_channels, 3);
  CHECK_EQ(im_num, lb_num);
  CHECK_LE(im_height, datum_height);
  CHECK_LE(im_width, datum_width);
  CHECK_GE(im_num, 1);

  //const int crop_size = param_.crop_size();

  // if (crop_size) {
  //   CHECK_EQ(crop_size, im_height);
  //   CHECK_EQ(crop_size, im_width);
  // } else {
  //   CHECK_EQ(datum_height, im_height);
  //   CHECK_EQ(datum_width, im_width);
  // }

  Dtype* transformed_data_pointer = transformed_data->mutable_cpu_data();
  Dtype* transformed_label_pointer = transformed_label->mutable_cpu_data();

  Transform_nv(datum, transformed_data_pointer, transformed_label_pointer, cnt); //call function 1
}

template<typename Dtype> void DataTransformer<Dtype>::Transform_nv(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label, int cnt) {
  //LOG(INFO) << "Function 1 is used"; std::cout.flush();

  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  //const Dtype scale = param_.scale();
  //const bool do_mirror = param_.mirror() && Rand(2);
  //const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  //const bool has_mean_values = mean_values_.size() > 0;
  int crop_x = param_.crop_size_x();
  int crop_y = param_.crop_size_y();

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  // Dtype* mean = NULL;
  // if (has_mean_file) {
  //   //CHECK_EQ(datum_channels, data_mean_.channels());
  //   CHECK_EQ(datum_height, data_mean_.height());
  //   CHECK_EQ(datum_width, data_mean_.width());
  //   mean = data_mean_.mutable_cpu_data();
  // }
  // if (has_mean_values) {
  //   CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
  //    "Specify either 1 mean_value or as many as channels: " << datum_channels;
  //   if (datum_channels > 1 && mean_values_.size() == 1) {
  //     // Replicate the mean_value for simplicity
  //     for (int c = 1; c < datum_channels; ++c) {
  //       mean_values_.push_back(mean_values_[0]);
  //     }
  //   }
  // }

  //before any transformation, get the image from datum
  //int height = datum_height;
  //int width = datum_width;
  Mat img = Mat::zeros(datum_height, datum_width, CV_8UC3);
  int offset = img.rows * img.cols;
  int dindex;
  Dtype d_element;
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      Vec3b& rgb = img.at<Vec3b>(i, j);
      for(int c = 0; c < 3; c++){
        dindex = c*offset + i*img.cols + j;
        if (has_uint8)
          d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
        else
          d_element = datum.float_data(dindex);
        rgb[c] = d_element;
      }
    }
  }
  
  //and a list of bbox
  int numOfBbox;
  if (has_uint8)
    numOfBbox = static_cast<Dtype>(static_cast<uint8_t>(data[3*offset]));
  else
    numOfBbox = datum.float_data(3*offset);
  //LOG(INFO) << "There are " << numOfBbox << " bounding boxes in image " << cnt;
  vector<vector<Dtype> > bboxlist;
  for(int i = 0; i < numOfBbox; i++){
    vector<Dtype> bbox(4);
    for(int b = 0; b < 4; b++){
      dindex = 3*offset + (i+1)*img.cols + b;
      if (has_uint8)
        d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
      else
        d_element = datum.float_data(dindex);
      bbox[b] = d_element;
    }
    //LOG(INFO) << bbox[0] << " " << bbox[1] << " " << bbox[2] << " " << bbox[3];
    bboxlist.push_back(bbox);
  }
  
  //TODO: start transform, different kinds
  Mat img_aug = Mat::zeros(crop_y, crop_x, CV_8UC3);
  //Mat img_aug;// = Mat::zeros(datum_height, datum_width, CV_8UC3);
  Mat img_temp; //size determined by scale
  vector<vector<Dtype> > bboxlist_aug;
  // We only do random transform as augmentation when training.
  if (phase_ == TRAIN) {
  //if(1) {
    as.scale = augmentation_scale(img, img_temp, bboxlist, bboxlist_aug);
    //LOG(INFO) << "scale DONE for " << cnt; std::cout.flush();
    as.crop = augmentation_crop(img_temp, img_aug, bboxlist_aug, bboxlist_aug);
    //LOG(INFO) << "crop DONE for " << cnt; std::cout.flush();
    as.flip = augmentation_flip(img_aug, img_aug, bboxlist_aug, bboxlist_aug);
    as.degree = augmentation_rotate(img_aug, img_aug, bboxlist_aug, bboxlist_aug);
    //as.degree = 0;
  }
  else {
    img_aug = img.clone();
    //LOG(INFO) << img_aug.cols;
    bboxlist_aug = bboxlist;
    as.scale = 1;
    as.crop = Size();
    as.flip = 0;
    as.degree = 0;
  }

  //copy tranformed img (img_aug) into transformed_data, do the mean-subtraction here
  //LOG(INFO) << "copy data for " << cnt << " size is " << img_aug.rows << "x" << img_aug.cols; std::cout.flush();
  offset = img_aug.rows * img_aug.cols;
  for (int i = 0; i < img_aug.rows; ++i) {
    for (int j = 0; j < img_aug.cols; ++j) {
      Vec3b& rgb = img_aug.at<Vec3b>(i, j);
      transformed_data[0*offset + i*img_aug.cols + j] = rgb[0] - 127;// - mean[0*offset + i*img.cols + j];
      transformed_data[1*offset + i*img_aug.cols + j] = rgb[1] - 127;// - mean[1*offset + i*img.cols + j];
      transformed_data[2*offset + i*img_aug.cols + j] = rgb[2] - 127;// - mean[2*offset + i*img.cols + j];
    }
  }
  //LOG(INFO) << "copy data done for " << cnt; std::cout.flush();
  // generate transformed_label based on bboxlist_aug
  // LOG(INFO) << "generating label map for " << cnt;
  //LOG(INFO) << "gen label for " << cnt; std::cout.flush();
  generateLabelMap(transformed_label, img_aug, bboxlist_aug, cnt);
  //LOG(INFO) << "gen label done for " << cnt; std::cout.flush();

  // visualize it: both before and after augmentation
  if(param_.visualize()){
    //LOG(INFO) << "begin visualize for " << cnt; std::cout.flush();
    visualize_bboxlist(img, img_aug, bboxlist, bboxlist_aug, transformed_label, as, cnt);  
    //LOG(INFO) << "visualize done for " << cnt; std::cout.flush();
  }
}

template<typename Dtype>
float DataTransformer<Dtype>::augmentation_scale(Mat& img_src, Mat& img_temp, vector<vector<Dtype> > bboxlist, vector<vector<Dtype> >& bboxlist_aug) {
  float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  float scale;
  //float scale = (param_.scale_max() - param_.scale_min()) * dice + param_.scale_min(); //linear shear into [scale_min, scale_max]
  if(dice < param_.scale_prob()) {
    img_temp = img_src.clone();
    bboxlist_aug = bboxlist;
    scale = 1;
  }
  else {
    float dice2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
    scale = (param_.scale_max() - param_.scale_min()) * dice2 + param_.scale_min(); //linear shear into [scale_min, scale_max]
    resize(img_src, img_temp, Size(), scale, scale, INTER_LINEAR);
    bboxlist_aug.clear();
    for(int i=0; i<bboxlist.size(); i++){ //for every bbox
      //LOG(INFO) << "bbox was " << bboxlist[i][0] << " " << bboxlist[i][1] << " " << bboxlist[i][2] << " " << bboxlist[i][3];
      vector<Dtype> scaled(4);
      scaled[0] = bboxlist[i][0] * scale;
      scaled[1] = bboxlist[i][1] * scale;
      scaled[2] = bboxlist[i][2] * scale;
      scaled[3] = bboxlist[i][3] * scale;
      bboxlist_aug.push_back(scaled);
      //LOG(INFO) << "and it is now " << scaled[0] << " " << scaled[1] << " " << scaled[2] << " " << scaled[3];
    }
  }
  return scale;
}

template<typename Dtype>
Size DataTransformer<Dtype>::augmentation_crop(Mat& img_temp, Mat& img_aug, vector<vector<Dtype> > bboxlist, vector<vector<Dtype> >& bboxlist_aug) {
  float dice_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  float dice_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  int crop_x = param_.crop_size_x();
  int crop_y = param_.crop_size_y();

  float x_offset = (img_temp.cols > crop_x ? int(dice_x * (img_temp.cols - crop_x) + 0.5) : 0);
  float y_offset = (img_temp.rows > crop_y ? int(dice_y * (img_temp.rows - crop_y) + 0.5) : 0);

  //LOG(INFO) << "Size of img_temp is " << img_temp.cols << " " << img_temp.rows;
  //LOG(INFO) << "ROI is " << x_offset << " " << y_offset << " " << min(800, img_temp.cols) << " " << min(256, img_temp.rows);
  
  Rect cropROI(x_offset, y_offset, min(crop_x, img_temp.cols), min(crop_y, img_temp.rows));
  Mat dstROI = img_aug(Rect(0, 0, min(crop_x, img_temp.cols), min(crop_y, img_temp.rows)));
  img_temp(cropROI).copyTo(dstROI);

  bboxlist_aug.clear();
  for(int i=0; i<bboxlist.size(); i++){ //for every bbox
    vector<Dtype> cropped(4);
    cropped[0] = bboxlist[i][0] - x_offset;
    cropped[1] = bboxlist[i][1] - y_offset;
    //to x1,y1,x2,y2 first
    cropped[2] = cropped[0] + bboxlist[i][2];
    cropped[3] = cropped[1] + bboxlist[i][3];
    //make sure they are in screen
    cropped[0] = std::max(Dtype(0), std::min(cropped[0], Dtype(crop_x-1)));
    cropped[1] = std::max(Dtype(0), std::min(cropped[1], Dtype(crop_y-1)));
    cropped[2] = std::max(Dtype(0), std::min(cropped[2], Dtype(crop_x-1)));
    cropped[3] = std::max(Dtype(0), std::min(cropped[3], Dtype(crop_y-1)));
    //LOG(INFO) << "rotated is " << rotated[0] << " " << rotated[1] << " " << rotated[2] << " " << rotated[3];

    //back to x1,y1,w,h format
    cropped[2] = cropped[2] - cropped[0];
    cropped[3] = cropped[3] - cropped[1];
    //LOG(INFO) << "rotated is " << rotated[0] << " " << rotated[1] << " " << rotated[2] << " " << rotated[3];

    if(cropped[2] > 0 && cropped[3] > 0)
      bboxlist_aug.push_back(cropped);
  }
  return Size(x_offset, y_offset);
}


template<typename Dtype>
bool DataTransformer<Dtype>::augmentation_flip(Mat& img_src, Mat& img_aug, vector<vector<Dtype> > bboxlist, vector<vector<Dtype> >& bboxlist_aug) {

  float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  bool doflip = (dice <= param_.flip_prob());
  if(doflip){
    flip(img_src, img_aug, 1);
    int w = img_src.cols;

    bboxlist_aug.clear();
    for(int i=0; i<bboxlist.size(); i++){ //for every bbox
      vector<Dtype> flipped(4);
      flipped[0] = w - bboxlist[i][0] - 1 - bboxlist[i][2];
      flipped[1] = bboxlist[i][1];
      flipped[2] = bboxlist[i][2];
      flipped[3] = bboxlist[i][3];
      bboxlist_aug.push_back(flipped);
    }
  }
  else {
    img_aug = img_src.clone();
    bboxlist_aug = bboxlist;
  }
  return doflip;
}

template<typename Dtype>
float DataTransformer<Dtype>::augmentation_rotate(Mat& img_src, Mat& img_aug, vector<vector<Dtype> > bboxlist, vector<vector<Dtype> >& bboxlist_aug) {
  
  float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  float degree = (dice - 0.5) * 2 * param_.max_rotate_degree();
  
  Point2f pt(img_src.cols/2.0, img_src.rows/2.0);
  Mat R(2,3,CV_32FC1);
  R = getRotationMatrix2D(pt, degree, 1.0);
  //LOG(INFO) << "R=[" << R.at<double>(0,0) << " " << R.at<double>(0,1) << " " << R.at<double>(0,2) << ";" 
  //          << R.at<double>(1,0) << " " << R.at<double>(1,1) << " " << R.at<double>(1,2) << "]";
  warpAffine(img_src, img_aug, R, Size(img_src.cols, img_src.rows));

  // adjust bbox for ratation TBD
  bboxlist_aug.clear();
  //Dtype abox[4] = {0, 0, 0, 0};
  //Dtype screenp[4] = {0, 0, img_src.cols, img_src.rows};
  int w = img_src.cols;
  int h = img_src.rows;
  for (int n = 0; n < bboxlist.size(); n++){
    //LOG(INFO) << "bboxlist is " << bboxlist[n][0] << " " << bboxlist[n][1] << " " << bboxlist[n][2] << " " << bboxlist[n][3];
    Mat center(3,1,CV_64FC1); //homogeneous 
    center.at<double>(0,0) = bboxlist[n][0] + bboxlist[n][2]/2.0;
    center.at<double>(1,0) = bboxlist[n][1] + bboxlist[n][3]/2.0;
    center.at<double>(2,0) = 1;
    //LOG(INFO) << "center is " << center.at<double>(0,0) << " " << center.at<double>(1,0) << " " << center.at<double>(2,0);

    Mat new_center(3,1,CV_64FC1);
    new_center = R * center;
    //LOG(INFO) << "new_center is " << new_center.at<double>(0,0) << " " << new_center.at<double>(1,0) << " " << new_center.at<double>(2,0);

    vector<Dtype> rotated(4); //in x1,y1,x2,y2 format first
    rotated[0] = new_center.at<double>(0,0) - bboxlist[n][2]/2.0;
    rotated[1] = new_center.at<double>(1,0) - bboxlist[n][3]/2.0;
    rotated[2] = new_center.at<double>(0,0) + bboxlist[n][2]/2.0;
    rotated[3] = new_center.at<double>(1,0) + bboxlist[n][3]/2.0;
    //LOG(INFO) << "rotated is " << rotated[0] << " " << rotated[1] << " " << rotated[2] << " " << rotated[3];

    //make sure it's on screen
    rotated[0] = std::max(Dtype(0), std::min(rotated[0], Dtype(w-1)));
    rotated[1] = std::max(Dtype(0), std::min(rotated[1], Dtype(h-1)));
    rotated[2] = std::max(Dtype(0), std::min(rotated[2], Dtype(w-1)));
    rotated[3] = std::max(Dtype(0), std::min(rotated[3], Dtype(h-1)));
    //LOG(INFO) << "rotated is " << rotated[0] << " " << rotated[1] << " " << rotated[2] << " " << rotated[3];

    //back to x1,y1,w,h format
    rotated[2] = rotated[2] - rotated[0];
    rotated[3] = rotated[3] - rotated[1];
    //LOG(INFO) << "rotated is " << rotated[0] << " " << rotated[1] << " " << rotated[2] << " " << rotated[3];

    if(rotated[2] > 0 && rotated[3] > 0)
      bboxlist_aug.push_back(rotated);
  }
  return degree;
}


template<typename Dtype>
void DataTransformer<Dtype>::generateLabelMap(Dtype* transformed_label, Mat img_aug, vector<vector<Dtype> > bboxlist_aug, int cnt) {
  //generate transformed_label based on bboxlist_aug
  //translating code in "fill_grid_with_bboxes" in m_utility.py
  int rezX = img_aug.cols;
  int rezY = img_aug.rows;
  int cell_width = param_.stride();
  int cell_height = param_.stride();
  float scale_cvg = param_.scale_cvg();
  int max_cvg_len = param_.max_cvg_len();
  int min_cvg_len = param_.min_cvg_len();
  bool opaque_coverage = param_.opaque_coverage();
  string coverage = param_.coverage();
  int grid_x = rezX / cell_width;
  int grid_y = rezY / cell_height;
  int channelOffset = grid_y * grid_x;

  //LOG(INFO) << "There are " << bboxlist_aug.size() << " bbox in " << cnt; std::cout.flush();

  // //clear out transformed_label, it may remain things for last batch
  //LOG(INFO) << "cleaning"; std::cout.flush();
  for (int g_x = 0; g_x < grid_x; g_x++){
    for (int g_y = 0; g_y < grid_y; g_y++){
      for (int i = 0; i < 5; i++){
        transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0;
      }
    }
  }
  // LOG(INFO) << "cleaning done"; std::cout.flush();


  for(int b = 0; b < bboxlist_aug.size(); b++){
    //the full bbox
    int x1 = bboxlist_aug[b][0];
    int y1 = bboxlist_aug[b][1];
    int x2 = x1 + bboxlist_aug[b][2];
    int y2 = y1 + bboxlist_aug[b][3];

    //make sure on screen
    x1 = max(0, min(x1, rezX-1));
    x2 = max(0, min(x2, rezX-1));
    y1 = max(0, min(y1, rezY-1));
    y2 = max(0, min(y2, rezY-1));

    int w = max(1, x2-x1);
    int h = max(1, y2-y1);
    float center_x = x1 + w/2.0;
    float center_y = y1 + h/2.0;

    float shrunk_width = w * scale_cvg;
    float shrunk_height = h * scale_cvg;

    if(!coverage.compare("gridbox_max")){
      shrunk_width = min((float)max_cvg_len, shrunk_width);
      shrunk_height = min((float)max_cvg_len, shrunk_height);
    }
    else if(!coverage.compare("gridbox_min")){
      shrunk_width = min((float)w, max((float)min_cvg_len, shrunk_width));
      shrunk_height = min((float)h, max((float)min_cvg_len, shrunk_height));
    }

    if(opaque_coverage){
      // Zero out coverage surface for whole bbox first.
      // This help avoids ambiguous coverage regions for very close cars.
      int box_cell_x1 = int((center_x - w/2)/cell_width);
      int box_cell_y1 = int((center_y - h/2)/cell_height);
      int box_cell_x2 = int((center_x + w/2)/cell_width);
      int box_cell_y2 = int((center_y + h/2)/cell_height);

      for (int g_x = box_cell_x1; g_x <= box_cell_x2; g_x++){
        for (int g_y = box_cell_y1; g_y <= box_cell_y2; g_y++){
          if (g_x < 0 || g_x >= grid_x || g_y < 0 || g_y >= grid_y) 
            continue;
          for (int i = 0; i < 5; i++)
            transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0;
        }
      }
    }

    //shrunk version
    float x1s = center_x - shrunk_width/2.0;
    float y1s = center_y - shrunk_height/2.0;
    float x2s = center_x + shrunk_width/2.0;
    float y2s = center_y + shrunk_height/2.0;

    //LOG(INFO) << "shrunk bbox: " << x1s << " " << y1s << " " << x2s << " " << y2s;

    int box_cell_x1 = int(x1s/cell_width);
    int box_cell_y1 = int(y1s/cell_height);
    int box_cell_x2 = int(x2s/cell_width);
    int box_cell_y2 = int(y2s/cell_height);

    //LOG(INFO) << "bbox cell: " << box_cell_x1 << " " << box_cell_y1 << " " << box_cell_x2 << " " << box_cell_y2;

    //int counter = 0;
    for (int g_x = box_cell_x1; g_x <= box_cell_x2; g_x++){
      for (int g_y = box_cell_y1; g_y <= box_cell_y2; g_y++){
        float this_cell_x = cell_width * (g_x + 0.5);
        float this_cell_y = cell_height * (g_y + 0.5);
        //cell is covered if center of cell lives inside bbox
        bool covered = (this_cell_x >= x1s && this_cell_x < x2s && this_cell_y >= y1s && this_cell_y < y2s);
        if(covered){
          //coverage
          transformed_label[channelOffset*0 + g_y*grid_x + g_x] = 1.0;
          //counter++;
          //bbox
          transformed_label[channelOffset*1 + g_y*grid_x + g_x] = (float(x1 - this_cell_x)/param_.bbox_norm_factor() + 1.0)/2.0; //((x1 - this_cell_x)/float(rezX) + 1)/2.0;
          transformed_label[channelOffset*2 + g_y*grid_x + g_x] = (float(y1 - this_cell_y)/param_.bbox_norm_factor() + 1.0)/2.0; //((y1 - this_cell_y)/float(rezY) + 1)/2.0;
          transformed_label[channelOffset*3 + g_y*grid_x + g_x] = (float(x2 - this_cell_x)/param_.bbox_norm_factor() + 1.0)/2.0; //((x2 - this_cell_x)/float(rezX) + 1)/2.0;
          transformed_label[channelOffset*4 + g_y*grid_x + g_x] = (float(y2 - this_cell_y)/param_.bbox_norm_factor() + 1.0)/2.0; //((y2 - this_cell_y)/float(rezY) + 1)/2.0;
        }
        //LOG(INFO) << this_cell_x << " and " << this_cell_y << " is covered? " << covered;
      }
    }
    //LOG(INFO) << counter << " grids are painted";
  }
}


void setLabel(Mat& im, const std::string label, const Point& org)
{
    int fontface = FONT_HERSHEY_SIMPLEX;
    double scale = 1;
    int thickness = 1;
    int baseline = 0;

    Size text = getTextSize(label, fontface, scale, thickness, &baseline);
    rectangle(im, org + Point(0, baseline), org + Point(text.width, -text.height), CV_RGB(0,0,0), CV_FILLED);
    putText(im, label, org, fontface, scale, CV_RGB(255,255,255), thickness, 20);
}

template<typename Dtype>
void DataTransformer<Dtype>::visualize_bboxlist(Mat& img, Mat& img_aug, vector<vector<Dtype> >& bboxlist, vector<vector<Dtype> >& bboxlist_aug, Dtype* transformed_label, 
                                                AugmentSelection as, int cnt) {
  Mat img_vis = Mat::zeros(img.rows*2, img.cols, CV_8UC3);
  //copy image content
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      Vec3b& rgb = img.at<Vec3b>(i, j);
      Vec3b& rgb_vis_upper = img_vis.at<Vec3b>(i, j);
      rgb_vis_upper = rgb;
    }
  }
  for (int i = 0; i < img_aug.rows; ++i) {
    for (int j = 0; j < img_aug.cols; ++j) {
      Vec3b& rgb_aug = img_aug.at<Vec3b>(i, j);
      Vec3b& rgb_vis_lower = img_vis.at<Vec3b>(i + img.rows, j);
      rgb_vis_lower = rgb_aug;
    }
  }

  for(int i=0; i<bboxlist.size(); i++){
    Point p1(bboxlist[i][0], bboxlist[i][1]);
    Point p2(bboxlist[i][0] + bboxlist[i][2], bboxlist[i][1] + bboxlist[i][3]);
    Scalar color(0, 255, 0);
    rectangle(img_vis, p1, p2, color, 2, 8);
  }
  // for(int i=0; i<bboxlist.size(); i++){
  //   Point p1(bboxlist_aug[i][0], bboxlist_aug[i][1]+img.rows);
  //   Point p2(bboxlist_aug[i][0] + bboxlist_aug[i][2], bboxlist_aug[i][1] + bboxlist_aug[i][3]+img.rows);
  //   Scalar color(0, 255, 255);
  //   rectangle(img_vis, p1, p2, color, 2, 8);
  // }

  // draw transformed label
  int grid_x = img_aug.cols / param_.stride();
  int grid_y = img_aug.rows / param_.stride();
  int channelOffset = grid_x * grid_y;
  int cell_width = param_.stride();
  int cell_height = param_.stride();

  int bboxCounter = 0;
  for (int cell_y = 0; cell_y < grid_y; cell_y++) {
    for (int cell_x = 0; cell_x < grid_x; cell_x++){
      int idx = cell_y * grid_x + cell_x;
      float cvg = transformed_label[channelOffset*0 + idx];
      
      Scalar color(0, 0, int(255*cvg));

      // draw coverage dot
      int x = int((float(cell_x) + 0.5) * cell_width);
      int y = int((float(cell_y) + 0.5) * cell_height);

      if(cvg > 0.5){
        //LOG(INFO) << "cell_x,cell_y = " << cell_x << " " << cell_y << ", cvg = " << cvg;
        circle(img_vis, Point(x,y+img.rows), 2, color, -1); //-1 means filled
      }

      // draw bounding box
      if(transformed_label[channelOffset*1 + idx] != 0 && transformed_label[channelOffset*2 + idx] != 0 &&
         transformed_label[channelOffset*3 + idx] != 0 && transformed_label[channelOffset*4 + idx] != 0){
        int x1 = int(((transformed_label[channelOffset*1 + idx]*2.0-1.0) * param_.bbox_norm_factor()) + x); //int(((transformed_label[channelOffset*1 + idx]*2-1) * img.cols) + x);
        int y1 = int(((transformed_label[channelOffset*2 + idx]*2.0-1.0) * param_.bbox_norm_factor()) + y); //int(((transformed_label[channelOffset*2 + idx]*2-1) * img.rows) + y);
        int x2 = int(((transformed_label[channelOffset*3 + idx]*2.0-1.0) * param_.bbox_norm_factor()) + x); //int(((transformed_label[channelOffset*3 + idx]*2-1) * img.cols) + x);
        int y2 = int(((transformed_label[channelOffset*4 + idx]*2.0-1.0) * param_.bbox_norm_factor()) + y); //int(((transformed_label[channelOffset*4 + idx]*2-1) * img.rows) + y);

        rectangle(img_vis, Point(x1,y1+img.rows), Point(x2,y2+img.rows), Scalar(0,255,0), 1);
        bboxCounter++;
      }

    }
  }
  //LOG(INFO) << bboxCounter << " bboxes are drawn";

  // draw text
  if(phase_ == TRAIN){
    std::stringstream ss;
    ss << "Augmenting with:" << (as.flip ? "flip" : "no flip") << "; Rotate " << as.degree << " deg; scaling: " << as.scale << "; crop: " 
       << as.crop.height << "," << as.crop.width;
    std::string str_info = ss.str();
    setLabel(img_vis, str_info, Point(0, img_vis.rows-50));

    rectangle(img_vis, Point(0,0+img.rows), Point(param_.crop_size_x(), param_.crop_size_y()+img.rows), Scalar(255,255,255), 1);

    char imagename [100];
    sprintf(imagename, "%s/augment_%d.jpg", param_.img_header().c_str(), cnt);
    //LOG(INFO) << "filename is " << imagename;
    imwrite(imagename, img_vis);
  }
  else {
    string str_info = "no augmentation for testing";
    setLabel(img_vis, str_info, Point(0, img_vis.rows-50));

    char imagename [100];
    sprintf(imagename, "%s/augment_%d.jpg", param_.img_header().c_str(), cnt);
    //LOG(INFO) << "filename is " << imagename;
    imwrite(imagename, img_vis);
  }
}




template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == TRAIN && param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
