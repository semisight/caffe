#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
//#include <opencv2/highgui/highgui.h>

using namespace cv;
//using namespace std;

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
NVDataLayer<Dtype>::~NVDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void NVDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Initialize DB
  db_.reset(db::GetDB(this->layer_param_.nvdata_param().backend()));
  db_->Open(this->layer_param_.nvdata_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());

  // Check if we should randomly skip a few data points
  if (this->layer_param_.nvdata_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.nvdata_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      cursor_->Next();
    }
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(cursor_->value());

  //LOG(INFO) << "Datum dimension: " << datum.channels() << datum.height() << datum.width();

  bool force_color = this->layer_param_.nvdata_param().force_encoded_color();
  if ((force_color && DecodeDatum(&datum, true)) ||
      DecodeDatumNative(&datum)) {
    LOG(INFO) << "Decoding Datum";
  }

  // image
  int crop_size = this->layer_param_.transform_param().crop_size();
  int crop_size_x = this->layer_param_.transform_param().crop_size_x();
  int crop_size_y = this->layer_param_.transform_param().crop_size_y();

  if (crop_size > 0) {
    top[0]->Reshape(this->layer_param_.nvdata_param().batch_size(),
        datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(this->layer_param_.nvdata_param().batch_size(),
        datum.channels(), crop_size, crop_size);
    this->transformed_data_.Reshape(1, datum.channels(), crop_size, crop_size);
  } 
  else {
    if(this->phase_ == TRAIN){
      top[0]->Reshape(
          this->layer_param_.nvdata_param().batch_size(), 
          //3, datum.height(), datum.width());
          3, crop_size_y, crop_size_x);
      this->prefetch_data_.Reshape(this->layer_param_.nvdata_param().batch_size(),
          //3, datum.height(), datum.width());
          3, crop_size_y, crop_size_x);
      this->transformed_data_.Reshape(1, 
          //3, datum.height(), datum.width());
          3, crop_size_y, crop_size_x);
    }
    else {
      top[0]->Reshape(this->layer_param_.nvdata_param().batch_size(), 3, datum.height(), datum.width());
      this->prefetch_data_.Reshape(this->layer_param_.nvdata_param().batch_size(), 3, datum.height(), datum.width());
      this->transformed_data_.Reshape(1, 3, datum.height(), datum.width());
    }
  }

  //LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

  // label
  if (this->output_labels_) { //output_labels_ will be 1 if there are 2 tops in prototxt, see base_data_layer.cpp
    const int stride = this->layer_param_.transform_param().stride();
    if(this->phase_ == TRAIN){
      top[1]->Reshape(this->layer_param_.nvdata_param().batch_size(), 
                    //5, datum.height()/stride, datum.width()/stride);
                    5, crop_size_y/stride, crop_size_x/stride);
      this->prefetch_label_.Reshape(this->layer_param_.nvdata_param().batch_size(), 
                    //5, datum.height()/stride, datum.width()/stride);
                    5, crop_size_y/stride, crop_size_x/stride);
      this->transformed_label_.Reshape(1, 
                    //5, datum.height()/stride, datum.width()/stride);
                    5, crop_size_y/stride, crop_size_x/stride);
    }
    else {
      top[1]->Reshape(this->layer_param_.nvdata_param().batch_size(), 5, datum.height()/stride, datum.width()/stride);
      this->prefetch_label_.Reshape(this->layer_param_.nvdata_param().batch_size(), 5, datum.height()/stride, datum.width()/stride);
      this->transformed_label_.Reshape(1, 5, datum.height()/stride, datum.width()/stride);
    }
    LOG(INFO) << "output label size: " << top[1]->num() << "," << top[1]->channels() << "," << top[1]->height() << "," << top[1]->width();
  }
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void NVDataLayer<Dtype>::InternalThreadEntry() {
  static int cnt = -1;
  cnt++;
  //LOG(INFO) << "InternalThreadEntry = " << cnt; 
  std::cout.flush();
  
  //LOG(INFO) << "this->prefetch_data_.count() = " << this->prefetch_data_.count();
  //LOG(INFO) << "this->output_labels_ = " << this->output_labels_; //always zero

  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  CHECK(this->transformed_label_.count());

  // Reshape on single input batches for inputs of varying dimension.
  const int batch_size = this->layer_param_.nvdata_param().batch_size();
  const int crop_size = this->layer_param_.transform_param().crop_size();

  bool force_color = this->layer_param_.nvdata_param().force_encoded_color();
  if (batch_size == 1 && crop_size == 0) {
    Datum datum;
    datum.ParseFromString(cursor_->value());
    if (datum.encoded()) {
      if (force_color) {
        DecodeDatum(&datum, true);
      } else {
        DecodeDatumNative(&datum);
      }
    }
    this->prefetch_data_.Reshape(1, datum.channels(), datum.height(), datum.width());
    this->transformed_data_.Reshape(1, 3, datum.height(), datum.width());
  }

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a blob
    Datum datum;
    datum.ParseFromString(cursor_->value());

    // fetch 4-dim datum from lmdb
    cv::Mat cv_img;
    if (datum.encoded()) {
      if (force_color) {
        cv_img = DecodeDatumToCVMat(datum, true);
      } else {
        cv_img = DecodeDatumToCVMatNative(datum);
      }
      if (cv_img.channels() != this->transformed_data_.channels()) {
        LOG(WARNING) << "Your dataset contains encoded images with mixed "
        << "channel sizes. Consider adding a 'force_color' flag to the "
        << "model definition, or rebuild your dataset using "
        << "convert_imageset.";
      }
    }
    read_time += timer.MicroSeconds();
    timer.Start();

    // Apply data transformations (mirror, scale, crop...)
    // for both image and label map
    int offset_data = this->prefetch_data_.offset(item_id);
    int offset_label = this->prefetch_label_.offset(item_id);
    //LOG(INFO) << "prefetch label size " << this->prefetch_label_.count();
    //LOG(INFO) << "offset of data and label are " << offset_data << " and " << offset_label;
    this->transformed_data_.set_cpu_data(top_data + offset_data);
    this->transformed_label_.set_cpu_data(top_label + offset_label);

    if (datum.encoded()) {
      //LOG(INFO) << "cv_img is being used for number " << cnt;
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    } else {
      //LOG(INFO) << "datum is being used for size " << datum.channels() << " " << datum.height() << " " << datum.width();
      this->data_transformer_->Transform_nv(datum, &(this->transformed_data_), &(this->transformed_label_), cnt*batch_size+item_id);
      //LOG(INFO) << "datum is used for number " << cnt;
    }

    //label
    // if (this->output_labels_) {
    //   top_label[item_id] = datum.label();
    // }
    trans_time += timer.MicroSeconds();
    // go to the next iter
    cursor_->Next();
    if (!cursor_->valid()) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      cursor_->SeekToFirst();
    }
  }  //batch size
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";

  std::cout.flush();
}

// template <typename Dtype>
// void NVDataLayer<Dtype>::generateLabelMap() {
//   LOG(INFO) << "dim of this->prefetch_data is: " << this->prefetch_data_.num()    << "," << this->prefetch_data_.channels() << "," 
//                                                   << this->prefetch_data_.height() << "," << this->prefetch_data_.width();
//   LOG(INFO) << "dim of this->prefetch_label is: " << this->prefetch_label_.num()    << "," << this->prefetch_label_.channels() << "," 
//                                                   << this->prefetch_label_.height() << "," << this->prefetch_label_.width();

// }


INSTANTIATE_CLASS(NVDataLayer);
REGISTER_LAYER_CLASS(NVData);

}  // namespace caffe
