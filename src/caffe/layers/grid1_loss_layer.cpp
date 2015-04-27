#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <string>

namespace caffe {

template <typename Dtype>
void Grid1LossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
                                    const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void Grid1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());

  int blobSize = bottom[0]->count();
  float *label = (float *) bottom[1]->cpu_data();
  float lambda = 0.33;

  for (int i=0; i<blobSize; i++) {
      diff_.mutable_cpu_data()[i] *= (label[i] ? (1 + lambda) : lambda);
  }

  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

#define gaga 1
#ifdef gaga
  //////////////////////////////////////////////////////////////////////
  // DEBUG CODE:
  ////////////////////////////////////////////////////////////////////// 
  static int print_cnt;

  float *data0 = (float *) bottom[0]->cpu_data();
  float *data1 = (float *) bottom[1]->cpu_data(); // <- label
  
  int numImgs = bottom[0]->num();
  count = blobSize/numImgs;
  int blob_w = bottom[0]->width();
  int blob_h = bottom[0]->height();

  // Figure out dimentionality of data
  if (count == 4320) {
      // is 120x36
      blob_w = 120;
      blob_h = 36;
  } else if (count == 1080) {
      blob_w = 60;
      blob_h = 18;
  } else if (blob_w==1 && blob_h==1) {
      blob_w = sqrt(count);
      blob_h = blob_w;
  }

  string predictStr;
  string labelStr;


  if (print_cnt % 256 == 0) {
      int all_zeros = 1;
      for (int i=0; i<blob_h; i++) {
          for (int j=0; j<blob_w; j++) {
              if (data0[i*blob_w + j] != 0) all_zeros = 0;
          }
      }
      if (all_zeros) {
          LOG(INFO) << "!!! GRID1_LOSS: ALL ZEROS !!!";
      }
    
      LOG(INFO) << "loss = " << loss << " count=" << count << " numImgs= " << numImgs;
      LOG(INFO) << "h = " << bottom[0]->height() << " w=" << bottom[0]->width() << " channels = " << bottom[0]->channels();
      LOG(INFO) << "GRID1 predict, label ";

      for (int i=0; i<blob_h; i+=blob_h/8) {
          predictStr = "";
          labelStr = "";

          for (int j=0; j<blob_w; j+=blob_w/8) {
              char astr[10];
              sprintf(astr, "%0.1f " , data0[i*blob_w + j]);
              predictStr += astr;
              sprintf(astr, "%0.1f " , data1[i*blob_w + j]);
              labelStr += astr;
          }
          LOG(INFO) << "  " << predictStr << " " << labelStr;
      }
      LOG(INFO) << " ";
  }
  print_cnt++;
#endif
}

template <typename Dtype>
void Grid1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(Grid1LossLayer);
#endif

INSTANTIATE_CLASS(Grid1LossLayer);
REGISTER_LAYER_CLASS(Grid1Loss);

}  // namespace caffe
