#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

  //////////////////////////////////////////////////////////////////////
  // DEBUG CODE:
  ////////////////////////////////////////////////////////////////////// 
  int blobSize = bottom[0]->count();
  float *data0 = (float *) bottom[0]->cpu_data();
  float *data1 = (float *) bottom[1]->cpu_data(); // <- label

  int numImgs = bottom[0]->num();
  count = blobSize/numImgs;
  int edge = sqrt(count);

  char predictStr[128];
  char labelStr[128];

#ifdef gaga
  LOG(INFO) << "loss = " << loss << " count=" << count << " edge = " << edge;
  LOG(INFO) << "GRID1 predict, label";

  {
      for (int i=0; i<edge; i++) {
          strcpy(predictStr, "");
          strcpy(labelStr, "");

          for (int j=0; j<edge; j++) {
              char astr[100];
              sprintf(astr, "%0.1f ", data0[i*edge + j]);
              strcat(predictStr, astr);
              sprintf(astr, "%0.1f ", data1[i*edge + j]);
              strcat(labelStr, astr);
          }
          LOG(INFO) << "  " << predictStr << " " << labelStr;
      }
      
  }
  LOG(INFO) << " ";
#endif


  int all_zeros = 1;
  for (int i=0; i<edge; i++) {
      for (int j=0; j<edge; j++) {
          if (data0[i*edge + j] != 0) all_zeros = 0;
      }
  }

  if (all_zeros) {
      LOG(INFO) << "!!! EUCLIDEAN_LOSS: ALL ZEROS !!!";
  }
 
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
