#include <vector>
#include <algorithm>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype> string my_debug_symbol_bbox_l2n(Dtype value);
template <typename Dtype> Dtype IOU(Dtype x1, Dtype y1, Dtype x2, Dtype y2, Dtype u1, Dtype v1, Dtype u2, Dtype v2);
template <typename Dtype> string my_debug_symbol_IOU(Dtype value);

template <typename Dtype>
void L2NLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void L2NLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

  //start to print debug info
  if(this->layer_param_.l2nloss_param().debug_info()){
    //////////////////////////////////////////////////////////////////////
    // DEBUG CODE:
    ////////////////////////////////////////////////////////////////////// 
    static int print_cnt;

    float *data0 = (float *) bottom[0]->cpu_data();
    float *data1 = (float *) bottom[1]->cpu_data(); // <- label
    int blobSize = bottom[0]->count();
    
    int numImgs = bottom[0]->num();
    count = blobSize/numImgs;
    int blob_w = bottom[0]->width();
    int blob_h = bottom[0]->height();
    int blob_c = bottom[0]->channels();
    Dtype loss = top[0]->cpu_data()[0];
    int stride = blob_w * blob_h;

    string predictStr;
    string labelStr;

    if (print_cnt % 50 == 0) {
        LOG(INFO) << "loss = " << loss << " count=" << count << " numImgs= " << numImgs << " top[0]->cpu_diff()[0] " << top[0]->cpu_diff()[0];
        LOG(INFO) << "h = " << bottom[0]->height() << " w=" << bottom[0]->width() << " channels = " << bottom[0]->channels();
        LOG(INFO) << "L2N debugging: difference between pred and gt";

        for (int i=0; i<blob_h; i++) {
            predictStr = "";
            for (int j=0; j<blob_w; j++) {
                Dtype absDiff = 0;
                if(data0[0*stride + i*blob_w + j] == 0 && data1[0*stride + i*blob_w + j] == 0 && data0[1*stride + i*blob_w + j] == 0 && data1[1*stride + i*blob_w + j] == 0 &&
                   data0[2*stride + i*blob_w + j] == 0 && data1[2*stride + i*blob_w + j] == 0 && data0[3*stride + i*blob_w + j] == 0 && data1[3*stride + i*blob_w + j] == 0) {
                  predictStr.append("-");
                }
                else {
                  for (int c=0; c<blob_c; c++){
                      Dtype value_pred = data0[c*stride + i*blob_w + j];
                      Dtype value_label = data1[c*stride + i*blob_w + j];
                      Dtype diff = value_pred - value_label;
                      absDiff += diff > 0 ? diff : -diff;
                  }
                  predictStr.append(my_debug_symbol_bbox_l2n(absDiff));
                }
            }
            LOG(INFO) << "  " << predictStr;
        }
        LOG(INFO) << "L2N debugging: IOU";

        for (int i=0; i<blob_h; i++) {
            predictStr = "";
            for (int j=0; j<blob_w; j++) {
                Dtype value = IOU(data0[0*stride+i*blob_w+j], data0[1*stride+i*blob_w+j], data0[2*stride+i*blob_w+j], data0[3*stride+i*blob_w+j], 
                                  data1[0*stride+i*blob_w+j], data1[1*stride+i*blob_w+j], data1[2*stride+i*blob_w+j], data1[3*stride+i*blob_w+j]);
                if(data0[0*stride + i*blob_w + j] == 0 && data1[0*stride + i*blob_w + j] == 0 && data0[1*stride + i*blob_w + j] == 0 && data1[1*stride + i*blob_w + j] == 0 &&
                   data0[2*stride + i*blob_w + j] == 0 && data1[2*stride + i*blob_w + j] == 0 && data0[3*stride + i*blob_w + j] == 0 && data1[3*stride + i*blob_w + j] == 0) {
                  predictStr.append("-");
                }
                else predictStr.append(my_debug_symbol_IOU(value));
            }
            LOG(INFO) << "  " << predictStr;
        }
        LOG(INFO) << " ";
    }
    print_cnt++;
  } // end if debug
}

template <typename Dtype>
Dtype IOU(Dtype x1, Dtype y1, Dtype x2, Dtype y2, Dtype u1, Dtype v1, Dtype u2, Dtype v2) {
  Dtype I_x = std::max(std::min(x2, u2) - std::max(x1, u1), Dtype(0));
  Dtype I_y = std::max(std::min(y2, v2) - std::max(y1, v1), Dtype(0));
  Dtype I = I_x * I_y;
  Dtype A = (y2-y1) * (x2-x1);
  Dtype B = (v2-v1) * (u2-u1);
  Dtype U = A + B - I;
  return I/U;
}

template <typename Dtype>
string my_debug_symbol_bbox_l2n(Dtype value) {
  string ans;
  if(value == 1) ans="|";
  else if(value >= 0.155) ans="X";
  else if(value >= 0.145) ans="F";
  else if(value >= 0.135) ans="E";
  else if(value >= 0.125) ans="D";
  else if(value >= 0.115) ans="C";
  else if(value >= 0.105) ans="B";
  else if(value >= 0.095) ans="A";
  else if(value >= 0.085) ans="9";
  else if(value >= 0.075) ans="8";
  else if(value >= 0.065) ans="7";
  else if(value >= 0.055) ans="6";
  else if(value >= 0.045) ans="5";
  else if(value >= 0.035) ans="4";
  else if(value >= 0.025) ans="3";
  else if(value >= 0.015) ans="2";
  else if(value >= 0.005) ans="1";
  else ans = "o";

  return ans;
}

template <typename Dtype>
string my_debug_symbol_IOU(Dtype value) {
  string ans;
  if(value >= 0.9) ans="9";
  else if(value >= 0.8) ans="8";
  else if(value >= 0.7) ans="7";
  else if(value >= 0.6) ans="6";
  else if(value >= 0.5) ans="5";
  else if(value >= 0.4) ans="4";
  else if(value >= 0.3) ans="3";
  else if(value >= 0.2) ans="2";
  else if(value >= 0.1) ans="1";
  else ans = "0";
  return ans;
}

template <typename Dtype>
void L2NLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU(L2NLossLayer);
#endif

INSTANTIATE_CLASS(L2NLossLayer);
REGISTER_LAYER_CLASS(L2NLoss);

}  // namespace caffe
