#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype> string my_debug_symbol_bbox(Dtype value);

template <typename Dtype>
void L1LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  sign_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void L1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  caffe_cpu_sign(count, diff_.cpu_data(), sign_.mutable_cpu_data());
  Dtype abs_sum = caffe_cpu_asum(count, diff_.cpu_data());
  Dtype loss = abs_sum / bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;


  if(this->layer_param_.l1loss_param().debug_info()){
    //////////////////////////////////////////////////////////////////////
    // DEBUG CODE:
    ////////////////////////////////////////////////////////////////////// 
    static int print_cnt;

    float *data0 = (float *) bottom[0]->cpu_data();
    float *data1 = (float *) bottom[1]->cpu_data(); // <- label
    int blobSize = bottom[0]->count();
    
    int numImgs = bottom[0]->num(); // only shows first training image of this batch actually
    count = blobSize/numImgs;
    int blob_w = bottom[0]->width();
    int blob_h = bottom[0]->height();
    int blob_c = bottom[0]->channels();

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

    if (print_cnt % 5 == 0) {

        LOG(INFO) << "loss = " << loss << " count=" << count << " numImgs= " << numImgs;
        LOG(INFO) << "h = " << bottom[0]->height() << " w=" << bottom[0]->width() << " channels = " << bottom[0]->channels();
        LOG(INFO) << "L1 net-output, label";

        // int c=2;
        // for (int y=0; y<blob_h; y+=blob_h/8) {
        //     predictStr = "";
        //     labelStr = "";

        //     for (int x=0; x<blob_w; x+=blob_w/8) {
        //         char astr[10];
        //         sprintf(astr, "%0.2f " , data0[c*stride + y*blob_w + x]); // capture the 3rd coordinate == x2
        //         predictStr += astr;
        //         sprintf(astr, "%0.2f " , data1[c*stride + y*blob_w + x]);
        //         labelStr += astr;
        //     }
        //     LOG(INFO) << "  " << predictStr << " " << labelStr;
        // }
        // LOG(INFO) << " ";
        int stride = blob_w * blob_h;

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
                  predictStr.append(my_debug_symbol_bbox(absDiff));
                }
            }
            LOG(INFO) << "  " << predictStr;
        }
        LOG(INFO) << " ";
    } // end if print_cnt % 5
    print_cnt++;
  } // end if debug
}

template <typename Dtype>
string my_debug_symbol_bbox(Dtype value) {
  string ans;
  if(value >= 0.095) ans="X";
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
void L1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          sign_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(L1LossLayer);
#endif

INSTANTIATE_CLASS(L1LossLayer);
REGISTER_LAYER_CLASS(L1Loss);

}  // namespace caffe
