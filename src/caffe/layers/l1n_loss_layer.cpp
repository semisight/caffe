#include <vector>
#include <algorithm>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype> string my_debug_symbol_bbox_l1n(Dtype value);
template <typename Dtype> string my_debug_symbol_norm(Dtype value);
template <typename Dtype> Dtype IOU(Dtype x1, Dtype y1, Dtype x2, Dtype y2, Dtype u1, Dtype v1, Dtype u2, Dtype v2);
template <typename Dtype> string my_debug_symbol_IOU(Dtype value);

template <typename Dtype>
void L1NLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  //LOG(INFO) << "before any reshape1";
  LossLayer<Dtype>::Reshape(bottom, top);
  //LOG(INFO) << "before any reshape2";

  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";

  CHECK_EQ(bottom[0]->channels(), 4) << "b_pred must have 4 channels";
  CHECK_EQ(bottom[1]->channels(), 4) << "b_gt must have 4 channels";

  //LOG(INFO) << "before any reshape";

  diff_.ReshapeLike(*bottom[0]); // diff_ = b[0] - b[1]
  sign_.ReshapeLike(*bottom[0]); // sign_ = sign(diff_)

  //LOG(INFO) << "diff and sign done";

  // reshape norm into 1/4 size
  int N=bottom[0]->num(),
      //C=bottom[0]->channels(),
      H=bottom[0]->height(),
      W=bottom[0]->width();

  norm_h_.Reshape(N, 1, H, W);
  norm_w_.Reshape(N, 1, H, W);
}

template <typename Dtype>
void L1NLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int nChannel = bottom[0]->channels();
  int nBatch = bottom[0]->num();
  int spatialLength = bottom[0]->width() * bottom[0]->height();

  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  caffe_cpu_sign(count, diff_.cpu_data(), sign_.mutable_cpu_data());

  //calc norm
  Dtype N0 = this->layer_param_.l1nloss_param().norm_ref();
  //Dtype IOU_thre = this->layer_param_.l1nloss_param().iouthre();

  const Dtype* label = bottom[1]->cpu_data(); // note that the label is hardcoded at position 1!!!!!!!
  for(int n=0; n<nBatch; n++){ //each sample is independent
    for(int i=0; i<spatialLength; i++){ //for each location
      Dtype norm_value_w;
      Dtype norm_value_h;
      Dtype l = Dtype(0);

      if(label[(n*nChannel)*spatialLength + i] == Dtype(0) &&
          label[(n*nChannel+1)*spatialLength + i] == Dtype(0) &&
          label[(n*nChannel+2)*spatialLength + i] == Dtype(0) &&
          label[(n*nChannel+3)*spatialLength + i] == Dtype(0)) { // if not in the bbox covered area, forward loss contribution is 0. And norm factor is set to 1
        norm_value_w = norm_value_h = Dtype(1);
        //l = Dtype(0);
      }
      else { // covered
        norm_value_w = ((label[(n*nChannel+2)*spatialLength + i] - label[(n*nChannel+0)*spatialLength + i])) / N0;
        norm_value_h = ((label[(n*nChannel+3)*spatialLength + i] - label[(n*nChannel+1)*spatialLength + i])) / N0;
        //norm_value = norm_value > 1   ? 1   : norm_value;
        norm_value_w = norm_value_w < 0.1 ? 0.1 : norm_value_w;
        norm_value_h = norm_value_h < 0.1 ? 0.1 : norm_value_h;

        l += (diff_.cpu_data()[(n*nChannel+0)*spatialLength + i] > 0 ? diff_.cpu_data()[(n*nChannel+0)*spatialLength + i] : -diff_.cpu_data()[(n*nChannel+0)*spatialLength + i]) * norm_value_w;
        l += (diff_.cpu_data()[(n*nChannel+1)*spatialLength + i] > 0 ? diff_.cpu_data()[(n*nChannel+1)*spatialLength + i] : -diff_.cpu_data()[(n*nChannel+1)*spatialLength + i]) * norm_value_h;
        l += (diff_.cpu_data()[(n*nChannel+2)*spatialLength + i] > 0 ? diff_.cpu_data()[(n*nChannel+2)*spatialLength + i] : -diff_.cpu_data()[(n*nChannel+2)*spatialLength + i]) * norm_value_w;
        l += (diff_.cpu_data()[(n*nChannel+3)*spatialLength + i] > 0 ? diff_.cpu_data()[(n*nChannel+3)*spatialLength + i] : -diff_.cpu_data()[(n*nChannel+3)*spatialLength + i]) * norm_value_h;
          //LOG(INFO) << "dim " << c << " : diff_ is" << diff_.cpu_data()[(n*nChannel+c)*spatialLength + i] << " l = " << l;
      } // else
      norm_w_.mutable_cpu_data()[n*spatialLength + i] = norm_value_w; // norm has only 1 channel
      norm_h_.mutable_cpu_data()[n*spatialLength + i] = norm_value_h;

      top[0]->mutable_cpu_data()[0] += l;
    } //for
  } //for

  LOG(INFO) << "loss is " << top[0]->mutable_cpu_data()[0] << " divided by " << nBatch << " debug_info = " << this->layer_param_.l1nloss_param().debug_info();
  top[0]->mutable_cpu_data()[0] /= nBatch;
  //Dtype abs_sum = caffe_cpu_asum(count, diff_.cpu_data());
  //Dtype loss = abs_sum / bottom[0]->num();
  //top[0]->mutable_cpu_data()[0] = loss;


  if(this->layer_param_.l1nloss_param().debug_info()){
    //////////////////////////////////////////////////////////////////////
    // DEBUG CODE:
    ////////////////////////////////////////////////////////////////////// 
    static int print_cnt;

    float *data0 = (float *) bottom[0]->cpu_data();
    float *data1 = (float *) bottom[1]->cpu_data(); // <- label
    float *norm_pointer_w = (float *) norm_w_.cpu_data();
    float *norm_pointer_h = (float *) norm_h_.cpu_data();
    int blobSize = bottom[0]->count();
    
    int numImgs = bottom[0]->num();
    count = blobSize/numImgs;
    int blob_w = bottom[0]->width();
    int blob_h = bottom[0]->height();
    int blob_c = bottom[0]->channels();
    Dtype loss = top[0]->cpu_data()[0];

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
      
        LOG(INFO) << "loss = " << loss << " count=" << count << " numImgs= " << numImgs << " top[0]->cpu_diff()[0] " << top[0]->cpu_diff()[0];
        LOG(INFO) << "h = " << bottom[0]->height() << " w=" << bottom[0]->width() << " channels = " << bottom[0]->channels() << " N0 = " << N0;
        LOG(INFO) << "L1N debugging: difference between pred and gt";

        // for (int i=0; i<blob_h; i+=blob_h/8) {
        //     predictStr = "";
        //     labelStr = "";

        //     for (int j=0; j<blob_w; j+=blob_w/8) {
        //         char astr[10];
        //         sprintf(astr, "%0.1f " , data0[i*blob_w + j]);
        //         predictStr += astr;
        //         sprintf(astr, "%0.1f " , data1[i*blob_w + j]);
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
                  predictStr.append(my_debug_symbol_bbox_l1n(absDiff));
                }
            }
            LOG(INFO) << "  " << predictStr;
        }
        LOG(INFO) << "L1N debugging: norm factor";

        for (int i=0; i<blob_h; i++) {
            predictStr = "";
            for (int j=0; j<blob_w; j++) {
                Dtype value = (norm_pointer_h[i*blob_w + j] + norm_pointer_w[i*blob_w + j])/2;
                predictStr.append(my_debug_symbol_norm(value));
            }
            LOG(INFO) << "  " << predictStr;
        }

        LOG(INFO) << "L1N debugging: IOU";

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
string my_debug_symbol_bbox_l1n(Dtype value) {
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
string my_debug_symbol_norm(Dtype value) {
  string ans;
  if(value == 1) ans="|";
  else if(value >= 0.85) ans="9";
  else if(value >= 0.75) ans="8";
  else if(value >= 0.65) ans="7";
  else if(value >= 0.55) ans="6";
  else if(value >= 0.45) ans="5";
  else if(value >= 0.35) ans="4";
  else if(value >= 0.25) ans="3";
  else if(value >= 0.15) ans="2";
  else if(value >= 0.05) ans="1";
  else ans = "-";

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
void L1NLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  int count = bottom[0]->count();
  int nChannel = bottom[0]->channels();
  int nBatch = bottom[0]->num();
  int spatialLength = bottom[0]->width() * bottom[0]->height();


  for (int s = 0; s < 2; ++s) { // 0 is prediction, 1 is gt
    if (propagate_down[s]) {
      const Dtype sign = (s == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[s]->num();

      //LOG(INFO) << "L1N Back: alpha = " << alpha;

      for(int n=0; n<nBatch; n++){ //each sample is independent
        for(int i=0; i<spatialLength; i++){ //for each location
          bottom[s]->mutable_cpu_diff()[(n*nChannel+0)*spatialLength + i] = sign_.cpu_data()[(n*nChannel+0)*spatialLength + i] * norm_w_.cpu_data()[n*spatialLength + i];
          bottom[s]->mutable_cpu_diff()[(n*nChannel+1)*spatialLength + i] = sign_.cpu_data()[(n*nChannel+1)*spatialLength + i] * norm_h_.cpu_data()[n*spatialLength + i];
          bottom[s]->mutable_cpu_diff()[(n*nChannel+2)*spatialLength + i] = sign_.cpu_data()[(n*nChannel+2)*spatialLength + i] * norm_w_.cpu_data()[n*spatialLength + i];
          bottom[s]->mutable_cpu_diff()[(n*nChannel+3)*spatialLength + i] = sign_.cpu_data()[(n*nChannel+3)*spatialLength + i] * norm_h_.cpu_data()[n*spatialLength + i];
            //if(sign_.cpu_data()[(n*nChannel+c)*spatialLength + i] != 0) 
              //LOG(INFO) << "channel " << c << ": grad: " << bottom[s]->mutable_cpu_diff()[(n*nChannel+c)*spatialLength + i] << "\t sign: " 
               //         << sign_.cpu_data()[(n*nChannel+c)*spatialLength + i] << "\t norm: " << norm_.cpu_data()[n*spatialLength + i] << "\t diff: " << diff_.cpu_data()[(n*nChannel+c)*spatialLength + i];
        }
      }
      
      caffe_scal(count, alpha, bottom[s]->mutable_cpu_diff());
      // caffe_cpu_axpby(
      //     bottom[i]->count(),              // count
      //     alpha,                              // alpha
      //     sign_.cpu_data(),                   // a
      //     Dtype(0),                           // beta
      //     bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(L1NLossLayer);
#endif

INSTANTIATE_CLASS(L1NLossLayer);
REGISTER_LAYER_CLASS(L1NLoss);

}  // namespace caffe
