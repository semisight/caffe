#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype> string my_debug_symbol(Dtype value);

template <typename Dtype>
void L1NLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";

  CHECK_EQ(bottom[1]->channels(), 4) << "b_pred must have 4 channels";
  CHECK_EQ(bottom[2]->channels(), 4) << "b_gt must have 4 channels";

  diff_.ReshapeLike(*bottom[0]); // diff_ = b[0] - b[1]
  sign_.ReshapeLike(*bottom[0]); // sign_ = sign(diff_)

  // reshape norm into 1/4 size
  int N=bottom[0]->num(),
      //C=bottom[0]->channels(),
      H=bottom[0]->height(),
      W=bottom[0]->width();

  norm_.Reshape(N, 1, H, W);
  rawLoss_.Reshape(N, 1, H, W);
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
  Dtype N0 = this->layer_param_.l1nloss_param().norm_ref(); //0.7

  const Dtype* label = bottom[1]->cpu_data(); // note that the label is hardcoded at position 1!!!!!!!
  for(int n=0; n<nBatch; n++){ //each sample is independent
    for(int i=0; i<spatialLength; i++){ //for each location
      Dtype norm_value;
      Dtype l = Dtype(0);
      if(label[(n*nChannel)*spatialLength + i] == Dtype(0) &&
          label[(n*nChannel+1)*spatialLength + i] == Dtype(0) &&
          label[(n*nChannel+2)*spatialLength + i] == Dtype(0) &&
          label[(n*nChannel+3)*spatialLength + i] == Dtype(0)) { // if not in the bbox covered area, forward loss contribution is 0. And norm factor is set to 1
        norm_value = Dtype(1);
        //l = Dtype(0);
      }
      else {
        norm_value = ((label[(n*nChannel+2)*spatialLength + i] - label[(n*nChannel)*spatialLength + i]) *
             (label[(n*nChannel+3)*spatialLength + i] - label[(n*nChannel+1)*spatialLength + i])) / N0;
        
        for(int c=0; c<nChannel; c++)  l += abs(diff_.cpu_data()[(n*nChannel+c)*spatialLength + i]);
        l /= norm_value;
      }

      norm_.mutable_cpu_data()[n*spatialLength + i] = n;
      top[0]->mutable_cpu_data()[0] += l;
    }
  }

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
    int blobSize = bottom[0]->count();
    
    int numImgs = bottom[0]->num();
    count = blobSize/numImgs;
    int blob_w = bottom[0]->width();
    int blob_h = bottom[0]->height();
    int blob_d = bottom[0]->channels();
    Dtype loss = top[0]->mutable_cpu_data()[0];

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

    if (print_cnt % 20 == 0) {

        int all_zeros = 1;
        int stride = blob_w*blob_h;
        for (int c=0; c<blob_d; c++) {
            for (int y=0; y<blob_h; y++) {
                for (int x=0; x<blob_w; x++) {
                    if (data0[c*stride + y*blob_w + x] != 0) all_zeros = 0;
                }
            }
        }

        if (all_zeros) {
            LOG(INFO) << "!!! L1_LOSS: ALL ZEROS !!!";
        }
      
        LOG(INFO) << "loss = " << loss << " count=" << count << " numImgs= " << numImgs << " top[0]->cpu_diff()[0] " << top[0]->cpu_diff()[0];
        LOG(INFO) << "h = " << bottom[0]->height() << " w=" << bottom[0]->width() << " channels = " << bottom[0]->channels();
        LOG(INFO) << "GRID1 predict, label ";

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
        for (int i=0; i<blob_h; i+=2) {
            predictStr = "";
            for (int j=0; j<blob_w; j+=2) {
                Dtype value = (data0[i*blob_w + j] + data0[(i+1)*blob_w + j] + data0[i*blob_w + j+1] + data0[(i+1)*blob_w + j+1]) / 4;
                predictStr.append(my_debug_symbol(value));
            }
            LOG(INFO) << "  " << predictStr;
        }
        LOG(INFO) << " ";
        for (int i=0; i<blob_h; i+=2) {
            labelStr = "";
            for (int j=0; j<blob_w; j+=2) {
                Dtype value = (data1[i*blob_w + j] + data1[(i+1)*blob_w + j] + data1[i*blob_w + j+1] + data1[(i+1)*blob_w + j+1]) / 4;
                labelStr.append(my_debug_symbol(value));
            }
            LOG(INFO) << "  " << labelStr;
        }
        LOG(INFO) << " ";
    }
    print_cnt++;
  } // end if debug
}

template <typename Dtype>
string my_debug_symbol(Dtype value) {
  string ans;
  if(value >= 0.95) ans="X";
  else if(value >= 0.85) ans="9";
  else if(value >= 0.75) ans="8";
  else if(value >= 0.65) ans="7";
  else if(value >= 0.55) ans="6";
  else if(value >= 0.45) ans="5";
  else if(value >= 0.35) ans="4";
  else if(value >= 0.25) ans="3";
  else if(value >= 0.15) ans="2";
  else if(value >= 0.05) ans="1";
  else if(value >= -0.05) ans="-";
  else ans = "-";

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

      for(int n=0; n<nBatch; n++){ //each sample is independent
        for(int i=0; i<spatialLength; i++){ //for each location
          for(int c=0; c<nChannel; c++) {
            bottom[s]->mutable_cpu_diff()[(n*nChannel+c)*spatialLength + i] = 
              sign_.cpu_data()[(n*nChannel+c)*spatialLength + i] / norm_.cpu_data()[(n*nChannel)*spatialLength + i];
          }
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
