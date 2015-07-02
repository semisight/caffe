#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include <iomanip>
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <string>


namespace caffe {

template <typename Dtype> string my_debug_symbol(Dtype value);

template <typename Dtype>
void NVMaskLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
                                    const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  grad_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void NVMaskLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());

  int blobSize = bottom[0]->count();
  float *label = (float *) bottom[1]->cpu_data();
  float lambda_0 = this->layer_param_.nvmaskloss_param().lambda_0();
  float lambda_1 = this->layer_param_.nvmaskloss_param().lambda_1();

  float cvg_threshold = this->layer_param_.nvmaskloss_param().cvg_threshold();
  if(!(this->layer_param_.nvmaskloss_param().negative_mining())) { //ordinray 
    for (int i=0; i<blobSize; i++) {
      diff_.mutable_cpu_data()[i] *= (label[i] ? lambda_1 : lambda_0);
      grad_.mutable_cpu_data()[i] = diff_.mutable_cpu_data()[i] * (label[i] ? lambda_1 : lambda_0);
    }
  }
  else { // negative mining
    float cvg_margin = this->layer_param_.nvmaskloss_param().cvg_margin();
    for (int i=0; i<blobSize; i++) {
      if( (bottom[0]->cpu_data()[i] < cvg_threshold - cvg_margin  &&  bottom[1]->cpu_data()[i] < cvg_threshold)  ||
          (bottom[0]->cpu_data()[i] > cvg_threshold + cvg_margin  &&  bottom[1]->cpu_data()[i] > cvg_threshold) ) { //(1)satisfy the condition to be ignore. (2) ground truth can be fractional
        diff_.mutable_cpu_data()[i] = 0;
        grad_.mutable_cpu_data()[i] = 0;
      }
      else {
        diff_.mutable_cpu_data()[i] *= (label[i] ? lambda_1 : lambda_0);
        grad_.mutable_cpu_data()[i] = diff_.mutable_cpu_data()[i] * (label[i] ? lambda_1 : lambda_0);
      }
    }
  }

  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

  //LOG(INFO) << this->layer_param_.grid1loss_param().debug_info();

  if(this->layer_param_.nvmaskloss_param().debug_info()){
    //////////////////////////////////////////////////////////////////////
    // DEBUG CODE:
    ////////////////////////////////////////////////////////////////////// 
    static int print_cnt;
    static int iteration_training;
    static int iteration_testing;
    if(this->phase_ == TRAIN) iteration_training++;
    if(this->phase_ == TEST) iteration_testing++;
    print_cnt++;

    float *pred = (float *) bottom[0]->cpu_data();
    float *gt = (float *) bottom[1]->cpu_data(); // <- label
    int numImgs = bottom[0]->num();
    count = blobSize/numImgs;
    int blob_w = bottom[0]->width();
    int blob_h = bottom[0]->height();

    if (print_cnt % this->layer_param_.nvmaskloss_param().debug_period() == 0) {
        string predictStr;
        string labelStr;
      
        for (int i=0; i<blob_h; i++) {
            predictStr = "";
            for (int j=0; j<blob_w; j++) {
                Dtype value = pred[i*blob_w + j];
                predictStr.append(my_debug_symbol(value));
            }
            LOG(INFO) << "  " << predictStr;
        }
        LOG(INFO) << " ";
        for (int i=0; i<blob_h; i++) {
            labelStr = "";
            for (int j=0; j<blob_w; j++) {
                Dtype value = gt[i*blob_w + j];
                labelStr.append(my_debug_symbol(value));
            }
            LOG(INFO) << "  " << labelStr;
        }
        LOG(INFO) << " ";
    }

    //calculate 0-1 loss, for every iteration
    int TP = 0, TN = 0, FP = 0, FN = 0;
    for (int n=0; n<numImgs; n++) {
      for (int i=0; i<count; i++) {
        if(gt[n*count+i] < cvg_threshold && pred[n*count+i] < cvg_threshold) TN++;
        if(gt[n*count+i] > cvg_threshold && pred[n*count+i] > cvg_threshold) TP++;
        if(gt[n*count+i] < cvg_threshold && pred[n*count+i] > cvg_threshold) FP++;
        if(gt[n*count+i] > cvg_threshold && pred[n*count+i] < cvg_threshold) FN++;
      }
    }
              
    //LOG(INFO) << "h = " << bottom[0]->height() << " w=" << bottom[0]->width() << " channels = " << bottom[0]->channels();
    float recall = (TP+FP != 0) ? float(TP)/(TP+FP) : 1;
    float precision = (TP+FN != 0) ? float(TP)/(TP+FN) : 1;

    LOG(INFO) << (this->phase_ == TRAIN ? "#Train " : "#Test") 
              << (this->phase_ == TRAIN ? iteration_training : iteration_testing) 
              << "\t recall: " << std::setw(6) << std::setprecision(3) << std::setfill(' ') << recall 
              << "\t precision: " << std::setw(6) << std::setprecision(3) << std::setfill(' ') << precision 
              << "\t mAP: " << std::setw(6) << std::setprecision(3) << std::setfill(' ') << recall*precision
              << "\t\tloss =" << loss << " count=" << count << " numImgs=" << numImgs << " lambda_0=" << lambda_0 << " lambda_1=" << lambda_1 << " loss weight=" << top[0]->cpu_diff()[0] 
              << " negative_mining:" << this->layer_param_.nvmaskloss_param().negative_mining();
   
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
void NVMaskLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          grad_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(NVMaskLossLayer);
#endif

INSTANTIATE_CLASS(NVMaskLossLayer);
REGISTER_LAYER_CLASS(NVMaskLoss);

}  // namespace caffe
