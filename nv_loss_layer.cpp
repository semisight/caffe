#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <string>

namespace caffe {

template <typename Dtype>
void NVLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
                                    const vector<Blob<Dtype>*>& top) {
  // we have 4 bottoms here (in order): [0]c_pred in 1 channel, [1]b_pred in 4 channel, [2]b_gt in 4 channel, [3]n in 1 channel
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[3]->count(1)) << "Inputs must have the same dimension. (c_pred and n)";
  CHECK_EQ(bottom[1]->count(1), bottom[2]->count(1)) << "Inputs must have the same dimension. (b_pred and b_gt)";
  diff4ch_.ReshapeLike(*bottom[1]);
  absdiff_.ReshapeLike(*bottom[0]);
  cutoff_.ReshapeLike(*bottom[0]);
  sign_.ReshapeLike(*bottom[1]);
  rawLoss_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void NVLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //compute diff4ch_(L1), absdiff_, cutoff_, sign_, and L for top[0]
  CHECK_EQ(bottom[1]->channels(), 4) << "b_pred must have 4 channels";
  CHECK_EQ(bottom[2]->channels(), 4) << "b_gt must have 4 channels";
  int nChannel = bottom[2]->channels();
  int nBatch = bottom[2]->num();
  int spatialLength = bottom[0]->width() * bottom[0]->height();
  int count = bottom[0]->count();
  int count_4ch = bottom[1]->count();
  float cut_thre = this->layer_param_.nvloss_param().cutoff(); //0.7

  //(1) diff4ch
  caffe_sub(
      count_4ch,
      bottom[1]->cpu_data(),
      bottom[2]->cpu_data(),
      diff4ch_.mutable_cpu_data()); // pred - gt
  
  //(2) sign
  caffe_cpu_sign(count, diff4ch_.cpu_data(), sign_.mutable_cpu_data());

  //(3) cutoff
  for(int i=0; i<count; i++){
    cutoff_.mutable_cpu_data()[i] = bottom[0]->cpu_data()[i] >= cut_thre ? Dtype(1) : Dtype(0);
  }

  //(4) absdiff
  for(int n=0; n<nBatch; n++){ //each sample is independent
    for(int i=0; i<spatialLength; i++){ //for each location
      Dtype inc = 0;
      for(int c=0; c<nChannel; c++){
        inc += (sign_.cpu_data()[(n*nChannel+c)*spatialLength + i] > 0) ? 
                diff4ch_.cpu_data()[(n*nChannel+c)*spatialLength + i] : -diff4ch_.cpu_data()[(n*nChannel+c)*spatialLength + i];
      }
      absdiff_.mutable_cpu_data()[n*spatialLength + i] = inc;
    }
  }

  // (5) loss
  Dtype loss = 0;
  caffe_mul(count, cutoff_.cpu_data(), absdiff_.cpu_data(), rawLoss_.mutable_cpu_data());
  caffe_div(count, rawLoss_.cpu_data(), bottom[3]->cpu_data(), rawLoss_.mutable_cpu_data());
  loss = caffe_cpu_asum(count, rawLoss_.cpu_data());
  loss /= nBatch;
  top[0]->mutable_cpu_data()[0] = loss;

  if(this->layer_param_.nvloss_param().debug() == 1){
    //////////////////////////////////////////////////////////////////////
    // DEBUG CODE:
    ////////////////////////////////////////////////////////////////////// 
    // static int print_cnt;

    // float *data0 = (float *) bottom[0]->cpu_data();
    // float *data1 = (float *) bottom[1]->cpu_data(); // <- label
    
    // int numImgs = bottom[0]->num();
    // count = blobSize/numImgs;
    // int blob_w = bottom[0]->width();
    // int blob_h = bottom[0]->height();

    // // Figure out dimentionality of data
    // if (count == 4320) {
    //     // is 120x36
    //     blob_w = 120;
    //     blob_h = 36;
    // } else if (count == 1080) {
    //     blob_w = 60;
    //     blob_h = 18;
    // } else if (blob_w==1 && blob_h==1) {
    //     blob_w = sqrt(count);
    //     blob_h = blob_w;
    // }

    // string predictStr;
    // string labelStr;

    // if (print_cnt % 200 == 0) {
    //     int all_zeros = 1;
    //     for (int i=0; i<blob_h; i++) {
    //         for (int j=0; j<blob_w; j++) {
    //             if (data0[i*blob_w + j] != 0) all_zeros = 0;
    //         }
    //     }
    //     if (all_zeros) {
    //         LOG(INFO) << "!!! GRID1_LOSS: ALL ZEROS !!!";
    //     }
      
    //     LOG(INFO) << "loss = " << loss << " count=" << count << " numImgs= " << numImgs << " lambda = " << lambda << " top[0]->cpu_diff()[0] " << top[0]->cpu_diff()[0];
    //     LOG(INFO) << "h = " << bottom[0]->height() << " w=" << bottom[0]->width() << " channels = " << bottom[0]->channels();
    //     LOG(INFO) << "GRID1 predict, label ";

    //     for (int i=0; i<blob_h; i+=blob_h/8) {
    //         predictStr = "";
    //         labelStr = "";

    //         for (int j=0; j<blob_w; j+=blob_w/8) {
    //             char astr[10];
    //             sprintf(astr, "%0.1f " , data0[i*blob_w + j]);
    //             predictStr += astr;
    //             sprintf(astr, "%0.1f " , data1[i*blob_w + j]);
    //             labelStr += astr;
    //         }
    //         LOG(INFO) << "  " << predictStr << " " << labelStr;
    //     }
    //     LOG(INFO) << " ";
    // }
    // print_cnt++;
  } //end if debug
}

template <typename Dtype>
void NVLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  int nChannel = bottom[2]->channels();
  int nBatch = bottom[2]->num();
  int spatialLength = bottom[0]->width() * bottom[0]->height();
  int count = bottom[0]->count();
  //int count_4ch = bottom[1]->count();
  
  // (1) bottom[0]->mutable_cpu_diff() and bottom[3]->mutable_cpu_diff()
  caffe_div(count, absdiff_.cpu_data(), bottom[3]->cpu_data(), bottom[0]->mutable_cpu_diff());
  caffe_scal(count, Dtype(1.0)/Dtype(nBatch), bottom[0]->mutable_cpu_diff()); //normalize to batchsize

  caffe_mul(count, bottom[0]->cpu_diff(), cutoff_.cpu_data(), bottom[3]->mutable_cpu_diff()); //multiply cutoff
  caffe_div(count, bottom[3]->cpu_diff(), bottom[3]->cpu_data(), bottom[3]->mutable_cpu_diff()); //divide n again
  caffe_scal(count, Dtype(-1), bottom[3]->mutable_cpu_diff());

  Dtype norm_term = Dtype(1)/Dtype(nBatch);
  Dtype norm_term_neg = Dtype(-1)/Dtype(nBatch);
  
  // (2) bottom[1]->mutable_cpu_diff() and bottom[2]->mutable_cpu_diff()
  for(int n=0; n<nBatch; n++){ //each sample is independent
    for(int i=0; i<spatialLength; i++){ //for each location
      if(cutoff_.cpu_data()[n*spatialLength + i] > 0){
        for(int c=0; c<nChannel; c++){
          bottom[1]->mutable_cpu_diff()[(n*nChannel+c)*spatialLength + i] = (sign_.cpu_data()[(n*nChannel+c)*spatialLength + i] > 0) ? 
                    norm_term/bottom[3]->cpu_data()[n*spatialLength + i] : norm_term_neg/bottom[3]->cpu_data()[n*spatialLength + i];
          bottom[2]->mutable_cpu_diff()[(n*nChannel+c)*spatialLength + i] = -bottom[1]->cpu_diff()[(n*nChannel+c)*spatialLength + i];
        }
      }
      else {
        for(int c=0; c<nChannel; c++){
          bottom[1]->mutable_cpu_diff()[(n*nChannel+c)*spatialLength + i] = bottom[2]->mutable_cpu_diff()[(n*nChannel+c)*spatialLength + i] = Dtype(0);
        }
      }
    }
  }

}

#ifdef CPU_ONLY
STUB_GPU(NVLossLayer);
#endif

INSTANTIATE_CLASS(NVLossLayer);
REGISTER_LAYER_CLASS(NVLoss);

}  // namespace caffe
