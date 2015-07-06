#ifndef MULTIBOX_CONFIDENCE_LOSS_LAYER_HPP_
#define MULTIBOX_CONFIDENCE_LOSS_LAYER_HPP_

#include "caffe/loss_layers.hpp"

using namespace caffe;

template <typename Dtype>
class MultiboxConfidenceLossLayer : public LossLayer<Dtype> {
 public:
  explicit MultiboxConfidenceLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiboxConfidenceLoss"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

#endif
