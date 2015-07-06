#ifndef MULTIBOX_BBOX_LOSS_LAYER_HPP_
#define MULTIBOX_BBOX_LOSS_LAYER_HPP_

#include "caffe/loss_layers.hpp"

using namespace caffe;

/**
 * @brief Computes the MultiboxBbox (L2) loss @f$
 */
template <typename Dtype>
class MultiboxBboxLossLayer : public LossLayer<Dtype> {
 public:
  explicit MultiboxBboxLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiboxBboxLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }

  /**
   * Unlike most loss layers, in the MultiboxBboxLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc MultiboxBboxLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the MultiboxBbox error gradient w.r.t. the inputs.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
};

#endif
