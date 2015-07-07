#ifndef L2N_LOSS_LAYER_HPP_
#define L2N_LOSS_LAYER_HPP_

#include "caffe/loss_layers.hpp"

using namespace caffe;

/**
 * @hacked L2Loss. Nothing changed. add some dubug info@f$
 */
template <typename Dtype>
class L2NLossLayer : public LossLayer<Dtype> {
 public:
  explicit L2NLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "L2NLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  /**
   * Unlike most loss layers, in the NVLossLayer we can backpropagate
   * to 4 inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc L1NLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //  const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the NV normalized error gradient w.r.t. the inputs.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    //const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
};

#endif
