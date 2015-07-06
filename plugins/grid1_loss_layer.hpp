#ifndef GRID1_LOSS_LAYER_HPP_
#define GRID1_LOSS_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/loss_layers.hpp"

using namespace caffe;

/**
 * @brief Computes the Grid1 (L2) loss @f$
 */
template <typename Dtype>
class Grid1LossLayer : public LossLayer<Dtype> {
 public:
  explicit Grid1LossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_(), grad_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Grid1Loss"; }
  /**
   * Unlike most loss layers, in the Grid1LossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc Grid1LossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //  const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the Grid1 error gradient w.r.t. the inputs.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    //const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> grad_;
};

#endif
