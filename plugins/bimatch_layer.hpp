#ifndef BIMATCH_LAYER_HPP_
#define BIMATCH_LAYER_HPP_

#include "caffe/neuron_layers.hpp"

using namespace caffe;

/**
 * @brief Computes bipartite match between net outputs and lables,
 *        for Multibox object detection
 */
template <typename Dtype>
class BimatchLayer : public NeuronLayer<Dtype> {
 public:
  explicit BimatchLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Bimatch"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }
};

#endif
