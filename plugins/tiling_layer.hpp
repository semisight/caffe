#ifndef TILING_LAYER_HPP_
#define TILING_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

using namespace caffe;

/**
 * @brief for transforming a C x H x W blob into a
 * C/t^2 x H*t x W*t blob where t is the tile dimension, and each pixel in
 * the input is turned into a txt grid from the channels in row major order.
 *
 */
template <typename Dtype>
class TilingLayer : public Layer<Dtype> {
 public:
  explicit TilingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Tiling"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  virtual inline int MaxTopBlobs() const {
      return 1;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    //  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    //const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int input_channels_, output_channels_;
  int input_width_, input_height_;
  int output_width_, output_height_;
  int count_per_input_map_, count_per_output_map_;
  int tile_dim_, tile_dim_sq_;
};

#endif
