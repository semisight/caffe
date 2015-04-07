#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MultiboxBboxLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  //  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
  //      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

// Hardcode a bunch of stuff just so we can test this out
#define COORDS_PER_BOX 4
#define MAX_GROUNDTRUTH_BOXES 2
#define MAX_PREDICTIONS 4

template <typename Dtype>
void MultiboxBboxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  int num_predicted_boxes = dim / COORDS_PER_BOX;
  const Dtype* prediction = bottom[0]->cpu_data();
  const Dtype* groundtruth_bboxes = bottom[1]->cpu_data();
  const Dtype* bipartite_match = bottom[2]->cpu_data();

  // bipartite matching code
  int n;
  for (n=0; n<num; n++) {
      if (n<1) {
          LOG(INFO) << "MBOX_BBOX_LOSS: bipartite_match={"
                    << bipartite_match[0]<< ","
                    << bipartite_match[1]<< ","
                    << bipartite_match[2]<< ","
                    << bipartite_match[3]<< "}";
      }
      for (int i=0; i < num_predicted_boxes; i++) {
          int matching_groundtruth_box = bipartite_match[i];

          // backpropagate only if match
          if (matching_groundtruth_box == -1) {
              for (int x=0; x<COORDS_PER_BOX; x++) {
                  diff_.mutable_cpu_data()[n*dim+i*COORDS_PER_BOX+x] = 0;
              }
          } else {
              for (int x=0; x<COORDS_PER_BOX; x++) {
                  Dtype gt_coord = groundtruth_bboxes[n*dim+i*COORDS_PER_BOX+x];
                  Dtype net_coord = prediction[n*dim+matching_groundtruth_box*COORDS_PER_BOX+x];
                  diff_.mutable_cpu_data()[n*dim+i] = net_coord - gt_coord;
                  if (n<1) {
                      LOG(INFO) << "MBOX_BBOX_LOSS: loss[n="<<n<<"][i="<<i<<"]="<<diff_.mutable_cpu_data()[n*dim+i];
                  }
              }
          }
      }
  }  

  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

  LOG(INFO) << "MBOX_BBOX_LOSS: loss=" << loss;
}
  
template <typename Dtype>
void MultiboxBboxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MultiboxBboxLossLayer);
#endif

INSTANTIATE_CLASS(MultiboxBboxLossLayer);
REGISTER_LAYER_CLASS(MultiboxBboxLoss);

}  // namespace caffe

#ifdef gaga
  if (0) {
      //////////////////////////////////////////////////////////////////////
      // Initialize Priors
      // 4 network outputs
      //////////////////////////////////////////////////////////////////////
      int num_gridcells_x = 2;
      int num_gridcells_y = 2;
      //0.0  0.5  1.0
      // +-----+-----+
      // | +-+ | +-+ |
      // | +-+ | +-+ |
      // +-----+-----+
      // | +-+ | +-+ |
      // | +-+ | +-+ |
      // +-----+-----+
      const Dtype priors[4][COORDS_PER_BOX];
      const Dtype cell_size_x = 1.0 / num_gridcells_x;
      const Dtype cell_size_y = 1.0 / num_gridcells_y;
      const Dtype center_offs_x = cell_size_x/2;
      const Dtype center_offs_y = cell_size_y/2;

      for (int x = 0; x < 2; x++) {
          for (int y = 0; y < 2; y++) {
              Dtype cell_origin_x = x*cell_size_x;
              Dtype cell_origin_y = y*cell_size_y;
              Dtype x1 = cell_origin_x + cell_size_x/3;
              Dtype y1 = cell_origin_y + cell_size_y/3;
              Dtype x2 = x1 + cell_size_x/3;
              Dtype y2 = y1 + cell_size_y/3;
              priors[y*2+x][0] = x1;
              priors[y*2+x][1] = y1;
              priors[y*2+x][2] = x2;
              priors[y*2+x][3] = y2;
          }
      }

      //////////////////////////////////////////////////////////////////////
      // Compute true coordinates
      //////////////////////////////////////////////////////////////////////
      Dtype trueCoords[4][COORDS_PER_BOX];
      for (int n=0; n<num; n++) {
          for (int i=0; i<dim/COORDS_PER_BOX; i++) {
              for (int j=0; j<num_groundtruth_bboxes; j++) {
                  for (int x=0; x<COORDS_PER_BOX; x++) {
                      ;
                  }
              }
          }
      }
  }
#endif
