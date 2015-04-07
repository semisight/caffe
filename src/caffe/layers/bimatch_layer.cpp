#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

#define COORDS_PER_BOX 4
#define MAX_GROUNDTRUTH_BOXES 2
#define MAX_PREDICTIONS 4

template <typename Dtype>
void BimatchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void BimatchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  int num_predicted_boxes = dim / COORDS_PER_BOX;

  const Dtype* prediction = bottom[0]->cpu_data();
  const Dtype* groundtruth_bboxes = bottom[1]->cpu_data();
  Dtype* bipartite_match = top[0]->mutable_cpu_data();

  // bipartite matching code
  for (int n=0; n<num; n++) {
      int bi_offs = n * num_predicted_boxes;
      // num * {[4], [4], n_gt}
      int num_groundtruth_bboxes = groundtruth_bboxes[n * (MAX_GROUNDTRUTH_BOXES * COORDS_PER_BOX + 1) + MAX_GROUNDTRUTH_BOXES * COORDS_PER_BOX];
      for (int i=0; i<num_predicted_boxes; i++) {
          bipartite_match[bi_offs + i] = -1;
      }
      char amsg[200];

      for (int j=0; j<num_groundtruth_bboxes; j++) {
          Dtype g_x1 = groundtruth_bboxes[n*dim + j*COORDS_PER_BOX + 0];
          Dtype g_y1 = groundtruth_bboxes[n*dim + j*COORDS_PER_BOX + 1];
          Dtype g_x2 = groundtruth_bboxes[n*dim + j*COORDS_PER_BOX + 2];
          Dtype g_y2 = groundtruth_bboxes[n*dim + j*COORDS_PER_BOX + 3];
          
          if (n<1){
              LOG(INFO) << "MBOX(fwd): num_gt_boxes = " << num_groundtruth_bboxes;
              sprintf(amsg,  "gt[%d] = (%1.2f %1.2f %1.2f %1.2f)", j, g_x1, g_y1, g_x2, g_y2);
              LOG(INFO) << "MBOX(fwd): " << amsg;
          }

          Dtype min_dist = -1;
          int min_idx = -1;

          for (int i=0; i<num_predicted_boxes; i++) {
              Dtype n_x1 = prediction[n*dim + i*COORDS_PER_BOX +0];
              Dtype n_y1 = prediction[n*dim + i*COORDS_PER_BOX +1];
              Dtype n_x2 = prediction[n*dim + i*COORDS_PER_BOX +2];
              Dtype n_y2 = prediction[n*dim + i*COORDS_PER_BOX +3];

              float distance = (pow(g_x1 - n_x1, 2) + 
                                pow(g_y1 - n_y1, 2) +
                                pow(g_x2 - n_x2, 2) + 
                                pow(g_y2 - n_y2, 2));

              if (n<1){
                  sprintf(amsg,  "net[%d] = (%1.2f %1.2f %1.2f %1.2f)    distance to gt[%d] is %2.2f",  i, n_x1, n_y1, n_x2, n_y2, j, distance);
                  LOG(INFO) << "MBOX(fwd): " << amsg;
              }

              if (bipartite_match[bi_offs + i] == -1) {
                  // -1 indicates unpaired so far
                  if (min_dist == -1 || distance < min_dist) {
                      min_dist = distance;
                      min_idx = i;
                  }
              }
          }

          assert(min_idx>=0 && min_dist!=1e10);
          bipartite_match[bi_offs + min_idx] = j; // mark as used
          if (n<1) {
              sprintf(amsg,  "MBOX(fwd): gt[%d] is matched to i[%d]", j, min_idx);
              LOG(INFO) << amsg;
          }
      }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BimatchLayer);
#endif

INSTANTIATE_CLASS(BimatchLayer);
REGISTER_LAYER_CLASS(Bimatch);

}  // namespace caffe
