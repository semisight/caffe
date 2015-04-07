// [based on multinomial logistic loss]
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

// label_boxes[j]   ground truth bboxes, up to j objects
// label_nboxes     number of objects, jMax
// match[i]         for each net output, what is it matched with in ground truth
//
// bottom[0] is the network prediction for confidence that a given bbox is good
// bottom[1] is the calculated bipartite match between the predicted bboxes and
//           the groundtruth bboxes (labels)
//           match[i] = -1 if no match else is set to j, the object
//           match[i] = j, j is the index of the ground truth box
//
// Derivative:
//
// dF/dCi = (Sum[j] (Xij * Ci)) / (Ci (1-Ci))
// 
//        for i in numPredictions:
//            grad[i] = (match[i]==-1) ? 0 : 1/(1 - C[i])
// 
// Ci = The network's confidence that a given output is an object
//      We're starting with 4 of these.
//      Ci is in the range of [0.0 .. 1.0]
//
// i = 1..K, the number of network predictions
// j = 1..M, the number of true objects in the image
//
// Xij = Ground truth assignment of prediction to object.
//       Xij has either 0 or 1 value.
//       Each object is associated with a single network output.
//
//  Sum_i(Xij) = 1
//
// Example:
// ========
// network_prediction[0] -> -
// network_prediction[1] -> obj0
// network_prediction[2] -> obj1
// network_prediction[3] -> -
//
//   i j
// X_0_0  0
// X_0_1  0
// X_0_2  0
//
// X_1_0  1
// X_1_1  0
// X_1_2  0
//
// X_2_0  0
// X_2_1  1
// X_2_2  0
//
// X_3_0  0
// X_3_1  0
// X_3_2  0


namespace caffe {

template <typename Dtype>
void MultiboxConfidenceLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  //  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
  //      "MULTIBOX_CONFIDENCE_LOSS layer inputs must have the same count.";
}

template <typename Dtype>
void MultiboxConfidenceLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_confidences = bottom[0]->cpu_data();
  const Dtype* bipartite_match = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Dtype loss = 0;
  char amsg[200];

 // for each image in the batch:
  int n;
  for (n = 0; n < num; ++n) {
      if (n<1) {
          LOG(INFO) << "MBOX_BBOX_CONF: bipartite_match = {"
                    << bipartite_match[0]<< ","
                    << bipartite_match[1]<< ","
                    << bipartite_match[2]<< ","
                    << bipartite_match[3]<< "}";
          sprintf(amsg, "%1.2f %1.2f %1.2f %1.2f",
                  bottom_confidences[n*dim + 0],
                  bottom_confidences[n*dim + 1],
                  bottom_confidences[n*dim + 2],
                  bottom_confidences[n*dim + 3]);
          LOG(INFO) << "                network_conf = {" << amsg << "}";
      }

      // for each prediction in image
      for (int i = 0; i < dim; i++) {
          int j = static_cast<int>(bipartite_match[n*dim +i]);
          if (j==-1) {
              // this output matches no objects
              loss -= log(1-bottom_confidences[n*dim +i]);
          } else {
              // this output is matched with an object
              loss -= log(bottom_confidences[n*dim +i]);
          }
      }
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void MultiboxConfidenceLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_confidences = bottom[0]->cpu_data();
    const Dtype* bipartite_match = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    const Dtype scale = - top[0]->cpu_diff()[0] / num;

    // for each image in the batch:
    for (int n = 0; n < num; ++n) {
        for (int i = 0; i < dim; i++) {
            int j = static_cast<int>(bipartite_match[n*dim +i]);
            if (j == -1) {
                bottom_diff[n*dim + i] = Dtype(0);
            } else { 
                Dtype denom = std::max((1 - bottom_confidences[n*dim+i]),
                                       Dtype(kLOG_THRESHOLD));
                bottom_diff[n*dim + i] = scale/denom;
            }
        }
    }
  }
}

INSTANTIATE_CLASS(MultiboxConfidenceLossLayer);
REGISTER_LAYER_CLASS(MultiboxConfidenceLoss);

}  // namespace caffe
