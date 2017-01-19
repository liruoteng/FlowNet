#include <algorithm> 
#include <vector>

#include "caffe/layers/reform_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {
    CHECK_NE(top[0], bottom[0])<<"layer does not support in-place computation";
    const BlobShape& top_blob_shape = this->layer_param_.reform_param().shape();
    const int top_num_axes = top_blob_shape.dim_size();
    CHECK_EQ(top_num_axes, 4) << "top blob must have 4 dimensions";
    int constant_count = 1;
    int top_dim = 0;
    for (int i = 0; i< top_num_axes; ++i) {
        top_dim = top_blob_shape.dim(i);
        constant_count *= top_dim;
    }
    patch_size_ = bottom[0]->shape(2);
    CHECK_EQ(bottom[0]->count(0, 1), constant_count) << "bottom count must equal to top count";
}

template <typename Dtype> 
void ReformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {

    CHECK_EQ(bottom[0]->shape(2), bottom[0]->shape(3)) << "only support square for now";
    const BlobShape& tbs = this->layer_param_.reform_param().shape();
    const int top_num_axes = tbs.dim_size();
    vector<int> patch_dim(top_num_axes);
    patch_dim[0] = 1;
    patch_dim[1] = 1;
    patch_dim[2] = patch_size_;
    patch_dim[3] = patch_size_;
    top_shape_ = vector<int>(top_num_axes,0);
    int top_shape_index = 0;
    vector<int> top_shape(top_num_axes, 0);
    for (int i = 0; i < top_num_axes; ++i) {
        top_shape_[top_shape_index++] = tbs.dim(i) * patch_dim[i];
    }
    top[0]->Reshape(top_shape_);
    CHECK_EQ(top[0]->shape(0), 1)<<"1";
    CHECK_EQ(top[0]->shape(1), 2)<<"2";
    CHECK_EQ(top[0]->shape(2), 384)<<"3";
    CHECK_EQ(top[0]->shape(3), 512)<<"4";
}

template <typename Dtype>
void ReformLayer<Dtype>::reform_copy(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top, 
        const Dtype* src_data, 
        Dtype* dest_data,
        bool is_forward)  {
    int slice_num = bottom[0]->shape(0);
    int offset_length = top[0]->num_axes();
    int height = top[0]->shape(2);
    int width = top[0]->shape(3);
    CHECK_EQ(height, 384) << "height";
    CHECK_EQ(width, 512) << "width";
    CHECK_EQ(slice_num,96) << "slice_num";
    CHECK_EQ(patch_size_, 64) << "patch size";
    vector<int> bottom_offset(offset_length, 0);
    vector<int> top_offset(offset_length, 0);
    for (int i = 0; i < slice_num; ++i) {
        for (int r = 0; r < patch_size_; ++r) {
            //bottom_offset[0] = 0;
            bottom_offset[0] = i;
            bottom_offset[2] = r;
            //bottom_offset[3] = 0;
            // top
            top_offset[1] = i * patch_size_ * patch_size_/ (height * width) ; // channel
            top_offset[2] = r + ((i%48) * patch_size_ / width) * patch_size_; // height
            top_offset[3] = (i * patch_size_ ) % width;
            // do the copy
            if (is_forward) {
                caffe_copy(patch_size_, src_data + bottom[0]->offset(bottom_offset),
                    dest_data + top[0]->offset(top_offset));
            } else {
                caffe_copy(patch_size_, src_data + top[0]->offset(top_offset),
                    dest_data + bottom[0]->offset(bottom_offset));
            }
        }  // for loop <r>
    }  // for loop <i>
}  // reform_copy()

template < typename Dtype>
void ReformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    reform_copy(bottom, top, bottom_data, top_data, true);
}

template <typename Dtype>
void ReformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, 
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    if (propagate_down[0]) {
        caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
        reform_copy(bottom, top, top_diff, bottom_diff, false);
    }
}

#ifdef CPU_ONLY
STUB_GPU(ReformLayer);
#endif

INSTANTIATE_CLASS(ReformLayer);
REGISTER_LAYER_CLASS(Reform);

}  // namespace caffe