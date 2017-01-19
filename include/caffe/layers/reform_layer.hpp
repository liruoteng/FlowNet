#ifndef CAFFE_REFORM_LAYER_HPP_
#define CAFFE_REFORM_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/** 
 * Take re-join each channel of the bottom blob into a full size channel
 */

template <typename Dtype>
class ReformLayer : public Layer<Dtype> {
public: 
	explicit ReformLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const {return "Reform"; }
	virtual inline int ExactNumBottomBlobs() const {return 1; }
	virtual inline int ExactNumTopBlobs() const {return 1; }

protected: 

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	/* GPU version to be implemented
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
		*/
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	/* GPU version to be implemented
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, 
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		*/
	int count_;
	int patch_size_;
	vector<int> top_shape_;

private:
	void reform_copy(const vector<Blob<Dtype>*>& bottom, 
		const vector<Blob<Dtype>*>& top,
		const Dtype* src_data, 
		Dtype* dest_data,
		bool is_forward);
	
};

}  // namespace caffe

#endif  // CAFFE_REFORM_LAYER_HPP_