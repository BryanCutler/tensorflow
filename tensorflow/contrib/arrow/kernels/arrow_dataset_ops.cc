/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *     ==============================================================================*/

#include "tensorflow/core/framework/dataset.h"
#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"

#define CHECK_ARROW(arrow_status)              \
  do {                                         \
     arrow::Status _s = (arrow_status);        \
     if (!_s.ok()) {                           \
       return errors::Internal(_s.ToString()); \
     }                                         \
  } while (false)                              \

namespace tensorflow {

class ArrowDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  explicit ArrowDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    // TODO: check types and shapes
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* host_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("host", &host_tensor));
    OP_REQUIRES(ctx, host_tensor->dims() == 0,
       errors::InvalidArgument("`host` must be a scalar.")); 
    string host = host_tensor->flat<string>()(0);
    
    const Tensor* columns_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("columns", &columns_tensor));
    OP_REQUIRES(
        ctx, columns_tensor->dims() <= 1,
        errors::InvalidArgument("`columns` must be a scalar or a vector."));
    
    // TODO: should this be col names?
    std::vector<int32> columns;
    columns.reserve(columns_tensor->NumElements());
    for (int32 i = 0; i < static_cast<int32>(columns_tensor->NumElements()); ++i) {
      columns.push_back(columns_tensor->flat<int32>()(i));
    }

    *output = new Dataset(ctx, host, columns, output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx,
            const string& host,
            const std::vector<int32>& columns,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          host_(host),
          columns_(columns),
          output_types_(output_types),
          output_shapes_(output_shapes) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Arrow")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override { return "ArrowDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);

        // Intialize and read first batch
        if (current_batch_ == nullptr) {
          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        }

        // Try to go to next batch if consumed all rows
        if (current_row_idx_ >= current_batch_->num_rows()) {
          TF_RETURN_IF_ERROR(NextStreamLocked());
        }

        // Check if reached end of stream
        if (current_batch_ == nullptr) {
          ResetStreamsLocked();
          *end_of_sequence = true;
        }

        for (size_t i = 0; i < dataset()->columns_.size(); i++) {
          int32 column = dataset()->columns_[i];
          DataType dt = dataset()->output_types_[i];
          Tensor out_tensor(ctx->allocator({}), dt, {});
          TF_RETURN_IF_ERROR(GetElementAsTensor(column, dt, &out_tensor));
          out_tensors->emplace_back(std::move(out_tensor));
          ++current_row_idx_;
          *end_of_sequence = false;
        }        
 
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        return errors::Unimplemented("SaveInternal is currently not supported");
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        return errors::Unimplemented("RestoreInternal is currently not supported");      
      }

     private:
      Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        CHECK_ARROW(arrow::io::ReadableFile::Open(dataset()->host_, &in_file_));
        CHECK_ARROW(arrow::ipc::RecordBatchFileReader::Open(in_file_.get(), &reader_));
        num_batches_ = reader_->num_record_batches();
        if (num_batches_ > 0) {
          CHECK_ARROW(reader_->ReadRecordBatch(current_batch_idx_, &current_batch_));
        }
        return Status::OK();
      }

      Status NextStreamLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (current_batch_idx_ < num_batches_) {
          CHECK_ARROW(reader_->ReadRecordBatch(current_batch_idx_, &current_batch_));
          current_row_idx_ = 0;
          ++current_batch_idx_;
        }
        return Status::OK();
      }

      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        reader_.reset();
        in_file_.reset();
        current_batch_ = nullptr;
        current_batch_idx_ = 0;
        current_row_idx_ = 0;
        num_batches_ = 0;
      }

      Status GetElementAsTensor(int32 column, DataType dt, Tensor *out_tensor) EXCLUSIVE_LOCKS_REQUIRED(mu_) {


        return Status::OK();
      }

     private:
      mutex mu_;
      std::shared_ptr<arrow::io::ReadableFile> in_file_ GUARDED_BY(mu_);
      std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader_ GUARDED_BY(mu_);
      std::shared_ptr<arrow::RecordBatch> current_batch_ GUARDED_BY(mu_) = nullptr;
      int64_t current_batch_idx_ GUARDED_BY(mu_) = 0;
      int64_t current_row_idx_ GUARDED_BY(mu_) = 0;
      int num_batches_ GUARDED_BY(mu_) = 0;
    };

   private:
    const string host_;
    const std::vector<int32> columns_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
 };

 private:
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("ArrowDataset").Device(DEVICE_CPU),
                        ArrowDatasetOp);

}  // namespace tensorflow
