#include "translator.h"
#include "translator.cu"
#include "util.h"
#include <mpi.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE       8

#define CUDA_MALLOC(TENSOR_NAME, ...)                           \
  cudaMallocManaged((void **)&TENSOR_NAME, sizeof(Tensor));     \
  TENSOR_NAME->init_cuda({__VA_ARGS__});

#define CUDA_DELETE(TENSOR_NAME)                                \
  cudaFree(TENSOR_NAME->buf);                                   \
  cudaFree(TENSOR_NAME);

static int BATCH_SIZE;
static int HIDDEN_SIZE = 256;
static int INPUT_VOCAB_SIZE = 4345;
static int OUTPUT_VOCAB_SIZE = 2803;

/*
 * Tensor
 * @brief : A multi-dimensional matrix containing elements of a single data type.
 *
 * @member buf    : Data buffer containing elements
 * @member shape  : Shape of tensor from outermost dimension to innermost dimension
                    - e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
 */
Tensor::Tensor(std::vector<int> shape_) {
  ndim = shape_.size();
  for (int i=0; i<ndim; ++i) {
    shape[i] = shape_[i];
  }
  int N_ = num_elem();
  buf = (float *)calloc(N_, sizeof(float));
}

Tensor::Tensor(std::vector<int> shape_, float *buf_) {
  ndim = shape_.size();
  for (int i=0; i<ndim; ++i) {
    shape[i] = shape_[i];
  }
  int N_ = num_elem();
  buf = (float *) malloc(N_ * sizeof(float));
  for (int n = 0; n < N_; ++n) {
    buf[n] = buf_[n];
  }
}

void Tensor::init_cuda(std::vector<int> shape_) {
  ndim = shape_.size();
  for (int i = 0; i < ndim; ++i) shape[i] = shape_[i];
  cudaMalloc((void **)&buf, sizeof(float) * num_elem());
}

Tensor::~Tensor() {
  if (buf != nullptr) free(buf);
}

__host__ __device__ int Tensor::num_elem() {
  int sz = 1;
  for (int i=0; i<ndim; ++i){
    sz *= shape[i];
  }
  return sz;
}

void Tensor::fill_zeros() {
  int N_ = num_elem();
  for (int n=0; n<N_; ++n) {
    buf[n] = 0.0;
  }
}

// Parameters
Tensor *eW_emb;
Tensor *eW_ir, *eW_iz, *eW_in;
Tensor *eW_hr, *eW_hz, *eW_hn;
Tensor *eb_ir, *eb_iz, *eb_in;
Tensor *eb_hr, *eb_hz, *eb_hn;
Tensor *dW_emb;
Tensor *dW_ir, *dW_iz, *dW_in;
Tensor *dW_hr, *dW_hz, *dW_hn;
Tensor *db_ir, *db_iz, *db_in;
Tensor *db_hr, *db_hz, *db_hn;
Tensor *dW_attn, *db_attn, *dW_attn_comb, *db_attn_comb, *dW_out, *db_out;

// Encoder Activations
Tensor *encoder_embidx, *encoder_hidden, *encoder_outputs;
Tensor *encoder_embedded;
Tensor *encoder_rtmp1, *encoder_rtmp2, *encoder_rtmp3, *encoder_rtmp4, *encoder_rtmp5, *encoder_rt;
Tensor *encoder_ztmp1, *encoder_ztmp2, *encoder_ztmp3, *encoder_ztmp4, *encoder_ztmp5, *encoder_zt;
Tensor *encoder_ntmp1, *encoder_ntmp2, *encoder_ntmp3, *encoder_ntmp4, *encoder_ntmp5, *encoder_ntmp6, *encoder_nt;
Tensor *encoder_htmp1, *encoder_htmp2, *encoder_htmp3, *encoder_ht;

// Decoder Activations
Tensor *decoder_embidx, *decoder_hidden, *decoder_embedded, *decoder_embhid;
Tensor *decoder_attn, *decoder_attn_weights, *decoder_attn_applied, *decoder_embattn;
Tensor *decoder_attn_comb, *decoder_relu;
Tensor *decoder_rtmp1, *decoder_rtmp2, *decoder_rtmp3, *decoder_rtmp4, *decoder_rtmp5, *decoder_rt;
Tensor *decoder_ztmp1, *decoder_ztmp2, *decoder_ztmp3, *decoder_ztmp4, *decoder_ztmp5, *decoder_zt;
Tensor *decoder_ntmp1, *decoder_ntmp2, *decoder_ntmp3, *decoder_ntmp4, *decoder_ntmp5, *decoder_ntmp6, *decoder_nt;
Tensor *decoder_htmp1, *decoder_htmp2, *decoder_htmp3, *decoder_ht;
Tensor *decoder_out, *decoder_logsoftmax, *decoder_outputs;

// Miscellaneous Variables
int *running_batches;
float *in_buf, *out_buf, *dev_params;

float *parameter = NULL;
size_t parameter_binary_size = 0;

// Initializations
void init_buffers(float *_in_buf, float *_out_buf);
void init_running_batches(int *runnings);
void init_encoder(Tensor *_hidden, Tensor *_outputs);
void init_decoder(Tensor *_embidx);

// Operations
void check_encoder_termination(float *input, int *runnings, int word_idx);
void fetch_words(float *input, Tensor *output, int word_idx);
void embedding(Tensor *input, Tensor *weight, Tensor *output);
void matvec(Tensor *input, Tensor *weight, Tensor *output);
void elemwise_add(Tensor *input1, Tensor *input2, Tensor *output);
void elemwise_sigmoid(Tensor *input, Tensor *output);
void elemwise_tanh(Tensor *input, Tensor *output);
void elemwise_mult(Tensor *input1, Tensor *input2, Tensor *output);
void elemwise_oneminus(Tensor *input, Tensor *output);
void select(Tensor *input_true, Tensor *input_false, Tensor *output, int *choices);
void copy_encoder_outputs(Tensor *input, Tensor *output, int *choices, int word_idx);
void concat(Tensor *input1, Tensor *input2, Tensor *output);
void linear(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output);
void softmax(Tensor *input, Tensor *output);
void bmm(Tensor *input, Tensor *weight, Tensor *output);
void relu(Tensor *input, Tensor *output);
void top_one(Tensor *input, Tensor *output);
void log_softmax(Tensor *input, Tensor *output);
void check_decoder_termination(Tensor *outputs, Tensor *embidx, float *out_buf, int *runnings, int word_idx);

/*
 * translator
 * @brief : French to English translator.
 *          Translate N sentences in French into N sentences in English
 *
 * @param [in1] input  : a tensor of size [N x MAX_LENGTH]. French tokens are stored in this tensor.
 * @param [out] output : a tensor of size [N x MAX_LENGTH]. English tokens will be stored in this tensor.
 */
void translator(Tensor *input, Tensor *output, int N){
  int mpi_rank, mpi_world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  int input_size_per_node = input->num_elem() / mpi_world_size;
  MPI_Scatter(
    &input->buf[mpi_rank * input_size_per_node], input_size_per_node, MPI_FLOAT,
    &input->buf[mpi_rank * input_size_per_node], input_size_per_node, MPI_FLOAT,
    0, MPI_COMM_WORLD);

  cudaSetDevice(0);

  cudaMemcpy(
    dev_params,
    parameter,
    sizeof(float) * parameter_binary_size,
    cudaMemcpyHostToDevice);

  init_weights<<<1, 1>>>(eW_emb, eW_ir, eW_iz, eW_in, eW_hr, eW_hz, eW_hn, eb_ir, eb_iz, eb_in, eb_hr, eb_hz, eb_hn, dW_emb, dW_ir, dW_iz, dW_in, dW_hr, dW_hz, dW_hn, db_ir, db_iz, db_in, db_hr, db_hz, db_hn, dW_attn, db_attn, dW_attn_comb, db_attn_comb, dW_out, db_out, dev_params);

  init_buffers(in_buf, out_buf);

  cudaMemcpy(
    in_buf,
    &input->buf[mpi_rank * input_size_per_node],
    sizeof(float) * input_size_per_node,
    cudaMemcpyHostToDevice);

  // Encoder init
  init_encoder(encoder_hidden, encoder_outputs);
  init_running_batches(running_batches);

  // Encoder
  for (int work_idx = 0; work_idx < MAX_LENGTH; ++work_idx) {

    check_encoder_termination(in_buf, running_batches, work_idx);

    fetch_words(in_buf, encoder_embidx, work_idx);
    embedding(encoder_embidx, eW_emb, encoder_embedded);

    // GRU
    // r_t
    matvec(encoder_embedded, eW_ir, encoder_rtmp1);
    elemwise_add(encoder_rtmp1, eb_ir, encoder_rtmp2);
    matvec(encoder_hidden, eW_hr, encoder_rtmp3);
    elemwise_add(encoder_rtmp3, eb_hr, encoder_rtmp4);
    elemwise_add(encoder_rtmp2, encoder_rtmp4, encoder_rtmp5);
    elemwise_sigmoid(encoder_rtmp5, encoder_rt);

    // z_t
    matvec(encoder_embedded, eW_iz, encoder_ztmp1);
    elemwise_add(encoder_ztmp1, eb_iz, encoder_ztmp2);
    matvec(encoder_hidden, eW_hz, encoder_ztmp3);
    elemwise_add(encoder_ztmp3, eb_hz, encoder_ztmp4);
    elemwise_add(encoder_ztmp2, encoder_ztmp4, encoder_ztmp5);
    elemwise_sigmoid(encoder_ztmp5, encoder_zt);

    // n_t
    matvec(encoder_embedded, eW_in, encoder_ntmp1);
    elemwise_add(encoder_ntmp1, eb_in, encoder_ntmp2);
    matvec(encoder_hidden, eW_hn, encoder_ntmp3);
    elemwise_add(encoder_ntmp3, eb_hn, encoder_ntmp4);
    elemwise_mult(encoder_rt, encoder_ntmp4, encoder_ntmp5);
    elemwise_add(encoder_ntmp2, encoder_ntmp5, encoder_ntmp6);
    elemwise_tanh(encoder_ntmp6, encoder_nt);

    // h_t
    elemwise_oneminus(encoder_zt, encoder_htmp1);
    elemwise_mult(encoder_htmp1, encoder_nt, encoder_htmp2);
    elemwise_mult(encoder_zt, encoder_hidden, encoder_htmp3);
    elemwise_add(encoder_htmp2, encoder_htmp3, encoder_ht);
    select(encoder_ht, encoder_hidden, encoder_hidden, running_batches);

    copy_encoder_outputs(encoder_hidden, encoder_outputs, running_batches, work_idx);
  } // end Encoder loop

  // Decoder init
  decoder_hidden = encoder_hidden;

  init_decoder(decoder_embidx);
  init_running_batches(running_batches);

  // Decoder
  for (int work_idx = 0; work_idx < MAX_LENGTH; ++work_idx) {

    // Embedding
    embedding(decoder_embidx, dW_emb, decoder_embedded);

    // Attention
    concat(decoder_embedded, decoder_hidden, decoder_embhid);
    linear(decoder_embhid, dW_attn, db_attn, decoder_attn);
    softmax(decoder_attn, decoder_attn_weights);
    bmm(decoder_attn_weights, encoder_outputs, decoder_attn_applied);
    concat(decoder_embedded, decoder_attn_applied, decoder_embattn);
    linear(decoder_embattn, dW_attn_comb, db_attn_comb, decoder_attn_comb);
    relu(decoder_attn_comb, decoder_relu);

    // GRU
    // r_t
    matvec(decoder_relu, dW_ir, decoder_rtmp1);
    elemwise_add(decoder_rtmp1, db_ir, decoder_rtmp2);
    matvec(decoder_hidden, dW_hr, decoder_rtmp3);
    elemwise_add(decoder_rtmp3, db_hr, decoder_rtmp4);
    elemwise_add(decoder_rtmp2, decoder_rtmp4, decoder_rtmp5);
    elemwise_sigmoid(decoder_rtmp5, decoder_rt);

    // z_t
    matvec(decoder_relu, dW_iz, decoder_ztmp1);
    elemwise_add(decoder_ztmp1, db_iz, decoder_ztmp2);
    matvec(decoder_hidden, dW_hz, decoder_ztmp3);
    elemwise_add(decoder_ztmp3, db_hz, decoder_ztmp4);
    elemwise_add(decoder_ztmp2, decoder_ztmp4, decoder_ztmp5);
    elemwise_sigmoid(decoder_ztmp5, decoder_zt);

    // n_t
    matvec(decoder_relu, dW_in, decoder_ntmp1);
    elemwise_add(decoder_ntmp1, db_in, decoder_ntmp2);
    matvec(decoder_hidden, dW_hn, decoder_ntmp3);
    elemwise_add(decoder_ntmp3, db_hn, decoder_ntmp4);
    elemwise_mult(decoder_rt, decoder_ntmp4, decoder_ntmp5);
    elemwise_add(decoder_ntmp2, decoder_ntmp5, decoder_ntmp6);
    elemwise_tanh(decoder_ntmp6, decoder_nt);

    // h_t
    elemwise_oneminus(decoder_zt, decoder_htmp1);
    elemwise_mult(decoder_htmp1, decoder_nt, decoder_htmp2);
    elemwise_mult(decoder_zt, decoder_hidden, decoder_htmp3);
    elemwise_add(decoder_htmp2, decoder_htmp3, decoder_hidden);

    // Select output token
    linear(decoder_hidden, dW_out, db_out, decoder_out);
    log_softmax(decoder_out, decoder_logsoftmax);
    top_one(decoder_logsoftmax, decoder_outputs);

    check_decoder_termination(decoder_outputs, decoder_embidx, out_buf, running_batches, work_idx);
  } // end Decoder loop

  int output_size_per_node = output->num_elem() / mpi_world_size;

  cudaMemcpy(
    &output->buf[mpi_rank * output_size_per_node],
    out_buf,
    sizeof(float) * output_size_per_node,
    cudaMemcpyDeviceToHost);

  MPI_Gather(
    &output->buf[mpi_rank * output_size_per_node], output_size_per_node, MPI_FLOAT,
    &output->buf[mpi_rank * output_size_per_node], output_size_per_node, MPI_FLOAT,
    0, MPI_COMM_WORLD);
}

void init_buffers(float *_in_buf, float *_out_buf) {
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(
    (BATCH_SIZE + blockDim.x - 1) / blockDim.x,
    (MAX_LENGTH + blockDim.y - 1) / blockDim.y);
  _init_buffers<<<gridDim, blockDim>>>(_in_buf, _out_buf, BATCH_SIZE);
}

void init_running_batches(int *runnings) {
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim((BATCH_SIZE + blockDim.x - 1) / blockDim.x);
  _init_running_batches<<<gridDim, blockDim>>>(runnings, BATCH_SIZE);
}

void init_encoder(Tensor *_hidden, Tensor *_outputs) {
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(
    (_outputs->shape[0] + blockDim.x - 1) / blockDim.x,
    (_outputs->shape[2] + blockDim.y - 1) / blockDim.y);
  _init_encoder<<<gridDim, blockDim>>>(_hidden, _outputs);
}

void init_decoder(Tensor *_embidx) {
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim((_embidx->shape[0] + blockDim.x - 1) / blockDim.x);
  _init_decoder<<<gridDim, blockDim>>>(_embidx);
}

void check_encoder_termination(float *input, int *runnings, int word_idx) {
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(BATCH_SIZE / blockDim.x);
  _check_encoder_termination<<<gridDim, blockDim>>>(input, runnings, word_idx, BATCH_SIZE);
}

void fetch_words(float *input, Tensor *output, int word_idx) {
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim((BATCH_SIZE + blockDim.x - 1) / blockDim.x);
  _fetch_words<<<gridDim, blockDim>>>(input, output, word_idx, BATCH_SIZE);
}

void embedding(Tensor *input, Tensor *weight, Tensor *output) {
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(
    (input->shape[0] + blockDim.x - 1) / blockDim.x,
    (weight->shape[1] + blockDim.y - 1) / blockDim.y);
  _embedding<<<gridDim, blockDim>>>(input, weight, output);
}

void matvec(Tensor *input, Tensor *weight, Tensor *output) {
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(
    (input->shape[0] + blockDim.x - 1) / blockDim.x,
    (weight->shape[0] + blockDim.y - 1) / blockDim.y);
  _matvec<<<gridDim, blockDim>>>(input, weight, output);
}

void elemwise_add(Tensor *input1, Tensor *input2, Tensor *output) {
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(
    (input1->shape[0] + blockDim.x - 1) / blockDim.x,
    (input1->shape[1] + blockDim.y - 1) / blockDim.y);
  _elemwise_add<<<gridDim, blockDim>>>(input1, input2, output);
}

void elemwise_sigmoid(Tensor *input, Tensor *output) {
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(
    (input->num_elem() + blockDim.x - 1) / blockDim.x);
  _elemwise_sigmoid<<<gridDim, blockDim>>>(input, output);
}

void elemwise_tanh(Tensor *input, Tensor *output) {
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(
    (input->num_elem() + blockDim.x - 1) / blockDim.x);
  _elemwise_tanh<<<gridDim, blockDim>>>(input, output);
}

void elemwise_mult(Tensor *input1, Tensor *input2, Tensor *output) {
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(
    (input1->num_elem() + blockDim.x - 1) / blockDim.x);
  _elemwise_mult<<<gridDim, blockDim>>>(input1, input2, output);
}

void elemwise_oneminus(Tensor *input, Tensor *output) {
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(
    (input->num_elem() + blockDim.x - 1) / blockDim.x);
  _elemwise_oneminus<<<gridDim, blockDim>>>(input, output);
}

void select(Tensor *input_true, Tensor *input_false, Tensor *output, int *choices) {
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(
    (input_true->shape[0] + blockDim.x - 1) / blockDim.x,
    (input_true->shape[1] + blockDim.y - 1) / blockDim.y);
  _select<<<gridDim, blockDim>>>(input_true, input_false, output, choices);
}

void copy_encoder_outputs(Tensor *input, Tensor *output, int *choices, int word_idx) {
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(
    (input->shape[0] + blockDim.x - 1) / blockDim.x,
    (input->shape[1] + blockDim.y - 1) / blockDim.y);
  _copy_encoder_outputs<<<gridDim, blockDim>>>(input, output, choices, word_idx);
}

void concat(Tensor *input1, Tensor *input2, Tensor *output) {
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(
    (input1->shape[0] + blockDim.x - 1) / blockDim.x,
    (2 * input1->shape[1] + blockDim.y - 1) / blockDim.y);
  _concat<<<gridDim, blockDim>>>(input1, input2, output);
}

void linear(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output) {
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(
    (input->shape[0] + blockDim.x - 1) / blockDim.x,
    (weight->shape[0] + blockDim.y - 1) / blockDim.y);
  _linear<<<gridDim, blockDim>>>(input, weight, bias, output);
}

void softmax(Tensor *input, Tensor *output) {
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(
    (input->num_elem() + blockDim.x - 1) / blockDim.x);
  _softmax<<<gridDim, blockDim>>>(input, output);
}

void bmm(Tensor *input, Tensor *weight, Tensor *output) {
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(
    (input->shape[0] + blockDim.x - 1) / blockDim.x,
    (weight->shape[2] + blockDim.y - 1) / blockDim.y);
  _bmm<<<gridDim, blockDim>>>(input, weight, output);
}

void relu(Tensor *input, Tensor *output) {
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(
    (input->num_elem() + blockDim.x - 1) / blockDim.x);
  _relu<<<gridDim, blockDim>>>(input, output);
}

void top_one(Tensor *input, Tensor *output) {
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(
    (input->shape[0] + blockDim.x - 1) / blockDim.x);
  _top_one<<<gridDim, blockDim>>>(input, output);
}

void log_softmax(Tensor *input, Tensor *output) {
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(
    (input->shape[0] + blockDim.x - 1) / blockDim.x);
  _log_softmax<<<gridDim, blockDim>>>(input, output);
}

void check_decoder_termination(Tensor *outputs, Tensor *embidx, float *out_buf, int *runnings, int word_idx) {
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(BATCH_SIZE / blockDim.x);
  _check_decoder_termination<<<gridDim, blockDim>>>(outputs, embidx, out_buf, runnings, word_idx);
}

/*
 * initialize_translator
 * @brief : initialize translator. load the parameter binary file and store parameters into Tensors
 *
 * @param [in1] parameter_fname  : the name of the binary file where parameters are stored
 */
void initialize_translator(const char *parameter_fname, int N){
  int mpi_rank, mpi_world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  BATCH_SIZE = N / mpi_world_size;

  if (mpi_rank == 0) {
    parameter = (float *) read_binary(parameter_fname, &parameter_binary_size);
  }

  MPI_Bcast(&parameter_binary_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  if (mpi_rank != 0) {
    parameter = new float[parameter_binary_size];
  }

  MPI_Bcast(parameter, parameter_binary_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

  cudaSetDevice(0);

  CUDA_MALLOC(eW_emb, INPUT_VOCAB_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(eW_ir, HIDDEN_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(eW_iz, HIDDEN_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(eW_in, HIDDEN_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(eW_hr, HIDDEN_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(eW_hz, HIDDEN_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(eW_hn, HIDDEN_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(eb_ir, HIDDEN_SIZE);
  CUDA_MALLOC(eb_iz, HIDDEN_SIZE);
  CUDA_MALLOC(eb_in, HIDDEN_SIZE);
  CUDA_MALLOC(eb_hr, HIDDEN_SIZE);
  CUDA_MALLOC(eb_hz, HIDDEN_SIZE);
  CUDA_MALLOC(eb_hn, HIDDEN_SIZE);
  CUDA_MALLOC(dW_emb, OUTPUT_VOCAB_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(dW_ir, HIDDEN_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(dW_iz, HIDDEN_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(dW_in, HIDDEN_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(dW_hr, HIDDEN_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(dW_hz, HIDDEN_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(dW_hn, HIDDEN_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(db_ir, HIDDEN_SIZE);
  CUDA_MALLOC(db_iz, HIDDEN_SIZE);
  CUDA_MALLOC(db_in, HIDDEN_SIZE);
  CUDA_MALLOC(db_hr, HIDDEN_SIZE);
  CUDA_MALLOC(db_hz, HIDDEN_SIZE);
  CUDA_MALLOC(db_hn, HIDDEN_SIZE);
  CUDA_MALLOC(dW_attn, MAX_LENGTH, 2 * HIDDEN_SIZE);
  CUDA_MALLOC(db_attn, MAX_LENGTH);
  CUDA_MALLOC(dW_attn_comb, HIDDEN_SIZE, 2 * HIDDEN_SIZE);
  CUDA_MALLOC(db_attn_comb, HIDDEN_SIZE);
  CUDA_MALLOC(dW_out, OUTPUT_VOCAB_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(db_out, OUTPUT_VOCAB_SIZE);

  CUDA_MALLOC(encoder_embidx, BATCH_SIZE);
  CUDA_MALLOC(encoder_hidden, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_outputs, BATCH_SIZE, MAX_LENGTH, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_embedded, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_rtmp1, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_rtmp2, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_rtmp3, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_rtmp4, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_rtmp5, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_rt, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_ztmp1, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_ztmp2, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_ztmp3, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_ztmp4, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_ztmp5, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_zt, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_ntmp1, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_ntmp2, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_ntmp3, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_ntmp4, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_ntmp5, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_ntmp6, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_nt, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_htmp1, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_htmp2, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_htmp3, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(encoder_ht, BATCH_SIZE, HIDDEN_SIZE);

  CUDA_MALLOC(decoder_embidx, BATCH_SIZE);
  CUDA_MALLOC(decoder_hidden, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_embedded, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_embhid, BATCH_SIZE, 2 * HIDDEN_SIZE);
  CUDA_MALLOC(decoder_attn, BATCH_SIZE, MAX_LENGTH);
  CUDA_MALLOC(decoder_attn_weights, BATCH_SIZE, MAX_LENGTH);
  CUDA_MALLOC(decoder_attn_applied, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_embattn, BATCH_SIZE, 2 * HIDDEN_SIZE);
  CUDA_MALLOC(decoder_attn_comb, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_relu, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_rtmp1, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_rtmp2, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_rtmp3, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_rtmp4, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_rtmp5, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_rt, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_ztmp1, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_ztmp2, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_ztmp3, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_ztmp4, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_ztmp5, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_zt, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_ntmp1, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_ntmp2, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_ntmp3, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_ntmp4, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_ntmp5, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_ntmp6, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_nt, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_htmp1, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_htmp2, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_htmp3, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_ht, BATCH_SIZE, HIDDEN_SIZE);
  CUDA_MALLOC(decoder_out, BATCH_SIZE, OUTPUT_VOCAB_SIZE);
  CUDA_MALLOC(decoder_logsoftmax, BATCH_SIZE, OUTPUT_VOCAB_SIZE);
  CUDA_MALLOC(decoder_outputs, BATCH_SIZE);

  cudaMalloc((void **)&running_batches, sizeof(int) * BATCH_SIZE);

  cudaMalloc((void **)&in_buf, sizeof(float) * N * MAX_LENGTH);
  cudaMalloc((void **)&out_buf, sizeof(float) * N * MAX_LENGTH);
  cudaMalloc((void **)&dev_params, sizeof(float) * parameter_binary_size);
}

/*
 * finalize_translator
 * @brief : free all dynamically allocated variables
 */
void finalize_translator(){
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  fprintf(stderr, "\n");

  // free parameters
  CUDA_DELETE(eW_emb);
  CUDA_DELETE(eW_ir);
  CUDA_DELETE(eW_iz);
  CUDA_DELETE(eW_in);
  CUDA_DELETE(eW_hr);
  CUDA_DELETE(eW_hz);
  CUDA_DELETE(eW_hn);
  CUDA_DELETE(eb_ir);
  CUDA_DELETE(eb_iz);
  CUDA_DELETE(eb_in);
  CUDA_DELETE(eb_hr);
  CUDA_DELETE(eb_hz);
  CUDA_DELETE(eb_hn);
  CUDA_DELETE(dW_emb);
  CUDA_DELETE(dW_ir);
  CUDA_DELETE(dW_iz);
  CUDA_DELETE(dW_in);
  CUDA_DELETE(dW_hr);
  CUDA_DELETE(dW_hz);
  CUDA_DELETE(dW_hn);
  CUDA_DELETE(db_ir);
  CUDA_DELETE(db_iz);
  CUDA_DELETE(db_in);
  CUDA_DELETE(db_hr);
  CUDA_DELETE(db_hz);
  CUDA_DELETE(db_hn);
  CUDA_DELETE(dW_attn);
  CUDA_DELETE(db_attn);
  CUDA_DELETE(dW_attn_comb);
  CUDA_DELETE(db_attn_comb);
  CUDA_DELETE(dW_out);
  CUDA_DELETE(db_out);

  // free encoder activations
  CUDA_DELETE(encoder_embidx);
  CUDA_DELETE(encoder_hidden);
  CUDA_DELETE(encoder_outputs);
  CUDA_DELETE(encoder_embedded);
  CUDA_DELETE(encoder_rtmp1);
  CUDA_DELETE(encoder_rtmp2);
  CUDA_DELETE(encoder_rtmp3);
  CUDA_DELETE(encoder_rtmp4);
  CUDA_DELETE(encoder_rtmp5);
  CUDA_DELETE(encoder_rt);
  CUDA_DELETE(encoder_ztmp1);
  CUDA_DELETE(encoder_ztmp2);
  CUDA_DELETE(encoder_ztmp3);
  CUDA_DELETE(encoder_ztmp4);
  CUDA_DELETE(encoder_ztmp5);
  CUDA_DELETE(encoder_zt);
  CUDA_DELETE(encoder_ntmp1);
  CUDA_DELETE(encoder_ntmp2);
  CUDA_DELETE(encoder_ntmp3);
  CUDA_DELETE(encoder_ntmp4);
  CUDA_DELETE(encoder_ntmp5);
  CUDA_DELETE(encoder_ntmp6);
  CUDA_DELETE(encoder_nt);
  CUDA_DELETE(encoder_htmp1);
  CUDA_DELETE(encoder_htmp2);
  CUDA_DELETE(encoder_htmp3);
  CUDA_DELETE(encoder_ht);

  // free decoder activations
  CUDA_DELETE(decoder_embidx);
  CUDA_DELETE(decoder_embedded);
  CUDA_DELETE(decoder_embhid);
  CUDA_DELETE(decoder_attn);
  CUDA_DELETE(decoder_attn_weights);
  CUDA_DELETE(decoder_attn_applied);
  CUDA_DELETE(decoder_embattn);
  CUDA_DELETE(decoder_attn_comb);
  CUDA_DELETE(decoder_relu);
  CUDA_DELETE(decoder_rtmp1);
  CUDA_DELETE(decoder_rtmp2);
  CUDA_DELETE(decoder_rtmp3);
  CUDA_DELETE(decoder_rtmp4);
  CUDA_DELETE(decoder_rtmp5);
  CUDA_DELETE(decoder_rt);
  CUDA_DELETE(decoder_ztmp1);
  CUDA_DELETE(decoder_ztmp2);
  CUDA_DELETE(decoder_ztmp3);
  CUDA_DELETE(decoder_ztmp4);
  CUDA_DELETE(decoder_ztmp5);
  CUDA_DELETE(decoder_zt);
  CUDA_DELETE(decoder_ntmp1);
  CUDA_DELETE(decoder_ntmp2);
  CUDA_DELETE(decoder_ntmp3);
  CUDA_DELETE(decoder_ntmp4);
  CUDA_DELETE(decoder_ntmp5);
  CUDA_DELETE(decoder_ntmp6);
  CUDA_DELETE(decoder_nt);
  CUDA_DELETE(decoder_htmp1);
  CUDA_DELETE(decoder_htmp2);
  CUDA_DELETE(decoder_htmp3);
  CUDA_DELETE(decoder_ht);
  CUDA_DELETE(decoder_out);
  CUDA_DELETE(decoder_logsoftmax);
  CUDA_DELETE(decoder_outputs);

  // free misc. variables
  cudaFree(running_batches);
  cudaFree(in_buf);
  cudaFree(out_buf);
  cudaFree(dev_params);

  delete[] parameter;
}
