#include "translator.h"
#include "util.h"
#include <mpi.h>
#include <math.h>
#include <cuda_runtime.h>

#define SOS_TOKEN     0
#define EOS_TOKEN     1

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

__global__ void init_weights(
  Tensor *_eW_emb,
  Tensor *_eW_ir,
  Tensor *_eW_iz,
  Tensor *_eW_in,
  Tensor *_eW_hr,
  Tensor *_eW_hz,
  Tensor *_eW_hn,
  Tensor *_eb_ir,
  Tensor *_eb_iz,
  Tensor *_eb_in,
  Tensor *_eb_hr,
  Tensor *_eb_hz,
  Tensor *_eb_hn,
  Tensor *_dW_emb,
  Tensor *_dW_ir,
  Tensor *_dW_iz,
  Tensor *_dW_in,
  Tensor *_dW_hr,
  Tensor *_dW_hz,
  Tensor *_dW_hn,
  Tensor *_db_ir,
  Tensor *_db_iz,
  Tensor *_db_in,
  Tensor *_db_hr,
  Tensor *_db_hz,
  Tensor *_db_hn,
  Tensor *_dW_attn,
  Tensor *_db_attn,
  Tensor *_dW_attn_comb,
  Tensor *_db_attn_comb,
  Tensor *_dW_out,
  Tensor *_db_out,
  float *param) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i > 0) return;

  _eW_emb->buf = param + OFFSET0;
  _eW_ir->buf = param + OFFSET1;
  _eW_iz->buf = param + OFFSET2;
  _eW_in->buf = param + OFFSET3;
  _eW_hr->buf = param + OFFSET4;
  _eW_hz->buf = param + OFFSET5;
  _eW_hn->buf = param + OFFSET6;
  _eb_ir->buf = param + OFFSET7;
  _eb_iz->buf = param + OFFSET8;
  _eb_in->buf = param + OFFSET9;
  _eb_hr->buf = param + OFFSET10;
  _eb_hz->buf = param + OFFSET11;
  _eb_hn->buf = param + OFFSET12;
  _dW_emb->buf = param + OFFSET13;
  _dW_ir->buf = param + OFFSET14;
  _dW_iz->buf = param + OFFSET15;
  _dW_in->buf = param + OFFSET16;
  _dW_hr->buf = param + OFFSET17;
  _dW_hz->buf = param + OFFSET18;
  _dW_hn->buf = param + OFFSET19;
  _db_ir->buf = param + OFFSET20;
  _db_iz->buf = param + OFFSET21;
  _db_in->buf = param + OFFSET22;
  _db_hr->buf = param + OFFSET23;
  _db_hz->buf = param + OFFSET24;
  _db_hn->buf = param + OFFSET25;
  _dW_attn->buf = param + OFFSET26;
  _db_attn->buf = param + OFFSET27;
  _dW_attn_comb->buf = param + OFFSET28;
  _db_attn_comb->buf = param + OFFSET29;
  _dW_out->buf = param + OFFSET30;
  _db_out->buf = param + OFFSET31;
}

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

__global__ void _init_buffers(float *_in_buf, float *_out_buf, int batch_size) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;

  if (b >= batch_size || n >= MAX_LENGTH) return;

  int idx = b * MAX_LENGTH + n;
  _in_buf[idx] = 0.0;
  _out_buf[idx] = 0.0;
}

void init_buffers(float *_in_buf, float *_out_buf) {
  dim3 blockDim(32, 32);
  dim3 gridDim(
    (BATCH_SIZE + blockDim.x - 1) / blockDim.x,
    (MAX_LENGTH + blockDim.y - 1) / blockDim.y);
  _init_buffers<<<gridDim, blockDim>>>(_in_buf, _out_buf, BATCH_SIZE);
}

__global__ void _init_running_batches(int *runnings, int batch_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_size) runnings[i] = 1;
}

void init_running_batches(int *runnings) {
  dim3 blockDim(32);
  dim3 gridDim((BATCH_SIZE + blockDim.x - 1) / blockDim.x);
  _init_running_batches<<<gridDim, blockDim>>>(runnings, BATCH_SIZE);
}

__global__ void _init_encoder(Tensor *_hidden, Tensor *_outputs) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;

  int B_ = _outputs->shape[0];
  int N_ = _outputs->shape[1];
  int H_ = _outputs->shape[2];

  if (b >= B_ || h >= H_) return;

  _hidden->buf[b * H_ + h] = 0.0;
  for (int n = 0; n < N_; ++n) _outputs->buf[(b * N_ + n) * H_ + h] = 0.0;
}

void init_encoder(Tensor *_hidden, Tensor *_outputs) {
  dim3 blockDim(32, 32);
  dim3 gridDim(
    (_outputs->shape[0] + blockDim.x - 1) / blockDim.x,
    (_outputs->shape[2] + blockDim.y - 1) / blockDim.y);
  _init_encoder<<<gridDim, blockDim>>>(_hidden, _outputs);
}

__global__ void _init_decoder(Tensor *_embidx) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;

  if (b >= _embidx->shape[0]) return;

  _embidx->buf[b] = SOS_TOKEN;
}

void init_decoder(Tensor *_embidx) {
  dim3 blockDim(32);
  dim3 gridDim((_embidx->shape[0] + blockDim.x - 1) / blockDim.x);
  _init_decoder<<<gridDim, blockDim>>>(_embidx);
}

__global__ void _check_encoder_termination(float *input, int *runnings, int word_idx, int batch_size) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;

  if (b >= batch_size || !runnings[b]) return;

  runnings[b] = input[b * MAX_LENGTH + word_idx] != 0.0;
}

void check_encoder_termination(float *input, int *runnings, int word_idx) {
  dim3 blockDim(32);
  dim3 gridDim(BATCH_SIZE / blockDim.x);
  _check_encoder_termination<<<gridDim, blockDim>>>(input, runnings, word_idx, BATCH_SIZE);
}

__global__ void _fetch_words(float *input, Tensor *output, int word_idx, int batch_size) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < batch_size) output->buf[b] = input[b * MAX_LENGTH + word_idx];
}

void fetch_words(float *input, Tensor *output, int word_idx) {
  dim3 blockDim(32);
  dim3 gridDim((BATCH_SIZE + blockDim.x - 1) / blockDim.x);
  _fetch_words<<<gridDim, blockDim>>>(input, output, word_idx, BATCH_SIZE);
}

/*
 * embedding
 * @brief : A simple lookup table that stores embeddings of a fixed dictionary and size.
 *
 * @param [in1] input  : embedding index   [B_]
 * @param [in2] weight : a matrix of size  [M_ x H_]
 * @param [out] output : a vectors of size [B_ x H_]
 */
__global__ void _embedding(Tensor *input, Tensor *weight, Tensor *output){
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;

  int B_ = input->shape[0];
  int H_ = weight->shape[1];

  if (b >= B_ || h >= H_) return;

  int ei = input->buf[b];
  output->buf[b * H_ + h] = weight->buf[ei * H_ + h];
}

void embedding(Tensor *input, Tensor *weight, Tensor *output) {
  dim3 blockDim(32, 32);
  dim3 gridDim(
    (input->shape[0] + blockDim.x - 1) / blockDim.x,
    (weight->shape[1] + blockDim.y - 1) / blockDim.y);
  _embedding<<<gridDim, blockDim>>>(input, weight, output);
}

/*
 * matvec
 * @brief : Perform a matrix-vector product of the matrix and the vector
 *
 * @param [in1] input  : a vectors of size [B_ x K_]
 * @param [in2] weight : a matrix of size  [M_ x K_]
 * @param [out] output : a vectors of size [B_ x M_]
 */
__global__ void _matvec(Tensor *input, Tensor *weight, Tensor *output) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;

  int B_ = input->shape[0];
  int M_ = weight->shape[0];
  int K_ = weight->shape[1];

  if (b >= B_ || m >= M_) return;

  float c = 0.0;
  for (int k = 0; k < K_; ++k) {
    float w = weight->buf[m * K_ + k];
    float i = input->buf[b * K_ + k];
    c += w * i;
  }
  output->buf[b * M_ + m] = c;
}

void matvec(Tensor *input, Tensor *weight, Tensor *output) {
  dim3 blockDim(32, 32);
  dim3 gridDim(
    (input->shape[0] + blockDim.x - 1) / blockDim.x,
    (weight->shape[0] + blockDim.y - 1) / blockDim.y);
  _matvec<<<gridDim, blockDim>>>(input, weight, output);
}

/*
 * elemwise_add
 * @brief : Element-by-element addition of tensors
 *
 * @param [in1] input1 : a vectors of size   [B_ * K _]
 * @param [in2] input2 : a vector(s) of size [K_] or [B_ * K _]
 * @param [out] output : a vectors of size   [B_ * K _]
 */
__global__ void _elemwise_add(Tensor *input1, Tensor *input2, Tensor *output){
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  int B_ = input1->shape[0];
  int K_ = input1->shape[1];

  if (b >= B_ || k >= K_) return;

  int index = b * K_ + k;
  output->buf[index] = input1->buf[index] + input2->buf[(input2->ndim == 1) ? k : index];
}

void elemwise_add(Tensor *input1, Tensor *input2, Tensor *output) {
  dim3 blockDim(32, 32);
  dim3 gridDim(
    (input1->shape[0] + blockDim.x - 1) / blockDim.x,
    (input1->shape[1] + blockDim.y - 1) / blockDim.y);
  _elemwise_add<<<gridDim, blockDim>>>(input1, input2, output);
}

/*
 * elemwise_sigmoid
 * @brief : Apply the element-wise sigmoid function. sigmoid(x) = 1 / (1 + exp(-x))
 *
 * @param [in1] input
 * @param [out] output
 */
__global__ void _elemwise_sigmoid(Tensor *input, Tensor *output) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  int N_ = input->num_elem();

  if (n >= N_) return;

  float x = input->buf[n];
  output->buf[n] = 1.0 / (1.0 + expf(-x));
}

void elemwise_sigmoid(Tensor *input, Tensor *output) {
  dim3 blockDim(32);
  dim3 gridDim(
    (input->num_elem() + blockDim.x - 1) / blockDim.x);
  _elemwise_sigmoid<<<gridDim, blockDim>>>(input, output);
}

/*
 * elemwise_tanh
 * @brief : Apply the Hyperbolic Tangent (Tanh) function element-wise.
 *
 * @param [in1] input
 * @param [out] output
 */
__global__ void _elemwise_tanh(Tensor *input, Tensor *output) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  int N_ = input->num_elem();

  if (n >= N_) return;

  float x = input->buf[n];
  output->buf[n] = tanhf(x);
}

void elemwise_tanh(Tensor *input, Tensor *output) {
  dim3 blockDim(32);
  dim3 gridDim(
    (input->num_elem() + blockDim.x - 1) / blockDim.x);
  _elemwise_tanh<<<gridDim, blockDim>>>(input, output);
}

/*
 * elemwise_mult
 * @brief : Element-by-element multiplication of tensors.
 *
 * @param [in1] input1
 * @param [in2] input2
 * @param [out] output
 */
__global__ void _elemwise_mult(Tensor *input1, Tensor *input2, Tensor *output) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  int N_ = input1->num_elem();

  if (n >= N_) return;

  float x1 = input1->buf[n];
  float x2 = input2->buf[n];
  output->buf[n] = x1 * x2;
}

void elemwise_mult(Tensor *input1, Tensor *input2, Tensor *output) {
  dim3 blockDim(32);
  dim3 gridDim(
    (input1->num_elem() + blockDim.x - 1) / blockDim.x);
  _elemwise_mult<<<gridDim, blockDim>>>(input1, input2, output);
}

/*
 * elemwise_oneminus
 * @brief : Apply the element-wise oneminus function. oneminus(x) = 1.0 - x
 *
 * @param [in1] input
 * @param [out] output
 */
__global__ void _elemwise_oneminus(Tensor *input, Tensor *output) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  int N_ = input->num_elem();

  if (n >= N_) return;

  float x = input->buf[n];
  output->buf[n] = 1.0 - x;
}

void elemwise_oneminus(Tensor *input, Tensor *output) {
  dim3 blockDim(32);
  dim3 gridDim(
    (input->num_elem() + blockDim.x - 1) / blockDim.x);
  _elemwise_oneminus<<<gridDim, blockDim>>>(input, output);
}

/*
 * select
 * @brief : Copies the data from i-th row from input_true if choices[i] is true, else from input_false
 *
 * @param [in1] input_true  : a tensor of size [B_ x N_]
 * @param [in2] input_false : a tensor of size [B_ x N_]
 * @param [out] output      : a tensor of size [B_ x N_]
 * @param [in3] choices     : an array of size [B_]
 */
__global__ void _select(Tensor *input_true, Tensor *input_false, Tensor *output, int *choices) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;

  int B_ = input_true->shape[0];
  int N_ = input_true->shape[1];

  if (b >= B_ || n >= N_) return;

  int choice = choices[b];
  int idx = b * N_ + n;
  output->buf[idx] = choice ? input_true->buf[idx] : input_false->buf[idx];
}

void select(Tensor *input_true, Tensor *input_false, Tensor *output, int *choices) {
  dim3 blockDim(32, 32);
  dim3 gridDim(
    (input_true->shape[0] + blockDim.x - 1) / blockDim.x,
    (input_true->shape[1] + blockDim.y - 1) / blockDim.y);
  _select<<<gridDim, blockDim>>>(input_true, input_false, output, choices);
}

/*
 * copy_encoder_outputs
 * @brief : Copy input vector of chosen batch into i-th row of the output matrix
 *
 * @param [in1] input   : a vectors of size  [B_ x N_]
 * @param [in2] i       : row index
 * @param [in3] choices : an array of size   [B_]
 * @param [out] output  : a matrices of size [B_ x MAX_LENGTH x N_]
 */
__global__ void _copy_encoder_outputs(Tensor *input, Tensor *output, int *choices, int word_idx) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;

  int B_ = input->shape[0];
  int N_ = input->shape[1];

  if (b >= B_ || n >= N_) return;

  if (!choices[b]) return;
  output->buf[(b * MAX_LENGTH + word_idx) * N_ + n] = input->buf[b * N_ + n];
}

void copy_encoder_outputs(Tensor *input, Tensor *output, int *choices, int word_idx) {
  dim3 blockDim(32, 32);
  dim3 gridDim(
    (input->shape[0] + blockDim.x - 1) / blockDim.x,
    (input->shape[1] + blockDim.y - 1) / blockDim.y);
  _copy_encoder_outputs<<<gridDim, blockDim>>>(input, output, choices, word_idx);
}

/*
 * concat
 * @brief : Concatenate the two input tensors
 *
 * @param [in1] input1 : a vectors of size [B x K_]
 * @param [in2] input2 : a vectors of size [B x K_]
 * @param [out] output : a vectors of size [B x 2 * K_]
 */
__global__ void _concat(Tensor *input1, Tensor *input2, Tensor *output) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  int B_ = input1->shape[0];
  int K_ = input1->shape[1];

  if (b >= B_ || k >= 2 * K_) return;

  output->buf[b * 2 * K_ + k] = k < K_ ? input1->buf[b * K_ + k] : input2->buf[b * K_ + k - K_];
}

void concat(Tensor *input1, Tensor *input2, Tensor *output) {
  dim3 blockDim(32, 32);
  dim3 gridDim(
    (input1->shape[0] + blockDim.x - 1) / blockDim.x,
    (2 * input1->shape[1] + blockDim.y - 1) / blockDim.y);
  _concat<<<gridDim, blockDim>>>(input1, input2, output);
}

/*
 * linear
 * @brief : Apply a linear transformation to the incoming data: linear(x) = x A + b.
 *
 * @param [in1] input  : a vectors of size [B_ x K_]
 * @param [in2] weight : a matrix of size  [N_ x K_]
 * @param [in3] bias   : a vector of size  [N_]
 * @param [out] output : a vectors of size [B_ x N_]
 */
__global__ void _linear(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;

  int B_ = input->shape[0];
  int K_ = weight->shape[1];
  int N_ = weight->shape[0];

  if (b >= B_ || n >= N_) return;

  float c = bias->buf[n];
  for (int k = 0; k < K_; ++k) {
    c += input->buf[b * K_ + k] * weight->buf[n * K_ + k];
  }
  output->buf[b * N_ + n] = c;
}

void linear(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output) {
  dim3 blockDim(32, 32);
  dim3 gridDim(
    (input->shape[0] + blockDim.x - 1) / blockDim.x,
    (weight->shape[0] + blockDim.y - 1) / blockDim.y);
  _linear<<<gridDim, blockDim>>>(input, weight, bias, output);
}

/*
 * softmax
 * @brief : Apply the Softmax function to an n-dimensional input Tensor rescaling them
 *          so that the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.
 *          softmax(xi) = exp(xi) / sum of exp(xi)
 *
 * @param [in1] input  : a vectors of size [B_ x N_]
 * @param [out] output : a vectors of size [B_ x N_]
 */
__global__ void _softmax(Tensor *input, Tensor *output) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;

  int B_ = input->shape[0];
  int N_ = input->shape[1];

  if (b >= B_) return;

  float sum = 0.0;
  for (int n = 0; n < N_; ++n) {
    sum += expf(input->buf[b * N_ + n]);
  }
  for (int n = 0; n < N_; ++n) {
    output->buf[b * N_ + n] = expf(input->buf[b * N_ + n]) / sum;
  }
}

void softmax(Tensor *input, Tensor *output) {
  dim3 blockDim(32);
  dim3 gridDim(
    (input->num_elem() + blockDim.x - 1) / blockDim.x);
  _softmax<<<gridDim, blockDim>>>(input, output);
}

/*
 * bmm
 * @brief : Perform a batch matrix-matrix product of matrices stored in input and weight.
 *          However, bmm performs matrix-vector product in this project.
 *
 * @param [in1] input  : a vectors of size  [B_ x K_]
 * @param [in2] weight : a matrices of size [B_ x K_ x N_]
 * @param [out] output : a vectors of size  [B_ x N_]
 */
__global__ void _bmm(Tensor *input, Tensor *weight, Tensor *output) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;

  int B_ = input->shape[0];
  int K_ = weight->shape[1];
  int N_ = weight->shape[2];

  if (b >= B_ || n >= N_) return;

  float c = 0.0;
  for (int k = 0; k < K_; ++k) {
    c += input->buf[b * K_ + k] * weight->buf[(b * K_ + k) * N_ + n];
  }
  output->buf[b * N_ + n] = c;
}

void bmm(Tensor *input, Tensor *weight, Tensor *output) {
  dim3 blockDim(32, 32);
  dim3 gridDim(
    (input->shape[0] + blockDim.x - 1) / blockDim.x,
    (weight->shape[2] + blockDim.y - 1) / blockDim.y);
  _bmm<<<gridDim, blockDim>>>(input, weight, output);
}

/*
 * relu
 * @brief : Apply the rectified linear unit function element-wise. relu(x) = max(0,x)
 *
 * @param [in1] input  : a vectors of size [B_ * K_]
 * @param [out] output : a vectors of size [B_ * K_]
 */
__global__ void _relu(Tensor *input, Tensor *output) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  int N_ = input->num_elem();

  if (n >= N_) return;

  float x = input->buf[n];
  if (x < 0.0) output->buf[n] = 0.0;
  else output->buf[n] = x;
}

void relu(Tensor *input, Tensor *output) {
  dim3 blockDim(32);
  dim3 gridDim(
    (input->num_elem() + blockDim.x - 1) / blockDim.x);
  _relu<<<gridDim, blockDim>>>(input, output);
}

/*
 * top_one
 * @brief : Return the largest element of the given input tensor.
 *
 * @param  [in1] input  : a vectors of size [B_ x N_]
 * @param  [out] output : a indices of size [B_]
 */
__global__ void _top_one(Tensor *input, Tensor *output) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;

  int B_ = input->shape[0];
  int N_ = input->shape[1];

  if (b >= B_) return;

  int topi = 0;
  float topval = input->buf[b * N_];

  for (int n = 1; n < N_; ++n) {
    float x = input->buf[b * N_ + n];
    if (x >= topval) {
      topi = n;
      topval = x;
    }
  }

  output->buf[b] = topi;
}

void top_one(Tensor *input, Tensor *output) {
  dim3 blockDim(32);
  dim3 gridDim(
    (input->shape[0] + blockDim.x - 1) / blockDim.x);
  _top_one<<<gridDim, blockDim>>>(input, output);
}

/*
 * log_softmax
 * @brief : Apply the log_softmax function to an input tensor. logsoftmax(x) = log(softmax(x))
 *
 * @param [in1] input  : a vectors of size [B_ * K_]
 * @param [out] output : a vectors of size [B_ * K_]
 */
__global__ void _log_softmax(Tensor *input, Tensor *output) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;

  int B_ = input->shape[0];
  int K_ = input->shape[1];

  if (b >= B_) return;

  float sum = 0.0;
  for (int k = 0; k < K_; ++k) {
    sum += expf(input->buf[b * K_ + k]);
  }

  float log_sum = logf(sum);
  for (int k = 0; k < K_; ++k) {
    output->buf[b * K_ + k] = input->buf[b * K_ + k] - log_sum;
  }
}

void log_softmax(Tensor *input, Tensor *output) {
  dim3 blockDim(32);
  dim3 gridDim(
    (input->shape[0] + blockDim.x - 1) / blockDim.x);
  _log_softmax<<<gridDim, blockDim>>>(input, output);
}

__global__ void _check_decoder_termination(Tensor *outputs, Tensor *embidx, float *output, int *runnings, int word_idx) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;

  if (b >= outputs->shape[0] || !runnings[b]) return;

  int topi = outputs->buf[b];
  output[b * MAX_LENGTH + word_idx] = topi;
  embidx->buf[b] = topi;

  if (topi == EOS_TOKEN) runnings[b] = 0;
}

void check_decoder_termination(Tensor *outputs, Tensor *embidx, float *out_buf, int *runnings, int word_idx) {
  dim3 blockDim(32);
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
