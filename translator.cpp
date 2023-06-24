#include "translator.h"
#include "translator.cu"
#include "util.h"
#include <omp.h>
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
Tensor **eW_emb;
Tensor **eW_ir, **eW_iz, **eW_in;
Tensor **eW_hr, **eW_hz, **eW_hn;
Tensor **eb_ir, **eb_iz, **eb_in;
Tensor **eb_hr, **eb_hz, **eb_hn;
Tensor **dW_emb;
Tensor **dW_ir, **dW_iz, **dW_in;
Tensor **dW_hr, **dW_hz, **dW_hn;
Tensor **db_ir, **db_iz, **db_in;
Tensor **db_hr, **db_hz, **db_hn;
Tensor **dW_attn, **db_attn, **dW_attn_comb, **db_attn_comb, **dW_out, **db_out;

// Encoder Activations
Tensor **encoder_embidx, **encoder_hidden, **encoder_outputs;
Tensor **encoder_embedded;
Tensor **encoder_rtmp1, **encoder_rtmp2, **encoder_rtmp3, **encoder_rtmp4, **encoder_rtmp5, **encoder_rt;
Tensor **encoder_ztmp1, **encoder_ztmp2, **encoder_ztmp3, **encoder_ztmp4, **encoder_ztmp5, **encoder_zt;
Tensor **encoder_ntmp1, **encoder_ntmp2, **encoder_ntmp3, **encoder_ntmp4, **encoder_ntmp5, **encoder_ntmp6, **encoder_nt;
Tensor **encoder_htmp1, **encoder_htmp2, **encoder_htmp3, **encoder_ht;

// Decoder Activations
Tensor **decoder_embidx, **decoder_hidden, **decoder_embedded, **decoder_embhid;
Tensor **decoder_attn, **decoder_attn_weights, **decoder_attn_applied, **decoder_embattn;
Tensor **decoder_attn_comb, **decoder_relu;
Tensor **decoder_rtmp1, **decoder_rtmp2, **decoder_rtmp3, **decoder_rtmp4, **decoder_rtmp5, **decoder_rt;
Tensor **decoder_ztmp1, **decoder_ztmp2, **decoder_ztmp3, **decoder_ztmp4, **decoder_ztmp5, **decoder_zt;
Tensor **decoder_ntmp1, **decoder_ntmp2, **decoder_ntmp3, **decoder_ntmp4, **decoder_ntmp5, **decoder_ntmp6, **decoder_nt;
Tensor **decoder_htmp1, **decoder_htmp2, **decoder_htmp3, **decoder_ht;
Tensor **decoder_out, **decoder_logsoftmax, **decoder_outputs;

// Miscellaneous Variables
int size_per_node;
int num_devices, size_per_device, **running_batches;
float **in_buf, **out_buf, **dev_params;

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
  int node;
  MPI_Comm_rank(MPI_COMM_WORLD, &node);

  MPI_Scatter(
    &input->buf[node * size_per_node], size_per_node, MPI_FLOAT,
    &input->buf[node * size_per_node], size_per_node, MPI_FLOAT,
    0, MPI_COMM_WORLD);

  #pragma omp parallel num_threads(num_devices)
  {
    int dev = omp_get_thread_num();
    cudaSetDevice(dev);

    cudaMemcpy(
      dev_params[dev],
      parameter,
      sizeof(float) * parameter_binary_size,
      cudaMemcpyHostToDevice);

    init_weights<<<1, 1>>>(eW_emb[dev], eW_ir[dev], eW_iz[dev], eW_in[dev], eW_hr[dev], eW_hz[dev], eW_hn[dev], eb_ir[dev], eb_iz[dev], eb_in[dev], eb_hr[dev], eb_hz[dev], eb_hn[dev], dW_emb[dev], dW_ir[dev], dW_iz[dev], dW_in[dev], dW_hr[dev], dW_hz[dev], dW_hn[dev], db_ir[dev], db_iz[dev], db_in[dev], db_hr[dev], db_hz[dev], db_hn[dev], dW_attn[dev], db_attn[dev], dW_attn_comb[dev], db_attn_comb[dev], dW_out[dev], db_out[dev], dev_params[dev]);

    init_buffers(in_buf[dev], out_buf[dev]);

    cudaMemcpy(
      in_buf[dev],
      &input->buf[node * size_per_node + dev * size_per_device],
      sizeof(float) * size_per_device,
      cudaMemcpyHostToDevice);

    // Encoder init
    init_encoder(encoder_hidden[dev], encoder_outputs[dev]);
    init_running_batches(running_batches[dev]);

    // Encoder
    for (int work_idx = 0; work_idx < MAX_LENGTH; ++work_idx) {

      check_encoder_termination(in_buf[dev], running_batches[dev], work_idx);

      fetch_words(in_buf[dev], encoder_embidx[dev], work_idx);
      embedding(encoder_embidx[dev], eW_emb[dev], encoder_embedded[dev]);

      // GRU
      // r_t
      matvec(encoder_embedded[dev], eW_ir[dev], encoder_rtmp1[dev]);
      elemwise_add(encoder_rtmp1[dev], eb_ir[dev], encoder_rtmp2[dev]);
      matvec(encoder_hidden[dev], eW_hr[dev], encoder_rtmp3[dev]);
      elemwise_add(encoder_rtmp3[dev], eb_hr[dev], encoder_rtmp4[dev]);
      elemwise_add(encoder_rtmp2[dev], encoder_rtmp4[dev], encoder_rtmp5[dev]);
      elemwise_sigmoid(encoder_rtmp5[dev], encoder_rt[dev]);

      // z_t
      matvec(encoder_embedded[dev], eW_iz[dev], encoder_ztmp1[dev]);
      elemwise_add(encoder_ztmp1[dev], eb_iz[dev], encoder_ztmp2[dev]);
      matvec(encoder_hidden[dev], eW_hz[dev], encoder_ztmp3[dev]);
      elemwise_add(encoder_ztmp3[dev], eb_hz[dev], encoder_ztmp4[dev]);
      elemwise_add(encoder_ztmp2[dev], encoder_ztmp4[dev], encoder_ztmp5[dev]);
      elemwise_sigmoid(encoder_ztmp5[dev], encoder_zt[dev]);

      // n_t
      matvec(encoder_embedded[dev], eW_in[dev], encoder_ntmp1[dev]);
      elemwise_add(encoder_ntmp1[dev], eb_in[dev], encoder_ntmp2[dev]);
      matvec(encoder_hidden[dev], eW_hn[dev], encoder_ntmp3[dev]);
      elemwise_add(encoder_ntmp3[dev], eb_hn[dev], encoder_ntmp4[dev]);
      elemwise_mult(encoder_rt[dev], encoder_ntmp4[dev], encoder_ntmp5[dev]);
      elemwise_add(encoder_ntmp2[dev], encoder_ntmp5[dev], encoder_ntmp6[dev]);
      elemwise_tanh(encoder_ntmp6[dev], encoder_nt[dev]);

      // h_t
      elemwise_oneminus(encoder_zt[dev], encoder_htmp1[dev]);
      elemwise_mult(encoder_htmp1[dev], encoder_nt[dev], encoder_htmp2[dev]);
      elemwise_mult(encoder_zt[dev], encoder_hidden[dev], encoder_htmp3[dev]);
      elemwise_add(encoder_htmp2[dev], encoder_htmp3[dev], encoder_ht[dev]);
      select(encoder_ht[dev], encoder_hidden[dev], encoder_hidden[dev], running_batches[dev]);

      copy_encoder_outputs(encoder_hidden[dev], encoder_outputs[dev], running_batches[dev], work_idx);
    } // end Encoder loop

    // Decoder init
    decoder_hidden[dev] = encoder_hidden[dev];

    init_decoder(decoder_embidx[dev]);
    init_running_batches(running_batches[dev]);

    // Decoder
    for (int work_idx = 0; work_idx < MAX_LENGTH; ++work_idx) {

      // Embedding
      embedding(decoder_embidx[dev], dW_emb[dev], decoder_embedded[dev]);

      // Attention
      concat(decoder_embedded[dev], decoder_hidden[dev], decoder_embhid[dev]);
      linear(decoder_embhid[dev], dW_attn[dev], db_attn[dev], decoder_attn[dev]);
      softmax(decoder_attn[dev], decoder_attn_weights[dev]);
      bmm(decoder_attn_weights[dev], encoder_outputs[dev], decoder_attn_applied[dev]);
      concat(decoder_embedded[dev], decoder_attn_applied[dev], decoder_embattn[dev]);
      linear(decoder_embattn[dev], dW_attn_comb[dev], db_attn_comb[dev], decoder_attn_comb[dev]);
      relu(decoder_attn_comb[dev], decoder_relu[dev]);

      // GRU
      // r_t
      matvec(decoder_relu[dev], dW_ir[dev], decoder_rtmp1[dev]);
      elemwise_add(decoder_rtmp1[dev], db_ir[dev], decoder_rtmp2[dev]);
      matvec(decoder_hidden[dev], dW_hr[dev], decoder_rtmp3[dev]);
      elemwise_add(decoder_rtmp3[dev], db_hr[dev], decoder_rtmp4[dev]);
      elemwise_add(decoder_rtmp2[dev], decoder_rtmp4[dev], decoder_rtmp5[dev]);
      elemwise_sigmoid(decoder_rtmp5[dev], decoder_rt[dev]);

      // z_t
      matvec(decoder_relu[dev], dW_iz[dev], decoder_ztmp1[dev]);
      elemwise_add(decoder_ztmp1[dev], db_iz[dev], decoder_ztmp2[dev]);
      matvec(decoder_hidden[dev], dW_hz[dev], decoder_ztmp3[dev]);
      elemwise_add(decoder_ztmp3[dev], db_hz[dev], decoder_ztmp4[dev]);
      elemwise_add(decoder_ztmp2[dev], decoder_ztmp4[dev], decoder_ztmp5[dev]);
      elemwise_sigmoid(decoder_ztmp5[dev], decoder_zt[dev]);

      // n_t
      matvec(decoder_relu[dev], dW_in[dev], decoder_ntmp1[dev]);
      elemwise_add(decoder_ntmp1[dev], db_in[dev], decoder_ntmp2[dev]);
      matvec(decoder_hidden[dev], dW_hn[dev], decoder_ntmp3[dev]);
      elemwise_add(decoder_ntmp3[dev], db_hn[dev], decoder_ntmp4[dev]);
      elemwise_mult(decoder_rt[dev], decoder_ntmp4[dev], decoder_ntmp5[dev]);
      elemwise_add(decoder_ntmp2[dev], decoder_ntmp5[dev], decoder_ntmp6[dev]);
      elemwise_tanh(decoder_ntmp6[dev], decoder_nt[dev]);

      // h_t
      elemwise_oneminus(decoder_zt[dev], decoder_htmp1[dev]);
      elemwise_mult(decoder_htmp1[dev], decoder_nt[dev], decoder_htmp2[dev]);
      elemwise_mult(decoder_zt[dev], decoder_hidden[dev], decoder_htmp3[dev]);
      elemwise_add(decoder_htmp2[dev], decoder_htmp3[dev], decoder_hidden[dev]);

      // Select output token
      linear(decoder_hidden[dev], dW_out[dev], db_out[dev], decoder_out[dev]);
      log_softmax(decoder_out[dev], decoder_logsoftmax[dev]);
      top_one(decoder_logsoftmax[dev], decoder_outputs[dev]);

      check_decoder_termination(decoder_outputs[dev], decoder_embidx[dev], out_buf[dev], running_batches[dev], work_idx);
    } // end Decoder loop

    cudaMemcpy(
      &output->buf[node * size_per_node + dev * size_per_device],
      out_buf[dev],
      sizeof(float) * size_per_device,
      cudaMemcpyDeviceToHost);
  }

  MPI_Gather(
    &output->buf[node * size_per_node], size_per_node, MPI_FLOAT,
    &output->buf[node * size_per_node], size_per_node, MPI_FLOAT,
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
  int node, num_nodes;
  MPI_Comm_rank(MPI_COMM_WORLD, &node);
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

  cudaGetDeviceCount(&num_devices);

  size_per_node = N * MAX_LENGTH / num_nodes;
  size_per_device = size_per_node / num_devices;
  BATCH_SIZE = N / num_nodes / num_devices;

  if (node == 0) {
    parameter = (float *) read_binary(parameter_fname, &parameter_binary_size);
  }

  MPI_Bcast(&parameter_binary_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  if (node != 0) {
    parameter = new float[parameter_binary_size];
  }

  MPI_Bcast(parameter, parameter_binary_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

  eW_emb = new Tensor *[num_devices];
  eW_ir = new Tensor *[num_devices];
  eW_iz = new Tensor *[num_devices];
  eW_in = new Tensor *[num_devices];
  eW_hr = new Tensor *[num_devices];
  eW_hz = new Tensor *[num_devices];
  eW_hn = new Tensor *[num_devices];
  eb_ir = new Tensor *[num_devices];
  eb_iz = new Tensor *[num_devices];
  eb_in = new Tensor *[num_devices];
  eb_hr = new Tensor *[num_devices];
  eb_hz = new Tensor *[num_devices];
  eb_hn = new Tensor *[num_devices];
  dW_emb = new Tensor *[num_devices];
  dW_ir = new Tensor *[num_devices];
  dW_iz = new Tensor *[num_devices];
  dW_in = new Tensor *[num_devices];
  dW_hr = new Tensor *[num_devices];
  dW_hz = new Tensor *[num_devices];
  dW_hn = new Tensor *[num_devices];
  db_ir = new Tensor *[num_devices];
  db_iz = new Tensor *[num_devices];
  db_in = new Tensor *[num_devices];
  db_hr = new Tensor *[num_devices];
  db_hz = new Tensor *[num_devices];
  db_hn = new Tensor *[num_devices];
  dW_attn = new Tensor *[num_devices];
  db_attn = new Tensor *[num_devices];
  dW_attn_comb = new Tensor *[num_devices];
  db_attn_comb = new Tensor *[num_devices];
  dW_out = new Tensor *[num_devices];
  db_out = new Tensor *[num_devices];

  encoder_embidx = new Tensor *[num_devices];
  encoder_hidden = new Tensor *[num_devices];
  encoder_outputs = new Tensor *[num_devices];
  encoder_embedded = new Tensor *[num_devices];
  encoder_rtmp1 = new Tensor *[num_devices];
  encoder_rtmp2 = new Tensor *[num_devices];
  encoder_rtmp3 = new Tensor *[num_devices];
  encoder_rtmp4 = new Tensor *[num_devices];
  encoder_rtmp5 = new Tensor *[num_devices];
  encoder_rt = new Tensor *[num_devices];
  encoder_ztmp1 = new Tensor *[num_devices];
  encoder_ztmp2 = new Tensor *[num_devices];
  encoder_ztmp3 = new Tensor *[num_devices];
  encoder_ztmp4 = new Tensor *[num_devices];
  encoder_ztmp5 = new Tensor *[num_devices];
  encoder_zt = new Tensor *[num_devices];
  encoder_ntmp1 = new Tensor *[num_devices];
  encoder_ntmp2 = new Tensor *[num_devices];
  encoder_ntmp3 = new Tensor *[num_devices];
  encoder_ntmp4 = new Tensor *[num_devices];
  encoder_ntmp5 = new Tensor *[num_devices];
  encoder_ntmp6 = new Tensor *[num_devices];
  encoder_nt = new Tensor *[num_devices];
  encoder_htmp1 = new Tensor *[num_devices];
  encoder_htmp2 = new Tensor *[num_devices];
  encoder_htmp3 = new Tensor *[num_devices];
  encoder_ht = new Tensor *[num_devices];

  decoder_embidx = new Tensor *[num_devices];
  decoder_hidden = new Tensor *[num_devices];
  decoder_embedded = new Tensor *[num_devices];
  decoder_embhid = new Tensor *[num_devices];
  decoder_attn = new Tensor *[num_devices];
  decoder_attn_weights = new Tensor *[num_devices];
  decoder_attn_applied = new Tensor *[num_devices];
  decoder_embattn = new Tensor *[num_devices];
  decoder_attn_comb = new Tensor *[num_devices];
  decoder_relu = new Tensor *[num_devices];
  decoder_rtmp1 = new Tensor *[num_devices];
  decoder_rtmp2 = new Tensor *[num_devices];
  decoder_rtmp3 = new Tensor *[num_devices];
  decoder_rtmp4 = new Tensor *[num_devices];
  decoder_rtmp5 = new Tensor *[num_devices];
  decoder_rt = new Tensor *[num_devices];
  decoder_ztmp1 = new Tensor *[num_devices];
  decoder_ztmp2 = new Tensor *[num_devices];
  decoder_ztmp3 = new Tensor *[num_devices];
  decoder_ztmp4 = new Tensor *[num_devices];
  decoder_ztmp5 = new Tensor *[num_devices];
  decoder_zt = new Tensor *[num_devices];
  decoder_ntmp1 = new Tensor *[num_devices];
  decoder_ntmp2 = new Tensor *[num_devices];
  decoder_ntmp3 = new Tensor *[num_devices];
  decoder_ntmp4 = new Tensor *[num_devices];
  decoder_ntmp5 = new Tensor *[num_devices];
  decoder_ntmp6 = new Tensor *[num_devices];
  decoder_nt = new Tensor *[num_devices];
  decoder_htmp1 = new Tensor *[num_devices];
  decoder_htmp2 = new Tensor *[num_devices];
  decoder_htmp3 = new Tensor *[num_devices];
  decoder_ht = new Tensor *[num_devices];
  decoder_out = new Tensor *[num_devices];
  decoder_logsoftmax = new Tensor *[num_devices];
  decoder_outputs = new Tensor *[num_devices];

  running_batches = new int *[num_devices];
  in_buf = new float *[num_devices];
  out_buf = new float *[num_devices];
  dev_params = new float *[num_devices];

  for (int dev = 0; dev < num_devices; ++dev) {
    cudaSetDevice(dev);

    CUDA_MALLOC(eW_emb[dev], INPUT_VOCAB_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(eW_ir[dev], HIDDEN_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(eW_iz[dev], HIDDEN_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(eW_in[dev], HIDDEN_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(eW_hr[dev], HIDDEN_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(eW_hz[dev], HIDDEN_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(eW_hn[dev], HIDDEN_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(eb_ir[dev], HIDDEN_SIZE);
    CUDA_MALLOC(eb_iz[dev], HIDDEN_SIZE);
    CUDA_MALLOC(eb_in[dev], HIDDEN_SIZE);
    CUDA_MALLOC(eb_hr[dev], HIDDEN_SIZE);
    CUDA_MALLOC(eb_hz[dev], HIDDEN_SIZE);
    CUDA_MALLOC(eb_hn[dev], HIDDEN_SIZE);
    CUDA_MALLOC(dW_emb[dev], OUTPUT_VOCAB_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(dW_ir[dev], HIDDEN_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(dW_iz[dev], HIDDEN_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(dW_in[dev], HIDDEN_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(dW_hr[dev], HIDDEN_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(dW_hz[dev], HIDDEN_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(dW_hn[dev], HIDDEN_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(db_ir[dev], HIDDEN_SIZE);
    CUDA_MALLOC(db_iz[dev], HIDDEN_SIZE);
    CUDA_MALLOC(db_in[dev], HIDDEN_SIZE);
    CUDA_MALLOC(db_hr[dev], HIDDEN_SIZE);
    CUDA_MALLOC(db_hz[dev], HIDDEN_SIZE);
    CUDA_MALLOC(db_hn[dev], HIDDEN_SIZE);
    CUDA_MALLOC(dW_attn[dev], MAX_LENGTH, 2 * HIDDEN_SIZE);
    CUDA_MALLOC(db_attn[dev], MAX_LENGTH);
    CUDA_MALLOC(dW_attn_comb[dev], HIDDEN_SIZE, 2 * HIDDEN_SIZE);
    CUDA_MALLOC(db_attn_comb[dev], HIDDEN_SIZE);
    CUDA_MALLOC(dW_out[dev], OUTPUT_VOCAB_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(db_out[dev], OUTPUT_VOCAB_SIZE);

    CUDA_MALLOC(encoder_embidx[dev], BATCH_SIZE);
    CUDA_MALLOC(encoder_hidden[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_outputs[dev], BATCH_SIZE, MAX_LENGTH, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_embedded[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_rtmp1[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_rtmp2[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_rtmp3[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_rtmp4[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_rtmp5[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_rt[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_ztmp1[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_ztmp2[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_ztmp3[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_ztmp4[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_ztmp5[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_zt[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_ntmp1[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_ntmp2[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_ntmp3[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_ntmp4[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_ntmp5[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_ntmp6[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_nt[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_htmp1[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_htmp2[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_htmp3[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(encoder_ht[dev], BATCH_SIZE, HIDDEN_SIZE);

    CUDA_MALLOC(decoder_embidx[dev], BATCH_SIZE);
    CUDA_MALLOC(decoder_hidden[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_embedded[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_embhid[dev], BATCH_SIZE, 2 * HIDDEN_SIZE);
    CUDA_MALLOC(decoder_attn[dev], BATCH_SIZE, MAX_LENGTH);
    CUDA_MALLOC(decoder_attn_weights[dev], BATCH_SIZE, MAX_LENGTH);
    CUDA_MALLOC(decoder_attn_applied[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_embattn[dev], BATCH_SIZE, 2 * HIDDEN_SIZE);
    CUDA_MALLOC(decoder_attn_comb[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_relu[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_rtmp1[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_rtmp2[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_rtmp3[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_rtmp4[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_rtmp5[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_rt[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_ztmp1[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_ztmp2[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_ztmp3[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_ztmp4[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_ztmp5[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_zt[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_ntmp1[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_ntmp2[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_ntmp3[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_ntmp4[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_ntmp5[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_ntmp6[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_nt[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_htmp1[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_htmp2[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_htmp3[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_ht[dev], BATCH_SIZE, HIDDEN_SIZE);
    CUDA_MALLOC(decoder_out[dev], BATCH_SIZE, OUTPUT_VOCAB_SIZE);
    CUDA_MALLOC(decoder_logsoftmax[dev], BATCH_SIZE, OUTPUT_VOCAB_SIZE);
    CUDA_MALLOC(decoder_outputs[dev], BATCH_SIZE);

    cudaMalloc((void **)&running_batches[dev], sizeof(int) * BATCH_SIZE);

    cudaMalloc((void **)&in_buf[dev], sizeof(float) * size_per_device);
    cudaMalloc((void **)&out_buf[dev], sizeof(float) * size_per_device);
    cudaMalloc((void **)&dev_params[dev], sizeof(float) * parameter_binary_size);
  }
}

/*
 * finalize_translator
 * @brief : free all dynamically allocated variables
 */
void finalize_translator(){
  for (int dev = 0; dev < num_devices; ++dev) {
    cudaSetDevice(dev);

    // free parameters
    CUDA_DELETE(eW_emb[dev]);
    CUDA_DELETE(eW_ir[dev]);
    CUDA_DELETE(eW_iz[dev]);
    CUDA_DELETE(eW_in[dev]);
    CUDA_DELETE(eW_hr[dev]);
    CUDA_DELETE(eW_hz[dev]);
    CUDA_DELETE(eW_hn[dev]);
    CUDA_DELETE(eb_ir[dev]);
    CUDA_DELETE(eb_iz[dev]);
    CUDA_DELETE(eb_in[dev]);
    CUDA_DELETE(eb_hr[dev]);
    CUDA_DELETE(eb_hz[dev]);
    CUDA_DELETE(eb_hn[dev]);
    CUDA_DELETE(dW_emb[dev]);
    CUDA_DELETE(dW_ir[dev]);
    CUDA_DELETE(dW_iz[dev]);
    CUDA_DELETE(dW_in[dev]);
    CUDA_DELETE(dW_hr[dev]);
    CUDA_DELETE(dW_hz[dev]);
    CUDA_DELETE(dW_hn[dev]);
    CUDA_DELETE(db_ir[dev]);
    CUDA_DELETE(db_iz[dev]);
    CUDA_DELETE(db_in[dev]);
    CUDA_DELETE(db_hr[dev]);
    CUDA_DELETE(db_hz[dev]);
    CUDA_DELETE(db_hn[dev]);
    CUDA_DELETE(dW_attn[dev]);
    CUDA_DELETE(db_attn[dev]);
    CUDA_DELETE(dW_attn_comb[dev]);
    CUDA_DELETE(db_attn_comb[dev]);
    CUDA_DELETE(dW_out[dev]);
    CUDA_DELETE(db_out[dev]);

    // free encoder activations
    CUDA_DELETE(encoder_embidx[dev]);
    CUDA_DELETE(encoder_hidden[dev]);
    CUDA_DELETE(encoder_outputs[dev]);
    CUDA_DELETE(encoder_embedded[dev]);
    CUDA_DELETE(encoder_rtmp1[dev]);
    CUDA_DELETE(encoder_rtmp2[dev]);
    CUDA_DELETE(encoder_rtmp3[dev]);
    CUDA_DELETE(encoder_rtmp4[dev]);
    CUDA_DELETE(encoder_rtmp5[dev]);
    CUDA_DELETE(encoder_rt[dev]);
    CUDA_DELETE(encoder_ztmp1[dev]);
    CUDA_DELETE(encoder_ztmp2[dev]);
    CUDA_DELETE(encoder_ztmp3[dev]);
    CUDA_DELETE(encoder_ztmp4[dev]);
    CUDA_DELETE(encoder_ztmp5[dev]);
    CUDA_DELETE(encoder_zt[dev]);
    CUDA_DELETE(encoder_ntmp1[dev]);
    CUDA_DELETE(encoder_ntmp2[dev]);
    CUDA_DELETE(encoder_ntmp3[dev]);
    CUDA_DELETE(encoder_ntmp4[dev]);
    CUDA_DELETE(encoder_ntmp5[dev]);
    CUDA_DELETE(encoder_ntmp6[dev]);
    CUDA_DELETE(encoder_nt[dev]);
    CUDA_DELETE(encoder_htmp1[dev]);
    CUDA_DELETE(encoder_htmp2[dev]);
    CUDA_DELETE(encoder_htmp3[dev]);
    CUDA_DELETE(encoder_ht[dev]);

    // free decoder activations
    CUDA_DELETE(decoder_embidx[dev]);
    CUDA_DELETE(decoder_embedded[dev]);
    CUDA_DELETE(decoder_embhid[dev]);
    CUDA_DELETE(decoder_attn[dev]);
    CUDA_DELETE(decoder_attn_weights[dev]);
    CUDA_DELETE(decoder_attn_applied[dev]);
    CUDA_DELETE(decoder_embattn[dev]);
    CUDA_DELETE(decoder_attn_comb[dev]);
    CUDA_DELETE(decoder_relu[dev]);
    CUDA_DELETE(decoder_rtmp1[dev]);
    CUDA_DELETE(decoder_rtmp2[dev]);
    CUDA_DELETE(decoder_rtmp3[dev]);
    CUDA_DELETE(decoder_rtmp4[dev]);
    CUDA_DELETE(decoder_rtmp5[dev]);
    CUDA_DELETE(decoder_rt[dev]);
    CUDA_DELETE(decoder_ztmp1[dev]);
    CUDA_DELETE(decoder_ztmp2[dev]);
    CUDA_DELETE(decoder_ztmp3[dev]);
    CUDA_DELETE(decoder_ztmp4[dev]);
    CUDA_DELETE(decoder_ztmp5[dev]);
    CUDA_DELETE(decoder_zt[dev]);
    CUDA_DELETE(decoder_ntmp1[dev]);
    CUDA_DELETE(decoder_ntmp2[dev]);
    CUDA_DELETE(decoder_ntmp3[dev]);
    CUDA_DELETE(decoder_ntmp4[dev]);
    CUDA_DELETE(decoder_ntmp5[dev]);
    CUDA_DELETE(decoder_ntmp6[dev]);
    CUDA_DELETE(decoder_nt[dev]);
    CUDA_DELETE(decoder_htmp1[dev]);
    CUDA_DELETE(decoder_htmp2[dev]);
    CUDA_DELETE(decoder_htmp3[dev]);
    CUDA_DELETE(decoder_ht[dev]);
    CUDA_DELETE(decoder_out[dev]);
    CUDA_DELETE(decoder_logsoftmax[dev]);
    CUDA_DELETE(decoder_outputs[dev]);

    // free misc. variables
    cudaFree(running_batches[dev]);
    cudaFree(in_buf[dev]);
    cudaFree(out_buf[dev]);
    cudaFree(dev_params[dev]);
  }

  // free parameters
  delete[] eW_emb;
  delete[] eW_ir;
  delete[] eW_iz;
  delete[] eW_in;
  delete[] eW_hr;
  delete[] eW_hz;
  delete[] eW_hn;
  delete[] eb_ir;
  delete[] eb_iz;
  delete[] eb_in;
  delete[] eb_hr;
  delete[] eb_hz;
  delete[] eb_hn;
  delete[] dW_emb;
  delete[] dW_ir;
  delete[] dW_iz;
  delete[] dW_in;
  delete[] dW_hr;
  delete[] dW_hz;
  delete[] dW_hn;
  delete[] db_ir;
  delete[] db_iz;
  delete[] db_in;
  delete[] db_hr;
  delete[] db_hz;
  delete[] db_hn;
  delete[] dW_attn;
  delete[] db_attn;
  delete[] dW_attn_comb;
  delete[] db_attn_comb;
  delete[] dW_out;
  delete[] db_out;

  // free encoder activations
  delete[] encoder_embidx;
  delete[] encoder_hidden;
  delete[] encoder_outputs;
  delete[] encoder_embedded;
  delete[] encoder_rtmp1;
  delete[] encoder_rtmp2;
  delete[] encoder_rtmp3;
  delete[] encoder_rtmp4;
  delete[] encoder_rtmp5;
  delete[] encoder_rt;
  delete[] encoder_ztmp1;
  delete[] encoder_ztmp2;
  delete[] encoder_ztmp3;
  delete[] encoder_ztmp4;
  delete[] encoder_ztmp5;
  delete[] encoder_zt;
  delete[] encoder_ntmp1;
  delete[] encoder_ntmp2;
  delete[] encoder_ntmp3;
  delete[] encoder_ntmp4;
  delete[] encoder_ntmp5;
  delete[] encoder_ntmp6;
  delete[] encoder_nt;
  delete[] encoder_htmp1;
  delete[] encoder_htmp2;
  delete[] encoder_htmp3;
  delete[] encoder_ht;

  // free decoder activations
  delete[] decoder_embidx;
  delete[] decoder_embedded;
  delete[] decoder_embhid;
  delete[] decoder_attn;
  delete[] decoder_attn_weights;
  delete[] decoder_attn_applied;
  delete[] decoder_embattn;
  delete[] decoder_attn_comb;
  delete[] decoder_relu;
  delete[] decoder_rtmp1;
  delete[] decoder_rtmp2;
  delete[] decoder_rtmp3;
  delete[] decoder_rtmp4;
  delete[] decoder_rtmp5;
  delete[] decoder_rt;
  delete[] decoder_ztmp1;
  delete[] decoder_ztmp2;
  delete[] decoder_ztmp3;
  delete[] decoder_ztmp4;
  delete[] decoder_ztmp5;
  delete[] decoder_zt;
  delete[] decoder_ntmp1;
  delete[] decoder_ntmp2;
  delete[] decoder_ntmp3;
  delete[] decoder_ntmp4;
  delete[] decoder_ntmp5;
  delete[] decoder_ntmp6;
  delete[] decoder_nt;
  delete[] decoder_htmp1;
  delete[] decoder_htmp2;
  delete[] decoder_htmp3;
  delete[] decoder_ht;
  delete[] decoder_out;
  delete[] decoder_logsoftmax;
  delete[] decoder_outputs;

  // free misc. variables
  delete[] running_batches;
  delete[] in_buf;
  delete[] out_buf;
  delete[] dev_params;

  delete[] parameter;
}
