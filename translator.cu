#include "translator.h"
#include <math.h>

#define SOS_TOKEN        0
#define EOS_TOKEN        1

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

__global__ void _init_buffers(float *_in_buf, float *_out_buf, int batch_size) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;

  if (b >= batch_size || n >= MAX_LENGTH) return;

  int idx = b * MAX_LENGTH + n;
  _in_buf[idx] = 0.0;
  _out_buf[idx] = 0.0;
}

__global__ void _init_running_batches(int *runnings, int batch_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_size) runnings[i] = 1;
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

__global__ void _init_decoder(Tensor *_embidx) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;

  if (b >= _embidx->shape[0]) return;

  _embidx->buf[b] = SOS_TOKEN;
}

__global__ void _check_encoder_termination(float *input, int *runnings, int word_idx, int batch_size) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;

  if (b >= batch_size || !runnings[b]) return;

  runnings[b] = input[b * MAX_LENGTH + word_idx] != 0.0;
}

__global__ void _fetch_words(float *input, Tensor *output, int word_idx, int batch_size) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < batch_size) output->buf[b] = input[b * MAX_LENGTH + word_idx];
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
    c += weight->buf[m * K_ + k] * input->buf[b * K_ + k];
  }
  output->buf[b * M_ + m] = c;
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

  output->buf[n] = 1.0 / (1.0 + expf(-input->buf[n]));
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

  output->buf[n] = tanhf(input->buf[n]);
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

  output->buf[n] = input1->buf[n] * input2->buf[n];
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

  output->buf[n] = 1.0 - input->buf[n];
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
  output->buf[n] = x < 0.0 ? 0.0 : x;
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

__global__ void _check_decoder_termination(Tensor *outputs, Tensor *embidx, float *output, int *runnings, int word_idx) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;

  if (b >= outputs->shape[0] || !runnings[b]) return;

  int topi = outputs->buf[b];
  output[b * MAX_LENGTH + word_idx] = topi;
  embidx->buf[b] = topi;

  if (topi == EOS_TOKEN) runnings[b] = 0;
}
