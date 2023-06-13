#include "translator.h"
#include "util.h"
#include <mpi.h>
#include <math.h>

#define BATCH_SIZE  1

static int SOS_token = 0;
static int EOS_token = 1;
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

Tensor::~Tensor() {
  if (buf != nullptr) free(buf);
}

int Tensor::num_elem() {
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

// Operations
void embedding(Tensor *input, Tensor *weight, Tensor *output);
void matvec(Tensor *input, Tensor *weight, Tensor *output);
void elemwise_add(Tensor *input1, Tensor *input2, Tensor *output);
void elemwise_sigmoid(Tensor *input, Tensor *output);
void elemwise_tanh(Tensor *input, Tensor *output);
void elemwise_mult(Tensor *input1, Tensor *input2, Tensor *output);
void elemwise_oneminus(Tensor *input, Tensor *output);
void copy_encoder_outputs(Tensor *input, Tensor *output, int i);
void concat(Tensor *input1, Tensor *input2, Tensor *output);
void linear(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output);
void softmax(Tensor *input, Tensor *output);
void bmm(Tensor *input, Tensor *weight, Tensor *output);
void relu(Tensor *input, Tensor *output);
void top_one(Tensor *input, Tensor *output);
void log_softmax(Tensor *input, Tensor *output);

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

  // N sentences
  for (
    int n = mpi_rank * N / mpi_world_size;
    n < (mpi_rank + 1) * N / mpi_world_size;
    n += BATCH_SIZE) {

    // Encoder init
    encoder_hidden->fill_zeros();
    encoder_outputs->fill_zeros();

    int terminate_encoder = 0;

    // Encoder
    for (int i = 0; i < MAX_LENGTH && !terminate_encoder; ++i) {

      // Embedding
      for (int b = 0; b < BATCH_SIZE; b++) {
        encoder_embidx->buf[b] = input->buf[(n + b) * MAX_LENGTH + i];
      }
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
      elemwise_add(encoder_htmp2, encoder_htmp3, encoder_hidden);

      copy_encoder_outputs(encoder_hidden, encoder_outputs, i);

      terminate_encoder = 1;
      for (int b = 0; b < BATCH_SIZE; ++b) {
        terminate_encoder = terminate_encoder && (input->buf[(n + b) * MAX_LENGTH + i + 1] == 0.0);
      }
    } // end Encoder loop

    // Decoder init
    decoder_hidden = encoder_hidden;

    for (int b = 0; b < BATCH_SIZE; ++b) {
      decoder_embidx->buf[b] = SOS_token;
    }

    int terminate_decoder = 0;
    int terminated_decoders[BATCH_SIZE] = { 0 };

    // Decoder
    for (int i = 0; i < MAX_LENGTH && !terminate_decoder; ++i) {

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

      for (int b = 0; b < BATCH_SIZE; b++) {
        if (terminated_decoders[b]) continue;

        int topi = decoder_outputs->buf[b];
        output->buf[(n + b) * MAX_LENGTH + i] = topi;
        decoder_embidx->buf[b] = topi;
        if (topi == EOS_token) terminated_decoders[b] = 1;
      }

      terminate_decoder = 1;
      for (int b = 0; b < BATCH_SIZE; b++) {
        terminate_decoder = terminate_decoder && terminated_decoders[b];
      }
    } // end Decoder loop
  } // end N input sentences loop

  int output_size_per_node = output->num_elem() / mpi_world_size;
  MPI_Gather(
    &output->buf[mpi_rank * output_size_per_node], output_size_per_node, MPI_FLOAT,
    &output->buf[mpi_rank * output_size_per_node], output_size_per_node, MPI_FLOAT,
    0, MPI_COMM_WORLD);
}

/*
 * embedding
 * @brief : A simple lookup table that stores embeddings of a fixed dictionary and size.
 *
 * @param [in1] input  : embedding index   [B_]
 * @param [in2] weight : a matrix of size  [M_ x H_]
 * @param [out] output : a vectors of size [B_ x H_]
 */
void embedding(Tensor *input, Tensor *weight, Tensor *output){
  int B_ = input->shape[0];
  int H_ = weight->shape[1];

  for (int b = 0; b < B_; ++b) {
    int ei = input->buf[b];
    for (int h = 0; h < H_; ++h) {
      output->buf[b * H_ + h] = weight->buf[ei * H_ + h];
    }
  }
}

/*
 * matvec
 * @brief : Perform a matrix-vector product of the matrix and the vector
 *
 * @param [in1] input  : a vectors of size [B_ x K_]
 * @param [in2] weight : a matrix of size  [M_ x K_]
 * @param [out] output : a vectors of size [B_ x M_]
 */
void matvec(Tensor *input, Tensor *weight, Tensor *output) {
  int B_ = input->shape[0];
  int M_ = weight->shape[0];
  int K_ = weight->shape[1];

  for (int b = 0; b < B_; ++b) {
    for (int m = 0; m < M_; ++m) {
      float c = 0.0;
      for (int k = 0; k < K_; ++k) {
        float w = weight->buf[m * K_ + k];
        float i = input->buf[b * K_ + k];
        c += w * i;
      }
      output->buf[b * M_ + m] = c;
    }
  }
}

/*
 * elemwise_add
 * @brief : Element-by-element addition of tensors
 *
 * @param [in1] input1 : a vectors of size   [B_ * K _]
 * @param [in2] input2 : a vector(s) of size [K_] or [B_ * K _]
 * @param [out] output : a vectors of size   [B_ * K _]
 */
void elemwise_add(Tensor *input1, Tensor *input2, Tensor *output){
  int B_ = input1->shape[0];
  int K_ = input1->shape[1];

  for (int b = 0; b < B_; ++b) {
    for (int k = 0; k < K_; ++k) {
      int index = b * K_ + k;
      output->buf[index] = input1->buf[index] + input2->buf[(input2->ndim == 1) ? k : index];
    }
  }
}

/*
 * elemwise_sigmoid
 * @brief : Apply the element-wise sigmoid function. sigmoid(x) = 1 / (1 + exp(-x))
 *
 * @param [in1] input
 * @param [out] output
 */
void elemwise_sigmoid(Tensor *input, Tensor *output) {
  int N_ = input->num_elem();

  for (int n = 0; n < N_; ++n) {
    float x = input->buf[n];
    output->buf[n] = 1.0 / (1.0 + expf(-x));
  }
}

/*
 * elemwise_tanh
 * @brief : Apply the Hyperbolic Tangent (Tanh) function element-wise.
 *
 * @param [in1] input
 * @param [out] output
 */
void elemwise_tanh(Tensor *input, Tensor *output) {
  int N_ = input->num_elem();

  for (int n = 0; n < N_; ++n) {
    float x = input->buf[n];
    output->buf[n] = tanhf(x);
  }
}

/*
 * elemwise_mult
 * @brief : Element-by-element multiplication of tensors.
 *
 * @param [in1] input1
 * @param [in2] input2
 * @param [out] output
 */
void elemwise_mult(Tensor *input1, Tensor *input2, Tensor *output) {
  int N_ = input1->num_elem();

  for (int n = 0; n < N_; ++n) {
    float x1 = input1->buf[n];
    float x2 = input2->buf[n];
    output->buf[n] = x1 * x2;
  }
}

/*
 * elemwise_oneminus
 * @brief : Apply the element-wise oneminus function. oneminus(x) = 1.0 - x
 *
 * @param [in1] input
 * @param [out] output
 */
void elemwise_oneminus(Tensor *input, Tensor *output) {
  int N_ = input->num_elem();

  for (int n = 0; n < N_; ++n) {
    float x = input->buf[n];
    output->buf[n] = 1.0 - x;
  }
}

/*
 * copy_encoder_outputs
 * @brief : Copy input vector into i-th row of the output matrix
 *
 * @param [in1] input  : a vectors of size  [B_ x N_]
 * @param [in2] i      : row index
 * @param [out] output : a matrices of size [B_ x MAX_LENGTH x N_]
 */
void copy_encoder_outputs(Tensor *input, Tensor *output, int i) {
  int B_ = input->shape[0];
  int N_ = input->shape[1];

  for (int b = 0; b < B_; ++b) {
    for (int n = 0; n < N_; ++n) {
      output->buf[(b * MAX_LENGTH + i) * HIDDEN_SIZE + n] = input->buf[b * N_ + n];
    }
  }
}

/*
 * concat
 * @brief : Concatenate the two input tensors
 *
 * @param [in1] input1 : a vectors of size [B x K_]
 * @param [in2] input2 : a vectors of size [B x K_]
 * @param [out] output : a vectors of size [B x 2 * K_]
 */
void concat(Tensor *input1, Tensor *input2, Tensor *output) {
  int B_ = input1->shape[0];
  int K_ = input1->shape[1];

  for (int b = 0; b < B_; ++b) {
    for (int k = 0; k < K_; ++k) {
      output->buf[b * 2 * K_ + k] = input1->buf[b * K_ + k];
    }
    for (int k = K_; k < 2 * K_; ++k) {
      output->buf[b * 2 * K_ + k] = input2->buf[b * K_ + k - K_];
    }
  }
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
void linear(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output) {
  int B_ = input->shape[0];
  int K_ = weight->shape[1];
  int N_ = weight->shape[0];

  for (int b = 0; b < B_; ++b) {
    for (int n = 0; n < N_; ++n) {
      float c = bias->buf[n];
      for (int k = 0; k < K_; ++k) {
        c += input->buf[b * K_ + k] * weight->buf[n * K_ + k];
      }
      output->buf[b * N_ + n] = c;
    }
  }
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
void softmax(Tensor *input, Tensor *output) {
  int B_ = input->shape[0];
  int N_ = input->shape[1];

  for (int b = 0; b < B_; ++b) {
    float sum = 0.0;
    for (int n = 0; n < N_; ++n) {
      sum += expf(input->buf[b * N_ + n]);
    }
    for (int n = 0; n < N_; ++n) {
      output->buf[b * N_ + n] = expf(input->buf[b * N_ + n]) / sum;
    }
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
void bmm(Tensor *input, Tensor *weight, Tensor *output) {
  int B_ = input->shape[0];
  int K_ = weight->shape[1];
  int N_ = weight->shape[2];

  for (int b = 0; b < B_; ++b) {
    for (int n = 0; n < N_; ++n) {
      float c = 0.0;
      for (int k = 0; k < K_; ++k) {
        c += input->buf[b * K_ + k] * weight->buf[(b * K_ + k) * N_ + n];
      }
      output->buf[b * N_ + n] = c;
    }
  }
}

/*
 * relu
 * @brief : Apply the rectified linear unit function element-wise. relu(x) = max(0,x)
 *
 * @param [in1] input  : a vectors of size [B_ * K_]
 * @param [out] output : a vectors of size [B_ * K_]
 */
void relu(Tensor *input, Tensor *output) {
  int N_ = input->num_elem();

  for (int n = 0; n < N_; ++n) {
    float x = input->buf[n];
    if (x < 0.0) output->buf[n] = 0.0;
    else output->buf[n] = x;
  }
}

/*
 * top_one
 * @brief : Return the largest element of the given input tensor.
 *
 * @param  [in1] input  : a vectors of size [B_ x N_]
 * @param  [out] output : a indices of size [B_]
 */
void top_one(Tensor *input, Tensor *output) {
  int B_ = input->shape[0];
  int N_ = input->shape[1];

  for (int b = 0; b < B_; ++b) {
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
}

/*
 * log_softmax
 * @brief : Apply the log_softmax function to an input tensor. logsoftmax(x) = log(softmax(x))
 *
 * @param [in1] input  : a vectors of size [B_ * K_]
 * @param [out] output : a vectors of size [B_ * K_]
 */
void log_softmax(Tensor *input, Tensor *output) {
  int B_ = input->shape[0];
  int K_ = input->shape[1];

  for (int b = 0; b < B_; ++b) {
    float sum = 0.0;
    for (int k = 0; k < K_; ++k) {
      sum += expf(input->buf[b * K_ + k]);
    }
    
    float log_sum = logf(sum);
    for (int k = 0; k < K_; ++k) {
      output->buf[b * K_ + k] = input->buf[b * K_ + k] - log_sum;
    }
  }
}

/*
 * initialize_translator
 * @brief : initialize translator. load the parameter binary file and store parameters into Tensors
 *
 * @param [in1] parameter_fname  : the name of the binary file where parameters are stored
 */
void initialize_translator(const char *parameter_fname, int N){
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  size_t parameter_binary_size = 0;
  float *parameter = NULL;

  if (mpi_rank == 0) {
    parameter = (float *) read_binary(parameter_fname, &parameter_binary_size);
  }

  MPI_Bcast(&parameter_binary_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  if (mpi_rank != 0) {
    parameter = new float[parameter_binary_size];
  }

  MPI_Bcast(parameter, parameter_binary_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

  eW_emb = new Tensor({INPUT_VOCAB_SIZE, HIDDEN_SIZE}, parameter + OFFSET0);
  eW_ir = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET1);
  eW_iz = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET2);
  eW_in = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET3);
  eW_hr = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET4);
  eW_hz = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET5);
  eW_hn = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET6);
  eb_ir = new Tensor({HIDDEN_SIZE}, parameter + OFFSET7);
  eb_iz = new Tensor({HIDDEN_SIZE}, parameter + OFFSET8);
  eb_in = new Tensor({HIDDEN_SIZE}, parameter + OFFSET9);
  eb_hr = new Tensor({HIDDEN_SIZE}, parameter + OFFSET10);
  eb_hz = new Tensor({HIDDEN_SIZE}, parameter + OFFSET11);
  eb_hn = new Tensor({HIDDEN_SIZE}, parameter + OFFSET12);
  dW_emb = new Tensor({OUTPUT_VOCAB_SIZE, HIDDEN_SIZE}, parameter + OFFSET13);
  dW_ir = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET14);
  dW_iz = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET15);
  dW_in = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET16);
  dW_hr = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET17);
  dW_hz = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET18);
  dW_hn = new Tensor({HIDDEN_SIZE, HIDDEN_SIZE}, parameter + OFFSET19);
  db_ir = new Tensor({HIDDEN_SIZE}, parameter + OFFSET20);
  db_iz = new Tensor({HIDDEN_SIZE}, parameter + OFFSET21);
  db_in = new Tensor({HIDDEN_SIZE}, parameter + OFFSET22);
  db_hr = new Tensor({HIDDEN_SIZE}, parameter + OFFSET23);
  db_hz = new Tensor({HIDDEN_SIZE}, parameter + OFFSET24);
  db_hn = new Tensor({HIDDEN_SIZE}, parameter + OFFSET25);
  dW_attn = new Tensor({MAX_LENGTH, 2 * HIDDEN_SIZE}, parameter + OFFSET26);
  db_attn = new Tensor({MAX_LENGTH}, parameter + OFFSET27);
  dW_attn_comb = new Tensor({HIDDEN_SIZE, 2 * HIDDEN_SIZE}, parameter + OFFSET28);
  db_attn_comb = new Tensor({HIDDEN_SIZE}, parameter + OFFSET29);
  dW_out = new Tensor({OUTPUT_VOCAB_SIZE, HIDDEN_SIZE}, parameter + OFFSET30);
  db_out = new Tensor({OUTPUT_VOCAB_SIZE}, parameter + OFFSET31);

  encoder_embidx = new Tensor({BATCH_SIZE});
  encoder_hidden = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_outputs = new Tensor({BATCH_SIZE, MAX_LENGTH, HIDDEN_SIZE});
  encoder_embedded = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_rtmp1 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_rtmp2 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_rtmp3 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_rtmp4 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_rtmp5 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_rt = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_ztmp1 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_ztmp2 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_ztmp3 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_ztmp4 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_ztmp5 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_zt = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_ntmp1 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_ntmp2 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_ntmp3 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_ntmp4 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_ntmp5 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_ntmp6 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_nt = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_htmp1 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_htmp2 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_htmp3 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  encoder_ht = new Tensor({BATCH_SIZE, HIDDEN_SIZE});

  decoder_embidx = new Tensor({BATCH_SIZE});
  decoder_hidden = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_embedded = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_embhid = new Tensor({BATCH_SIZE, 2 * HIDDEN_SIZE});
  decoder_attn = new Tensor({BATCH_SIZE, MAX_LENGTH});
  decoder_attn_weights = new Tensor ({BATCH_SIZE, MAX_LENGTH});
  decoder_attn_applied = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_embattn = new Tensor({BATCH_SIZE, 2 * HIDDEN_SIZE});
  decoder_attn_comb = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_relu = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_rtmp1 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_rtmp2 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_rtmp3 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_rtmp4 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_rtmp5 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_rt = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_ztmp1 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_ztmp2 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_ztmp3 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_ztmp4 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_ztmp5 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_zt = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_ntmp1 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_ntmp2 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_ntmp3 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_ntmp4 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_ntmp5 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_ntmp6 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_nt = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_htmp1 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_htmp2 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_htmp3 = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_ht = new Tensor({BATCH_SIZE, HIDDEN_SIZE});
  decoder_out = new Tensor({BATCH_SIZE, OUTPUT_VOCAB_SIZE});
  decoder_logsoftmax = new Tensor({BATCH_SIZE, OUTPUT_VOCAB_SIZE});
  decoder_outputs = new Tensor({BATCH_SIZE});

  delete[] parameter;
}

/*
 * finalize_translator
 * @brief : free all dynamically allocated variables
 */
void finalize_translator(){
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    fprintf(stderr, "\n");

    // free parameters
    delete eW_emb;
    delete eW_ir;
    delete eW_iz;
    delete eW_in;
    delete eW_hr;
    delete eW_hz;
    delete eW_hn;
    delete eb_ir;
    delete eb_iz;
    delete eb_in;
    delete eb_hr;
    delete eb_hz;
    delete eb_hn;
    delete dW_emb;
    delete dW_ir;
    delete dW_iz;
    delete dW_in;
    delete dW_hr;
    delete dW_hz;
    delete dW_hn;
    delete db_ir;
    delete db_iz;
    delete db_in;
    delete db_hr;
    delete db_hz;
    delete db_hn;
    delete dW_attn;
    delete db_attn;
    delete dW_attn_comb;
    delete db_attn_comb;
    delete dW_out;
    delete db_out;

    // free encoder activations
    delete encoder_embidx;
    delete encoder_hidden;
    delete encoder_outputs;
    delete encoder_embedded;
    delete encoder_rtmp1;
    delete encoder_rtmp2;
    delete encoder_rtmp3;
    delete encoder_rtmp4;
    delete encoder_rtmp5;
    delete encoder_rt;
    delete encoder_ztmp1;
    delete encoder_ztmp2;
    delete encoder_ztmp3;
    delete encoder_ztmp4;
    delete encoder_ztmp5;
    delete encoder_zt;
    delete encoder_ntmp1;
    delete encoder_ntmp2;
    delete encoder_ntmp3;
    delete encoder_ntmp4;
    delete encoder_ntmp5;
    delete encoder_ntmp6;
    delete encoder_nt;
    delete encoder_htmp1;
    delete encoder_htmp2;
    delete encoder_htmp3;
    delete encoder_ht;

    // free decoder activations
    delete decoder_embidx;
    delete decoder_embedded;
    delete decoder_embhid;
    delete decoder_attn;
    delete decoder_attn_weights;
    delete decoder_attn_applied;
    delete decoder_embattn;
    delete decoder_attn_comb;
    delete decoder_relu;
    delete decoder_rtmp1;
    delete decoder_rtmp2;
    delete decoder_rtmp3;
    delete decoder_rtmp4;
    delete decoder_rtmp5;
    delete decoder_rt;
    delete decoder_ztmp1;
    delete decoder_ztmp2;
    delete decoder_ztmp3;
    delete decoder_ztmp4;
    delete decoder_ztmp5;
    delete decoder_zt;
    delete decoder_ntmp1;
    delete decoder_ntmp2;
    delete decoder_ntmp3;
    delete decoder_ntmp4;
    delete decoder_ntmp5;
    delete decoder_ntmp6;
    delete decoder_nt;
    delete decoder_htmp1;
    delete decoder_htmp2;
    delete decoder_htmp3;
    delete decoder_ht;
    delete decoder_out;
    delete decoder_logsoftmax;
    delete decoder_outputs;
  }
}
