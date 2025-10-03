Quantization {#dev_guide_attributes_quantization}
=================================================

@anchor dgaq_intro
## Introduction

Some primitives support input and output tensors with int8 data types,
both signed and unsigned, enabling reduced-precision inference on
supported hardware.

Similarly, some primitives support OFP8-compliant f8 types (8-bit
floating-point formats) designed to accelerate AI workloads, including
training and inference of large neural networks. Lowering precision to
8 bits with f8 enables faster computation and reduced memory usage.

Related materials:
- [Lower Numerical Precision Deep Learning Inference and Training](https://www.intel.com/content/dam/develop/external/us/en/documents/lower-numerical-precision-deep-learning-jan2018-754765.pdf)
- INT8 example with annotations: @ref dev_guide_inference_int8
- f8 example with annotations: @ref matmul_f8_quantization_cpp
- [OFP8 standard 8-bit floating-point](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)

## Supported Quantization Models

oneDNN supports integer quantization (int8) and floating-point
quantization (f8).

@note The guide below does not cover how the appropriate scaling factors can be
found. Refer to the materials in the [Introduction](@ref dgaq_intro).

### int8 Quantization

The primary int8 quantization model that the library assumes is the following:
\f[
    x_{f32}[:] = scale_{x} \cdot (x_{int8}[:] - zp_{x})
\f]

where \f$scale_{x}\f$ is a *scaling factor* in float format,
\f$zp_{x}\f$ is the *zero point* in int32 format, and
\f$[:]\f$ is used to denote elementwise application of the formula
to the arrays. In order to provide best performance, oneDNN does not
compute those scaling factors and zero-points as part of primitive
computation. Those should be computed and provided by the user.

To support int8 quantization, primitives should be created and executed as
follows (for details on the API and examples, see the following sections):

- During primitive descriptor creation:
    - If one or multiple inputs are int8 (signed or not),
      then the primitive will behave as a quantized integer operation.
    - The dimensionality of the scaling factors and zero-point
      should be provided using masks and groups (e.g. one scale per tensor, one scale per channel, etc.).
- During primitive execution:
    - The user must provide the actual quantization parameters as arguments to the execute function.
    Scales are `f32` values, and zero-points are `s32` values.

### f8 Quantization

For f8 quantization, oneDNN follows the
[OFP8 specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)
and supports two 8-bit floating-point formats:

- **f8_e5m2**: 1 sign + 5 exponent + 2 mantissa bits (max value: 57,344)
- **f8_e4m3**: 1 sign + 4 exponent + 3 mantissa bits (max value: 448)

The f8 quantization model uses simple scaling without zero-points:
\f[
    x_{f32}[:] = scale_{x} \cdot x_{f8}[:]
\f]

where \f$scale_{x}\f$ is a scaling factor applied during dequantization.

To support f8 quantization, primitives should be created and executed as
follows (for details on the API and examples, see the following sections):

- During primitive descriptor creation:
    - Specify f8 data types (`f8_e5m2` or `f8_e4m3`) for inputs.
    - Configure scaling factors using the same mask and groups-based approach as int8,
    but zero-points are not applicable.
- During primitive execution:
    - Provide scaling factors for dequantization from f8 to the computation precision
    as arguments to the execute function.

### Numerical behavior

**int8 Quantization Behavior:**
Primitive implementations may convert int8 inputs to wider data types
(e.g., int16 or int32) without affecting accuracy. During execution,
primitives avoid integer overflow by using wider data types (e.g., int32)
for intermediate values and accumulators. Results are converted back to the
target data type before writing to output memory objects.

@warning
int8 computation behavior may vary slightly depending on the architecture.
For more details, refer to @ref dev_guide_int8_computations.

**f8 Quantization Behavior:**

@warning
TODO: Need content here.

**Conversion Behavior:**
When converting to integral data types, implementations typically saturate.
For floating-point data types, underflow/overflow can occur. To force
saturation in floating-point data types, use
@ref dev_guide_attributes_post_ops_eltwise with clip algorithm.

**Post-Operations:**
When multiple operations are fused in a single primitive using the
[post ops attribute](@ref dev_guide_attributes_post_ops), they are computed
in f32 precision. Destination quantization parameters are applied after
the post-ops as follows:

\f[
   \dst[:] = post\_ops(OP(src[:], weights[:], ...)) / scale_{\dst} + zp_{\dst}
\f]

Quantizing/dequantizing values between post-operations can be achieved
using [eltwise](@ref dev_guide_attributes_post_ops_eltwise),
[binary](@ref dev_guide_attributes_post_ops_binary), or the scale parameter
of the appropriate post-operation.

### int8 Convolution Quantization Breakdown

Consider a convolution with bias. The tensors are represented as:

- \f$\src_{f32}[:] = scale_{\src} \cdot (\src_{int8}[:] - zp_{\src})\f$
- \f$\weights_{f32}[:] = scale_{\weights} \cdot \weights_{int8}[:]\f$
- \f$\dst_{f32}[:] = scale_{\dst} \cdot (\dst_{int8}[:] - zp_{\dst})\f$

Here the \f$\src_{f32}, \weights_{f32}, \dst_{f32}\f$ are not
computed at all, the whole work happens with int8 tensors.So the task
is to compute the \f$\dst_{int8}\f$ tensor, using the \f$\src_{int8}\f$,
\f$\weights_{int8}\f$ tensors passed at execution time, as well as the
corresponding quantization parameters \f$scale_{\src}\f$,
\f$scale_{\weights}\f$, \f$scale_{\dst}\f$, and \f$zp_{\src}\f$,
\f$zp_{\dst}\f$.
Mathematically, the computations are:

\f[
   \dst_{int8}[:] =
      \operatorname{f32\_to\_int8}(
         (scale_{\src} \cdot scale_{\weights} \cdot
         \operatorname{s32\_to\_f32}(conv_{s32}(\src_{int8}, \weights_{int8})
	   - zp_{\src} \cdot comp_{s32}) + bias_{f32}) / scale_{\dst}
           + zp_{\dst} )
\f]

where

- \f$\operatorname{conv}_{s32}\f$ is just a regular convolution which takes
  source and weights with int8 data type and compute the result in int32 data
  type (int32 is chosen to avoid overflows during the computations);

- \f$comp_{s32}\f$ is a compensation term to account for
  \f$\src\f$ non-zero zero-point. This term is computed by the oneDNN
  library and can typically be pre-computed ahead of time, for example
  during weights reorder.

- \f$\operatorname{f32\_to\_s8}()\f$ converts an `f32` value to `s8` with
  potential saturation if the values are out of the range of the int8 data
  type.

- \f$\operatorname{s32\_to\_f32}()\f$ converts an `int8` value to
  `f32` with potential rounding. This conversion is typically
  necessary to apply `f32` scaling factors.

#### Per-Channel Scaling Specifics

Some of the primitives have limited support of multiple scales for a quantized
tensor. The most popular use case is the @ref dev_guide_convolution primitive
that supports per-output-channel scaling factors for the weights, meaning that
the actual convolution computations would need to scale different output
channels differently. This is possible without significant performance loss
because the per-output-channel re-quantization is only required at the very end
of the computations. It seems impossible to implement the same trick for the
input channels, since that would require re-quantization for every input
data point.

- \f$\src_{f32}(n, ic, ih, iw) = scale_{\src} \cdot \src_{int8}(n, ic, ih, iw)\f$

- \f$\weights_{f32}(oc, ic, kh, kw) = scale_{\weights}(oc) \cdot
  \weights_{int8}(oc, ic, kh, kw)\f$

- \f$\dst_{f32}(n, oc, oh, ow) = scale_{\dst} \cdot
  \dst_{int8}(n, oc, oh, ow)\f$

Note that now the weights' scaling factor depends on \f$oc\f$.

To compute the \f$\dst_{int8}\f$ we need to perform the following:

\f[

    \dst_{int8}(n, oc, oh, ow) =
        \operatorname{f32\_to\_int8}(
            \frac{scale_{\src} \cdot scale_{\weights}(oc) \cdot
            conv_{s32}(\src_{int8}, \weights_{int8})|_{(n, oc, oh, ow)} +
            \bias_{f32}}{scale_{\dst}}
        ).
\f]

The user is responsible for preparing quantized weights accordingly. To do that,
oneDNN provides reorders that can perform per-channel scaling:

\f[

    \weights_{int8}(oc, ic, kh, kw) =
        \operatorname{f32\_to\_int8}(
            \weights_{f32}(oc, ic, kh, kw) / scale_{weights}(oc)
        ).
\f]

## Scaling and Zero-Point API and Usage

The library API supports both int8 and f8 quantization models described above.
The API was designed to be flexible enough to accommodate different
quantization schemes. As long as users can fit their model into the given
functionality everything should work fine. We designed a minimal and simple
yet powerful enough quantization API that supports both scaling factors and
zero-points (for int8) as well as f8 floating-point quantization.

For int8 quantization, the most common data types are
#dnnl::memory::data_type::s8 and #dnnl::memory::data_type::u8. Both scaling
factors and zero-points are supported and maintained by users separately from
oneDNN memory objects.

For f8 quantization, the supported data types are
#dnnl::memory::data_type::f8_e5m2 and #dnnl::memory::data_type::f8_e4m3.
Only scaling factors are supported.

The library essentially extends the ability of the primitives to scale the
output before storing the result to the memory with the destination data type.
That's exactly the minimum that we need to support int8 inference (check the
equations above--only \f$output\_scale\f$ is non-standard).

@warning
TODO: What is this about -- (check the equations above--only \f$output\_scale\f$ is non-standard)?

The scaling happens in the single precision floating point data type
(#dnnl::memory::data_type::f32). Before storing, the result is downconverted
to the destination data type with saturation if required. The rounding happens
according to the current HW setting (for instance, on CPU according to the
MXCSR register).

@anchor dev_guide_attributes_quantization_scales
### Argument Scaling

The library uses @ref dev_guide_attributes API for setting the scaling factors
for most of the primitives. The supporting attributes can be found in the
documentation for each primitive. The unsupported cases are handled according
to the
[attributes error handling section](@ref dev_guide_attributes_error_handling).

#### Available Scaling API Methods

oneDNN provides the following methods for setting scaling factors:

~~~cpp
// Legacy method with simple mask-based scaling
void dnnl::primitive_attr::set_scales_mask(int arg, int mask);

// Generic method with groups support
void dnnl::primitive_attr::set_scales(int arg, int mask,
                                      const memory::dims &groups,
                                      memory::data_type data_type = memory::data_type::f32,
                                      bool is_on_host = false);

// Convenience method for single host-side scalar
void dnnl::primitive_attr::set_host_scale(int arg,
                                          memory::data_type data_type = memory::data_type::f32);
~~~

##### Concepts

Arguments (`arg`) specify which primitive input/output to scale:
- `DNNL_ARG_SRC`: Source tensor
- `DNNL_ARG_WEIGHTS`: Weight tensor
- `DNNL_ARG_DST`: Destination tensor
- `DNNL_ARG_BIAS`: Bias tensor (limited support)

Mask (`mask`) controls which dimensions get individual scaling factors:
- `0`: Single scale for entire tensor (global scaling)
- `1 << dim`: Scale varies along dimension `dim`
- `(1 << dim1) + (1 << dim2)`: Scales vary along multiple dimensions

Groups (`groups`) divide dimensions into blocks for block-wise quantization:
- `{}`: No grouping (default)
- `{G}`: Single group
- `{G1, G2, ...}`: Multi-dimensional grouping

The scaling parameters support multiple data types to accommodate
different quantization workflows and precisions requirements:
- `f32`
- `bf16`, `f16`
- `f8_e5m2`, `f8_e4m3`
- `e8m0`

Additionally, scales can be specified as residing on host or device memory
(refer to [the section below](@ref host-side-scalars-and-zero-points) for
more details):
- `is_on_host = false`: Scale values are in device memory
- `is_on_host = true`: Scale values are in host memory


#### Supported Scaling Patterns

oneDNN supports several scaling patterns to support different quantization
schemes. These patterns apply to both int8 and f8 quantization.

* **Global scaling** (`mask=0`) uses a single scale factor for the entire
  tensor, making it the simplest approach.
* **Per-channel scaling** (`mask=1<<dim`) applies different scale factors
  along a specific dimension, commonly used for CNN weights.
* **Multi-dimensional scaling** (`mask=(1<<dim1)+(1<<dim2)`) provides
  independent scales along multiple tensor dimensions, useful for complex
  activations where both batch and channel dimensions need separate scaling.
* **Group-based quantization** subdivides tensor dimensions into smaller
  blocks with individual scale factors, important for large transformer
  models and advanced quantization techniques.

##### Global Scaling

In the simplest case, when there is only one common scale the attribute changes
the op behavior from
\f[
    \dst[:] = Op(...)
\f]

to

\f[
    \dst[:] = scale \cdot Op(...).
\f]

~~~cpp
// Using full set_scales API (recommended)
attr.set_scales(DNNL_ARG_SRC, 0, {}, dnnl::memory::data_type::f32,
                false /*on device*/);

// Using convenience set_host_scale API for single host-side scalar
attr.set_host_scale(DNNL_ARG_SRC, dnnl::memory::data_type::f32);

// Using legacy set_scales_mask API
attr.set_scales_mask(DNNL_ARG_SRC, 0);

// Tensor: [N, C, H, W] = [2, 3, 4, 4]
// Scales: 1 value
// Usage: All elements use same scale
~~~

@note For more details on global scaling with a single scale value residing on
host, use @ref host-side-scalars-and-zero-points "host-side scalar scaling"
(`set_host_scale`) to avoid device memory transfer overhead.

Global scaling is demonstrated in
[Examples 2](#example-2-convolution-with-per-output-channel-quantization)
and [3](#example-3-comprehensive-asymmetric-quantization-with-zero-points)
below.

##### Per-Channel Scaling

Per-channel scaling applies different scale factors along specific tensor
dimensions. For instance, it is commonly used for CNN weights where each
output channel has its own scale.

~~~cpp
// Scale per output channel (dimension 0 of weights)
attr.set_scales(DNNL_ARG_WEIGHTS, 1 << 0, {}, dnnl::memory::data_type::f32,
                false /*on device*/);

// Tensor: [OC, IC, H, W] = [64, 128, 3, 3]
// Scales: 64 values (one per output channel)
// Usage: Each output channel gets its own scaling factor
~~~

Per-channel scaling is demonstrated in
[Examples 1](#example-1-weights-quantization-with-per-output-channel-scaling),
[2](#example-2-convolution-with-per-output-channel-quantization), and
[3](#example-3-comprehensive-asymmetric-quantization-with-zero-points) below.
It's also used in @ref inference_int8_matmul_cpp for weights quantization.

##### Group-Based Quantization

Groups enable block-wise quantization by subdividing tensor dimensions into
smaller blocks, each with its own scale. This might help balance accuracy
and efficiency by providing more granular quantization than global scaling.

~~~cpp
// Weight shape: [K, N] = [1024, 512] with groups [32, 1]
// Creates 32 blocks of size [32, 512] each with its own scale
std::vector<dnnl::memory::dim_t> groups = {32, 1};
attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), groups,
                memory::data_type::f32, false);

// Tensor: [K, N] = [1024, 512]
// Scales: 32 values (one per group)
// Usage: Each group gets its own scaling factor
~~~

Group-based quantization is demonstrated in
[Examples 4](#example-4-matmul-with-advanced-quantization)
and [5](#example-5-matmul-with-precomputed-reductions-and-advanced-quantization)
below.
See also @ref weights_decompression_matmul_cpp for a complete implementation.

##### Multi-Dimensional Scaling

Multi-dimensional scaling applies scales across multiple tensor dimensions
simultaneously.

For scales per dimensions \f$d_i\f$, set `mask = `\f$\sum_{d_i} 2^{d_i}\f$.

Resulting scale count without groups: \f$\prod_{d_i} D_{d_i}\f$, with groups:
\f$\prod_{d_i} G_{d_i}\f$.

~~~cpp
// Scale varies along batch and channel dimensions
attr.set_scales(DNNL_ARG_SRC, (1 << 0) + (1 << 1), {},
                dnnl::memory::data_type::f32, false);

// Tensor: [N, C, H, W] = [8, 64, 32, 32]
// Scales needed: 8 * 64 = 512 values
// Usage: Each (batch, channel) combination gets its own scale
~~~

Multi-dimensional scaling is demonstrated in
[Examples 4](#example-4-matmul-with-advanced-quantization)
and [5](#example-5-matmul-with-precomputed-reductions-and-advanced-quantization)
below.
See also @ref weights_decompression_matmul_cpp for a complete implementation.

@anchor dev_guide_attributes_quantization_zero_points
### Argument Zero-Points

Zero-points handle the quantization case where the quantized integer range
does not center around zero.

The library uses @ref dev_guide_attributes API for setting zero-points for
most primitives. The supporting attributes can be found in the documentation
for each primitive. The unsupported cases are handled according to the
[attributes error handling section](@ref dev_guide_attributes_error_handling).

#### Available Zero-Point API Methods

oneDNN provides the following methods for setting zero-points:

~~~cpp
// Legacy method with simple mask-based zero-points
void dnnl::primitive_attr::set_zero_points_mask(int arg, int mask);

// Generic method with groups support
void dnnl::primitive_attr::set_zero_points(int arg, int mask,
                                          const memory::dims &groups,
                                          memory::data_type data_type = memory::data_type::s32,
                                          bool is_on_host = false);

// Convenience method for single host-side scalar
void dnnl::primitive_attr::set_host_zero_point(int arg,
                                              memory::data_type data_type = memory::data_type::s32);
~~~

##### Zero-Point Concepts

Arguments (`arg`) specify which primitive input/output to apply zero-points:
- `DNNL_ARG_SRC`: Source tensor zero-points
- `DNNL_ARG_WEIGHTS`: Weight tensor zero-points
- `DNNL_ARG_DST`: Destination tensor zero-points

Mask (`mask`) and Groups (`groups`) follow the same semantics as scaling
factors.

Data Types (`data_type`) supported for zero-points:
- `s32`
- `s8`, `u8`
- `s4`, `u4`

Additionally, zero-point can be specified as residing on host or device memory
(refer to [the section below](@ref host-side-scalars-and-zero-points) for
more details):
- `is_on_host = false`: Zero-point value is in device memory
- `is_on_host = true`: Zero-point value is in host memory

#### Supported Zero-Point Patterns

Zero-point patterns mirror the scaling patterns described above. The same mask
and groups concepts apply:

- **Global zero-point** (`mask=0`): Single zero-point for entire tensor
- **Per-channel zero-points** (`mask=1<<dim`): Different zero-points per
  channel
- **Group-based zero-points** (`mask` with `groups`): Block-wise zero-points
- **Multi-dimensional zero-points** (`mask=(1<<dim1)+(1<<dim2)`):
  Independent zero-points across multiple dimensions

~~~cpp
// Global zero-point
attr.set_zero_points(DNNL_ARG_SRC, 0, {}, memory::data_type::s32, false);

// Per-channel zero-points
attr.set_zero_points(DNNL_ARG_WEIGHTS, 1 << 0, {}, memory::data_type::s8,
                     false);

// Group-based zero-points
std::vector<dnnl::memory::dim_t> groups = {64, 1};
attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), groups,
                     memory::data_type::s32, false);
~~~

Zero-point usage is demonstrated in
[Examples 2](#example-2-convolution-with-per-output-channel-quantization),
[3](#example-3-comprehensive-asymmetric-quantization-with-zero-points), and
[5](#example-5-matmul-with-precomputed-reductions-and-advanced-quantization)
below.
See also @ref inference_int8_matmul_cpp and @ref weights_decompression_matmul_cpp
for complete implementations.

@anchor host-side-scalars-and-zero-points
#### Special Case: Host-side Scalar Scale and Zero-point

When using the GPU engine, host-side scalar scales and zero-points are
supported to reduce copying of data from host to device. A memory object
for scale or zero-point host value should be created as a host-side scalar
(see @ref dev_guide_host_side_scalars for details) and passed to the primitive
execution function. The host scales or zero-points attributes should also
be set using the following API:

~~~cpp
dnnl::primitive_attr attr;
attr.set_host_scale(DNNL_ARG_DST,
           memory::data_type::f32);

attr.set_host_zero_point(DNNL_ARG_DST,
           memory::data_type::s32);
~~~

## Examples

### Example 1: weights quantization with per-output-channel scaling

~~~cpp
   // weights dimensions
   const int OC, IC, KH, KW;

   // original f32 weights in plain format
   dnnl::memory::desc wei_plain_f32_md(
           {OC, IC, KH, KW},                 // dims
           dnnl::memory::data_type::f32,     // the data originally in f32
           dnnl::memory::format_tag::hwigo   // the plain memory format
           );

   // the scaling factors for quantized weights
   // An unique scale for each output-channel.
   std::vector<float> wei_scales(OC) = { /* values */ };

   // optional: zero-points for asymmetric quantization
   // std::vector<int8_t> wei_zero_points(OC) = { /* values */ };

   // int8 convolution primitive descriptor
   dnnl::convolution_forward::primitive_desc conv_pd(
       /* see the next example */);

   // query the convolution weights memory descriptor
   dnnl::memory::desc wei_conv_s8_md = conv_pd.weights_desc();

   // prepare the attributes for the reorder
   dnnl::primitive_attr attr;
   const int quantization_mask = 0
       | (1 << 0);  // scale per OC dimension, which is the dim #0
   attr.set_scales_mask(DNNL_ARG_DST, quantization_mask);

   // optional: set zero-points for asymmetric weights quantization
   // attr.set_zero_points_mask(DNNL_ARG_DST, quantization_mask);

   // create reorder that would perform:
   //   wei_s8(oc, ic, kh, kw) <- wei_f32(oc, ic, kh, kw) / scale(oc) [- zp(oc)]
   // including the data format conversion.
   auto wei_reorder_pd = dnnl::reorder::primitive_desc(
           wei_plain_f32_md, engine, // source
           wei_conv_s8_md, engine, // destination,
           attr);
   auto wei_reorder = dnnl::reorder(wei_reorder_pd);

// ...
~~~

### Example 2: convolution with per-output-channel quantization

This example is complementary to the previous example (which should ideally be
the first one). Let's say we want to create an int8 convolution with per-output
channel scaling and zero-points for both source and destination tensors.

~~~cpp
   const float src_scale; // src_f32[:] = src_scale * (src_s8[:] - src_zp)
   const int32_t src_zp;  // source zero-point for asymmetric quantization
   const float dst_scale; // dst_f32[:] = dst_scale * (dst_s8[:] - dst_zp)
   const int32_t dst_zp;  // destination zero-point

   // the scaling factors for quantized weights (as declared above)
   // An unique scale for each output-channel.
   std::vector<float> wei_scales(OC) = {...};

   // optional: per-channel zero-points for weights
   // (asymmetric weight quantization)
   // std::vector<int8_t> wei_zero_points(OC) = {...};


   // Src, weights, and dst memory descriptors for convolution,
   // with memory format tag == any to allow a convolution implementation
   // to chose the appropriate memory format

   dnnl::memory::desc src_conv_s8_any_md(
           {BATCH, IC, IH, IW},          // dims
           dnnl::memory::data_type::s8,  // the data originally in s8
           dnnl::memory::format_tag::any // let convolution to choose
           );

   dnnl::memory::desc wei_conv_s8_any_md(
           {OC, IC, KH, KW},             // dims
           dnnl::memory::data_type::s8,  // the data originally in s8
           dnnl::memory::format_tag::any // let convolution to choose
           );

   dnnl::memory::desc dst_conv_s8_any_md(...);  // ditto

   // prepare the attributes for the convolution
   dnnl::primitive_attr attr;
   const int data_mask = 0; // scale and zero-point per tensor for source and destination
   const int wei_mask = 0
       | (1 << 0); // scale per OC dimension, which is the dim #0 on weights tensor:
                   // (   OC, IC, KH, KW)
                   //      0   1   2   3

   // Configure scaling factors
   attr.set_scales_mask(DNNL_ARG_SRC, data_mask);
   attr.set_scales_mask(DNNL_ARG_WEIGHTS, wei_mask);
   attr.set_scales_mask(DNNL_ARG_DST, data_mask);

   // Configure zero-points for asymmetric quantization
   attr.set_zero_points_mask(DNNL_ARG_SRC, data_mask);
       // global source zero-point
   attr.set_zero_points_mask(DNNL_ARG_DST, data_mask);
       // global destination zero-point
   // optional: per-channel weight zero-points
   // attr.set_zero_points_mask(DNNL_ARG_WEIGHTS, wei_mask);

   // create a convolution primitive descriptor
   auto conv_pd = dnnl::convolution_forward::primitive_desc(
           dnnl::prop_kind::forward_inference,
           dnnl::algorithm::convolution_direct,
           src_conv_s8_any_md,                     // what's important is that
           wei_conv_s8_any_md,                     // we specified that we want
           dst_conv_s8_any_md,                     // computations in s8
           strides, padding_l, padding_r,
           dnnl::padding_kind::zero,
           attr);   // the attributes describe the quantization flow

   // Execute the convolution with runtime quantization parameters
   auto conv = dnnl::convolution_forward(conv_pd);

   // Create memory objects for quantization parameters
   auto src_scale_mem = /* create memory with src_scale */;
   auto src_zp_mem = /* create memory with src_zp */;
   auto wei_scales_mem = /* create memory with wei_scales */;
   auto dst_scale_mem = /* create memory with dst_scale */;
   auto dst_zp_mem = /* create memory with dst_zp */;

   conv.execute(stream, {
       {DNNL_ARG_SRC, src_memory},
       {DNNL_ARG_WEIGHTS, wei_memory},
       {DNNL_ARG_DST, dst_memory},
       {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scale_mem},
       {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_mem},
       {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scale_mem},
       {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zp_mem},
       {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zp_mem}
       // optional: {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wei_zp_mem}
   });
// ...
~~~

### Example 3: comprehensive asymmetric quantization with zero-points

This example demonstrates asymmetric quantization using zero-points alongside
scaling factors for a complete quantization workflow including reorder
operations.

~~~cpp
   // Quantization parameters
   const float src_scale = 0.1f;
   const int32_t src_zp = 128;     // common for u8 inputs

   const float dst_scale = 0.2f;
   const int32_t dst_zp = -10;     // can be negative for s8 outputs

   // Per-channel weights quantization
   const int OC = 256;
   std::vector<float> wei_scales(OC);
   std::vector<int8_t> wei_zero_points(OC);
   // ... initialize wei_scales and wei_zero_points

   // Memory descriptors
   dnnl::memory::desc src_u8_md({BATCH, IC, IH, IW},
                                dnnl::memory::data_type::u8,
                                dnnl::memory::format_tag::nhwc);

   dnnl::memory::desc wei_s8_md({OC, IC, KH, KW},
                                dnnl::memory::data_type::s8,
                                dnnl::memory::format_tag::any);

   dnnl::memory::desc dst_s8_md({BATCH, OC, OH, OW},
                                dnnl::memory::data_type::s8,
                                dnnl::memory::format_tag::any);

   // Configure quantization attributes
   dnnl::primitive_attr attr;

   // Source: global scale and zero-point (u8 asymmetric)
   attr.set_scales_mask(DNNL_ARG_SRC, 0);        // global scale
   attr.set_zero_points_mask(DNNL_ARG_SRC, 0);   // global zero-point

   // Weights: per-channel scales and zero-points (s8 asymmetric per-channel)
   const int wei_mask = 1 << 0;  // per output channel
   attr.set_scales_mask(DNNL_ARG_WEIGHTS, wei_mask);     // per-channel scales
   attr.set_zero_points(DNNL_ARG_WEIGHTS, wei_mask, {},
                         dnnl::memory::data_type::s8);
                         // per-channel s8 zero-points

   // Destination: global scale and zero-point (s8 asymmetric)
   attr.set_scales_mask(DNNL_ARG_DST, 0);        // global scale
   attr.set_zero_points_mask(DNNL_ARG_DST, 0);   // global zero-point

   // Create primitive
   auto conv_pd = dnnl::convolution_forward::primitive_desc(
           dnnl::prop_kind::forward_inference,
           dnnl::algorithm::convolution_direct,
           src_u8_md, wei_s8_md, dst_s8_md,
           strides, padding_l, padding_r,
           dnnl::padding_kind::zero,
           attr);
   auto conv = dnnl::convolution_forward(conv_pd);

   // Create runtime quantization parameter memories
   auto src_scale_mem = dnnl::memory({{1}, dnnl::memory::data_type::f32,
                                     dnnl::memory::format_tag::x}, engine);
   write_to_dnnl_memory(&src_scale, src_scale_mem);

   auto src_zp_mem = dnnl::memory({{1}, dnnl::memory::data_type::s32,
                                  dnnl::memory::format_tag::x}, engine);
   write_to_dnnl_memory(&src_zp, src_zp_mem);

   auto wei_scales_mem = dnnl::memory({{OC}, dnnl::memory::data_type::f32,
                                      dnnl::memory::format_tag::x}, engine);
   write_to_dnnl_memory(wei_scales.data(), wei_scales_mem);

   auto wei_zp_mem = dnnl::memory({{OC}, dnnl::memory::data_type::s8,
                                  dnnl::memory::format_tag::x}, engine);
   write_to_dnnl_memory(wei_zero_points.data(), wei_zp_mem);

   auto dst_scale_mem = dnnl::memory({{1}, dnnl::memory::data_type::f32,
                                     dnnl::memory::format_tag::x}, engine);
   write_to_dnnl_memory(&dst_scale, dst_scale_mem);

   auto dst_zp_mem = dnnl::memory({{1}, dnnl::memory::data_type::s32,
                                  dnnl::memory::format_tag::x}, engine);
   write_to_dnnl_memory(&dst_zp, dst_zp_mem);

   // Execute with full asymmetric quantization
   conv.execute(stream, {
       {DNNL_ARG_SRC, src_memory},
       {DNNL_ARG_WEIGHTS, wei_memory},
       {DNNL_ARG_DST, dst_memory},
       {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scale_mem},
       {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_mem},
       {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scale_mem},
       {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zp_mem},
       {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wei_zp_mem},
       {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zp_mem}
   });
// ...
~~~

### Example 4: matmul with advanced quantization

This example describes a process of weights decompression, or
weights-only-quantization (WoQ), in matmul primitive which may be found when
running Large Language Models (LLM). The advanced quantization here refers to
additional grouping introduced over reduction dimension besides traditional
per-N quantization.

~~~cpp
   // Src, weights, and dst memory descriptors for matmul.
   // Consider simple 2D matmul case.
   dnnl::memory::desc src_f16_any_md(...);
   dnnl::memory::desc wei_s8_any_md(
           {K (256), N (512)},           // dims
           dnnl::memory::data_type::s8,  // the data originally in s8
           dnnl::memory::format_tag::any // let matmul to choose
           );
   dnnl::memory::desc dst_f16_any_md(...);

   // prepare the attributes
   dnnl::primitive_attr attr;
   // scale per K and N dimensions:
   const int wei_mask = (1 << 0) | (1 << 1);
   // K dimension specifies the group size of `128`. It means that each 128
   // elements over K dimension will share a single value. For a given example,
   // there will be two groups, thus, two values referring to a single N value.
   std::vector<dim_t> wei_groups = {128, 1}

   // the scaling factors for quantized weights (as declared above)
   // A unique scale for each gK (256 / 128 = 2) times N, total 1024 elements.
   std::vector<half> wei_scales(gK, N) = {...};

   attr.set_scales(DNNL_ARG_WEIGHTS, wei_mask, wei_groups, data_type::f16);

   // Additionally, to instruct the library to perform weights decompression,
   // fpmath mode must be set with a flag set to `true`:
   attr.set_fpmath_mode(fpmath_mode::f16, /* apply_to_int = */ true);

   // create a matmul primitive descriptor
   auto matmul_pd = dnnl::matmul::primitive_desc(
           engine,
           src_f16_any_md,
           wei_s8_any_md,
           dst_f16_any_md,
           attr);   // the attributes describe the quantization flow
// ...
~~~

### Example 5: matmul with precomputed reductions and advanced quantization

This example is a complementary addition to the one above. It describes a
process of dynamic quantization with weights's tensor asymmetric quantization
and external precomputed reductions of the source tensor. This shows asymmetric
quantization with zero-points for advanced use cases.

The case arises from the technique of quantizing source tensor on-the-fly (on
the application side) and passing both quantized source and weights tensors to
the library.

It's important that precomputed reductions appear from weights zero-points to
provide accurate result when zero-points datatype is s8, in which case it's
impossible to apply them on-the-fly without potential accuracy loss.

~~~cpp
   // Src, weights, and dst memory descriptors for matmul.
   // Consider simple 2D matmul case.
   dnnl::memory::desc src_u8_any_md(
           {M (64), K (256)},            // dims
           dnnl::memory::data_type::u8,  // the data originally in u8
           dnnl::memory::format_tag::any // let matmul to choose
           );
   dnnl::memory::desc wei_s8_any_md(
           {K (256), N (512)},           // dims
           dnnl::memory::data_type::s8,  // the data originally in s8
           dnnl::memory::format_tag::any // let matmul to choose
           );
   dnnl::memory::desc dst_f16_any_md(...);

   // prepare the attributes
   dnnl::primitive_attr attr;
   // scale per K and N dimensions:
   const int wei_mask = (1 << 0) | (1 << 1);
   // K dimension specifies the group size of `128`. It means that each 128
   // elements over K dimension will share a single value. For a given example,
   // there will be two groups, thus, two values referring to a single N value.
   std::vector<dim_t> wei_scales_groups = {128, 1}

   // The scaling factors for quantized weights (as declared above)
   // A unique scale for each scale_gK (256 / 128 = 2) times N, total 1024
   // elements.
   std::vector<half> wei_scales(scale_gK, N) = {...};

   attr.set_scales(DNNL_ARG_WEIGHTS, wei_mask, wei_scales_groups,
           data_type::f16);

   // Zero-points would have the same mask as grouping applies for them as well.
   // For example, let it use the different size of the group.
   std::vector<dim_t> wei_zp_groups = {64, 1};

   // The zero-point factors for quantized weights (as declared above)
   // A unique zero-point for each zp_gK (256 / 64 = 4) times N, total 2048
   // elements. Using s8 zero-points for weights.
   std::vector<int8_t> wei_zps(zp_gK, N) = {...};

   attr.set_zero_points(DNNL_ARG_WEIGHTS, wei_mask, wei_zp_groups,
           data_type::s8);

   // Source tensor with asymmetric quantization (u8 data with zero-point)
   const int src_mask = 0; // global zero-point and scale for source
   attr.set_scales(DNNL_ARG_SRC, src_mask, {}, data_type::f32);
   attr.set_zero_points(DNNL_ARG_SRC, src_mask, {}, data_type::s32);

   // Now, specify the precomputed reductions.
   // Note that it's specified for source tensor.
   // It means it should have full-size source tensor mask (which in this
   // example coincides with `wei_mask`), and groups would be over another
   // dimension, same as zero-points group size.
   std::vector<dim_t> src_pr_groups = {1, 64};

   // The precomputed reduction factors for quantized sources.
   // A unique reduction for each M times pr_gK (256 / 64 = 4), total 256
   // elements.
   std::vector<half> src_prs(M, pr_gK) = {...};

   attr.set_precomputed_reductions(DNNL_ARG_SRC, src_tensor_mask,
           src_pr_groups);

   // fpmath mode is not required in case of dynamic quantization as it's
   // treated as classical quantization case.

   // create a matmul primitive descriptor
   auto matmul_pd = dnnl::matmul::primitive_desc(
           engine,
           src_u8_any_md,    // asymmetric u8 source with zero-points
           wei_s8_any_md,    // asymmetric s8 weights with zero-points
                             // and groups
           dst_f16_any_md,
           attr);   // the attributes describe the quantization flow

   // Execute with runtime quantization parameters including zero-points
   auto matmul = dnnl::matmul(matmul_pd);
   matmul.execute(stream, {
       {DNNL_ARG_SRC, src_memory},
       {DNNL_ARG_WEIGHTS, wei_memory},
       {DNNL_ARG_DST, dst_memory},
       {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scale_mem},
       {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_mem},
       {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zp_mem},
       {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wei_zp_mem},
       {DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS | DNNL_ARG_SRC, src_pr_mem}
   });
// ...
~~~

### Example 6: f8 matmul with quantization and scaling

@ref fp8_matmul_scaling_cpp example demonstrates f8 quantization workflow
using both f8_e4m3 and f8_e5m2 formats. It shows the complete process from
f32 data to f8 quantization, matrix multiplication, and dequantization back
to f32.

@warning
TODO: would be rebased later to include link to actual example.


