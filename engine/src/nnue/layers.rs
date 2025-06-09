use std::io::{self, Read};

use crate::utils::simd::SimdAlign;

pub const ACTIVATION_RANGE: i32 = 127;
#[allow(dead_code)]
pub const WEIGHT_SCALE_SHIFT: i32 = 6;
pub const WEIGHT_SCALE: i32 = 64;
pub const OUTPUT_WEIGHT_SCALE: i32 = 32;

#[derive(Clone, Debug)]
pub struct Transformer<const IN: usize, const OUT: usize> {
    pub weights: SimdAlign<[[i16; OUT]; IN]>,
    pub biases: SimdAlign<[i16; OUT]>,
}

#[derive(Clone, Debug)]
pub struct Linear<const IN: usize, const OUT: usize> {
    pub weights: SimdAlign<[[i8; IN]; OUT]>,
    pub biases: SimdAlign<[i32; OUT]>,
}

#[derive(Clone, Debug)]
pub struct LinearOut<const IN: usize> {
    pub weights: SimdAlign<[i8; IN]>,
    pub bias: i32,
}

impl<const IN: usize, const OUT: usize> Transformer<IN, OUT> {
    pub(super) fn load_from(reader: &mut impl Read) -> io::Result<Self> {
        let mut weights = Wrapped([[0i16; OUT]; IN]);
        let mut biases = [0i16; OUT];
        reader.read_exact(bytemuck::bytes_of_mut(&mut weights))?;
        reader.read_exact(bytemuck::cast_slice_mut(&mut biases))?;

        Ok(Self {
            weights: weights.0.map(|col| col.map(i16::from_le)).into(),
            biases: biases.map(i16::from_le).into(),
        })
    }

    #[inline]
    pub fn reset(&self, dst: &mut [i16; OUT]) {
        *dst = *self.biases;
    }

    #[inline]
    pub fn add(&self, feature: usize, dst: &mut SimdAlign<[i16; OUT]>) {
        for (a, b) in dst.iter_mut().zip(&self.weights[feature]) {
            *a += *b;
        }
    }

    #[inline]
    pub fn sub(&self, feature: usize, dst: &mut SimdAlign<[i16; OUT]>) {
        for (a, b) in dst.iter_mut().zip(&self.weights[feature]) {
            *a -= *b;
        }
    }
}

#[cfg(target_feature = "avx2")]
const REGISTER_WIDTH_AVX2: usize = 32;

impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
    pub(super) fn load_from(reader: &mut impl Read) -> io::Result<Self> {
        let mut weights = Wrapped([[0i8; IN]; OUT]);
        let mut biases = [0i32; OUT];
        reader.read_exact(bytemuck::bytes_of_mut(&mut weights))?;
        reader.read_exact(bytemuck::cast_slice_mut(&mut biases))?;

        Ok(Self {
            weights: weights.0.into(),
            biases: biases.map(i32::from_le).into(),
        })
    }

    #[inline]
    pub fn propagate(&self, input: &SimdAlign<[i8; IN]>, output: &mut SimdAlign<[i32; OUT]>) {
        #[cfg(target_feature = "avx2")]
        if Self::can_use_avx2() {
            self.propagate_avx2(input, output);
            return;
        }

        *output = self.biases;
        for (value, row) in output.iter_mut().zip(self.weights.iter()) {
            *value += dot(row, input);
        }
        for out in output.iter_mut() {
            *out /= WEIGHT_SCALE;
        }
    }

    #[inline(always)]
    #[cfg(target_feature = "avx2")]
    const fn can_use_avx2() -> bool {
        IN % REGISTER_WIDTH_AVX2 == 0 && OUT % 4 == 0
    }

    #[inline(always)]
    #[cfg(target_feature = "avx2")]
    fn propagate_avx2(&self, input: &SimdAlign<[i8; IN]>, output: &mut SimdAlign<[i32; OUT]>) {
        use core::arch::x86_64::{
            __m128i, __m256i, _mm_add_epi32, _mm_load_si128, _mm_srai_epi32, _mm_store_si128,
            _mm256_add_epi32, _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_hadd_epi32,
            _mm256_load_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi16,
            _mm256_setzero_si256,
        };

        unsafe {
            for (out_chunk, (weight_rows, bias_chunk)) in output
                .chunks_mut(4)
                .zip(self.weights.chunks(4).zip(self.biases.chunks(4)))
            {
                let mut sums = [_mm256_setzero_si256(); 4];
                for (m, in_chunk) in input.chunks(REGISTER_WIDTH_AVX2).enumerate() {
                    let in_chunk = _mm256_load_si256(in_chunk.as_ptr() as *const __m256i);
                    for (n, sum) in sums.iter_mut().enumerate() {
                        let weights = weight_rows
                            .get_unchecked(n)
                            .get_unchecked(m * REGISTER_WIDTH_AVX2..)
                            .as_ptr();

                        *sum = m256_add_dpbusd_epi32(
                            *sum,
                            in_chunk,
                            _mm256_load_si256(weights as *const __m256i),
                        );
                    }
                    let bias_chunk = _mm_load_si128(bias_chunk.as_ptr() as *const __m128i);
                    let output = m256_haddx4(sums[0], sums[1], sums[2], sums[3], bias_chunk);
                    let output = _mm_srai_epi32(output, WEIGHT_SCALE_SHIFT);
                    _mm_store_si128(out_chunk.as_mut_ptr() as *mut __m128i, output);
                }
            }
        }

        unsafe fn m256_add_dpbusd_epi32(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
            unsafe {
                let product0 = _mm256_maddubs_epi16(a, b);
                let one = _mm256_set1_epi16(1);
                let product0 = _mm256_madd_epi16(product0, one);

                _mm256_add_epi32(acc, product0)
            }
        }
        unsafe fn m256_haddx4(
            mut sum0: __m256i,
            sum1: __m256i,
            mut sum2: __m256i,
            sum3: __m256i,
            bias: __m128i,
        ) -> __m128i {
            unsafe {
                sum0 = _mm256_hadd_epi32(sum0, sum1);
                sum2 = _mm256_hadd_epi32(sum2, sum3);

                sum0 = _mm256_hadd_epi32(sum0, sum2);

                let sum128lo = _mm256_castsi256_si128(sum0);
                let sum128hi = _mm256_extracti128_si256(sum0, 1);

                _mm_add_epi32(_mm_add_epi32(sum128lo, sum128hi), bias)
            }
        }
    }
}

impl<const IN: usize> LinearOut<IN> {
    pub(super) fn load_from(reader: &mut impl Read) -> io::Result<Self> {
        let mut weights = [0i8; IN];
        let mut bias = 0;
        reader.read_exact(bytemuck::cast_slice_mut(&mut weights))?;
        reader.read_exact(bytemuck::bytes_of_mut(&mut bias))?;

        Ok(Self {
            weights: weights.into(),
            bias: i32::from_le(bias),
        })
    }

    #[inline]
    pub fn propagate(&self, input: &SimdAlign<[i8; IN]>) -> i32 {
        (self.bias + dot(&self.weights, input)) / OUTPUT_WEIGHT_SCALE
    }
}

#[inline]
pub fn cat<const A: usize, const B: usize, const C: usize>(
    a: &[i16; A],
    b: &[i16; B],
    output: &mut [i16; C],
) {
    output[..A].copy_from_slice(a);
    output[A..].copy_from_slice(b);
}

#[inline]
pub fn clipped_relu16<const SIZE: usize>(
    input: &SimdAlign<[i16; SIZE]>,
    output: &mut SimdAlign<[i8; SIZE]>,
) {
    #[cfg(target_feature = "avx2")]
    if SIZE % REGISTER_WIDTH_AVX2 == 0 {
        clipped_relu16_avx2(input, output);
        return;
    }

    for (output, &input) in output.iter_mut().zip(input.iter()) {
        *output = input.clamp(0, ACTIVATION_RANGE as i16) as i8;
    }
}

#[inline(always)]
#[cfg(target_feature = "avx2")]
pub fn clipped_relu16_avx2<const SIZE: usize>(
    input: &SimdAlign<[i16; SIZE]>,
    output: &mut SimdAlign<[i8; SIZE]>,
) {
    use core::arch::x86_64::{
        __m256i, _mm256_load_si256, _mm256_max_epi8, _mm256_packs_epi16, _mm256_permute4x64_epi64,
        _mm256_setzero_si256, _mm256_store_si256,
    };

    const INPUT_REGISTER_SIZE: usize = REGISTER_WIDTH_AVX2 / 2;
    const OUTPUT_REGISTER_SIZE: usize = REGISTER_WIDTH_AVX2;
    const CONTROL: i32 = 0b11011000;

    unsafe {
        let zero = _mm256_setzero_si256();
        for (out_chunk, in_chunk) in output
            .chunks_mut(OUTPUT_REGISTER_SIZE)
            .zip(input.chunks(2 * INPUT_REGISTER_SIZE))
        {
            let in_chunk1 =
                _mm256_load_si256(in_chunk.get_unchecked(0..).as_ptr() as *const __m256i);
            let in_chunk2 = _mm256_load_si256(
                in_chunk.get_unchecked(INPUT_REGISTER_SIZE..).as_ptr() as *const __m256i,
            );
            let result = _mm256_permute4x64_epi64(
                _mm256_max_epi8(_mm256_packs_epi16(in_chunk1, in_chunk2), zero),
                CONTROL,
            );
            _mm256_store_si256(out_chunk.as_mut_ptr() as *mut __m256i, result);
        }
    }
}

#[inline]
pub fn clipped_relu<const SIZE: usize>(
    input: &SimdAlign<[i32; SIZE]>,
    output: &mut SimdAlign<[i8; SIZE]>,
) {
    #[cfg(target_feature = "avx2")]
    if SIZE % REGISTER_WIDTH_AVX2 == 0 {
        clipped_relu_avx2(input, output);
        return;
    }

    for (output, &input) in output.iter_mut().zip(input.iter()) {
        *output = input.clamp(0, ACTIVATION_RANGE) as i8;
    }
}

#[inline(always)]
#[cfg(target_feature = "avx2")]
pub fn clipped_relu_avx2<const SIZE: usize>(
    input: &SimdAlign<[i32; SIZE]>,
    output: &mut SimdAlign<[i8; SIZE]>,
) {
    const INPUT_REGISTER_SIZE: usize = REGISTER_WIDTH_AVX2 / 4;
    const OUTPUT_REGISTER_SIZE: usize = REGISTER_WIDTH_AVX2;

    use core::arch::x86_64::{
        __m256i, _mm256_load_si256, _mm256_max_epi8, _mm256_packs_epi16, _mm256_packs_epi32,
        _mm256_permutevar8x32_epi32, _mm256_set_epi32, _mm256_setzero_si256, _mm256_store_si256,
    };

    unsafe {
        let zero = _mm256_setzero_si256();
        let control = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
        for (out_chunk, in_chunk) in output
            .chunks_mut(OUTPUT_REGISTER_SIZE)
            .zip(input.chunks(4 * INPUT_REGISTER_SIZE))
        {
            let step = INPUT_REGISTER_SIZE;
            let in_chunk1 = _mm256_packs_epi32(
                _mm256_load_si256(in_chunk.get_unchecked(0..).as_ptr() as *mut __m256i),
                _mm256_load_si256(in_chunk.get_unchecked(step..).as_ptr() as *mut __m256i),
            );
            let in_chunk2 = _mm256_packs_epi32(
                _mm256_load_si256(in_chunk.get_unchecked(2 * step..).as_ptr() as *mut __m256i),
                _mm256_load_si256(in_chunk.get_unchecked(3 * step..).as_ptr() as *mut __m256i),
            );
            let result = _mm256_permutevar8x32_epi32(
                _mm256_max_epi8(_mm256_packs_epi16(in_chunk1, in_chunk2), zero),
                control,
            );
            _mm256_store_si256(out_chunk.as_ptr() as *mut __m256i, result);
        }
    }
}

#[inline]
fn dot<const SIZE: usize>(a: &[i8; SIZE], b: &[i8; SIZE]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(&a, &b)| a as i32 * b as i32)
        .sum()
}

#[repr(transparent)]
#[derive(Copy, Clone)]
struct Wrapped<T, const N1: usize, const N2: usize>([[T; N1]; N2]);

unsafe impl<T, const N1: usize, const N2: usize> bytemuck::Zeroable for Wrapped<T, N1, N2> where
    T: bytemuck::Zeroable
{
}

unsafe impl<T, const N1: usize, const N2: usize> bytemuck::Pod for Wrapped<T, N1, N2> where
    T: bytemuck::Pod
{
}
