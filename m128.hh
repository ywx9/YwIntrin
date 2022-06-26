/// @file  ywintrin/m128.hh
/// @brief header for extended-vector with 128-bits without fp16
/// Copyright (c) Yw Ninefold @ Ywx9
/// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace yw {
using int1 = signed char;
using int2 = signed short;
using int4 = signed int;
using int8 = signed long long;
using nat1 = nat1;
using nat2 = unsigned short;
using nat4 = nat4;
using nat8 = unsigned long long;
using fat4 = float;
using fat8 = double;
}

#ifdef __yw__
#include <immintrin.h>
namespace yw::intrin::inline m128 {
using m128d = __m128d;
using m128f = __m128;
using m128i = __m128i;
using mmask8 = __mmask8;
using mmask16 = __mmask16;
using MANTISSA_NORM_ENUM = _MM_MANTISSA_NORM_ENUM;
inline m128i abs_epi16(m128i a) noexcept { return _mm_abs_epi16(a); }
inline m128i mask_abs_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_abs_epi16(src, k, a); }
inline m128i maskz_abs_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_abs_epi16(k, a); }
inline m128i abs_epi32(m128i a) noexcept { return _mm_abs_epi32(a); }
inline m128i mask_abs_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_abs_epi32(src, k, a); }
inline m128i maskz_abs_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_abs_epi32(k, a); }
inline m128i abs_epi64(m128i a) noexcept { return _mm_abs_epi64(a); }
inline m128i mask_abs_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_abs_epi64(src, k, a); }
inline m128i maskz_abs_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_abs_epi64(k, a); }
inline m128i abs_epi8(m128i a) noexcept { return _mm_abs_epi8(a); }
inline m128i mask_abs_epi8(m128i src, mmask16 k, m128i a) noexcept { return _mm_mask_abs_epi8(src, k, a); }
inline m128i maskz_abs_epi8(mmask16 k, m128i a) noexcept { return _mm_maskz_abs_epi8(k, a); }
inline m128i add_epi16(m128i a, m128i b) noexcept { return _mm_add_epi16(a, b); }
inline m128i mask_add_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_add_epi16(src, k, a, b); }
inline m128i maskz_add_epi16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_add_epi16(k, a, b); }
inline m128i add_epi32(m128i a, m128i b) noexcept { return _mm_add_epi32(a, b); }
inline m128i mask_add_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_add_epi32(src, k, a, b); }
inline m128i maskz_add_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_add_epi32(k, a, b); }
inline m128i add_epi64(m128i a, m128i b) noexcept { return _mm_add_epi64(a, b); }
inline m128i mask_add_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_add_epi64(src, k, a, b); }
inline m128i maskz_add_epi64(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_add_epi64(k, a, b); }
inline m128i add_epi8(m128i a, m128i b) noexcept { return _mm_add_epi8(a, b); }
inline m128i mask_add_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_add_epi8(src, k, a, b); }
inline m128i maskz_add_epi8(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_add_epi8(k, a, b); }
inline m128d add_pd(m128d a, m128d b) noexcept { return _mm_add_pd(a, b); }
inline m128d mask_add_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_add_pd(src, k, a, b); }
inline m128d maskz_add_pd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_add_pd(k, a, b); }
inline m128f add_ps(m128f a, m128f b) noexcept { return _mm_add_ps(a, b); }
inline m128f mask_add_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_add_ps(src, k, a, b); }
inline m128f maskz_add_ps(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_add_ps(k, a, b); }
template<int4 rounding> inline m128d add_round_sd(m128d a, m128d b) noexcept { return _mm_add_round_sd(a, b, rounding); }
template<int4 rounding> inline m128d mask_add_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_add_round_sd(src, k, a, b, rounding); }
template<int4 rounding> inline m128d maskz_add_round_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_add_round_sd(k, a, b, rounding); }
template<int4 rounding> inline m128f add_round_ss(m128f a, m128f b) noexcept { return _mm_add_round_ss(a, b, rounding); }
template<int4 rounding> inline m128f mask_add_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_add_round_ss(src, k, a, b, rounding); }
template<int4 rounding> inline m128f maskz_add_round_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_add_round_ss(k, a, b, rounding); }
inline m128d add_sd(m128d a, m128d b) noexcept { return _mm_add_sd(a, b); }
inline m128d mask_add_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_add_sd(src, k, a, b); }
inline m128d maskz_add_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_add_sd(k, a, b); }
inline m128f add_ss(m128f a, m128f b) noexcept { return _mm_add_ss(a, b); }
inline m128f mask_add_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_add_ss(src, k, a, b); }
inline m128f maskz_add_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_add_ss(k, a, b); }
inline m128i adds_epi16(m128i a, m128i b) noexcept { return _mm_adds_epi16(a, b); }
inline m128i mask_adds_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_adds_epi16(src, k, a, b); }
inline m128i maskz_adds_epi16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_adds_epi16(k, a, b); }
inline m128i adds_epi8(m128i a, m128i b) noexcept { return _mm_adds_epi8(a, b); }
inline m128i mask_adds_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_adds_epi8(src, k, a, b); }
inline m128i maskz_adds_epi8(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_adds_epi8(k, a, b); }
inline m128i adds_epu16(m128i a, m128i b) noexcept { return _mm_adds_epu16(a, b); }
inline m128i mask_adds_epu16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_adds_epu16(src, k, a, b); }
inline m128i maskz_adds_epu16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_adds_epu16(k, a, b); }
inline m128i adds_epu8(m128i a, m128i b) noexcept { return _mm_adds_epu8(a, b); }
inline m128i mask_adds_epu8(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_adds_epu8(src, k, a, b); }
inline m128i maskz_adds_epu8(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_adds_epu8(k, a, b); }
inline m128d addsub_pd(m128d a, m128d b) noexcept { return _mm_addsub_pd(a, b); }
inline m128f addsub_ps(m128f a, m128f b) noexcept { return _mm_addsub_ps(a, b); }
inline m128i aesdec_si128(m128i a, m128i RoundKey) noexcept { return _mm_aesdec_si128(a, RoundKey); }
inline nat1 aesdec128kl_u8(m128i* __odata, m128i __idata, const void* __h) noexcept { return _mm_aesdec128kl_u8(__odata, __idata, __h); }
inline nat1 aesdec256kl_u8(m128i* __odata, m128i __idata, const void* __h) noexcept { return _mm_aesdec256kl_u8(__odata, __idata, __h); }
inline m128i aesdeclast_si128(m128i a, m128i RoundKey) noexcept { return _mm_aesdeclast_si128(a, RoundKey); }
inline nat1 aesdecwide128kl_u8(m128i* __odata, const m128i* __idata, const void* __h) noexcept { return _mm_aesdecwide128kl_u8(__odata, __idata, __h); }
inline nat1 aesdecwide256kl_u8(m128i* __odata, const m128i* __idata, const void* __h) noexcept { return _mm_aesdecwide256kl_u8(__odata, __idata, __h); }
inline m128i aesenc_si128(m128i a, m128i RoundKey) noexcept { return _mm_aesenc_si128(a, RoundKey); }
inline nat1 aesenc128kl_u8(m128i* __odata, m128i __idata, const void* __h) noexcept { return _mm_aesenc128kl_u8(__odata, __idata, __h); }
inline nat1 aesenc256kl_u8(m128i* __odata, m128i __idata, const void* __h) noexcept { return _mm_aesenc256kl_u8(__odata, __idata, __h); }
inline m128i aesenclast_si128(m128i a, m128i RoundKey) noexcept { return _mm_aesenclast_si128(a, RoundKey); }
inline nat1 aesencwide128kl_u8(m128i* __odata, const m128i* __idata, const void* __h) noexcept { return _mm_aesencwide128kl_u8(__odata, __idata, __h); }
inline nat1 aesencwide256kl_u8(m128i* __odata, const m128i* __idata, const void* __h) noexcept { return _mm_aesencwide256kl_u8(__odata, __idata, __h); }
inline m128i aesimc_si128(m128i a) noexcept { return _mm_aesimc_si128(a); }
template<int4 imm8> inline m128i aeskeygenassist_si128(m128i a) noexcept { return _mm_aeskeygenassist_si128(a, imm8); }
template<int4 imm8> inline m128i alignr_epi32(m128i a, m128i b) noexcept { return _mm_alignr_epi32(a, b, imm8); }
template<int4 imm8> inline m128i mask_alignr_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_alignr_epi32(src, k, a, b, imm8); }
template<int4 imm8> inline m128i maskz_alignr_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_alignr_epi32(k, a, b, imm8); }
template<int4 imm8> inline m128i alignr_epi64(m128i a, m128i b) noexcept { return _mm_alignr_epi64(a, b, imm8); }
template<int4 imm8> inline m128i mask_alignr_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_alignr_epi64(src, k, a, b, imm8); }
template<int4 imm8> inline m128i maskz_alignr_epi64(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_alignr_epi64(k, a, b, imm8); }
template<int4 imm8> inline m128i alignr_epi8(m128i a, m128i b) noexcept { return _mm_alignr_epi8(a, b, imm8); }
template<int4 imm8> inline m128i mask_alignr_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_alignr_epi8(src, k, a, b, imm8); }
template<int4 imm8> inline m128i maskz_alignr_epi8(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_alignr_epi8(k, a, b, imm8); }
inline m128i mask_and_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_and_epi32(src, k, a, b); }
inline m128i maskz_and_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_and_epi32(k, a, b); }
inline m128i mask_and_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_and_epi64(src, k, a, b); }
inline m128i maskz_and_epi64(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_and_epi64(k, a, b); }
inline m128d and_pd(m128d a, m128d b) noexcept { return _mm_and_pd(a, b); }
inline m128d mask_and_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_and_pd(src, k, a, b); }
inline m128d maskz_and_pd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_and_pd(k, a, b); }
inline m128f and_ps(m128f a, m128f b) noexcept { return _mm_and_ps(a, b); }
inline m128f mask_and_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_and_ps(src, k, a, b); }
inline m128f maskz_and_ps(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_and_ps(k, a, b); }
inline m128i and_si128(m128i a, m128i b) noexcept { return _mm_and_si128(a, b); }
inline m128i mask_andnot_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_andnot_epi32(src, k, a, b); }
inline m128i maskz_andnot_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_andnot_epi32(k, a, b); }
inline m128i mask_andnot_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_andnot_epi64(src, k, a, b); }
inline m128i maskz_andnot_epi64(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_andnot_epi64(k, a, b); }
inline m128d andnot_pd(m128d a, m128d b) noexcept { return _mm_andnot_pd(a, b); }
inline m128d mask_andnot_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_andnot_pd(src, k, a, b); }
inline m128d maskz_andnot_pd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_andnot_pd(k, a, b); }
inline m128f andnot_ps(m128f a, m128f b) noexcept { return _mm_andnot_ps(a, b); }
inline m128f mask_andnot_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_andnot_ps(src, k, a, b); }
inline m128f maskz_andnot_ps(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_andnot_ps(k, a, b); }
inline m128i andnot_si128(m128i a, m128i b) noexcept { return _mm_andnot_si128(a, b); }
inline m128i avg_epu16(m128i a, m128i b) noexcept { return _mm_avg_epu16(a, b); }
inline m128i mask_avg_epu16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_avg_epu16(src, k, a, b); }
inline m128i maskz_avg_epu16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_avg_epu16(k, a, b); }
inline m128i avg_epu8(m128i a, m128i b) noexcept { return _mm_avg_epu8(a, b); }
inline m128i mask_avg_epu8(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_avg_epu8(src, k, a, b); }
inline m128i maskz_avg_epu8(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_avg_epu8(k, a, b); }
inline mmask16 bitshuffle_epi64_mask(m128i b, m128i c) noexcept { return _mm_bitshuffle_epi64_mask(b, c); }
inline mmask16 mask_bitshuffle_epi64_mask(mmask16 k, m128i b, m128i c) noexcept { return _mm_mask_bitshuffle_epi64_mask(k, b, c); }
template<int4 imm8> inline m128i blend_epi16(m128i a, m128i b) noexcept { return _mm_blend_epi16(a, b, imm8); }
inline m128i mask_blend_epi16(mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_blend_epi16(k, a, b); }
template<int4 imm8> inline m128i blend_epi32(m128i a, m128i b) noexcept { return _mm_blend_epi32(a, b, imm8); }
inline m128i mask_blend_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_blend_epi32(k, a, b); }
inline m128i mask_blend_epi64(mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_blend_epi64(k, a, b); }
inline m128i mask_blend_epi8(mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_blend_epi8(k, a, b); }
template<int4 imm8> inline m128d blend_pd(m128d a, m128d b) noexcept { return _mm_blend_pd(a, b, imm8); }
inline m128d mask_blend_pd(mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_blend_pd(k, a, b); }
template<int4 imm8> inline m128f blend_ps(m128f a, m128f b) noexcept { return _mm_blend_ps(a, b, imm8); }
inline m128f mask_blend_ps(mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_blend_ps(k, a, b); }
inline m128i blendv_epi8(m128i a, m128i b, m128i mask) noexcept { return _mm_blendv_epi8(a, b, mask); }
inline m128d blendv_pd(m128d a, m128d b, m128d mask) noexcept { return _mm_blendv_pd(a, b, mask); }
inline m128f blendv_ps(m128f a, m128f b, m128f mask) noexcept { return _mm_blendv_ps(a, b, mask); }
inline m128i broadcast_i32x2(m128i a) noexcept { return _mm_broadcast_i32x2(a); }
inline m128i mask_broadcast_i32x2(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_broadcast_i32x2(src, k, a); }
inline m128i maskz_broadcast_i32x2(mmask8 k, m128i a) noexcept { return _mm_maskz_broadcast_i32x2(k, a); }
inline m128f broadcast_ss(float const* mem_addr) noexcept { return _mm_broadcast_ss(mem_addr); }
inline m128i broadcastb_epi8(m128i a) noexcept { return _mm_broadcastb_epi8(a); }
inline m128i mask_broadcastb_epi8(m128i src, mmask16 k, m128i a) noexcept { return _mm_mask_broadcastb_epi8(src, k, a); }
inline m128i maskz_broadcastb_epi8(mmask16 k, m128i a) noexcept { return _mm_maskz_broadcastb_epi8(k, a); }
inline m128i broadcastd_epi32(m128i a) noexcept { return _mm_broadcastd_epi32(a); }
inline m128i mask_broadcastd_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_broadcastd_epi32(src, k, a); }
inline m128i maskz_broadcastd_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_broadcastd_epi32(k, a); }
inline m128i broadcastmb_epi64(mmask8 k) noexcept { return _mm_broadcastmb_epi64(k); }
inline m128i broadcastmw_epi32(mmask16 k) noexcept { return _mm_broadcastmw_epi32(k); }
inline m128i broadcastq_epi64(m128i a) noexcept { return _mm_broadcastq_epi64(a); }
inline m128i mask_broadcastq_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_broadcastq_epi64(src, k, a); }
inline m128i maskz_broadcastq_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_broadcastq_epi64(k, a); }
inline m128d broadcastsd_pd(m128d a) noexcept { return _mm_broadcastsd_pd(a); }
inline m128f broadcastss_ps(m128f a) noexcept { return _mm_broadcastss_ps(a); }
inline m128f mask_broadcastss_ps(m128f src, mmask8 k, m128f a) noexcept { return _mm_mask_broadcastss_ps(src, k, a); }
inline m128f maskz_broadcastss_ps(mmask8 k, m128f a) noexcept { return _mm_maskz_broadcastss_ps(k, a); }
inline m128i broadcastw_epi16(m128i a) noexcept { return _mm_broadcastw_epi16(a); }
inline m128i mask_broadcastw_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_broadcastw_epi16(src, k, a); }
inline m128i maskz_broadcastw_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_broadcastw_epi16(k, a); }
template<int4 imm8> inline m128i bslli_si128(m128i a) noexcept { return _mm_bslli_si128(a, imm8); }
template<int4 imm8> inline m128i bsrli_si128(m128i a) noexcept { return _mm_bsrli_si128(a, imm8); }
inline m128f castpd_ps(m128d a) noexcept { return _mm_castpd_ps(a); }
inline m128i castpd_si128(m128d a) noexcept { return _mm_castpd_si128(a); }
inline m128d castps_pd(m128f a) noexcept { return _mm_castps_pd(a); }
inline m128i castps_si128(m128f a) noexcept { return _mm_castps_si128(a); }
inline m128d castsi128_pd(m128i a) noexcept { return _mm_castsi128_pd(a); }
inline m128f castsi128_ps(m128i a) noexcept { return _mm_castsi128_ps(a); }
inline m128d ceil_pd(m128d a) noexcept { return _mm_ceil_pd(a); }
inline m128f ceil_ps(m128f a) noexcept { return _mm_ceil_ps(a); }
inline m128d ceil_sd(m128d a, m128d b) noexcept { return _mm_ceil_sd(a, b); }
inline m128f ceil_ss(m128f a, m128f b) noexcept { return _mm_ceil_ss(a, b); }
template<int4 imm8> inline m128i clmulepi64_si128(m128i a, m128i b) noexcept { return _mm_clmulepi64_si128(a, b, imm8); }
template<int4 imm8> inline mmask8 cmp_epi16_mask(m128i a, m128i b) noexcept { return _mm_cmp_epi16_mask(a, b, imm8); }
template<int4 imm8> inline mmask8 mask_cmp_epi16_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmp_epi16_mask(k1, a, b, imm8); }
template<int4 imm8> inline mmask8 cmp_epi32_mask(m128i a, m128i b) noexcept { return _mm_cmp_epi32_mask(a, b, imm8); }
template<int4 imm8> inline mmask8 mask_cmp_epi32_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmp_epi32_mask(k1, a, b, imm8); }
template<int4 imm8> inline mmask8 cmp_epi64_mask(m128i a, m128i b) noexcept { return _mm_cmp_epi64_mask(a, b, imm8); }
template<int4 imm8> inline mmask8 mask_cmp_epi64_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmp_epi64_mask(k1, a, b, imm8); }
template<int4 imm8> inline mmask16 cmp_epi8_mask(m128i a, m128i b) noexcept { return _mm_cmp_epi8_mask(a, b, imm8); }
template<int4 imm8> inline mmask16 mask_cmp_epi8_mask(mmask16 k1, m128i a, m128i b) noexcept { return _mm_mask_cmp_epi8_mask(k1, a, b, imm8); }
template<int4 imm8> inline mmask8 cmp_epu16_mask(m128i a, m128i b) noexcept { return _mm_cmp_epu16_mask(a, b, imm8); }
template<int4 imm8> inline mmask8 mask_cmp_epu16_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmp_epu16_mask(k1, a, b, imm8); }
template<int4 imm8> inline mmask8 cmp_epu32_mask(m128i a, m128i b) noexcept { return _mm_cmp_epu32_mask(a, b, imm8); }
template<int4 imm8> inline mmask8 mask_cmp_epu32_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmp_epu32_mask(k1, a, b, imm8); }
template<int4 imm8> inline mmask8 cmp_epu64_mask(m128i a, m128i b) noexcept { return _mm_cmp_epu64_mask(a, b, imm8); }
template<int4 imm8> inline mmask8 mask_cmp_epu64_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmp_epu64_mask(k1, a, b, imm8); }
template<int4 imm8> inline mmask16 cmp_epu8_mask(m128i a, m128i b) noexcept { return _mm_cmp_epu8_mask(a, b, imm8); }
template<int4 imm8> inline mmask16 mask_cmp_epu8_mask(mmask16 k1, m128i a, m128i b) noexcept { return _mm_mask_cmp_epu8_mask(k1, a, b, imm8); }
template<int4 imm8> inline m128d cmp_pd(m128d a, m128d b) noexcept { return _mm_cmp_pd(a, b, imm8); }
template<int4 imm8> inline mmask8 cmp_pd_mask(m128d a, m128d b) noexcept { return _mm_cmp_pd_mask(a, b, imm8); }
template<int4 imm8> inline mmask8 mask_cmp_pd_mask(mmask8 k1, m128d a, m128d b) noexcept { return _mm_mask_cmp_pd_mask(k1, a, b, imm8); }
template<int4 imm8> inline m128f cmp_ps(m128f a, m128f b) noexcept { return _mm_cmp_ps(a, b, imm8); }
template<int4 imm8> inline mmask8 cmp_ps_mask(m128f a, m128f b) noexcept { return _mm_cmp_ps_mask(a, b, imm8); }
template<int4 imm8> inline mmask8 mask_cmp_ps_mask(mmask8 k1, m128f a, m128f b) noexcept { return _mm_mask_cmp_ps_mask(k1, a, b, imm8); }
template<int4 imm8, int sae> inline mmask8 cmp_round_sd_mask(m128d a, m128d b) noexcept { return _mm_cmp_round_sd_mask(a, b, imm8, sae); }
template<int4 imm8, int sae> inline mmask8 mask_cmp_round_sd_mask(mmask8 k1, m128d a, m128d b) noexcept { return _mm_mask_cmp_round_sd_mask(k1, a, b, imm8, sae); }
template<int4 imm8, int sae> inline mmask8 cmp_round_ss_mask(m128f a, m128f b) noexcept { return _mm_cmp_round_ss_mask(a, b, imm8, sae); }
template<int4 imm8, int sae> inline mmask8 mask_cmp_round_ss_mask(mmask8 k1, m128f a, m128f b) noexcept { return _mm_mask_cmp_round_ss_mask(k1, a, b, imm8, sae); }
template<int4 imm8> inline m128d cmp_sd(m128d a, m128d b) noexcept { return _mm_cmp_sd(a, b, imm8); }
template<int4 imm8> inline mmask8 cmp_sd_mask(m128d a, m128d b) noexcept { return _mm_cmp_sd_mask(a, b, imm8); }
template<int4 imm8> inline mmask8 mask_cmp_sd_mask(mmask8 k1, m128d a, m128d b) noexcept { return _mm_mask_cmp_sd_mask(k1, a, b, imm8); }
template<int4 imm8> inline m128f cmp_ss(m128f a, m128f b) noexcept { return _mm_cmp_ss(a, b, imm8); }
template<int4 imm8> inline mmask8 cmp_ss_mask(m128f a, m128f b) noexcept { return _mm_cmp_ss_mask(a, b, imm8); }
template<int4 imm8> inline mmask8 mask_cmp_ss_mask(mmask8 k1, m128f a, m128f b) noexcept { return _mm_mask_cmp_ss_mask(k1, a, b, imm8); }
inline m128i cmpeq_epi16(m128i a, m128i b) noexcept { return _mm_cmpeq_epi16(a, b); }
inline mmask8 cmpeq_epi16_mask(m128i a, m128i b) noexcept { return _mm_cmpeq_epi16_mask(a, b); }
inline mmask8 mask_cmpeq_epi16_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpeq_epi16_mask(k1, a, b); }
inline m128i cmpeq_epi32(m128i a, m128i b) noexcept { return _mm_cmpeq_epi32(a, b); }
inline mmask8 cmpeq_epi32_mask(m128i a, m128i b) noexcept { return _mm_cmpeq_epi32_mask(a, b); }
inline mmask8 mask_cmpeq_epi32_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpeq_epi32_mask(k1, a, b); }
inline m128i cmpeq_epi64(m128i a, m128i b) noexcept { return _mm_cmpeq_epi64(a, b); }
inline mmask8 cmpeq_epi64_mask(m128i a, m128i b) noexcept { return _mm_cmpeq_epi64_mask(a, b); }
inline mmask8 mask_cmpeq_epi64_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpeq_epi64_mask(k1, a, b); }
inline m128i cmpeq_epi8(m128i a, m128i b) noexcept { return _mm_cmpeq_epi8(a, b); }
inline mmask16 cmpeq_epi8_mask(m128i a, m128i b) noexcept { return _mm_cmpeq_epi8_mask(a, b); }
inline mmask16 mask_cmpeq_epi8_mask(mmask16 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpeq_epi8_mask(k1, a, b); }
inline mmask8 cmpeq_epu16_mask(m128i a, m128i b) noexcept { return _mm_cmpeq_epu16_mask(a, b); }
inline mmask8 mask_cmpeq_epu16_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpeq_epu16_mask(k1, a, b); }
inline mmask8 cmpeq_epu32_mask(m128i a, m128i b) noexcept { return _mm_cmpeq_epu32_mask(a, b); }
inline mmask8 mask_cmpeq_epu32_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpeq_epu32_mask(k1, a, b); }
inline mmask8 cmpeq_epu64_mask(m128i a, m128i b) noexcept { return _mm_cmpeq_epu64_mask(a, b); }
inline mmask8 mask_cmpeq_epu64_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpeq_epu64_mask(k1, a, b); }
inline mmask16 cmpeq_epu8_mask(m128i a, m128i b) noexcept { return _mm_cmpeq_epu8_mask(a, b); }
inline mmask16 mask_cmpeq_epu8_mask(mmask16 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpeq_epu8_mask(k1, a, b); }
inline m128d cmpeq_pd(m128d a, m128d b) noexcept { return _mm_cmpeq_pd(a, b); }
inline m128f cmpeq_ps(m128f a, m128f b) noexcept { return _mm_cmpeq_ps(a, b); }
inline m128d cmpeq_sd(m128d a, m128d b) noexcept { return _mm_cmpeq_sd(a, b); }
inline m128f cmpeq_ss(m128f a, m128f b) noexcept { return _mm_cmpeq_ss(a, b); }
template<int4 imm8> inline int cmpestra(m128i a, int la, m128i b, int lb) noexcept { return _mm_cmpestra(a, la, b, lb, imm8); }
template<int4 imm8> inline int cmpestrc(m128i a, int la, m128i b, int lb) noexcept { return _mm_cmpestrc(a, la, b, lb, imm8); }
template<int4 imm8> inline int cmpestri(m128i a, int la, m128i b, int lb) noexcept { return _mm_cmpestri(a, la, b, lb, imm8); }
template<int4 imm8> inline m128i cmpestrm(m128i a, int la, m128i b, int lb) noexcept { return _mm_cmpestrm(a, la, b, lb, imm8); }
template<int4 imm8> inline int cmpestro(m128i a, int la, m128i b, int lb) noexcept { return _mm_cmpestro(a, la, b, lb, imm8); }
template<int4 imm8> inline int cmpestrs(m128i a, int la, m128i b, int lb) noexcept { return _mm_cmpestrs(a, la, b, lb, imm8); }
template<int4 imm8> inline int cmpestrz(m128i a, int la, m128i b, int lb) noexcept { return _mm_cmpestrz(a, la, b, lb, imm8); }
inline mmask8 cmpge_epi16_mask(m128i a, m128i b) noexcept { return _mm_cmpge_epi16_mask(a, b); }
inline mmask8 mask_cmpge_epi16_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpge_epi16_mask(k1, a, b); }
inline mmask8 cmpge_epi32_mask(m128i a, m128i b) noexcept { return _mm_cmpge_epi32_mask(a, b); }
inline mmask8 mask_cmpge_epi32_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpge_epi32_mask(k1, a, b); }
inline mmask8 cmpge_epi64_mask(m128i a, m128i b) noexcept { return _mm_cmpge_epi64_mask(a, b); }
inline mmask8 mask_cmpge_epi64_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpge_epi64_mask(k1, a, b); }
inline mmask16 cmpge_epi8_mask(m128i a, m128i b) noexcept { return _mm_cmpge_epi8_mask(a, b); }
inline mmask16 mask_cmpge_epi8_mask(mmask16 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpge_epi8_mask(k1, a, b); }
inline mmask8 cmpge_epu16_mask(m128i a, m128i b) noexcept { return _mm_cmpge_epu16_mask(a, b); }
inline mmask8 mask_cmpge_epu16_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpge_epu16_mask(k1, a, b); }
inline mmask8 cmpge_epu32_mask(m128i a, m128i b) noexcept { return _mm_cmpge_epu32_mask(a, b); }
inline mmask8 mask_cmpge_epu32_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpge_epu32_mask(k1, a, b); }
inline mmask8 cmpge_epu64_mask(m128i a, m128i b) noexcept { return _mm_cmpge_epu64_mask(a, b); }
inline mmask8 mask_cmpge_epu64_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpge_epu64_mask(k1, a, b); }
inline mmask16 cmpge_epu8_mask(m128i a, m128i b) noexcept { return _mm_cmpge_epu8_mask(a, b); }
inline mmask16 mask_cmpge_epu8_mask(mmask16 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpge_epu8_mask(k1, a, b); }
inline m128d cmpge_pd(m128d a, m128d b) noexcept { return _mm_cmpge_pd(a, b); }
inline m128f cmpge_ps(m128f a, m128f b) noexcept { return _mm_cmpge_ps(a, b); }
inline m128d cmpge_sd(m128d a, m128d b) noexcept { return _mm_cmpge_sd(a, b); }
inline m128f cmpge_ss(m128f a, m128f b) noexcept { return _mm_cmpge_ss(a, b); }
inline m128i cmpgt_epi16(m128i a, m128i b) noexcept { return _mm_cmpgt_epi16(a, b); }
inline mmask8 cmpgt_epi16_mask(m128i a, m128i b) noexcept { return _mm_cmpgt_epi16_mask(a, b); }
inline mmask8 mask_cmpgt_epi16_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpgt_epi16_mask(k1, a, b); }
inline m128i cmpgt_epi32(m128i a, m128i b) noexcept { return _mm_cmpgt_epi32(a, b); }
inline mmask8 cmpgt_epi32_mask(m128i a, m128i b) noexcept { return _mm_cmpgt_epi32_mask(a, b); }
inline mmask8 mask_cmpgt_epi32_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpgt_epi32_mask(k1, a, b); }
inline m128i cmpgt_epi64(m128i a, m128i b) noexcept { return _mm_cmpgt_epi64(a, b); }
inline mmask8 cmpgt_epi64_mask(m128i a, m128i b) noexcept { return _mm_cmpgt_epi64_mask(a, b); }
inline mmask8 mask_cmpgt_epi64_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpgt_epi64_mask(k1, a, b); }
inline m128i cmpgt_epi8(m128i a, m128i b) noexcept { return _mm_cmpgt_epi8(a, b); }
inline mmask16 cmpgt_epi8_mask(m128i a, m128i b) noexcept { return _mm_cmpgt_epi8_mask(a, b); }
inline mmask16 mask_cmpgt_epi8_mask(mmask16 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpgt_epi8_mask(k1, a, b); }
inline mmask8 cmpgt_epu16_mask(m128i a, m128i b) noexcept { return _mm_cmpgt_epu16_mask(a, b); }
inline mmask8 mask_cmpgt_epu16_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpgt_epu16_mask(k1, a, b); }
inline mmask8 cmpgt_epu32_mask(m128i a, m128i b) noexcept { return _mm_cmpgt_epu32_mask(a, b); }
inline mmask8 mask_cmpgt_epu32_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpgt_epu32_mask(k1, a, b); }
inline mmask8 cmpgt_epu64_mask(m128i a, m128i b) noexcept { return _mm_cmpgt_epu64_mask(a, b); }
inline mmask8 mask_cmpgt_epu64_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpgt_epu64_mask(k1, a, b); }
inline mmask16 cmpgt_epu8_mask(m128i a, m128i b) noexcept { return _mm_cmpgt_epu8_mask(a, b); }
inline mmask16 mask_cmpgt_epu8_mask(mmask16 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpgt_epu8_mask(k1, a, b); }
inline m128d cmpgt_pd(m128d a, m128d b) noexcept { return _mm_cmpgt_pd(a, b); }
inline m128f cmpgt_ps(m128f a, m128f b) noexcept { return _mm_cmpgt_ps(a, b); }
inline m128d cmpgt_sd(m128d a, m128d b) noexcept { return _mm_cmpgt_sd(a, b); }
inline m128f cmpgt_ss(m128f a, m128f b) noexcept { return _mm_cmpgt_ss(a, b); }
template<int4 imm8> inline int cmpistra(m128i a, m128i b) noexcept { return _mm_cmpistra(a, b, imm8); }
template<int4 imm8> inline int cmpistrc(m128i a, m128i b) noexcept { return _mm_cmpistrc(a, b, imm8); }
template<int4 imm8> inline int cmpistri(m128i a, m128i b) noexcept { return _mm_cmpistri(a, b, imm8); }
template<int4 imm8> inline m128i cmpistrm(m128i a, m128i b) noexcept { return _mm_cmpistrm(a, b, imm8); }
template<int4 imm8> inline int cmpistro(m128i a, m128i b) noexcept { return _mm_cmpistro(a, b, imm8); }
template<int4 imm8> inline int cmpistrs(m128i a, m128i b) noexcept { return _mm_cmpistrs(a, b, imm8); }
template<int4 imm8> inline int cmpistrz(m128i a, m128i b) noexcept { return _mm_cmpistrz(a, b, imm8); }
inline mmask8 cmple_epi16_mask(m128i a, m128i b) noexcept { return _mm_cmple_epi16_mask(a, b); }
inline mmask8 mask_cmple_epi16_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmple_epi16_mask(k1, a, b); }
inline mmask8 cmple_epi32_mask(m128i a, m128i b) noexcept { return _mm_cmple_epi32_mask(a, b); }
inline mmask8 mask_cmple_epi32_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmple_epi32_mask(k1, a, b); }
inline mmask8 cmple_epi64_mask(m128i a, m128i b) noexcept { return _mm_cmple_epi64_mask(a, b); }
inline mmask8 mask_cmple_epi64_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmple_epi64_mask(k1, a, b); }
inline mmask16 cmple_epi8_mask(m128i a, m128i b) noexcept { return _mm_cmple_epi8_mask(a, b); }
inline mmask16 mask_cmple_epi8_mask(mmask16 k1, m128i a, m128i b) noexcept { return _mm_mask_cmple_epi8_mask(k1, a, b); }
inline mmask8 cmple_epu16_mask(m128i a, m128i b) noexcept { return _mm_cmple_epu16_mask(a, b); }
inline mmask8 mask_cmple_epu16_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmple_epu16_mask(k1, a, b); }
inline mmask8 cmple_epu32_mask(m128i a, m128i b) noexcept { return _mm_cmple_epu32_mask(a, b); }
inline mmask8 mask_cmple_epu32_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmple_epu32_mask(k1, a, b); }
inline mmask8 cmple_epu64_mask(m128i a, m128i b) noexcept { return _mm_cmple_epu64_mask(a, b); }
inline mmask8 mask_cmple_epu64_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmple_epu64_mask(k1, a, b); }
inline mmask16 cmple_epu8_mask(m128i a, m128i b) noexcept { return _mm_cmple_epu8_mask(a, b); }
inline mmask16 mask_cmple_epu8_mask(mmask16 k1, m128i a, m128i b) noexcept { return _mm_mask_cmple_epu8_mask(k1, a, b); }
inline m128d cmple_pd(m128d a, m128d b) noexcept { return _mm_cmple_pd(a, b); }
inline m128f cmple_ps(m128f a, m128f b) noexcept { return _mm_cmple_ps(a, b); }
inline m128d cmple_sd(m128d a, m128d b) noexcept { return _mm_cmple_sd(a, b); }
inline m128f cmple_ss(m128f a, m128f b) noexcept { return _mm_cmple_ss(a, b); }
inline m128i cmplt_epi16(m128i a, m128i b) noexcept { return _mm_cmplt_epi16(a, b); }
inline mmask8 cmplt_epi16_mask(m128i a, m128i b) noexcept { return _mm_cmplt_epi16_mask(a, b); }
inline mmask8 mask_cmplt_epi16_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmplt_epi16_mask(k1, a, b); }
inline m128i cmplt_epi32(m128i a, m128i b) noexcept { return _mm_cmplt_epi32(a, b); }
inline mmask8 cmplt_epi32_mask(m128i a, m128i b) noexcept { return _mm_cmplt_epi32_mask(a, b); }
inline mmask8 mask_cmplt_epi32_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmplt_epi32_mask(k1, a, b); }
inline mmask8 cmplt_epi64_mask(m128i a, m128i b) noexcept { return _mm_cmplt_epi64_mask(a, b); }
inline mmask8 mask_cmplt_epi64_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmplt_epi64_mask(k1, a, b); }
inline m128i cmplt_epi8(m128i a, m128i b) noexcept { return _mm_cmplt_epi8(a, b); }
inline mmask16 cmplt_epi8_mask(m128i a, m128i b) noexcept { return _mm_cmplt_epi8_mask(a, b); }
inline mmask16 mask_cmplt_epi8_mask(mmask16 k1, m128i a, m128i b) noexcept { return _mm_mask_cmplt_epi8_mask(k1, a, b); }
inline mmask8 cmplt_epu16_mask(m128i a, m128i b) noexcept { return _mm_cmplt_epu16_mask(a, b); }
inline mmask8 mask_cmplt_epu16_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmplt_epu16_mask(k1, a, b); }
inline mmask8 cmplt_epu32_mask(m128i a, m128i b) noexcept { return _mm_cmplt_epu32_mask(a, b); }
inline mmask8 mask_cmplt_epu32_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmplt_epu32_mask(k1, a, b); }
inline mmask8 cmplt_epu64_mask(m128i a, m128i b) noexcept { return _mm_cmplt_epu64_mask(a, b); }
inline mmask8 mask_cmplt_epu64_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmplt_epu64_mask(k1, a, b); }
inline mmask16 cmplt_epu8_mask(m128i a, m128i b) noexcept { return _mm_cmplt_epu8_mask(a, b); }
inline mmask16 mask_cmplt_epu8_mask(mmask16 k1, m128i a, m128i b) noexcept { return _mm_mask_cmplt_epu8_mask(k1, a, b); }
inline m128d cmplt_pd(m128d a, m128d b) noexcept { return _mm_cmplt_pd(a, b); }
inline m128f cmplt_ps(m128f a, m128f b) noexcept { return _mm_cmplt_ps(a, b); }
inline m128d cmplt_sd(m128d a, m128d b) noexcept { return _mm_cmplt_sd(a, b); }
inline m128f cmplt_ss(m128f a, m128f b) noexcept { return _mm_cmplt_ss(a, b); }
inline mmask8 cmpneq_epi16_mask(m128i a, m128i b) noexcept { return _mm_cmpneq_epi16_mask(a, b); }
inline mmask8 mask_cmpneq_epi16_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpneq_epi16_mask(k1, a, b); }
inline mmask8 cmpneq_epi32_mask(m128i a, m128i b) noexcept { return _mm_cmpneq_epi32_mask(a, b); }
inline mmask8 mask_cmpneq_epi32_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpneq_epi32_mask(k1, a, b); }
inline mmask8 cmpneq_epi64_mask(m128i a, m128i b) noexcept { return _mm_cmpneq_epi64_mask(a, b); }
inline mmask8 mask_cmpneq_epi64_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpneq_epi64_mask(k1, a, b); }
inline mmask16 cmpneq_epi8_mask(m128i a, m128i b) noexcept { return _mm_cmpneq_epi8_mask(a, b); }
inline mmask16 mask_cmpneq_epi8_mask(mmask16 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpneq_epi8_mask(k1, a, b); }
inline mmask8 cmpneq_epu16_mask(m128i a, m128i b) noexcept { return _mm_cmpneq_epu16_mask(a, b); }
inline mmask8 mask_cmpneq_epu16_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpneq_epu16_mask(k1, a, b); }
inline mmask8 cmpneq_epu32_mask(m128i a, m128i b) noexcept { return _mm_cmpneq_epu32_mask(a, b); }
inline mmask8 mask_cmpneq_epu32_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpneq_epu32_mask(k1, a, b); }
inline mmask8 cmpneq_epu64_mask(m128i a, m128i b) noexcept { return _mm_cmpneq_epu64_mask(a, b); }
inline mmask8 mask_cmpneq_epu64_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpneq_epu64_mask(k1, a, b); }
inline mmask16 cmpneq_epu8_mask(m128i a, m128i b) noexcept { return _mm_cmpneq_epu8_mask(a, b); }
inline mmask16 mask_cmpneq_epu8_mask(mmask16 k1, m128i a, m128i b) noexcept { return _mm_mask_cmpneq_epu8_mask(k1, a, b); }
inline m128d cmpneq_pd(m128d a, m128d b) noexcept { return _mm_cmpneq_pd(a, b); }
inline m128f cmpneq_ps(m128f a, m128f b) noexcept { return _mm_cmpneq_ps(a, b); }
inline m128d cmpneq_sd(m128d a, m128d b) noexcept { return _mm_cmpneq_sd(a, b); }
inline m128f cmpneq_ss(m128f a, m128f b) noexcept { return _mm_cmpneq_ss(a, b); }
inline m128d cmpnge_pd(m128d a, m128d b) noexcept { return _mm_cmpnge_pd(a, b); }
inline m128f cmpnge_ps(m128f a, m128f b) noexcept { return _mm_cmpnge_ps(a, b); }
inline m128d cmpnge_sd(m128d a, m128d b) noexcept { return _mm_cmpnge_sd(a, b); }
inline m128f cmpnge_ss(m128f a, m128f b) noexcept { return _mm_cmpnge_ss(a, b); }
inline m128d cmpngt_pd(m128d a, m128d b) noexcept { return _mm_cmpngt_pd(a, b); }
inline m128f cmpngt_ps(m128f a, m128f b) noexcept { return _mm_cmpngt_ps(a, b); }
inline m128d cmpngt_sd(m128d a, m128d b) noexcept { return _mm_cmpngt_sd(a, b); }
inline m128f cmpngt_ss(m128f a, m128f b) noexcept { return _mm_cmpngt_ss(a, b); }
inline m128d cmpnle_pd(m128d a, m128d b) noexcept { return _mm_cmpnle_pd(a, b); }
inline m128f cmpnle_ps(m128f a, m128f b) noexcept { return _mm_cmpnle_ps(a, b); }
inline m128d cmpnle_sd(m128d a, m128d b) noexcept { return _mm_cmpnle_sd(a, b); }
inline m128f cmpnle_ss(m128f a, m128f b) noexcept { return _mm_cmpnle_ss(a, b); }
inline m128d cmpnlt_pd(m128d a, m128d b) noexcept { return _mm_cmpnlt_pd(a, b); }
inline m128f cmpnlt_ps(m128f a, m128f b) noexcept { return _mm_cmpnlt_ps(a, b); }
inline m128d cmpnlt_sd(m128d a, m128d b) noexcept { return _mm_cmpnlt_sd(a, b); }
inline m128f cmpnlt_ss(m128f a, m128f b) noexcept { return _mm_cmpnlt_ss(a, b); }
inline m128d cmpord_pd(m128d a, m128d b) noexcept { return _mm_cmpord_pd(a, b); }
inline m128f cmpord_ps(m128f a, m128f b) noexcept { return _mm_cmpord_ps(a, b); }
inline m128d cmpord_sd(m128d a, m128d b) noexcept { return _mm_cmpord_sd(a, b); }
inline m128f cmpord_ss(m128f a, m128f b) noexcept { return _mm_cmpord_ss(a, b); }
inline m128d cmpunord_pd(m128d a, m128d b) noexcept { return _mm_cmpunord_pd(a, b); }
inline m128f cmpunord_ps(m128f a, m128f b) noexcept { return _mm_cmpunord_ps(a, b); }
inline m128d cmpunord_sd(m128d a, m128d b) noexcept { return _mm_cmpunord_sd(a, b); }
inline m128f cmpunord_ss(m128f a, m128f b) noexcept { return _mm_cmpunord_ss(a, b); }
template<int4 imm8, int sae> inline int comi_round_sd(m128d a, m128d b) noexcept { return _mm_comi_round_sd(a, b, imm8, sae); }
template<int4 imm8, int sae> inline int comi_round_ss(m128f a, m128f b) noexcept { return _mm_comi_round_ss(a, b, imm8, sae); }
inline int comieq_sd(m128d a, m128d b) noexcept { return _mm_comieq_sd(a, b); }
inline int comieq_ss(m128f a, m128f b) noexcept { return _mm_comieq_ss(a, b); }
inline int comige_sd(m128d a, m128d b) noexcept { return _mm_comige_sd(a, b); }
inline int comige_ss(m128f a, m128f b) noexcept { return _mm_comige_ss(a, b); }
inline int comigt_sd(m128d a, m128d b) noexcept { return _mm_comigt_sd(a, b); }
inline int comigt_ss(m128f a, m128f b) noexcept { return _mm_comigt_ss(a, b); }
inline int comile_sd(m128d a, m128d b) noexcept { return _mm_comile_sd(a, b); }
inline int comile_ss(m128f a, m128f b) noexcept { return _mm_comile_ss(a, b); }
inline int comilt_sd(m128d a, m128d b) noexcept { return _mm_comilt_sd(a, b); }
inline int comilt_ss(m128f a, m128f b) noexcept { return _mm_comilt_ss(a, b); }
inline int comineq_sd(m128d a, m128d b) noexcept { return _mm_comineq_sd(a, b); }
inline int comineq_ss(m128f a, m128f b) noexcept { return _mm_comineq_ss(a, b); }
inline m128i mask_compress_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_compress_epi16(src, k, a); }
inline m128i maskz_compress_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_compress_epi16(k, a); }
inline m128i mask_compress_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_compress_epi32(src, k, a); }
inline m128i maskz_compress_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_compress_epi32(k, a); }
inline m128i mask_compress_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_compress_epi64(src, k, a); }
inline m128i maskz_compress_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_compress_epi64(k, a); }
inline m128i mask_compress_epi8(m128i src, mmask16 k, m128i a) noexcept { return _mm_mask_compress_epi8(src, k, a); }
inline m128i maskz_compress_epi8(mmask16 k, m128i a) noexcept { return _mm_maskz_compress_epi8(k, a); }
inline m128d mask_compress_pd(m128d src, mmask8 k, m128d a) noexcept { return _mm_mask_compress_pd(src, k, a); }
inline m128d maskz_compress_pd(mmask8 k, m128d a) noexcept { return _mm_maskz_compress_pd(k, a); }
inline m128f mask_compress_ps(m128f src, mmask8 k, m128f a) noexcept { return _mm_mask_compress_ps(src, k, a); }
inline m128f maskz_compress_ps(mmask8 k, m128f a) noexcept { return _mm_maskz_compress_ps(k, a); }
inline void mask_compressstoreu_epi16(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_compressstoreu_epi16(base_addr, k, a); }
inline void mask_compressstoreu_epi32(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_compressstoreu_epi32(base_addr, k, a); }
inline void mask_compressstoreu_epi64(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_compressstoreu_epi64(base_addr, k, a); }
inline void mask_compressstoreu_epi8(void* base_addr, mmask16 k, m128i a) noexcept { return _mm_mask_compressstoreu_epi8(base_addr, k, a); }
inline void mask_compressstoreu_pd(void* base_addr, mmask8 k, m128d a) noexcept { return _mm_mask_compressstoreu_pd(base_addr, k, a); }
inline void mask_compressstoreu_ps(void* base_addr, mmask8 k, m128f a) noexcept { return _mm_mask_compressstoreu_ps(base_addr, k, a); }
inline m128i conflict_epi32(m128i a) noexcept { return _mm_conflict_epi32(a); }
inline m128i mask_conflict_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_conflict_epi32(src, k, a); }
inline m128i maskz_conflict_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_conflict_epi32(k, a); }
inline m128i conflict_epi64(m128i a) noexcept { return _mm_conflict_epi64(a); }
inline m128i mask_conflict_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_conflict_epi64(src, k, a); }
inline m128i maskz_conflict_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_conflict_epi64(k, a); }
template<int4 rounding> inline m128f cvt_roundi32_ss(m128f a, int b) noexcept { return _mm_cvt_roundi32_ss(a, b, rounding); }
template<int4 rounding> inline m128d cvt_roundi64_sd(m128d a, int8 b) noexcept { return _mm_cvt_roundi64_sd(a, b, rounding); }
template<int4 rounding> inline m128f cvt_roundi64_ss(m128f a, int8 b) noexcept { return _mm_cvt_roundi64_ss(a, b, rounding); }
template<int4 rounding> inline int cvt_roundsd_i32(m128d a) noexcept { return _mm_cvt_roundsd_i32(a, rounding); }
template<int4 rounding> inline int8 cvt_roundsd_i64(m128d a) noexcept { return _mm_cvt_roundsd_i64(a, rounding); }
template<int4 rounding> inline int cvt_roundsd_si32(m128d a) noexcept { return _mm_cvt_roundsd_si32(a, rounding); }
template<int4 rounding> inline int8 cvt_roundsd_si64(m128d a) noexcept { return _mm_cvt_roundsd_si64(a, rounding); }
template<int4 rounding> inline m128f cvt_roundsd_ss(m128f a, m128d b) noexcept { return _mm_cvt_roundsd_ss(a, b, rounding); }
template<int4 rounding> inline m128f mask_cvt_roundsd_ss(m128f src, mmask8 k, m128f a, m128d b) noexcept { return _mm_mask_cvt_roundsd_ss(src, k, a, b, rounding); }
template<int4 rounding> inline m128f maskz_cvt_roundsd_ss(mmask8 k, m128f a, m128d b) noexcept { return _mm_maskz_cvt_roundsd_ss(k, a, b, rounding); }
template<int4 rounding> inline nat4 cvt_roundsd_u32(m128d a) noexcept { return _mm_cvt_roundsd_u32(a, rounding); }
template<int4 rounding> inline nat8 cvt_roundsd_u64(m128d a) noexcept { return _mm_cvt_roundsd_u64(a, rounding); }
template<int4 rounding> inline m128f cvt_roundsi32_ss(m128f a, int b) noexcept { return _mm_cvt_roundsi32_ss(a, b, rounding); }
template<int4 rounding> inline m128d cvt_roundsi64_sd(m128d a, int8 b) noexcept { return _mm_cvt_roundsi64_sd(a, b, rounding); }
template<int4 rounding> inline m128f cvt_roundsi64_ss(m128f a, int8 b) noexcept { return _mm_cvt_roundsi64_ss(a, b, rounding); }
template<int4 rounding> inline int cvt_roundss_i32(m128f a) noexcept { return _mm_cvt_roundss_i32(a, rounding); }
template<int4 rounding> inline int8 cvt_roundss_i64(m128f a) noexcept { return _mm_cvt_roundss_i64(a, rounding); }
template<int sae> inline m128d cvt_roundss_sd(m128d a, m128f b) noexcept { return _mm_cvt_roundss_sd(a, b, sae); }
template<int sae> inline m128d mask_cvt_roundss_sd(m128d src, mmask8 k, m128d a, m128f b) noexcept { return _mm_mask_cvt_roundss_sd(src, k, a, b, sae); }
template<int sae> inline m128d maskz_cvt_roundss_sd(mmask8 k, m128d a, m128f b) noexcept { return _mm_maskz_cvt_roundss_sd(k, a, b, sae); }
template<int4 rounding> inline int cvt_roundss_si32(m128f a) noexcept { return _mm_cvt_roundss_si32(a, rounding); }
template<int4 rounding> inline int8 cvt_roundss_si64(m128f a) noexcept { return _mm_cvt_roundss_si64(a, rounding); }
template<int4 rounding> inline nat4 cvt_roundss_u32(m128f a) noexcept { return _mm_cvt_roundss_u32(a, rounding); }
template<int4 rounding> inline nat8 cvt_roundss_u64(m128f a) noexcept { return _mm_cvt_roundss_u64(a, rounding); }
template<int4 rounding> inline m128f cvt_roundu32_ss(m128f a, nat4 b) noexcept { return _mm_cvt_roundu32_ss(a, b, rounding); }
template<int4 rounding> inline m128d cvt_roundu64_sd(m128d a, nat8 b) noexcept { return _mm_cvt_roundu64_sd(a, b, rounding); }
template<int4 rounding> inline m128f cvt_roundu64_ss(m128f a, nat8 b) noexcept { return _mm_cvt_roundu64_ss(a, b, rounding); }
inline m128f cvt_si2ss(m128f a, int b) noexcept { return _mm_cvt_si2ss(a, b); }
inline int cvt_ss2si(m128f a) noexcept { return _mm_cvt_ss2si(a); }
inline m128i cvtepi16_epi32(m128i a) noexcept { return _mm_cvtepi16_epi32(a); }
inline m128i mask_cvtepi16_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi16_epi32(src, k, a); }
inline m128i maskz_cvtepi16_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepi16_epi32(k, a); }
inline m128i cvtepi16_epi64(m128i a) noexcept { return _mm_cvtepi16_epi64(a); }
inline m128i mask_cvtepi16_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi16_epi64(src, k, a); }
inline m128i maskz_cvtepi16_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepi16_epi64(k, a); }
inline m128i cvtepi16_epi8(m128i a) noexcept { return _mm_cvtepi16_epi8(a); }
inline m128i mask_cvtepi16_epi8(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi16_epi8(src, k, a); }
inline m128i maskz_cvtepi16_epi8(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepi16_epi8(k, a); }
inline void mask_cvtepi16_storeu_epi8(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi16_storeu_epi8(base_addr, k, a); }
inline m128i cvtepi32_epi16(m128i a) noexcept { return _mm_cvtepi32_epi16(a); }
inline m128i mask_cvtepi32_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi32_epi16(src, k, a); }
inline m128i maskz_cvtepi32_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepi32_epi16(k, a); }
inline m128i cvtepi32_epi64(m128i a) noexcept { return _mm_cvtepi32_epi64(a); }
inline m128i mask_cvtepi32_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi32_epi64(src, k, a); }
inline m128i maskz_cvtepi32_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepi32_epi64(k, a); }
inline m128i cvtepi32_epi8(m128i a) noexcept { return _mm_cvtepi32_epi8(a); }
inline m128i mask_cvtepi32_epi8(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi32_epi8(src, k, a); }
inline m128i maskz_cvtepi32_epi8(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepi32_epi8(k, a); }
inline m128d cvtepi32_pd(m128i a) noexcept { return _mm_cvtepi32_pd(a); }
inline m128d mask_cvtepi32_pd(m128d src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi32_pd(src, k, a); }
inline m128d maskz_cvtepi32_pd(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepi32_pd(k, a); }
inline m128f cvtepi32_ps(m128i a) noexcept { return _mm_cvtepi32_ps(a); }
inline m128f mask_cvtepi32_ps(m128f src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi32_ps(src, k, a); }
inline m128f maskz_cvtepi32_ps(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepi32_ps(k, a); }
inline void mask_cvtepi32_storeu_epi16(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi32_storeu_epi16(base_addr, k, a); }
inline void mask_cvtepi32_storeu_epi8(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi32_storeu_epi8(base_addr, k, a); }
inline m128i cvtepi64_epi16(m128i a) noexcept { return _mm_cvtepi64_epi16(a); }
inline m128i mask_cvtepi64_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi64_epi16(src, k, a); }
inline m128i maskz_cvtepi64_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepi64_epi16(k, a); }
inline m128i cvtepi64_epi32(m128i a) noexcept { return _mm_cvtepi64_epi32(a); }
inline m128i mask_cvtepi64_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi64_epi32(src, k, a); }
inline m128i maskz_cvtepi64_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepi64_epi32(k, a); }
inline m128i cvtepi64_epi8(m128i a) noexcept { return _mm_cvtepi64_epi8(a); }
inline m128i mask_cvtepi64_epi8(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi64_epi8(src, k, a); }
inline m128i maskz_cvtepi64_epi8(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepi64_epi8(k, a); }
inline m128d cvtepi64_pd(m128i a) noexcept { return _mm_cvtepi64_pd(a); }
inline m128d mask_cvtepi64_pd(m128d src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi64_pd(src, k, a); }
inline m128d maskz_cvtepi64_pd(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepi64_pd(k, a); }
inline m128f cvtepi64_ps(m128i a) noexcept { return _mm_cvtepi64_ps(a); }
inline m128f mask_cvtepi64_ps(m128f src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi64_ps(src, k, a); }
inline m128f maskz_cvtepi64_ps(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepi64_ps(k, a); }
inline void mask_cvtepi64_storeu_epi16(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi64_storeu_epi16(base_addr, k, a); }
inline void mask_cvtepi64_storeu_epi32(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi64_storeu_epi32(base_addr, k, a); }
inline void mask_cvtepi64_storeu_epi8(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi64_storeu_epi8(base_addr, k, a); }
inline m128i cvtepi8_epi16(m128i a) noexcept { return _mm_cvtepi8_epi16(a); }
inline m128i mask_cvtepi8_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi8_epi16(src, k, a); }
inline m128i maskz_cvtepi8_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepi8_epi16(k, a); }
inline m128i cvtepi8_epi32(m128i a) noexcept { return _mm_cvtepi8_epi32(a); }
inline m128i mask_cvtepi8_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi8_epi32(src, k, a); }
inline m128i maskz_cvtepi8_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepi8_epi32(k, a); }
inline m128i cvtepi8_epi64(m128i a) noexcept { return _mm_cvtepi8_epi64(a); }
inline m128i mask_cvtepi8_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepi8_epi64(src, k, a); }
inline m128i maskz_cvtepi8_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepi8_epi64(k, a); }
inline m128i cvtepu16_epi32(m128i a) noexcept { return _mm_cvtepu16_epi32(a); }
inline m128i mask_cvtepu16_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepu16_epi32(src, k, a); }
inline m128i maskz_cvtepu16_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepu16_epi32(k, a); }
inline m128i cvtepu16_epi64(m128i a) noexcept { return _mm_cvtepu16_epi64(a); }
inline m128i mask_cvtepu16_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepu16_epi64(src, k, a); }
inline m128i maskz_cvtepu16_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepu16_epi64(k, a); }
inline m128i cvtepu32_epi64(m128i a) noexcept { return _mm_cvtepu32_epi64(a); }
inline m128i mask_cvtepu32_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepu32_epi64(src, k, a); }
inline m128i maskz_cvtepu32_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepu32_epi64(k, a); }
inline m128d cvtepu32_pd(m128i a) noexcept { return _mm_cvtepu32_pd(a); }
inline m128d mask_cvtepu32_pd(m128d src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepu32_pd(src, k, a); }
inline m128d maskz_cvtepu32_pd(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepu32_pd(k, a); }
inline m128d cvtepu64_pd(m128i a) noexcept { return _mm_cvtepu64_pd(a); }
inline m128d mask_cvtepu64_pd(m128d src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepu64_pd(src, k, a); }
inline m128d maskz_cvtepu64_pd(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepu64_pd(k, a); }
inline m128f cvtepu64_ps(m128i a) noexcept { return _mm_cvtepu64_ps(a); }
inline m128f mask_cvtepu64_ps(m128f src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepu64_ps(src, k, a); }
inline m128f maskz_cvtepu64_ps(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepu64_ps(k, a); }
inline m128i cvtepu8_epi16(m128i a) noexcept { return _mm_cvtepu8_epi16(a); }
inline m128i mask_cvtepu8_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepu8_epi16(src, k, a); }
inline m128i maskz_cvtepu8_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepu8_epi16(k, a); }
inline m128i cvtepu8_epi32(m128i a) noexcept { return _mm_cvtepu8_epi32(a); }
inline m128i mask_cvtepu8_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepu8_epi32(src, k, a); }
inline m128i maskz_cvtepu8_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepu8_epi32(k, a); }
inline m128i cvtepu8_epi64(m128i a) noexcept { return _mm_cvtepu8_epi64(a); }
inline m128i mask_cvtepu8_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtepu8_epi64(src, k, a); }
inline m128i maskz_cvtepu8_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtepu8_epi64(k, a); }
inline m128d cvti32_sd(m128d a, int b) noexcept { return _mm_cvti32_sd(a, b); }
inline m128f cvti32_ss(m128f a, int b) noexcept { return _mm_cvti32_ss(a, b); }
inline m128d cvti64_sd(m128d a, int8 b) noexcept { return _mm_cvti64_sd(a, b); }
inline m128f cvti64_ss(m128f a, int8 b) noexcept { return _mm_cvti64_ss(a, b); }
inline m128i cvtpd_epi32(m128d a) noexcept { return _mm_cvtpd_epi32(a); }
inline m128i mask_cvtpd_epi32(m128i src, mmask8 k, m128d a) noexcept { return _mm_mask_cvtpd_epi32(src, k, a); }
inline m128i maskz_cvtpd_epi32(mmask8 k, m128d a) noexcept { return _mm_maskz_cvtpd_epi32(k, a); }
inline m128i cvtpd_epi64(m128d a) noexcept { return _mm_cvtpd_epi64(a); }
inline m128i mask_cvtpd_epi64(m128i src, mmask8 k, m128d a) noexcept { return _mm_mask_cvtpd_epi64(src, k, a); }
inline m128i maskz_cvtpd_epi64(mmask8 k, m128d a) noexcept { return _mm_maskz_cvtpd_epi64(k, a); }
inline m128i cvtpd_epu32(m128d a) noexcept { return _mm_cvtpd_epu32(a); }
inline m128i mask_cvtpd_epu32(m128i src, mmask8 k, m128d a) noexcept { return _mm_mask_cvtpd_epu32(src, k, a); }
inline m128i maskz_cvtpd_epu32(mmask8 k, m128d a) noexcept { return _mm_maskz_cvtpd_epu32(k, a); }
inline m128i cvtpd_epu64(m128d a) noexcept { return _mm_cvtpd_epu64(a); }
inline m128i mask_cvtpd_epu64(m128i src, mmask8 k, m128d a) noexcept { return _mm_mask_cvtpd_epu64(src, k, a); }
inline m128i maskz_cvtpd_epu64(mmask8 k, m128d a) noexcept { return _mm_maskz_cvtpd_epu64(k, a); }
inline m128f cvtpd_ps(m128d a) noexcept { return _mm_cvtpd_ps(a); }
inline m128f mask_cvtpd_ps(m128f src, mmask8 k, m128d a) noexcept { return _mm_mask_cvtpd_ps(src, k, a); }
inline m128f maskz_cvtpd_ps(mmask8 k, m128d a) noexcept { return _mm_maskz_cvtpd_ps(k, a); }
inline m128f cvtph_ps(m128i a) noexcept { return _mm_cvtph_ps(a); }
inline m128f mask_cvtph_ps(m128f src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtph_ps(src, k, a); }
inline m128f maskz_cvtph_ps(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtph_ps(k, a); }
inline m128i cvtps_epi32(m128f a) noexcept { return _mm_cvtps_epi32(a); }
inline m128i mask_cvtps_epi32(m128i src, mmask8 k, m128f a) noexcept { return _mm_mask_cvtps_epi32(src, k, a); }
inline m128i maskz_cvtps_epi32(mmask8 k, m128f a) noexcept { return _mm_maskz_cvtps_epi32(k, a); }
inline m128i cvtps_epi64(m128f a) noexcept { return _mm_cvtps_epi64(a); }
inline m128i mask_cvtps_epi64(m128i src, mmask8 k, m128f a) noexcept { return _mm_mask_cvtps_epi64(src, k, a); }
inline m128i maskz_cvtps_epi64(mmask8 k, m128f a) noexcept { return _mm_maskz_cvtps_epi64(k, a); }
inline m128i cvtps_epu32(m128f a) noexcept { return _mm_cvtps_epu32(a); }
inline m128i mask_cvtps_epu32(m128i src, mmask8 k, m128f a) noexcept { return _mm_mask_cvtps_epu32(src, k, a); }
inline m128i maskz_cvtps_epu32(mmask8 k, m128f a) noexcept { return _mm_maskz_cvtps_epu32(k, a); }
inline m128i cvtps_epu64(m128f a) noexcept { return _mm_cvtps_epu64(a); }
inline m128i mask_cvtps_epu64(m128i src, mmask8 k, m128f a) noexcept { return _mm_mask_cvtps_epu64(src, k, a); }
inline m128i maskz_cvtps_epu64(mmask8 k, m128f a) noexcept { return _mm_maskz_cvtps_epu64(k, a); }
inline m128d cvtps_pd(m128f a) noexcept { return _mm_cvtps_pd(a); }
inline double cvtsd_f64(m128d a) noexcept { return _mm_cvtsd_f64(a); }
inline int cvtsd_i32(m128d a) noexcept { return _mm_cvtsd_i32(a); }
inline int8 cvtsd_i64(m128d a) noexcept { return _mm_cvtsd_i64(a); }
inline int cvtsd_si32(m128d a) noexcept { return _mm_cvtsd_si32(a); }
inline int8 cvtsd_si64(m128d a) noexcept { return _mm_cvtsd_si64(a); }
inline int8 cvtsd_si64x(m128d a) noexcept { return _mm_cvtsd_si64(a); }
inline m128f cvtsd_ss(m128f a, m128d b) noexcept { return _mm_cvtsd_ss(a, b); }
inline m128f mask_cvtsd_ss(m128f src, mmask8 k, m128f a, m128d b) noexcept { return _mm_mask_cvtsd_ss(src, k, a, b); }
inline m128f maskz_cvtsd_ss(mmask8 k, m128f a, m128d b) noexcept { return _mm_maskz_cvtsd_ss(k, a, b); }
inline nat4 cvtsd_u32(m128d a) noexcept { return _mm_cvtsd_u32(a); }
inline nat8 cvtsd_u64(m128d a) noexcept { return _mm_cvtsd_u64(a); }
inline m128i cvtsepi16_epi8(m128i a) noexcept { return _mm_cvtsepi16_epi8(a); }
inline m128i mask_cvtsepi16_epi8(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtsepi16_epi8(src, k, a); }
inline m128i maskz_cvtsepi16_epi8(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtsepi16_epi8(k, a); }
inline void mask_cvtsepi16_storeu_epi8(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_cvtsepi16_storeu_epi8(base_addr, k, a); }
inline m128i cvtsepi32_epi16(m128i a) noexcept { return _mm_cvtsepi32_epi16(a); }
inline m128i mask_cvtsepi32_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtsepi32_epi16(src, k, a); }
inline m128i maskz_cvtsepi32_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtsepi32_epi16(k, a); }
inline m128i cvtsepi32_epi8(m128i a) noexcept { return _mm_cvtsepi32_epi8(a); }
inline m128i mask_cvtsepi32_epi8(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtsepi32_epi8(src, k, a); }
inline m128i maskz_cvtsepi32_epi8(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtsepi32_epi8(k, a); }
inline void mask_cvtsepi32_storeu_epi16(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_cvtsepi32_storeu_epi16(base_addr, k, a); }
inline void mask_cvtsepi32_storeu_epi8(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_cvtsepi32_storeu_epi8(base_addr, k, a); }
inline m128i cvtsepi64_epi16(m128i a) noexcept { return _mm_cvtsepi64_epi16(a); }
inline m128i mask_cvtsepi64_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtsepi64_epi16(src, k, a); }
inline m128i maskz_cvtsepi64_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtsepi64_epi16(k, a); }
inline m128i cvtsepi64_epi32(m128i a) noexcept { return _mm_cvtsepi64_epi32(a); }
inline m128i mask_cvtsepi64_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtsepi64_epi32(src, k, a); }
inline m128i maskz_cvtsepi64_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtsepi64_epi32(k, a); }
inline m128i cvtsepi64_epi8(m128i a) noexcept { return _mm_cvtsepi64_epi8(a); }
inline m128i mask_cvtsepi64_epi8(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtsepi64_epi8(src, k, a); }
inline m128i maskz_cvtsepi64_epi8(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtsepi64_epi8(k, a); }
inline void mask_cvtsepi64_storeu_epi16(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_cvtsepi64_storeu_epi16(base_addr, k, a); }
inline void mask_cvtsepi64_storeu_epi32(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_cvtsepi64_storeu_epi32(base_addr, k, a); }
inline void mask_cvtsepi64_storeu_epi8(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_cvtsepi64_storeu_epi8(base_addr, k, a); }
inline int cvtsi128_si32(m128i a) noexcept { return _mm_cvtsi128_si32(a); }
inline int8 cvtsi128_si64(m128i a) noexcept { return _mm_cvtsi128_si64(a); }
inline int8 cvtsi128_si64x(m128i a) noexcept { return _mm_cvtsi128_si64(a); }
inline m128d cvtsi32_sd(m128d a, int b) noexcept { return _mm_cvtsi32_sd(a, b); }
inline m128i cvtsi32_si128(int a) noexcept { return _mm_cvtsi32_si128(a); }
inline m128f cvtsi32_ss(m128f a, int b) noexcept { return _mm_cvtsi32_ss(a, b); }
inline m128d cvtsi64_sd(m128d a, int8 b) noexcept { return _mm_cvtsi64_sd(a, b); }
inline m128i cvtsi64_si128(int8 a) noexcept { return _mm_cvtsi64_si128(a); }
inline m128f cvtsi64_ss(m128f a, int8 b) noexcept { return _mm_cvtsi64_ss(a, b); }
inline m128d cvtsi64x_sd(m128d a, int8 b) noexcept { return _mm_cvtsi64_sd(a, b); }
inline m128i cvtsi64x_si128(int8 a) noexcept { return _mm_cvtsi64_si128(a); }
inline float cvtss_f32(m128f a) noexcept { return _mm_cvtss_f32(a); }
inline int cvtss_i32(m128f a) noexcept { return _mm_cvtss_i32(a); }
inline int8 cvtss_i64(m128f a) noexcept { return _mm_cvtss_i64(a); }
inline m128d cvtss_sd(m128d a, m128f b) noexcept { return _mm_cvtss_sd(a, b); }
inline m128d mask_cvtss_sd(m128d src, mmask8 k, m128d a, m128f b) noexcept { return _mm_mask_cvtss_sd(src, k, a, b); }
inline m128d maskz_cvtss_sd(mmask8 k, m128d a, m128f b) noexcept { return _mm_maskz_cvtss_sd(k, a, b); }
inline int cvtss_si32(m128f a) noexcept { return _mm_cvtss_si32(a); }
inline int8 cvtss_si64(m128f a) noexcept { return _mm_cvtss_si64(a); }
inline nat4 cvtss_u32(m128f a) noexcept { return _mm_cvtss_u32(a); }
inline nat8 cvtss_u64(m128f a) noexcept { return _mm_cvtss_u64(a); }
template<int sae> inline int cvtt_roundsd_i32(m128d a) noexcept { return _mm_cvtt_roundsd_i32(a, sae); }
template<int sae> inline int8 cvtt_roundsd_i64(m128d a) noexcept { return _mm_cvtt_roundsd_i64(a, sae); }
template<int sae> inline int cvtt_roundsd_si32(m128d a) noexcept { return _mm_cvtt_roundsd_si32(a, sae); }
template<int sae> inline int8 cvtt_roundsd_si64(m128d a) noexcept { return _mm_cvtt_roundsd_si64(a, sae); }
template<int sae> inline nat4 cvtt_roundsd_u32(m128d a) noexcept { return _mm_cvtt_roundsd_u32(a, sae); }
template<int sae> inline nat8 cvtt_roundsd_u64(m128d a) noexcept { return _mm_cvtt_roundsd_u64(a, sae); }
template<int sae> inline int cvtt_roundss_i32(m128f a) noexcept { return _mm_cvtt_roundss_i32(a, sae); }
template<int sae> inline int8 cvtt_roundss_i64(m128f a) noexcept { return _mm_cvtt_roundss_i64(a, sae); }
template<int sae> inline int cvtt_roundss_si32(m128f a) noexcept { return _mm_cvtt_roundss_si32(a, sae); }
template<int sae> inline int8 cvtt_roundss_si64(m128f a) noexcept { return _mm_cvtt_roundss_si64(a, sae); }
template<int sae> inline nat4 cvtt_roundss_u32(m128f a) noexcept { return _mm_cvtt_roundss_u32(a, sae); }
template<int sae> inline nat8 cvtt_roundss_u64(m128f a) noexcept { return _mm_cvtt_roundss_u64(a, sae); }
inline int cvtt_ss2si(m128f a) noexcept { return _mm_cvtt_ss2si(a); }
inline m128i cvttpd_epi32(m128d a) noexcept { return _mm_cvttpd_epi32(a); }
inline m128i mask_cvttpd_epi32(m128i src, mmask8 k, m128d a) noexcept { return _mm_mask_cvttpd_epi32(src, k, a); }
inline m128i maskz_cvttpd_epi32(mmask8 k, m128d a) noexcept { return _mm_maskz_cvttpd_epi32(k, a); }
inline m128i cvttpd_epi64(m128d a) noexcept { return _mm_cvttpd_epi64(a); }
inline m128i mask_cvttpd_epi64(m128i src, mmask8 k, m128d a) noexcept { return _mm_mask_cvttpd_epi64(src, k, a); }
inline m128i maskz_cvttpd_epi64(mmask8 k, m128d a) noexcept { return _mm_maskz_cvttpd_epi64(k, a); }
inline m128i cvttpd_epu32(m128d a) noexcept { return _mm_cvttpd_epu32(a); }
inline m128i mask_cvttpd_epu32(m128i src, mmask8 k, m128d a) noexcept { return _mm_mask_cvttpd_epu32(src, k, a); }
inline m128i maskz_cvttpd_epu32(mmask8 k, m128d a) noexcept { return _mm_maskz_cvttpd_epu32(k, a); }
inline m128i cvttpd_epu64(m128d a) noexcept { return _mm_cvttpd_epu64(a); }
inline m128i mask_cvttpd_epu64(m128i src, mmask8 k, m128d a) noexcept { return _mm_mask_cvttpd_epu64(src, k, a); }
inline m128i maskz_cvttpd_epu64(mmask8 k, m128d a) noexcept { return _mm_maskz_cvttpd_epu64(k, a); }
inline m128i cvttps_epi32(m128f a) noexcept { return _mm_cvttps_epi32(a); }
inline m128i mask_cvttps_epi32(m128i src, mmask8 k, m128f a) noexcept { return _mm_mask_cvttps_epi32(src, k, a); }
inline m128i maskz_cvttps_epi32(mmask8 k, m128f a) noexcept { return _mm_maskz_cvttps_epi32(k, a); }
inline m128i cvttps_epi64(m128f a) noexcept { return _mm_cvttps_epi64(a); }
inline m128i mask_cvttps_epi64(m128i src, mmask8 k, m128f a) noexcept { return _mm_mask_cvttps_epi64(src, k, a); }
inline m128i maskz_cvttps_epi64(mmask8 k, m128f a) noexcept { return _mm_maskz_cvttps_epi64(k, a); }
inline m128i cvttps_epu32(m128f a) noexcept { return _mm_cvttps_epu32(a); }
inline m128i mask_cvttps_epu32(m128i src, mmask8 k, m128f a) noexcept { return _mm_mask_cvttps_epu32(src, k, a); }
inline m128i maskz_cvttps_epu32(mmask8 k, m128f a) noexcept { return _mm_maskz_cvttps_epu32(k, a); }
inline m128i cvttps_epu64(m128f a) noexcept { return _mm_cvttps_epu64(a); }
inline m128i mask_cvttps_epu64(m128i src, mmask8 k, m128f a) noexcept { return _mm_mask_cvttps_epu64(src, k, a); }
inline m128i maskz_cvttps_epu64(mmask8 k, m128f a) noexcept { return _mm_maskz_cvttps_epu64(k, a); }
inline int cvttsd_i32(m128d a) noexcept { return _mm_cvttsd_i32(a); }
inline int8 cvttsd_i64(m128d a) noexcept { return _mm_cvttsd_i64(a); }
inline int cvttsd_si32(m128d a) noexcept { return _mm_cvttsd_si32(a); }
inline int8 cvttsd_si64(m128d a) noexcept { return _mm_cvttsd_si64(a); }
inline int8 cvttsd_si64x(m128d a) noexcept { return _mm_cvttsd_si64(a); }
inline nat4 cvttsd_u32(m128d a) noexcept { return _mm_cvttsd_u32(a); }
inline nat8 cvttsd_u64(m128d a) noexcept { return _mm_cvttsd_u64(a); }
inline int cvttss_i32(m128f a) noexcept { return _mm_cvttss_i32(a); }
inline int8 cvttss_i64(m128f a) noexcept { return _mm_cvttss_i64(a); }
inline int cvttss_si32(m128f a) noexcept { return _mm_cvttss_si32(a); }
inline int8 cvttss_si64(m128f a) noexcept { return _mm_cvttss_si64(a); }
inline nat4 cvttss_u32(m128f a) noexcept { return _mm_cvttss_u32(a); }
inline nat8 cvttss_u64(m128f a) noexcept { return _mm_cvttss_u64(a); }
inline m128d cvtu32_sd(m128d a, nat4 b) noexcept { return _mm_cvtu32_sd(a, b); }
inline m128f cvtu32_ss(m128f a, nat4 b) noexcept { return _mm_cvtu32_ss(a, b); }
inline m128d cvtu64_sd(m128d a, nat8 b) noexcept { return _mm_cvtu64_sd(a, b); }
inline m128f cvtu64_ss(m128f a, nat8 b) noexcept { return _mm_cvtu64_ss(a, b); }
inline m128i cvtusepi16_epi8(m128i a) noexcept { return _mm_cvtusepi16_epi8(a); }
inline m128i mask_cvtusepi16_epi8(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtusepi16_epi8(src, k, a); }
inline m128i maskz_cvtusepi16_epi8(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtusepi16_epi8(k, a); }
inline void mask_cvtusepi16_storeu_epi8(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_cvtusepi16_storeu_epi8(base_addr, k, a); }
inline m128i cvtusepi32_epi16(m128i a) noexcept { return _mm_cvtusepi32_epi16(a); }
inline m128i mask_cvtusepi32_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtusepi32_epi16(src, k, a); }
inline m128i maskz_cvtusepi32_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtusepi32_epi16(k, a); }
inline m128i cvtusepi32_epi8(m128i a) noexcept { return _mm_cvtusepi32_epi8(a); }
inline m128i mask_cvtusepi32_epi8(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtusepi32_epi8(src, k, a); }
inline m128i maskz_cvtusepi32_epi8(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtusepi32_epi8(k, a); }
inline void mask_cvtusepi32_storeu_epi16(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_cvtusepi32_storeu_epi16(base_addr, k, a); }
inline void mask_cvtusepi32_storeu_epi8(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_cvtusepi32_storeu_epi8(base_addr, k, a); }
inline m128i cvtusepi64_epi16(m128i a) noexcept { return _mm_cvtusepi64_epi16(a); }
inline m128i mask_cvtusepi64_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtusepi64_epi16(src, k, a); }
inline m128i maskz_cvtusepi64_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtusepi64_epi16(k, a); }
inline m128i cvtusepi64_epi32(m128i a) noexcept { return _mm_cvtusepi64_epi32(a); }
inline m128i mask_cvtusepi64_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtusepi64_epi32(src, k, a); }
inline m128i maskz_cvtusepi64_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtusepi64_epi32(k, a); }
inline m128i cvtusepi64_epi8(m128i a) noexcept { return _mm_cvtusepi64_epi8(a); }
inline m128i mask_cvtusepi64_epi8(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_cvtusepi64_epi8(src, k, a); }
inline m128i maskz_cvtusepi64_epi8(mmask8 k, m128i a) noexcept { return _mm_maskz_cvtusepi64_epi8(k, a); }
inline void mask_cvtusepi64_storeu_epi16(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_cvtusepi64_storeu_epi16(base_addr, k, a); }
inline void mask_cvtusepi64_storeu_epi32(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_cvtusepi64_storeu_epi32(base_addr, k, a); }
inline void mask_cvtusepi64_storeu_epi8(void* base_addr, mmask8 k, m128i a) noexcept { return _mm_mask_cvtusepi64_storeu_epi8(base_addr, k, a); }
template<int4 imm8> inline m128i dbsad_epu8(m128i a, m128i b) noexcept { return _mm_dbsad_epu8(a, b, imm8); }
template<int4 imm8> inline m128i mask_dbsad_epu8(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_dbsad_epu8(src, k, a, b, imm8); }
template<int4 imm8> inline m128i maskz_dbsad_epu8(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_dbsad_epu8(k, a, b, imm8); }
inline m128d div_pd(m128d a, m128d b) noexcept { return _mm_div_pd(a, b); }
inline m128d mask_div_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_div_pd(src, k, a, b); }
inline m128d maskz_div_pd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_div_pd(k, a, b); }
inline m128f div_ps(m128f a, m128f b) noexcept { return _mm_div_ps(a, b); }
inline m128f mask_div_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_div_ps(src, k, a, b); }
inline m128f maskz_div_ps(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_div_ps(k, a, b); }
template<int4 rounding> inline m128d div_round_sd(m128d a, m128d b) noexcept { return _mm_div_round_sd(a, b, rounding); }
template<int4 rounding> inline m128d mask_div_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_div_round_sd(src, k, a, b, rounding); }
template<int4 rounding> inline m128d maskz_div_round_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_div_round_sd(k, a, b, rounding); }
template<int4 rounding> inline m128f div_round_ss(m128f a, m128f b) noexcept { return _mm_div_round_ss(a, b, rounding); }
template<int4 rounding> inline m128f mask_div_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_div_round_ss(src, k, a, b, rounding); }
template<int4 rounding> inline m128f maskz_div_round_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_div_round_ss(k, a, b, rounding); }
inline m128d div_sd(m128d a, m128d b) noexcept { return _mm_div_sd(a, b); }
inline m128d mask_div_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_div_sd(src, k, a, b); }
inline m128d maskz_div_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_div_sd(k, a, b); }
inline m128f div_ss(m128f a, m128f b) noexcept { return _mm_div_ss(a, b); }
inline m128f mask_div_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_div_ss(src, k, a, b); }
inline m128f maskz_div_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_div_ss(k, a, b); }
template<int4 imm8> inline m128d dp_pd(m128d a, m128d b) noexcept { return _mm_dp_pd(a, b, imm8); }
template<int4 imm8> inline m128f dp_ps(m128f a, m128f b) noexcept { return _mm_dp_ps(a, b, imm8); }
inline m128i dpbusd_avx_epi32(m128i src, m128i a, m128i b) noexcept { return _mm_dpbusd_avx_epi32(src, a, b); }
inline m128i dpbusd_epi32(m128i src, m128i a, m128i b) noexcept { return _mm_dpbusd_epi32(src, a, b); }
inline m128i mask_dpbusd_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_dpbusd_epi32(src, k, a, b); }
inline m128i maskz_dpbusd_epi32(mmask8 k, m128i src, m128i a, m128i b) noexcept { return _mm_maskz_dpbusd_epi32(k, src, a, b); }
inline m128i dpbusds_avx_epi32(m128i src, m128i a, m128i b) noexcept { return _mm_dpbusds_avx_epi32(src, a, b); }
inline m128i dpbusds_epi32(m128i src, m128i a, m128i b) noexcept { return _mm_dpbusds_epi32(src, a, b); }
inline m128i mask_dpbusds_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_dpbusds_epi32(src, k, a, b); }
inline m128i maskz_dpbusds_epi32(mmask8 k, m128i src, m128i a, m128i b) noexcept { return _mm_maskz_dpbusds_epi32(k, src, a, b); }
inline m128i dpwssd_avx_epi32(m128i src, m128i a, m128i b) noexcept { return _mm_dpwssd_avx_epi32(src, a, b); }
inline m128i dpwssd_epi32(m128i src, m128i a, m128i b) noexcept { return _mm_dpwssd_epi32(src, a, b); }
inline m128i mask_dpwssd_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_dpwssd_epi32(src, k, a, b); }
inline m128i maskz_dpwssd_epi32(mmask8 k, m128i src, m128i a, m128i b) noexcept { return _mm_maskz_dpwssd_epi32(k, src, a, b); }
inline m128i dpwssds_avx_epi32(m128i src, m128i a, m128i b) noexcept { return _mm_dpwssds_avx_epi32(src, a, b); }
inline m128i dpwssds_epi32(m128i src, m128i a, m128i b) noexcept { return _mm_dpwssds_epi32(src, a, b); }
inline m128i mask_dpwssds_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_dpwssds_epi32(src, k, a, b); }
inline m128i maskz_dpwssds_epi32(mmask8 k, m128i src, m128i a, m128i b) noexcept { return _mm_maskz_dpwssds_epi32(k, src, a, b); }
inline nat4 encodekey128_u32(nat4 __htype, m128i __key, void* __h) noexcept { return _mm_encodekey128_u32(__htype, __key, __h); }
inline nat4 encodekey256_u32(nat4 __htype, m128i __key_lo, m128i __key_hi, void* __h) noexcept { return _mm_encodekey256_u32(__htype, __key_lo, __key_hi, __h); }
inline m128i mask_expand_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_expand_epi16(src, k, a); }
inline m128i maskz_expand_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_expand_epi16(k, a); }
inline m128i mask_expand_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_expand_epi32(src, k, a); }
inline m128i maskz_expand_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_expand_epi32(k, a); }
inline m128i mask_expand_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_expand_epi64(src, k, a); }
inline m128i maskz_expand_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_expand_epi64(k, a); }
inline m128i mask_expand_epi8(m128i src, mmask16 k, m128i a) noexcept { return _mm_mask_expand_epi8(src, k, a); }
inline m128i maskz_expand_epi8(mmask16 k, m128i a) noexcept { return _mm_maskz_expand_epi8(k, a); }
inline m128d mask_expand_pd(m128d src, mmask8 k, m128d a) noexcept { return _mm_mask_expand_pd(src, k, a); }
inline m128d maskz_expand_pd(mmask8 k, m128d a) noexcept { return _mm_maskz_expand_pd(k, a); }
inline m128f mask_expand_ps(m128f src, mmask8 k, m128f a) noexcept { return _mm_mask_expand_ps(src, k, a); }
inline m128f maskz_expand_ps(mmask8 k, m128f a) noexcept { return _mm_maskz_expand_ps(k, a); }
inline m128i mask_expandloadu_epi16(m128i src, mmask8 k, const void* mem_addr) noexcept { return _mm_mask_expandloadu_epi16(src, k, mem_addr); }
inline m128i maskz_expandloadu_epi16(mmask8 k, const void* mem_addr) noexcept { return _mm_maskz_expandloadu_epi16(k, mem_addr); }
inline m128i mask_expandloadu_epi32(m128i src, mmask8 k, void const* mem_addr) noexcept { return _mm_mask_expandloadu_epi32(src, k, mem_addr); }
inline m128i maskz_expandloadu_epi32(mmask8 k, void const* mem_addr) noexcept { return _mm_maskz_expandloadu_epi32(k, mem_addr); }
inline m128i mask_expandloadu_epi64(m128i src, mmask8 k, void const* mem_addr) noexcept { return _mm_mask_expandloadu_epi64(src, k, mem_addr); }
inline m128i maskz_expandloadu_epi64(mmask8 k, void const* mem_addr) noexcept { return _mm_maskz_expandloadu_epi64(k, mem_addr); }
inline m128i mask_expandloadu_epi8(m128i src, mmask16 k, const void* mem_addr) noexcept { return _mm_mask_expandloadu_epi8(src, k, mem_addr); }
inline m128i maskz_expandloadu_epi8(mmask16 k, const void* mem_addr) noexcept { return _mm_maskz_expandloadu_epi8(k, mem_addr); }
inline m128d mask_expandloadu_pd(m128d src, mmask8 k, void const* mem_addr) noexcept { return _mm_mask_expandloadu_pd(src, k, mem_addr); }
inline m128d maskz_expandloadu_pd(mmask8 k, void const* mem_addr) noexcept { return _mm_maskz_expandloadu_pd(k, mem_addr); }
inline m128f mask_expandloadu_ps(m128f src, mmask8 k, void const* mem_addr) noexcept { return _mm_mask_expandloadu_ps(src, k, mem_addr); }
inline m128f maskz_expandloadu_ps(mmask8 k, void const* mem_addr) noexcept { return _mm_maskz_expandloadu_ps(k, mem_addr); }
template<int4 imm8> inline int extract_epi16(m128i a) noexcept { return _mm_extract_epi16(a, imm8); }
template<int4 imm8> inline int extract_epi32(m128i a) noexcept { return _mm_extract_epi32(a, imm8); }
template<int4 imm8> inline int8 extract_epi64(m128i a) noexcept { return _mm_extract_epi64(a, imm8); }
template<int4 imm8> inline int extract_epi8(m128i a) noexcept { return _mm_extract_epi8(a, imm8); }
template<int4 imm8> inline int extract_ps(m128f a) noexcept { return _mm_extract_ps(a, imm8); }
template<int4 imm8> inline m128d fixupimm_pd(m128d a, m128d b, m128i c) noexcept { return _mm_fixupimm_pd(a, b, c, imm8); }
template<int4 imm8> inline m128d mask_fixupimm_pd(m128d a, mmask8 k, m128d b, m128i c) noexcept { return _mm_mask_fixupimm_pd(a, k, b, c, imm8); }
template<int4 imm8> inline m128d maskz_fixupimm_pd(mmask8 k, m128d a, m128d b, m128i c) noexcept { return _mm_maskz_fixupimm_pd(k, a, b, c, imm8); }
template<int4 imm8> inline m128f fixupimm_ps(m128f a, m128f b, m128i c) noexcept { return _mm_fixupimm_ps(a, b, c, imm8); }
template<int4 imm8> inline m128f mask_fixupimm_ps(m128f a, mmask8 k, m128f b, m128i c) noexcept { return _mm_mask_fixupimm_ps(a, k, b, c, imm8); }
template<int4 imm8> inline m128f maskz_fixupimm_ps(mmask8 k, m128f a, m128f b, m128i c) noexcept { return _mm_maskz_fixupimm_ps(k, a, b, c, imm8); }
template<int4 imm8, int sae> inline m128d fixupimm_round_sd(m128d a, m128d b, m128i c) noexcept { return _mm_fixupimm_round_sd(a, b, c, imm8, sae); }
template<int4 imm8, int sae> inline m128d mask_fixupimm_round_sd(m128d a, mmask8 k, m128d b, m128i c) noexcept { return _mm_mask_fixupimm_round_sd(a, k, b, c, imm8, sae); }
template<int4 imm8, int sae> inline m128d maskz_fixupimm_round_sd(mmask8 k, m128d a, m128d b, m128i c) noexcept { return _mm_maskz_fixupimm_round_sd(k, a, b, c, imm8, sae); }
template<int4 imm8, int sae> inline m128f fixupimm_round_ss(m128f a, m128f b, m128i c) noexcept { return _mm_fixupimm_round_ss(a, b, c, imm8, sae); }
template<int4 imm8, int sae> inline m128f mask_fixupimm_round_ss(m128f a, mmask8 k, m128f b, m128i c) noexcept { return _mm_mask_fixupimm_round_ss(a, k, b, c, imm8, sae); }
template<int4 imm8, int sae> inline m128f maskz_fixupimm_round_ss(mmask8 k, m128f a, m128f b, m128i c) noexcept { return _mm_maskz_fixupimm_round_ss(k, a, b, c, imm8, sae); }
template<int4 imm8> inline m128d fixupimm_sd(m128d a, m128d b, m128i c) noexcept { return _mm_fixupimm_sd(a, b, c, imm8); }
template<int4 imm8> inline m128d mask_fixupimm_sd(m128d a, mmask8 k, m128d b, m128i c) noexcept { return _mm_mask_fixupimm_sd(a, k, b, c, imm8); }
template<int4 imm8> inline m128d maskz_fixupimm_sd(mmask8 k, m128d a, m128d b, m128i c) noexcept { return _mm_maskz_fixupimm_sd(k, a, b, c, imm8); }
template<int4 imm8> inline m128f fixupimm_ss(m128f a, m128f b, m128i c) noexcept { return _mm_fixupimm_ss(a, b, c, imm8); }
template<int4 imm8> inline m128f mask_fixupimm_ss(m128f a, mmask8 k, m128f b, m128i c) noexcept { return _mm_mask_fixupimm_ss(a, k, b, c, imm8); }
template<int4 imm8> inline m128f maskz_fixupimm_ss(mmask8 k, m128f a, m128f b, m128i c) noexcept { return _mm_maskz_fixupimm_ss(k, a, b, c, imm8); }
inline m128d floor_pd(m128d a) noexcept { return _mm_floor_pd(a); }
inline m128f floor_ps(m128f a) noexcept { return _mm_floor_ps(a); }
inline m128d floor_sd(m128d a, m128d b) noexcept { return _mm_floor_sd(a, b); }
inline m128f floor_ss(m128f a, m128f b) noexcept { return _mm_floor_ss(a, b); }
inline m128d fmadd_pd(m128d a, m128d b, m128d c) noexcept { return _mm_fmadd_pd(a, b, c); }
inline m128d mask_fmadd_pd(m128d a, mmask8 k, m128d b, m128d c) noexcept { return _mm_mask_fmadd_pd(a, k, b, c); }
inline m128d mask3_fmadd_pd(m128d a, m128d b, m128d c, mmask8 k) noexcept { return _mm_mask3_fmadd_pd(a, b, c, k); }
inline m128d maskz_fmadd_pd(mmask8 k, m128d a, m128d b, m128d c) noexcept { return _mm_maskz_fmadd_pd(k, a, b, c); }
inline m128f fmadd_ps(m128f a, m128f b, m128f c) noexcept { return _mm_fmadd_ps(a, b, c); }
inline m128f mask_fmadd_ps(m128f a, mmask8 k, m128f b, m128f c) noexcept { return _mm_mask_fmadd_ps(a, k, b, c); }
inline m128f mask3_fmadd_ps(m128f a, m128f b, m128f c, mmask8 k) noexcept { return _mm_mask3_fmadd_ps(a, b, c, k); }
inline m128f maskz_fmadd_ps(mmask8 k, m128f a, m128f b, m128f c) noexcept { return _mm_maskz_fmadd_ps(k, a, b, c); }
template<int4 rounding> inline m128d fmadd_round_sd(m128d a, m128d b, m128d c) noexcept { return _mm_fmadd_round_sd(a, b, c, rounding); }
template<int4 rounding> inline m128d mask_fmadd_round_sd(m128d a, mmask8 k, m128d b, m128d c) noexcept { return _mm_mask_fmadd_round_sd(a, k, b, c, rounding); }
template<int4 rounding> inline m128d mask3_fmadd_round_sd(m128d a, m128d b, m128d c, mmask8 k) noexcept { return _mm_mask3_fmadd_round_sd(a, b, c, k, rounding); }
template<int4 rounding> inline m128d maskz_fmadd_round_sd(mmask8 k, m128d a, m128d b, m128d c) noexcept { return _mm_maskz_fmadd_round_sd(k, a, b, c, rounding); }
template<int4 rounding> inline m128f fmadd_round_ss(m128f a, m128f b, m128f c) noexcept { return _mm_fmadd_round_ss(a, b, c, rounding); }
template<int4 rounding> inline m128f mask_fmadd_round_ss(m128f a, mmask8 k, m128f b, m128f c) noexcept { return _mm_mask_fmadd_round_ss(a, k, b, c, rounding); }
template<int4 rounding> inline m128f mask3_fmadd_round_ss(m128f a, m128f b, m128f c, mmask8 k) noexcept { return _mm_mask3_fmadd_round_ss(a, b, c, k, rounding); }
template<int4 rounding> inline m128f maskz_fmadd_round_ss(mmask8 k, m128f a, m128f b, m128f c) noexcept { return _mm_maskz_fmadd_round_ss(k, a, b, c, rounding); }
inline m128d fmadd_sd(m128d a, m128d b, m128d c) noexcept { return _mm_fmadd_sd(a, b, c); }
inline m128d mask_fmadd_sd(m128d a, mmask8 k, m128d b, m128d c) noexcept { return _mm_mask_fmadd_sd(a, k, b, c); }
inline m128d mask3_fmadd_sd(m128d a, m128d b, m128d c, mmask8 k) noexcept { return _mm_mask3_fmadd_sd(a, b, c, k); }
inline m128d maskz_fmadd_sd(mmask8 k, m128d a, m128d b, m128d c) noexcept { return _mm_maskz_fmadd_sd(k, a, b, c); }
inline m128f fmadd_ss(m128f a, m128f b, m128f c) noexcept { return _mm_fmadd_ss(a, b, c); }
inline m128f mask_fmadd_ss(m128f a, mmask8 k, m128f b, m128f c) noexcept { return _mm_mask_fmadd_ss(a, k, b, c); }
inline m128f mask3_fmadd_ss(m128f a, m128f b, m128f c, mmask8 k) noexcept { return _mm_mask3_fmadd_ss(a, b, c, k); }
inline m128f maskz_fmadd_ss(mmask8 k, m128f a, m128f b, m128f c) noexcept { return _mm_maskz_fmadd_ss(k, a, b, c); }
inline m128d fmaddsub_pd(m128d a, m128d b, m128d c) noexcept { return _mm_fmaddsub_pd(a, b, c); }
inline m128d mask_fmaddsub_pd(m128d a, mmask8 k, m128d b, m128d c) noexcept { return _mm_mask_fmaddsub_pd(a, k, b, c); }
inline m128d mask3_fmaddsub_pd(m128d a, m128d b, m128d c, mmask8 k) noexcept { return _mm_mask3_fmaddsub_pd(a, b, c, k); }
inline m128d maskz_fmaddsub_pd(mmask8 k, m128d a, m128d b, m128d c) noexcept { return _mm_maskz_fmaddsub_pd(k, a, b, c); }
inline m128f fmaddsub_ps(m128f a, m128f b, m128f c) noexcept { return _mm_fmaddsub_ps(a, b, c); }
inline m128f mask_fmaddsub_ps(m128f a, mmask8 k, m128f b, m128f c) noexcept { return _mm_mask_fmaddsub_ps(a, k, b, c); }
inline m128f mask3_fmaddsub_ps(m128f a, m128f b, m128f c, mmask8 k) noexcept { return _mm_mask3_fmaddsub_ps(a, b, c, k); }
inline m128f maskz_fmaddsub_ps(mmask8 k, m128f a, m128f b, m128f c) noexcept { return _mm_maskz_fmaddsub_ps(k, a, b, c); }
inline m128d fmsub_pd(m128d a, m128d b, m128d c) noexcept { return _mm_fmsub_pd(a, b, c); }
inline m128d mask_fmsub_pd(m128d a, mmask8 k, m128d b, m128d c) noexcept { return _mm_mask_fmsub_pd(a, k, b, c); }
inline m128d mask3_fmsub_pd(m128d a, m128d b, m128d c, mmask8 k) noexcept { return _mm_mask3_fmsub_pd(a, b, c, k); }
inline m128d maskz_fmsub_pd(mmask8 k, m128d a, m128d b, m128d c) noexcept { return _mm_maskz_fmsub_pd(k, a, b, c); }
inline m128f fmsub_ps(m128f a, m128f b, m128f c) noexcept { return _mm_fmsub_ps(a, b, c); }
inline m128f mask_fmsub_ps(m128f a, mmask8 k, m128f b, m128f c) noexcept { return _mm_mask_fmsub_ps(a, k, b, c); }
inline m128f mask3_fmsub_ps(m128f a, m128f b, m128f c, mmask8 k) noexcept { return _mm_mask3_fmsub_ps(a, b, c, k); }
inline m128f maskz_fmsub_ps(mmask8 k, m128f a, m128f b, m128f c) noexcept { return _mm_maskz_fmsub_ps(k, a, b, c); }
template<int4 rounding> inline m128d fmsub_round_sd(m128d a, m128d b, m128d c) noexcept { return _mm_fmsub_round_sd(a, b, c, rounding); }
template<int4 rounding> inline m128d mask_fmsub_round_sd(m128d a, mmask8 k, m128d b, m128d c) noexcept { return _mm_mask_fmsub_round_sd(a, k, b, c, rounding); }
template<int4 rounding> inline m128d mask3_fmsub_round_sd(m128d a, m128d b, m128d c, mmask8 k) noexcept { return _mm_mask3_fmsub_round_sd(a, b, c, k, rounding); }
template<int4 rounding> inline m128d maskz_fmsub_round_sd(mmask8 k, m128d a, m128d b, m128d c) noexcept { return _mm_maskz_fmsub_round_sd(k, a, b, c, rounding); }
template<int4 rounding> inline m128f fmsub_round_ss(m128f a, m128f b, m128f c) noexcept { return _mm_fmsub_round_ss(a, b, c, rounding); }
template<int4 rounding> inline m128f mask_fmsub_round_ss(m128f a, mmask8 k, m128f b, m128f c) noexcept { return _mm_mask_fmsub_round_ss(a, k, b, c, rounding); }
template<int4 rounding> inline m128f mask3_fmsub_round_ss(m128f a, m128f b, m128f c, mmask8 k) noexcept { return _mm_mask3_fmsub_round_ss(a, b, c, k, rounding); }
template<int4 rounding> inline m128f maskz_fmsub_round_ss(mmask8 k, m128f a, m128f b, m128f c) noexcept { return _mm_maskz_fmsub_round_ss(k, a, b, c, rounding); }
inline m128d fmsub_sd(m128d a, m128d b, m128d c) noexcept { return _mm_fmsub_sd(a, b, c); }
inline m128d mask_fmsub_sd(m128d a, mmask8 k, m128d b, m128d c) noexcept { return _mm_mask_fmsub_sd(a, k, b, c); }
inline m128d mask3_fmsub_sd(m128d a, m128d b, m128d c, mmask8 k) noexcept { return _mm_mask3_fmsub_sd(a, b, c, k); }
inline m128d maskz_fmsub_sd(mmask8 k, m128d a, m128d b, m128d c) noexcept { return _mm_maskz_fmsub_sd(k, a, b, c); }
inline m128f fmsub_ss(m128f a, m128f b, m128f c) noexcept { return _mm_fmsub_ss(a, b, c); }
inline m128f mask_fmsub_ss(m128f a, mmask8 k, m128f b, m128f c) noexcept { return _mm_mask_fmsub_ss(a, k, b, c); }
inline m128f mask3_fmsub_ss(m128f a, m128f b, m128f c, mmask8 k) noexcept { return _mm_mask3_fmsub_ss(a, b, c, k); }
inline m128f maskz_fmsub_ss(mmask8 k, m128f a, m128f b, m128f c) noexcept { return _mm_maskz_fmsub_ss(k, a, b, c); }
inline m128d fmsubadd_pd(m128d a, m128d b, m128d c) noexcept { return _mm_fmsubadd_pd(a, b, c); }
inline m128d mask_fmsubadd_pd(m128d a, mmask8 k, m128d b, m128d c) noexcept { return _mm_mask_fmsubadd_pd(a, k, b, c); }
inline m128d mask3_fmsubadd_pd(m128d a, m128d b, m128d c, mmask8 k) noexcept { return _mm_mask3_fmsubadd_pd(a, b, c, k); }
inline m128d maskz_fmsubadd_pd(mmask8 k, m128d a, m128d b, m128d c) noexcept { return _mm_maskz_fmsubadd_pd(k, a, b, c); }
inline m128f fmsubadd_ps(m128f a, m128f b, m128f c) noexcept { return _mm_fmsubadd_ps(a, b, c); }
inline m128f mask_fmsubadd_ps(m128f a, mmask8 k, m128f b, m128f c) noexcept { return _mm_mask_fmsubadd_ps(a, k, b, c); }
inline m128f mask3_fmsubadd_ps(m128f a, m128f b, m128f c, mmask8 k) noexcept { return _mm_mask3_fmsubadd_ps(a, b, c, k); }
inline m128f maskz_fmsubadd_ps(mmask8 k, m128f a, m128f b, m128f c) noexcept { return _mm_maskz_fmsubadd_ps(k, a, b, c); }
inline m128d fnmadd_pd(m128d a, m128d b, m128d c) noexcept { return _mm_fnmadd_pd(a, b, c); }
inline m128d mask_fnmadd_pd(m128d a, mmask8 k, m128d b, m128d c) noexcept { return _mm_mask_fnmadd_pd(a, k, b, c); }
inline m128d mask3_fnmadd_pd(m128d a, m128d b, m128d c, mmask8 k) noexcept { return _mm_mask3_fnmadd_pd(a, b, c, k); }
inline m128d maskz_fnmadd_pd(mmask8 k, m128d a, m128d b, m128d c) noexcept { return _mm_maskz_fnmadd_pd(k, a, b, c); }
inline m128f fnmadd_ps(m128f a, m128f b, m128f c) noexcept { return _mm_fnmadd_ps(a, b, c); }
inline m128f mask_fnmadd_ps(m128f a, mmask8 k, m128f b, m128f c) noexcept { return _mm_mask_fnmadd_ps(a, k, b, c); }
inline m128f mask3_fnmadd_ps(m128f a, m128f b, m128f c, mmask8 k) noexcept { return _mm_mask3_fnmadd_ps(a, b, c, k); }
inline m128f maskz_fnmadd_ps(mmask8 k, m128f a, m128f b, m128f c) noexcept { return _mm_maskz_fnmadd_ps(k, a, b, c); }
template<int4 rounding> inline m128d fnmadd_round_sd(m128d a, m128d b, m128d c) noexcept { return _mm_fnmadd_round_sd(a, b, c, rounding); }
template<int4 rounding> inline m128d mask_fnmadd_round_sd(m128d a, mmask8 k, m128d b, m128d c) noexcept { return _mm_mask_fnmadd_round_sd(a, k, b, c, rounding); }
template<int4 rounding> inline m128d mask3_fnmadd_round_sd(m128d a, m128d b, m128d c, mmask8 k) noexcept { return _mm_mask3_fnmadd_round_sd(a, b, c, k, rounding); }
template<int4 rounding> inline m128d maskz_fnmadd_round_sd(mmask8 k, m128d a, m128d b, m128d c) noexcept { return _mm_maskz_fnmadd_round_sd(k, a, b, c, rounding); }
template<int4 rounding> inline m128f fnmadd_round_ss(m128f a, m128f b, m128f c) noexcept { return _mm_fnmadd_round_ss(a, b, c, rounding); }
template<int4 rounding> inline m128f mask_fnmadd_round_ss(m128f a, mmask8 k, m128f b, m128f c) noexcept { return _mm_mask_fnmadd_round_ss(a, k, b, c, rounding); }
template<int4 rounding> inline m128f mask3_fnmadd_round_ss(m128f a, m128f b, m128f c, mmask8 k) noexcept { return _mm_mask3_fnmadd_round_ss(a, b, c, k, rounding); }
template<int4 rounding> inline m128f maskz_fnmadd_round_ss(mmask8 k, m128f a, m128f b, m128f c) noexcept { return _mm_maskz_fnmadd_round_ss(k, a, b, c, rounding); }
inline m128d fnmadd_sd(m128d a, m128d b, m128d c) noexcept { return _mm_fnmadd_sd(a, b, c); }
inline m128d mask_fnmadd_sd(m128d a, mmask8 k, m128d b, m128d c) noexcept { return _mm_mask_fnmadd_sd(a, k, b, c); }
inline m128d mask3_fnmadd_sd(m128d a, m128d b, m128d c, mmask8 k) noexcept { return _mm_mask3_fnmadd_sd(a, b, c, k); }
inline m128d maskz_fnmadd_sd(mmask8 k, m128d a, m128d b, m128d c) noexcept { return _mm_maskz_fnmadd_sd(k, a, b, c); }
inline m128f fnmadd_ss(m128f a, m128f b, m128f c) noexcept { return _mm_fnmadd_ss(a, b, c); }
inline m128f mask_fnmadd_ss(m128f a, mmask8 k, m128f b, m128f c) noexcept { return _mm_mask_fnmadd_ss(a, k, b, c); }
inline m128f mask3_fnmadd_ss(m128f a, m128f b, m128f c, mmask8 k) noexcept { return _mm_mask3_fnmadd_ss(a, b, c, k); }
inline m128f maskz_fnmadd_ss(mmask8 k, m128f a, m128f b, m128f c) noexcept { return _mm_maskz_fnmadd_ss(k, a, b, c); }
inline m128d fnmsub_pd(m128d a, m128d b, m128d c) noexcept { return _mm_fnmsub_pd(a, b, c); }
inline m128d mask_fnmsub_pd(m128d a, mmask8 k, m128d b, m128d c) noexcept { return _mm_mask_fnmsub_pd(a, k, b, c); }
inline m128d mask3_fnmsub_pd(m128d a, m128d b, m128d c, mmask8 k) noexcept { return _mm_mask3_fnmsub_pd(a, b, c, k); }
inline m128d maskz_fnmsub_pd(mmask8 k, m128d a, m128d b, m128d c) noexcept { return _mm_maskz_fnmsub_pd(k, a, b, c); }
inline m128f fnmsub_ps(m128f a, m128f b, m128f c) noexcept { return _mm_fnmsub_ps(a, b, c); }
inline m128f mask_fnmsub_ps(m128f a, mmask8 k, m128f b, m128f c) noexcept { return _mm_mask_fnmsub_ps(a, k, b, c); }
inline m128f mask3_fnmsub_ps(m128f a, m128f b, m128f c, mmask8 k) noexcept { return _mm_mask3_fnmsub_ps(a, b, c, k); }
inline m128f maskz_fnmsub_ps(mmask8 k, m128f a, m128f b, m128f c) noexcept { return _mm_maskz_fnmsub_ps(k, a, b, c); }
template<int4 rounding> inline m128d fnmsub_round_sd(m128d a, m128d b, m128d c) noexcept { return _mm_fnmsub_round_sd(a, b, c, rounding); }
template<int4 rounding> inline m128d mask_fnmsub_round_sd(m128d a, mmask8 k, m128d b, m128d c) noexcept { return _mm_mask_fnmsub_round_sd(a, k, b, c, rounding); }
template<int4 rounding> inline m128d mask3_fnmsub_round_sd(m128d a, m128d b, m128d c, mmask8 k) noexcept { return _mm_mask3_fnmsub_round_sd(a, b, c, k, rounding); }
template<int4 rounding> inline m128d maskz_fnmsub_round_sd(mmask8 k, m128d a, m128d b, m128d c) noexcept { return _mm_maskz_fnmsub_round_sd(k, a, b, c, rounding); }
template<int4 rounding> inline m128f fnmsub_round_ss(m128f a, m128f b, m128f c) noexcept { return _mm_fnmsub_round_ss(a, b, c, rounding); }
template<int4 rounding> inline m128f mask_fnmsub_round_ss(m128f a, mmask8 k, m128f b, m128f c) noexcept { return _mm_mask_fnmsub_round_ss(a, k, b, c, rounding); }
template<int4 rounding> inline m128f mask3_fnmsub_round_ss(m128f a, m128f b, m128f c, mmask8 k) noexcept { return _mm_mask3_fnmsub_round_ss(a, b, c, k, rounding); }
template<int4 rounding> inline m128f maskz_fnmsub_round_ss(mmask8 k, m128f a, m128f b, m128f c) noexcept { return _mm_maskz_fnmsub_round_ss(k, a, b, c, rounding); }
inline m128d fnmsub_sd(m128d a, m128d b, m128d c) noexcept { return _mm_fnmsub_sd(a, b, c); }
inline m128d mask_fnmsub_sd(m128d a, mmask8 k, m128d b, m128d c) noexcept { return _mm_mask_fnmsub_sd(a, k, b, c); }
inline m128d mask3_fnmsub_sd(m128d a, m128d b, m128d c, mmask8 k) noexcept { return _mm_mask3_fnmsub_sd(a, b, c, k); }
inline m128d maskz_fnmsub_sd(mmask8 k, m128d a, m128d b, m128d c) noexcept { return _mm_maskz_fnmsub_sd(k, a, b, c); }
inline m128f fnmsub_ss(m128f a, m128f b, m128f c) noexcept { return _mm_fnmsub_ss(a, b, c); }
inline m128f mask_fnmsub_ss(m128f a, mmask8 k, m128f b, m128f c) noexcept { return _mm_mask_fnmsub_ss(a, k, b, c); }
inline m128f mask3_fnmsub_ss(m128f a, m128f b, m128f c, mmask8 k) noexcept { return _mm_mask3_fnmsub_ss(a, b, c, k); }
inline m128f maskz_fnmsub_ss(mmask8 k, m128f a, m128f b, m128f c) noexcept { return _mm_maskz_fnmsub_ss(k, a, b, c); }
template<int4 imm8> inline mmask8 fpclass_pd_mask(m128d a) noexcept { return _mm_fpclass_pd_mask(a, imm8); }
template<int4 imm8> inline mmask8 mask_fpclass_pd_mask(mmask8 k1, m128d a) noexcept { return _mm_mask_fpclass_pd_mask(k1, a, imm8); }
template<int4 imm8> inline mmask8 fpclass_ps_mask(m128f a) noexcept { return _mm_fpclass_ps_mask(a, imm8); }
template<int4 imm8> inline mmask8 mask_fpclass_ps_mask(mmask8 k1, m128f a) noexcept { return _mm_mask_fpclass_ps_mask(k1, a, imm8); }
template<int4 imm8> inline mmask8 fpclass_sd_mask(m128d a) noexcept { return _mm_fpclass_sd_mask(a, imm8); }
template<int4 imm8> inline mmask8 mask_fpclass_sd_mask(mmask8 k1, m128d a) noexcept { return _mm_mask_fpclass_sd_mask(k1, a, imm8); }
template<int4 imm8> inline mmask8 fpclass_ss_mask(m128f a) noexcept { return _mm_fpclass_ss_mask(a, imm8); }
template<int4 imm8> inline mmask8 mask_fpclass_ss_mask(mmask8 k1, m128f a) noexcept { return _mm_mask_fpclass_ss_mask(k1, a, imm8); }
inline m128d getexp_pd(m128d a) noexcept { return _mm_getexp_pd(a); }
inline m128d mask_getexp_pd(m128d src, mmask8 k, m128d a) noexcept { return _mm_mask_getexp_pd(src, k, a); }
inline m128d maskz_getexp_pd(mmask8 k, m128d a) noexcept { return _mm_maskz_getexp_pd(k, a); }
inline m128f getexp_ps(m128f a) noexcept { return _mm_getexp_ps(a); }
inline m128f mask_getexp_ps(m128f src, mmask8 k, m128f a) noexcept { return _mm_mask_getexp_ps(src, k, a); }
inline m128f maskz_getexp_ps(mmask8 k, m128f a) noexcept { return _mm_maskz_getexp_ps(k, a); }
template<int sae> inline m128d getexp_round_sd(m128d a, m128d b) noexcept { return _mm_getexp_round_sd(a, b, sae); }
template<int sae> inline m128d mask_getexp_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_getexp_round_sd(src, k, a, b, sae); }
template<int sae> inline m128d maskz_getexp_round_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_getexp_round_sd(k, a, b, sae); }
template<int sae> inline m128f getexp_round_ss(m128f a, m128f b) noexcept { return _mm_getexp_round_ss(a, b, sae); }
template<int sae> inline m128f mask_getexp_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_getexp_round_ss(src, k, a, b, sae); }
template<int sae> inline m128f maskz_getexp_round_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_getexp_round_ss(k, a, b, sae); }
inline m128d getexp_sd(m128d a, m128d b) noexcept { return _mm_getexp_sd(a, b); }
inline m128d mask_getexp_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_getexp_sd(src, k, a, b); }
inline m128d maskz_getexp_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_getexp_sd(k, a, b); }
inline m128f getexp_ss(m128f a, m128f b) noexcept { return _mm_getexp_ss(a, b); }
inline m128f mask_getexp_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_getexp_ss(src, k, a, b); }
inline m128f maskz_getexp_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_getexp_ss(k, a, b); }
template<int sc> inline m128d getmant_pd(m128d a, MANTISSA_NORM_ENUM interv) noexcept { return _mm_getmant_pd(a, interv, sc); }
template<int sc> inline m128d mask_getmant_pd(m128d src, mmask8 k, m128d a, MANTISSA_NORM_ENUM interv) noexcept { return _mm_mask_getmant_pd(src, k, a, interv, sc); }
template<int sc> inline m128d maskz_getmant_pd(mmask8 k, m128d a, MANTISSA_NORM_ENUM interv) noexcept { return _mm_maskz_getmant_pd(k, a, interv, sc); }
template<int sc> inline m128f getmant_ps(m128f a, MANTISSA_NORM_ENUM interv) noexcept { return _mm_getmant_ps(a, interv, sc); }
template<int sc> inline m128f mask_getmant_ps(m128f src, mmask8 k, m128f a, MANTISSA_NORM_ENUM interv) noexcept { return _mm_mask_getmant_ps(src, k, a, interv, sc); }
template<int sc> inline m128f maskz_getmant_ps(mmask8 k, m128f a, MANTISSA_NORM_ENUM interv) noexcept { return _mm_maskz_getmant_ps(k, a, interv, sc); }
template<int sae, int sc> inline m128d getmant_round_sd(m128d a, m128d b, MANTISSA_NORM_ENUM interv) noexcept { return _mm_getmant_round_sd(a, b, interv, sc, sae); }
template<int sae, int sc> inline m128d mask_getmant_round_sd(m128d src, mmask8 k, m128d a, m128d b, MANTISSA_NORM_ENUM interv) noexcept { return _mm_mask_getmant_round_sd(src, k, a, b, interv, sc, sae); }
template<int sae, int sc> inline m128d maskz_getmant_round_sd(mmask8 k, m128d a, m128d b, MANTISSA_NORM_ENUM interv) noexcept { return _mm_maskz_getmant_round_sd(k, a, b, interv, sc, sae); }
template<int sae, int sc> inline m128f getmant_round_ss(m128f a, m128f b, MANTISSA_NORM_ENUM interv) noexcept { return _mm_getmant_round_ss(a, b, interv, sc, sae); }
template<int sae, int sc> inline m128f mask_getmant_round_ss(m128f src, mmask8 k, m128f a, m128f b, MANTISSA_NORM_ENUM interv) noexcept { return _mm_mask_getmant_round_ss(src, k, a, b, interv, sc, sae); }
template<int sae, int sc> inline m128f maskz_getmant_round_ss(mmask8 k, m128f a, m128f b, MANTISSA_NORM_ENUM interv) noexcept { return _mm_maskz_getmant_round_ss(k, a, b, interv, sc, sae); }
template<int sc> inline m128d getmant_sd(m128d a, m128d b, MANTISSA_NORM_ENUM interv) noexcept { return _mm_getmant_sd(a, b, interv, sc); }
template<int sc> inline m128d mask_getmant_sd(m128d src, mmask8 k, m128d a, m128d b, MANTISSA_NORM_ENUM interv) noexcept { return _mm_mask_getmant_sd(src, k, a, b, interv, sc); }
template<int sc> inline m128d maskz_getmant_sd(mmask8 k, m128d a, m128d b, MANTISSA_NORM_ENUM interv) noexcept { return _mm_maskz_getmant_sd(k, a, b, interv, sc); }
template<int sc> inline m128f getmant_ss(m128f a, m128f b, MANTISSA_NORM_ENUM interv) noexcept { return _mm_getmant_ss(a, b, interv, sc); }
template<int sc> inline m128f mask_getmant_ss(m128f src, mmask8 k, m128f a, m128f b, MANTISSA_NORM_ENUM interv) noexcept { return _mm_mask_getmant_ss(src, k, a, b, interv, sc); }
template<int sc> inline m128f maskz_getmant_ss(mmask8 k, m128f a, m128f b, MANTISSA_NORM_ENUM interv) noexcept { return _mm_maskz_getmant_ss(k, a, b, interv, sc); }
template<int b> inline m128i gf2p8affine_epi64_epi8(m128i x, m128i A) noexcept { return _mm_gf2p8affine_epi64_epi8(x, A, b); }
template<int b> inline m128i mask_gf2p8affine_epi64_epi8(m128i src, mmask16 k, m128i x, m128i A) noexcept { return _mm_mask_gf2p8affine_epi64_epi8(src, k, x, A, b); }
template<int b> inline m128i maskz_gf2p8affine_epi64_epi8(mmask16 k, m128i x, m128i A) noexcept { return _mm_maskz_gf2p8affine_epi64_epi8(k, x, A, b); }
template<int b> inline m128i gf2p8affineinv_epi64_epi8(m128i x, m128i A) noexcept { return _mm_gf2p8affineinv_epi64_epi8(x, A, b); }
template<int b> inline m128i mask_gf2p8affineinv_epi64_epi8(m128i src, mmask16 k, m128i x, m128i A) noexcept { return _mm_mask_gf2p8affineinv_epi64_epi8(src, k, x, A, b); }
template<int b> inline m128i maskz_gf2p8affineinv_epi64_epi8(mmask16 k, m128i x, m128i A) noexcept { return _mm_maskz_gf2p8affineinv_epi64_epi8(k, x, A, b); }
inline m128i gf2p8mul_epi8(m128i a, m128i b) noexcept { return _mm_gf2p8mul_epi8(a, b); }
inline m128i mask_gf2p8mul_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_gf2p8mul_epi8(src, k, a, b); }
inline m128i maskz_gf2p8mul_epi8(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_gf2p8mul_epi8(k, a, b); }
inline m128i hadd_epi16(m128i a, m128i b) noexcept { return _mm_hadd_epi16(a, b); }
inline m128i hadd_epi32(m128i a, m128i b) noexcept { return _mm_hadd_epi32(a, b); }
inline m128d hadd_pd(m128d a, m128d b) noexcept { return _mm_hadd_pd(a, b); }
inline m128f hadd_ps(m128f a, m128f b) noexcept { return _mm_hadd_ps(a, b); }
inline m128i hadds_epi16(m128i a, m128i b) noexcept { return _mm_hadds_epi16(a, b); }
inline m128i hsub_epi16(m128i a, m128i b) noexcept { return _mm_hsub_epi16(a, b); }
inline m128i hsub_epi32(m128i a, m128i b) noexcept { return _mm_hsub_epi32(a, b); }
inline m128d hsub_pd(m128d a, m128d b) noexcept { return _mm_hsub_pd(a, b); }
inline m128f hsub_ps(m128f a, m128f b) noexcept { return _mm_hsub_ps(a, b); }
inline m128i hsubs_epi16(m128i a, m128i b) noexcept { return _mm_hsubs_epi16(a, b); }
template<int scale> inline m128i i32gather_epi32(int const* base_addr, m128i vindex) noexcept { return _mm_i32gather_epi32(base_addr, vindex, scale); }
template<int scale> inline m128i mask_i32gather_epi32(m128i src, int const* base_addr, m128i vindex, m128i mask) noexcept { return _mm_mask_i32gather_epi32(src, base_addr, vindex, mask, scale); }
template<int scale> inline m128i mmask_i32gather_epi32(m128i src, mmask8 k, m128i vindex, void const* base_addr) noexcept { return _mm_mmask_i32gather_epi32(src, k, vindex, base_addr, scale); }
template<int scale> inline m128i i32gather_epi64(int8 const* base_addr, m128i vindex) noexcept { return _mm_i32gather_epi64(base_addr, vindex, scale); }
template<int scale> inline m128i mask_i32gather_epi64(m128i src, int8 const* base_addr, m128i vindex, m128i mask) noexcept { return _mm_mask_i32gather_epi64(src, base_addr, vindex, mask, scale); }
template<int scale> inline m128i mmask_i32gather_epi64(m128i src, mmask8 k, m128i vindex, void const* base_addr) noexcept { return _mm_mmask_i32gather_epi64(src, k, vindex, base_addr, scale); }
template<int scale> inline m128d i32gather_pd(double const* base_addr, m128i vindex) noexcept { return _mm_i32gather_pd(base_addr, vindex, scale); }
template<int scale> inline m128d mask_i32gather_pd(m128d src, double const* base_addr, m128i vindex, m128d mask) noexcept { return _mm_mask_i32gather_pd(src, base_addr, vindex, mask, scale); }
template<int scale> inline m128d mmask_i32gather_pd(m128d src, mmask8 k, m128i vindex, void const* base_addr) noexcept { return _mm_mmask_i32gather_pd(src, k, vindex, base_addr, scale); }
template<int scale> inline m128f i32gather_ps(float const* base_addr, m128i vindex) noexcept { return _mm_i32gather_ps(base_addr, vindex, scale); }
template<int scale> inline m128f mask_i32gather_ps(m128f src, float const* base_addr, m128i vindex, m128f mask) noexcept { return _mm_mask_i32gather_ps(src, base_addr, vindex, mask, scale); }
template<int scale> inline m128f mmask_i32gather_ps(m128f src, mmask8 k, m128i vindex, void const* base_addr) noexcept { return _mm_mmask_i32gather_ps(src, k, vindex, base_addr, scale); }
template<int scale> inline void i32scatter_epi32(void* base_addr, m128i vindex, m128i a) noexcept { return _mm_i32scatter_epi32(base_addr, vindex, a, scale); }
template<int scale> inline void mask_i32scatter_epi32(void* base_addr, mmask8 k, m128i vindex, m128i a) noexcept { return _mm_mask_i32scatter_epi32(base_addr, k, vindex, a, scale); }
template<int scale> inline void i32scatter_epi64(void* base_addr, m128i vindex, m128i a) noexcept { return _mm_i32scatter_epi64(base_addr, vindex, a, scale); }
template<int scale> inline void mask_i32scatter_epi64(void* base_addr, mmask8 k, m128i vindex, m128i a) noexcept { return _mm_mask_i32scatter_epi64(base_addr, k, vindex, a, scale); }
template<int scale> inline void i32scatter_pd(void* base_addr, m128i vindex, m128d a) noexcept { return _mm_i32scatter_pd(base_addr, vindex, a, scale); }
template<int scale> inline void mask_i32scatter_pd(void* base_addr, mmask8 k, m128i vindex, m128d a) noexcept { return _mm_mask_i32scatter_pd(base_addr, k, vindex, a, scale); }
template<int scale> inline void i32scatter_ps(void* base_addr, m128i vindex, m128f a) noexcept { return _mm_i32scatter_ps(base_addr, vindex, a, scale); }
template<int scale> inline void mask_i32scatter_ps(void* base_addr, mmask8 k, m128i vindex, m128f a) noexcept { return _mm_mask_i32scatter_ps(base_addr, k, vindex, a, scale); }
template<int scale> inline m128i i64gather_epi32(int const* base_addr, m128i vindex) noexcept { return _mm_i64gather_epi32(base_addr, vindex, scale); }
template<int scale> inline m128i mask_i64gather_epi32(m128i src, int const* base_addr, m128i vindex, m128i mask) noexcept { return _mm_mask_i64gather_epi32(src, base_addr, vindex, mask, scale); }
template<int scale> inline m128i mmask_i64gather_epi32(m128i src, mmask8 k, m128i vindex, void const* base_addr) noexcept { return _mm_mmask_i64gather_epi32(src, k, vindex, base_addr, scale); }
template<int scale> inline m128i i64gather_epi64(int8 const* base_addr, m128i vindex) noexcept { return _mm_i64gather_epi64(base_addr, vindex, scale); }
template<int scale> inline m128i mask_i64gather_epi64(m128i src, int8 const* base_addr, m128i vindex, m128i mask) noexcept { return _mm_mask_i64gather_epi64(src, base_addr, vindex, mask, scale); }
template<int scale> inline m128i mmask_i64gather_epi64(m128i src, mmask8 k, m128i vindex, void const* base_addr) noexcept { return _mm_mmask_i64gather_epi64(src, k, vindex, base_addr, scale); }
template<int scale> inline m128d i64gather_pd(double const* base_addr, m128i vindex) noexcept { return _mm_i64gather_pd(base_addr, vindex, scale); }
template<int scale> inline m128d mask_i64gather_pd(m128d src, double const* base_addr, m128i vindex, m128d mask) noexcept { return _mm_mask_i64gather_pd(src, base_addr, vindex, mask, scale); }
template<int scale> inline m128d mmask_i64gather_pd(m128d src, mmask8 k, m128i vindex, void const* base_addr) noexcept { return _mm_mmask_i64gather_pd(src, k, vindex, base_addr, scale); }
template<int scale> inline m128f i64gather_ps(float const* base_addr, m128i vindex) noexcept { return _mm_i64gather_ps(base_addr, vindex, scale); }
template<int scale> inline m128f mask_i64gather_ps(m128f src, float const* base_addr, m128i vindex, m128f mask) noexcept { return _mm_mask_i64gather_ps(src, base_addr, vindex, mask, scale); }
template<int scale> inline m128f mmask_i64gather_ps(m128f src, mmask8 k, m128i vindex, void const* base_addr) noexcept { return _mm_mmask_i64gather_ps(src, k, vindex, base_addr, scale); }
template<int scale> inline void i64scatter_epi32(void* base_addr, m128i vindex, m128i a) noexcept { return _mm_i64scatter_epi32(base_addr, vindex, a, scale); }
template<int scale> inline void mask_i64scatter_epi32(void* base_addr, mmask8 k, m128i vindex, m128i a) noexcept { return _mm_mask_i64scatter_epi32(base_addr, k, vindex, a, scale); }
template<int scale> inline void i64scatter_epi64(void* base_addr, m128i vindex, m128i a) noexcept { return _mm_i64scatter_epi64(base_addr, vindex, a, scale); }
template<int scale> inline void mask_i64scatter_epi64(void* base_addr, mmask8 k, m128i vindex, m128i a) noexcept { return _mm_mask_i64scatter_epi64(base_addr, k, vindex, a, scale); }
template<int scale> inline void i64scatter_pd(void* base_addr, m128i vindex, m128d a) noexcept { return _mm_i64scatter_pd(base_addr, vindex, a, scale); }
template<int scale> inline void mask_i64scatter_pd(void* base_addr, mmask8 k, m128i vindex, m128d a) noexcept { return _mm_mask_i64scatter_pd(base_addr, k, vindex, a, scale); }
template<int scale> inline void i64scatter_ps(void* base_addr, m128i vindex, m128f a) noexcept { return _mm_i64scatter_ps(base_addr, vindex, a, scale); }
template<int scale> inline void mask_i64scatter_ps(void* base_addr, mmask8 k, m128i vindex, m128f a) noexcept { return _mm_mask_i64scatter_ps(base_addr, k, vindex, a, scale); }
template<int4 imm8> inline m128i insert_epi16(m128i a, int i) noexcept { return _mm_insert_epi16(a, i, imm8); }
template<int4 imm8> inline m128i insert_epi32(m128i a, int i) noexcept { return _mm_insert_epi32(a, i, imm8); }
template<int4 imm8> inline m128i insert_epi64(m128i a, int8 i) noexcept { return _mm_insert_epi64(a, i, imm8); }
template<int4 imm8> inline m128i insert_epi8(m128i a, int i) noexcept { return _mm_insert_epi8(a, i, imm8); }
template<int4 imm8> inline m128f insert_ps(m128f a, m128f b) noexcept { return _mm_insert_ps(a, b, imm8); }
inline m128i lddqu_si128(m128i const* mem_addr) noexcept { return _mm_lddqu_si128(mem_addr); }
inline m128i mask_load_epi32(m128i src, mmask8 k, void const* mem_addr) noexcept { return _mm_mask_load_epi32(src, k, mem_addr); }
inline m128i maskz_load_epi32(mmask8 k, void const* mem_addr) noexcept { return _mm_maskz_load_epi32(k, mem_addr); }
inline m128i mask_load_epi64(m128i src, mmask8 k, void const* mem_addr) noexcept { return _mm_mask_load_epi64(src, k, mem_addr); }
inline m128i maskz_load_epi64(mmask8 k, void const* mem_addr) noexcept { return _mm_maskz_load_epi64(k, mem_addr); }
inline m128d load_pd(double const* mem_addr) noexcept { return _mm_load_pd(mem_addr); }
inline m128d mask_load_pd(m128d src, mmask8 k, void const* mem_addr) noexcept { return _mm_mask_load_pd(src, k, mem_addr); }
inline m128d maskz_load_pd(mmask8 k, void const* mem_addr) noexcept { return _mm_maskz_load_pd(k, mem_addr); }
inline m128d load_pd1(double const* mem_addr) noexcept { return _mm_load_pd1(mem_addr); }
inline m128f load_ps(float const* mem_addr) noexcept { return _mm_load_ps(mem_addr); }
inline m128f mask_load_ps(m128f src, mmask8 k, void const* mem_addr) noexcept { return _mm_mask_load_ps(src, k, mem_addr); }
inline m128f maskz_load_ps(mmask8 k, void const* mem_addr) noexcept { return _mm_maskz_load_ps(k, mem_addr); }
inline m128f load_ps1(float const* mem_addr) noexcept { return _mm_load_ps1(mem_addr); }
inline m128d load_sd(double const* mem_addr) noexcept { return _mm_load_sd(mem_addr); }
inline m128d mask_load_sd(m128d src, mmask8 k, const double* mem_addr) noexcept { return _mm_mask_load_sd(src, k, mem_addr); }
inline m128d maskz_load_sd(mmask8 k, const double* mem_addr) noexcept { return _mm_maskz_load_sd(k, mem_addr); }
inline m128i load_si128(m128i const* mem_addr) noexcept { return _mm_load_si128(mem_addr); }
inline m128f load_ss(float const* mem_addr) noexcept { return _mm_load_ss(mem_addr); }
inline m128f mask_load_ss(m128f src, mmask8 k, const float* mem_addr) noexcept { return _mm_mask_load_ss(src, k, mem_addr); }
inline m128f maskz_load_ss(mmask8 k, const float* mem_addr) noexcept { return _mm_maskz_load_ss(k, mem_addr); }
inline m128d load1_pd(double const* mem_addr) noexcept { return _mm_load1_pd(mem_addr); }
inline m128f load1_ps(float const* mem_addr) noexcept { return _mm_load1_ps(mem_addr); }
inline m128d loaddup_pd(double const* mem_addr) noexcept { return _mm_loaddup_pd(mem_addr); }
inline m128d loadh_pd(m128d a, double const* mem_addr) noexcept { return _mm_loadh_pd(a, mem_addr); }
inline void loadiwkey(nat4 __ctl, m128i __intkey, m128i __enkey_lo, m128i __enkey_hi) noexcept { return _mm_loadiwkey(__ctl, __intkey, __enkey_lo, __enkey_hi); }
inline m128i loadl_epi64(m128i const* mem_addr) noexcept { return _mm_loadl_epi64(mem_addr); }
inline m128d loadl_pd(m128d a, double const* mem_addr) noexcept { return _mm_loadl_pd(a, mem_addr); }
inline m128d loadr_pd(double const* mem_addr) noexcept { return _mm_loadr_pd(mem_addr); }
inline m128f loadr_ps(float const* mem_addr) noexcept { return _mm_loadr_ps(mem_addr); }
inline m128i loadu_epi16(void const* mem_addr) noexcept { return _mm_loadu_epi16(mem_addr); }
inline m128i mask_loadu_epi16(m128i src, mmask8 k, void const* mem_addr) noexcept { return _mm_mask_loadu_epi16(src, k, mem_addr); }
inline m128i maskz_loadu_epi16(mmask8 k, void const* mem_addr) noexcept { return _mm_maskz_loadu_epi16(k, mem_addr); }
inline m128i loadu_epi32(void const* mem_addr) noexcept { return _mm_loadu_epi32(mem_addr); }
inline m128i mask_loadu_epi32(m128i src, mmask8 k, void const* mem_addr) noexcept { return _mm_mask_loadu_epi32(src, k, mem_addr); }
inline m128i maskz_loadu_epi32(mmask8 k, void const* mem_addr) noexcept { return _mm_maskz_loadu_epi32(k, mem_addr); }
inline m128i loadu_epi64(void const* mem_addr) noexcept { return _mm_loadu_epi64(mem_addr); }
inline m128i mask_loadu_epi64(m128i src, mmask8 k, void const* mem_addr) noexcept { return _mm_mask_loadu_epi64(src, k, mem_addr); }
inline m128i maskz_loadu_epi64(mmask8 k, void const* mem_addr) noexcept { return _mm_maskz_loadu_epi64(k, mem_addr); }
inline m128i loadu_epi8(void const* mem_addr) noexcept { return _mm_loadu_epi8(mem_addr); }
inline m128i mask_loadu_epi8(m128i src, mmask16 k, void const* mem_addr) noexcept { return _mm_mask_loadu_epi8(src, k, mem_addr); }
inline m128i maskz_loadu_epi8(mmask16 k, void const* mem_addr) noexcept { return _mm_maskz_loadu_epi8(k, mem_addr); }
inline m128d loadu_pd(double const* mem_addr) noexcept { return _mm_loadu_pd(mem_addr); }
inline m128d mask_loadu_pd(m128d src, mmask8 k, void const* mem_addr) noexcept { return _mm_mask_loadu_pd(src, k, mem_addr); }
inline m128d maskz_loadu_pd(mmask8 k, void const* mem_addr) noexcept { return _mm_maskz_loadu_pd(k, mem_addr); }
inline m128f loadu_ps(float const* mem_addr) noexcept { return _mm_loadu_ps(mem_addr); }
inline m128f mask_loadu_ps(m128f src, mmask8 k, void const* mem_addr) noexcept { return _mm_mask_loadu_ps(src, k, mem_addr); }
inline m128f maskz_loadu_ps(mmask8 k, void const* mem_addr) noexcept { return _mm_maskz_loadu_ps(k, mem_addr); }
inline m128i loadu_si128(m128i const* mem_addr) noexcept { return _mm_loadu_si128(mem_addr); }
inline m128i loadu_si16(void const* mem_addr) noexcept { return _mm_loadu_si16(mem_addr); }
inline m128i loadu_si32(void const* mem_addr) noexcept { return _mm_loadu_si32(mem_addr); }
inline m128i loadu_si64(void const* mem_addr) noexcept { return _mm_loadu_si64(mem_addr); }
inline m128i lzcnt_epi32(m128i a) noexcept { return _mm_lzcnt_epi32(a); }
inline m128i mask_lzcnt_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_lzcnt_epi32(src, k, a); }
inline m128i maskz_lzcnt_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_lzcnt_epi32(k, a); }
inline m128i lzcnt_epi64(m128i a) noexcept { return _mm_lzcnt_epi64(a); }
inline m128i mask_lzcnt_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_lzcnt_epi64(src, k, a); }
inline m128i maskz_lzcnt_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_lzcnt_epi64(k, a); }
inline m128i madd_epi16(m128i a, m128i b) noexcept { return _mm_madd_epi16(a, b); }
inline m128i mask_madd_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_madd_epi16(src, k, a, b); }
inline m128i maskz_madd_epi16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_madd_epi16(k, a, b); }
inline m128i madd52hi_epu64(m128i a, m128i b, m128i c) noexcept { return _mm_madd52hi_epu64(a, b, c); }
inline m128i mask_madd52hi_epu64(m128i a, mmask8 k, m128i b, m128i c) noexcept { return _mm_mask_madd52hi_epu64(a, k, b, c); }
inline m128i maskz_madd52hi_epu64(mmask8 k, m128i a, m128i b, m128i c) noexcept { return _mm_maskz_madd52hi_epu64(k, a, b, c); }
inline m128i madd52lo_epu64(m128i a, m128i b, m128i c) noexcept { return _mm_madd52lo_epu64(a, b, c); }
inline m128i mask_madd52lo_epu64(m128i a, mmask8 k, m128i b, m128i c) noexcept { return _mm_mask_madd52lo_epu64(a, k, b, c); }
inline m128i maskz_madd52lo_epu64(mmask8 k, m128i a, m128i b, m128i c) noexcept { return _mm_maskz_madd52lo_epu64(k, a, b, c); }
inline m128i maddubs_epi16(m128i a, m128i b) noexcept { return _mm_maddubs_epi16(a, b); }
inline m128i mask_maddubs_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_maddubs_epi16(src, k, a, b); }
inline m128i maskz_maddubs_epi16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_maddubs_epi16(k, a, b); }
inline m128i maskload_epi32(int const* mem_addr, m128i mask) noexcept { return _mm_maskload_epi32(mem_addr, mask); }
inline m128i maskload_epi64(int8 const* mem_addr, m128i mask) noexcept { return _mm_maskload_epi64(mem_addr, mask); }
inline m128d maskload_pd(double const* mem_addr, m128i mask) noexcept { return _mm_maskload_pd(mem_addr, mask); }
inline m128f maskload_ps(float const* mem_addr, m128i mask) noexcept { return _mm_maskload_ps(mem_addr, mask); }
inline void maskmoveu_si128(m128i a, m128i mask, char* mem_addr) noexcept { return _mm_maskmoveu_si128(a, mask, mem_addr); }
inline void maskstore_epi32(int* mem_addr, m128i mask, m128i a) noexcept { return _mm_maskstore_epi32(mem_addr, mask, a); }
inline void maskstore_epi64(int8* mem_addr, m128i mask, m128i a) noexcept { return _mm_maskstore_epi64(mem_addr, mask, a); }
inline void maskstore_pd(double* mem_addr, m128i mask, m128d a) noexcept { return _mm_maskstore_pd(mem_addr, mask, a); }
inline void maskstore_ps(float* mem_addr, m128i mask, m128f a) noexcept { return _mm_maskstore_ps(mem_addr, mask, a); }
inline m128i mask_max_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_max_epi16(src, k, a, b); }
inline m128i maskz_max_epi16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_max_epi16(k, a, b); }
inline m128i max_epi16(m128i a, m128i b) noexcept { return _mm_max_epi16(a, b); }
inline m128i mask_max_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_max_epi32(src, k, a, b); }
inline m128i maskz_max_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_max_epi32(k, a, b); }
inline m128i max_epi32(m128i a, m128i b) noexcept { return _mm_max_epi32(a, b); }
inline m128i mask_max_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_max_epi64(src, k, a, b); }
inline m128i maskz_max_epi64(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_max_epi64(k, a, b); }
inline m128i max_epi64(m128i a, m128i b) noexcept { return _mm_max_epi64(a, b); }
inline m128i mask_max_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_max_epi8(src, k, a, b); }
inline m128i maskz_max_epi8(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_max_epi8(k, a, b); }
inline m128i max_epi8(m128i a, m128i b) noexcept { return _mm_max_epi8(a, b); }
inline m128i mask_max_epu16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_max_epu16(src, k, a, b); }
inline m128i maskz_max_epu16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_max_epu16(k, a, b); }
inline m128i max_epu16(m128i a, m128i b) noexcept { return _mm_max_epu16(a, b); }
inline m128i mask_max_epu32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_max_epu32(src, k, a, b); }
inline m128i maskz_max_epu32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_max_epu32(k, a, b); }
inline m128i max_epu32(m128i a, m128i b) noexcept { return _mm_max_epu32(a, b); }
inline m128i mask_max_epu64(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_max_epu64(src, k, a, b); }
inline m128i maskz_max_epu64(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_max_epu64(k, a, b); }
inline m128i max_epu64(m128i a, m128i b) noexcept { return _mm_max_epu64(a, b); }
inline m128i mask_max_epu8(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_max_epu8(src, k, a, b); }
inline m128i maskz_max_epu8(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_max_epu8(k, a, b); }
inline m128i max_epu8(m128i a, m128i b) noexcept { return _mm_max_epu8(a, b); }
inline m128d mask_max_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_max_pd(src, k, a, b); }
inline m128d maskz_max_pd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_max_pd(k, a, b); }
inline m128d max_pd(m128d a, m128d b) noexcept { return _mm_max_pd(a, b); }
inline m128f mask_max_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_max_ps(src, k, a, b); }
inline m128f maskz_max_ps(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_max_ps(k, a, b); }
inline m128f max_ps(m128f a, m128f b) noexcept { return _mm_max_ps(a, b); }
template<int sae> inline m128d mask_max_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_max_round_sd(src, k, a, b, sae); }
template<int sae> inline m128d maskz_max_round_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_max_round_sd(k, a, b, sae); }
template<int sae> inline m128d max_round_sd(m128d a, m128d b) noexcept { return _mm_max_round_sd(a, b, sae); }
template<int sae> inline m128f mask_max_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_max_round_ss(src, k, a, b, sae); }
template<int sae> inline m128f maskz_max_round_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_max_round_ss(k, a, b, sae); }
template<int sae> inline m128f max_round_ss(m128f a, m128f b) noexcept { return _mm_max_round_ss(a, b, sae); }
inline m128d mask_max_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_max_sd(src, k, a, b); }
inline m128d maskz_max_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_max_sd(k, a, b); }
inline m128d max_sd(m128d a, m128d b) noexcept { return _mm_max_sd(a, b); }
inline m128f mask_max_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_max_ss(src, k, a, b); }
inline m128f maskz_max_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_max_ss(k, a, b); }
inline m128f max_ss(m128f a, m128f b) noexcept { return _mm_max_ss(a, b); }
inline m128i mask_min_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_min_epi16(src, k, a, b); }
inline m128i maskz_min_epi16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_min_epi16(k, a, b); }
inline m128i min_epi16(m128i a, m128i b) noexcept { return _mm_min_epi16(a, b); }
inline m128i mask_min_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_min_epi32(src, k, a, b); }
inline m128i maskz_min_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_min_epi32(k, a, b); }
inline m128i min_epi32(m128i a, m128i b) noexcept { return _mm_min_epi32(a, b); }
inline m128i mask_min_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_min_epi64(src, k, a, b); }
inline m128i maskz_min_epi64(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_min_epi64(k, a, b); }
inline m128i min_epi64(m128i a, m128i b) noexcept { return _mm_min_epi64(a, b); }
inline m128i mask_min_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_min_epi8(src, k, a, b); }
inline m128i maskz_min_epi8(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_min_epi8(k, a, b); }
inline m128i min_epi8(m128i a, m128i b) noexcept { return _mm_min_epi8(a, b); }
inline m128i mask_min_epu16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_min_epu16(src, k, a, b); }
inline m128i maskz_min_epu16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_min_epu16(k, a, b); }
inline m128i min_epu16(m128i a, m128i b) noexcept { return _mm_min_epu16(a, b); }
inline m128i mask_min_epu32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_min_epu32(src, k, a, b); }
inline m128i maskz_min_epu32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_min_epu32(k, a, b); }
inline m128i min_epu32(m128i a, m128i b) noexcept { return _mm_min_epu32(a, b); }
inline m128i mask_min_epu64(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_min_epu64(src, k, a, b); }
inline m128i maskz_min_epu64(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_min_epu64(k, a, b); }
inline m128i min_epu64(m128i a, m128i b) noexcept { return _mm_min_epu64(a, b); }
inline m128i mask_min_epu8(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_min_epu8(src, k, a, b); }
inline m128i maskz_min_epu8(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_min_epu8(k, a, b); }
inline m128i min_epu8(m128i a, m128i b) noexcept { return _mm_min_epu8(a, b); }
inline m128d mask_min_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_min_pd(src, k, a, b); }
inline m128d maskz_min_pd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_min_pd(k, a, b); }
inline m128d min_pd(m128d a, m128d b) noexcept { return _mm_min_pd(a, b); }
inline m128f mask_min_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_min_ps(src, k, a, b); }
inline m128f maskz_min_ps(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_min_ps(k, a, b); }
inline m128f min_ps(m128f a, m128f b) noexcept { return _mm_min_ps(a, b); }
template<int sae> inline m128d mask_min_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_min_round_sd(src, k, a, b, sae); }
template<int sae> inline m128d maskz_min_round_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_min_round_sd(k, a, b, sae); }
template<int sae> inline m128d min_round_sd(m128d a, m128d b) noexcept { return _mm_min_round_sd(a, b, sae); }
template<int sae> inline m128f mask_min_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_min_round_ss(src, k, a, b, sae); }
template<int sae> inline m128f maskz_min_round_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_min_round_ss(k, a, b, sae); }
template<int sae> inline m128f min_round_ss(m128f a, m128f b) noexcept { return _mm_min_round_ss(a, b, sae); }
inline m128d mask_min_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_min_sd(src, k, a, b); }
inline m128d maskz_min_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_min_sd(k, a, b); }
inline m128d min_sd(m128d a, m128d b) noexcept { return _mm_min_sd(a, b); }
inline m128f mask_min_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_min_ss(src, k, a, b); }
inline m128f maskz_min_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_min_ss(k, a, b); }
inline m128f min_ss(m128f a, m128f b) noexcept { return _mm_min_ss(a, b); }
inline m128i minpos_epu16(m128i a) noexcept { return _mm_minpos_epu16(a); }
inline m128i mask_mov_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_mov_epi16(src, k, a); }
inline m128i maskz_mov_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_mov_epi16(k, a); }
inline m128i mask_mov_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_mov_epi32(src, k, a); }
inline m128i maskz_mov_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_mov_epi32(k, a); }
inline m128i mask_mov_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_mov_epi64(src, k, a); }
inline m128i maskz_mov_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_mov_epi64(k, a); }
inline m128i mask_mov_epi8(m128i src, mmask16 k, m128i a) noexcept { return _mm_mask_mov_epi8(src, k, a); }
inline m128i maskz_mov_epi8(mmask16 k, m128i a) noexcept { return _mm_maskz_mov_epi8(k, a); }
inline m128d mask_mov_pd(m128d src, mmask8 k, m128d a) noexcept { return _mm_mask_mov_pd(src, k, a); }
inline m128d maskz_mov_pd(mmask8 k, m128d a) noexcept { return _mm_maskz_mov_pd(k, a); }
inline m128f mask_mov_ps(m128f src, mmask8 k, m128f a) noexcept { return _mm_mask_mov_ps(src, k, a); }
inline m128f maskz_mov_ps(mmask8 k, m128f a) noexcept { return _mm_maskz_mov_ps(k, a); }
inline m128i move_epi64(m128i a) noexcept { return _mm_move_epi64(a); }
inline m128d mask_move_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_move_sd(src, k, a, b); }
inline m128d maskz_move_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_move_sd(k, a, b); }
inline m128d move_sd(m128d a, m128d b) noexcept { return _mm_move_sd(a, b); }
inline m128f mask_move_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_move_ss(src, k, a, b); }
inline m128f maskz_move_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_move_ss(k, a, b); }
inline m128f move_ss(m128f a, m128f b) noexcept { return _mm_move_ss(a, b); }
inline m128d mask_movedup_pd(m128d src, mmask8 k, m128d a) noexcept { return _mm_mask_movedup_pd(src, k, a); }
inline m128d maskz_movedup_pd(mmask8 k, m128d a) noexcept { return _mm_maskz_movedup_pd(k, a); }
inline m128d movedup_pd(m128d a) noexcept { return _mm_movedup_pd(a); }
inline m128f mask_movehdup_ps(m128f src, mmask8 k, m128f a) noexcept { return _mm_mask_movehdup_ps(src, k, a); }
inline m128f maskz_movehdup_ps(mmask8 k, m128f a) noexcept { return _mm_maskz_movehdup_ps(k, a); }
inline m128f movehdup_ps(m128f a) noexcept { return _mm_movehdup_ps(a); }
inline m128f movehl_ps(m128f a, m128f b) noexcept { return _mm_movehl_ps(a, b); }
inline m128f mask_moveldup_ps(m128f src, mmask8 k, m128f a) noexcept { return _mm_mask_moveldup_ps(src, k, a); }
inline m128f maskz_moveldup_ps(mmask8 k, m128f a) noexcept { return _mm_maskz_moveldup_ps(k, a); }
inline m128f moveldup_ps(m128f a) noexcept { return _mm_moveldup_ps(a); }
inline m128f movelh_ps(m128f a, m128f b) noexcept { return _mm_movelh_ps(a, b); }
inline int movemask_epi8(m128i a) noexcept { return _mm_movemask_epi8(a); }
inline int movemask_pd(m128d a) noexcept { return _mm_movemask_pd(a); }
inline int movemask_ps(m128f a) noexcept { return _mm_movemask_ps(a); }
inline mmask8 movepi16_mask(m128i a) noexcept { return _mm_movepi16_mask(a); }
inline mmask8 movepi32_mask(m128i a) noexcept { return _mm_movepi32_mask(a); }
inline mmask8 movepi64_mask(m128i a) noexcept { return _mm_movepi64_mask(a); }
inline mmask16 movepi8_mask(m128i a) noexcept { return _mm_movepi8_mask(a); }
inline m128i movm_epi16(mmask8 k) noexcept { return _mm_movm_epi16(k); }
inline m128i movm_epi32(mmask8 k) noexcept { return _mm_movm_epi32(k); }
inline m128i movm_epi64(mmask8 k) noexcept { return _mm_movm_epi64(k); }
inline m128i movm_epi8(mmask16 k) noexcept { return _mm_movm_epi8(k); }
template<int4 imm8> inline m128i mpsadbw_epu8(m128i a, m128i b) noexcept { return _mm_mpsadbw_epu8(a, b, imm8); }
inline m128i mask_mul_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_mul_epi32(src, k, a, b); }
inline m128i maskz_mul_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_mul_epi32(k, a, b); }
inline m128i mul_epi32(m128i a, m128i b) noexcept { return _mm_mul_epi32(a, b); }
inline m128i mask_mul_epu32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_mul_epu32(src, k, a, b); }
inline m128i maskz_mul_epu32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_mul_epu32(k, a, b); }
inline m128i mul_epu32(m128i a, m128i b) noexcept { return _mm_mul_epu32(a, b); }
inline m128d mask_mul_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_mul_pd(src, k, a, b); }
inline m128d maskz_mul_pd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_mul_pd(k, a, b); }
inline m128d mul_pd(m128d a, m128d b) noexcept { return _mm_mul_pd(a, b); }
inline m128f mask_mul_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_mul_ps(src, k, a, b); }
inline m128f maskz_mul_ps(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_mul_ps(k, a, b); }
inline m128f mul_ps(m128f a, m128f b) noexcept { return _mm_mul_ps(a, b); }
template<int4 rounding> inline m128d mask_mul_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_mul_round_sd(src, k, a, b, rounding); }
template<int4 rounding> inline m128d maskz_mul_round_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_mul_round_sd(k, a, b, rounding); }
template<int4 rounding> inline m128d mul_round_sd(m128d a, m128d b) noexcept { return _mm_mul_round_sd(a, b, rounding); }
template<int4 rounding> inline m128f mask_mul_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_mul_round_ss(src, k, a, b, rounding); }
template<int4 rounding> inline m128f maskz_mul_round_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_mul_round_ss(k, a, b, rounding); }
template<int4 rounding> inline m128f mul_round_ss(m128f a, m128f b) noexcept { return _mm_mul_round_ss(a, b, rounding); }
inline m128d mask_mul_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_mul_sd(src, k, a, b); }
inline m128d maskz_mul_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_mul_sd(k, a, b); }
inline m128d mul_sd(m128d a, m128d b) noexcept { return _mm_mul_sd(a, b); }
inline m128f mask_mul_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_mul_ss(src, k, a, b); }
inline m128f maskz_mul_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_mul_ss(k, a, b); }
inline m128f mul_ss(m128f a, m128f b) noexcept { return _mm_mul_ss(a, b); }
inline m128i mask_mulhi_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_mulhi_epi16(src, k, a, b); }
inline m128i maskz_mulhi_epi16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_mulhi_epi16(k, a, b); }
inline m128i mulhi_epi16(m128i a, m128i b) noexcept { return _mm_mulhi_epi16(a, b); }
inline m128i mask_mulhi_epu16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_mulhi_epu16(src, k, a, b); }
inline m128i maskz_mulhi_epu16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_mulhi_epu16(k, a, b); }
inline m128i mulhi_epu16(m128i a, m128i b) noexcept { return _mm_mulhi_epu16(a, b); }
inline m128i mask_mulhrs_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_mulhrs_epi16(src, k, a, b); }
inline m128i maskz_mulhrs_epi16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_mulhrs_epi16(k, a, b); }
inline m128i mulhrs_epi16(m128i a, m128i b) noexcept { return _mm_mulhrs_epi16(a, b); }
inline m128i mask_mullo_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_mullo_epi16(src, k, a, b); }
inline m128i maskz_mullo_epi16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_mullo_epi16(k, a, b); }
inline m128i mullo_epi16(m128i a, m128i b) noexcept { return _mm_mullo_epi16(a, b); }
inline m128i mask_mullo_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_mullo_epi32(src, k, a, b); }
inline m128i maskz_mullo_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_mullo_epi32(k, a, b); }
inline m128i mullo_epi32(m128i a, m128i b) noexcept { return _mm_mullo_epi32(a, b); }
inline m128i mask_mullo_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_mullo_epi64(src, k, a, b); }
inline m128i maskz_mullo_epi64(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_mullo_epi64(k, a, b); }
inline m128i mullo_epi64(m128i a, m128i b) noexcept { return _mm_mullo_epi64(a, b); }
inline m128i mask_multishift_epi64_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_multishift_epi64_epi8(src, k, a, b); }
inline m128i maskz_multishift_epi64_epi8(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_multishift_epi64_epi8(k, a, b); }
inline m128i multishift_epi64_epi8(m128i a, m128i b) noexcept { return _mm_multishift_epi64_epi8(a, b); }
inline m128i mask_or_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_or_epi32(src, k, a, b); }
inline m128i maskz_or_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_or_epi32(k, a, b); }
inline m128i mask_or_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_or_epi64(src, k, a, b); }
inline m128i maskz_or_epi64(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_or_epi64(k, a, b); }
inline m128d mask_or_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_or_pd(src, k, a, b); }
inline m128d maskz_or_pd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_or_pd(k, a, b); }
inline m128d or_pd(m128d a, m128d b) noexcept { return _mm_or_pd(a, b); }
inline m128f mask_or_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_or_ps(src, k, a, b); }
inline m128f maskz_or_ps(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_or_ps(k, a, b); }
inline m128f or_ps(m128f a, m128f b) noexcept { return _mm_or_ps(a, b); }
inline m128i or_si128(m128i a, m128i b) noexcept { return _mm_or_si128(a, b); }
inline m128i mask_packs_epi16(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_packs_epi16(src, k, a, b); }
inline m128i maskz_packs_epi16(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_packs_epi16(k, a, b); }
inline m128i packs_epi16(m128i a, m128i b) noexcept { return _mm_packs_epi16(a, b); }
inline m128i mask_packs_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_packs_epi32(src, k, a, b); }
inline m128i maskz_packs_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_packs_epi32(k, a, b); }
inline m128i packs_epi32(m128i a, m128i b) noexcept { return _mm_packs_epi32(a, b); }
inline m128i mask_packus_epi16(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_packus_epi16(src, k, a, b); }
inline m128i maskz_packus_epi16(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_packus_epi16(k, a, b); }
inline m128i packus_epi16(m128i a, m128i b) noexcept { return _mm_packus_epi16(a, b); }
inline m128i mask_packus_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_packus_epi32(src, k, a, b); }
inline m128i maskz_packus_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_packus_epi32(k, a, b); }
inline m128i packus_epi32(m128i a, m128i b) noexcept { return _mm_packus_epi32(a, b); }
template<int4 imm8> inline m128d mask_permute_pd(m128d src, mmask8 k, m128d a) noexcept { return _mm_mask_permute_pd(src, k, a, imm8); }
template<int4 imm8> inline m128d maskz_permute_pd(mmask8 k, m128d a) noexcept { return _mm_maskz_permute_pd(k, a, imm8); }
template<int4 imm8> inline m128d permute_pd(m128d a) noexcept { return _mm_permute_pd(a, imm8); }
template<int4 imm8> inline m128f mask_permute_ps(m128f src, mmask8 k, m128f a) noexcept { return _mm_mask_permute_ps(src, k, a, imm8); }
template<int4 imm8> inline m128f maskz_permute_ps(mmask8 k, m128f a) noexcept { return _mm_maskz_permute_ps(k, a, imm8); }
template<int4 imm8> inline m128f permute_ps(m128f a) noexcept { return _mm_permute_ps(a, imm8); }
inline m128d mask_permutevar_pd(m128d src, mmask8 k, m128d a, m128i b) noexcept { return _mm_mask_permutevar_pd(src, k, a, b); }
inline m128d maskz_permutevar_pd(mmask8 k, m128d a, m128i b) noexcept { return _mm_maskz_permutevar_pd(k, a, b); }
inline m128d permutevar_pd(m128d a, m128i b) noexcept { return _mm_permutevar_pd(a, b); }
inline m128f mask_permutevar_ps(m128f src, mmask8 k, m128f a, m128i b) noexcept { return _mm_mask_permutevar_ps(src, k, a, b); }
inline m128f maskz_permutevar_ps(mmask8 k, m128f a, m128i b) noexcept { return _mm_maskz_permutevar_ps(k, a, b); }
inline m128f permutevar_ps(m128f a, m128i b) noexcept { return _mm_permutevar_ps(a, b); }
inline m128i mask_permutex2var_epi16(m128i a, mmask8 k, m128i idx, m128i b) noexcept { return _mm_mask_permutex2var_epi16(a, k, idx, b); }
inline m128i mask2_permutex2var_epi16(m128i a, m128i idx, mmask8 k, m128i b) noexcept { return _mm_mask2_permutex2var_epi16(a, idx, k, b); }
inline m128i maskz_permutex2var_epi16(mmask8 k, m128i a, m128i idx, m128i b) noexcept { return _mm_maskz_permutex2var_epi16(k, a, idx, b); }
inline m128i permutex2var_epi16(m128i a, m128i idx, m128i b) noexcept { return _mm_permutex2var_epi16(a, idx, b); }
inline m128i mask_permutex2var_epi32(m128i a, mmask8 k, m128i idx, m128i b) noexcept { return _mm_mask_permutex2var_epi32(a, k, idx, b); }
inline m128i mask2_permutex2var_epi32(m128i a, m128i idx, mmask8 k, m128i b) noexcept { return _mm_mask2_permutex2var_epi32(a, idx, k, b); }
inline m128i maskz_permutex2var_epi32(mmask8 k, m128i a, m128i idx, m128i b) noexcept { return _mm_maskz_permutex2var_epi32(k, a, idx, b); }
inline m128i permutex2var_epi32(m128i a, m128i idx, m128i b) noexcept { return _mm_permutex2var_epi32(a, idx, b); }
inline m128i mask_permutex2var_epi64(m128i a, mmask8 k, m128i idx, m128i b) noexcept { return _mm_mask_permutex2var_epi64(a, k, idx, b); }
inline m128i mask2_permutex2var_epi64(m128i a, m128i idx, mmask8 k, m128i b) noexcept { return _mm_mask2_permutex2var_epi64(a, idx, k, b); }
inline m128i maskz_permutex2var_epi64(mmask8 k, m128i a, m128i idx, m128i b) noexcept { return _mm_maskz_permutex2var_epi64(k, a, idx, b); }
inline m128i permutex2var_epi64(m128i a, m128i idx, m128i b) noexcept { return _mm_permutex2var_epi64(a, idx, b); }
inline m128i mask_permutex2var_epi8(m128i a, mmask16 k, m128i idx, m128i b) noexcept { return _mm_mask_permutex2var_epi8(a, k, idx, b); }
inline m128i mask2_permutex2var_epi8(m128i a, m128i idx, mmask16 k, m128i b) noexcept { return _mm_mask2_permutex2var_epi8(a, idx, k, b); }
inline m128i maskz_permutex2var_epi8(mmask16 k, m128i a, m128i idx, m128i b) noexcept { return _mm_maskz_permutex2var_epi8(k, a, idx, b); }
inline m128i permutex2var_epi8(m128i a, m128i idx, m128i b) noexcept { return _mm_permutex2var_epi8(a, idx, b); }
inline m128d mask_permutex2var_pd(m128d a, mmask8 k, m128i idx, m128d b) noexcept { return _mm_mask_permutex2var_pd(a, k, idx, b); }
inline m128d mask2_permutex2var_pd(m128d a, m128i idx, mmask8 k, m128d b) noexcept { return _mm_mask2_permutex2var_pd(a, idx, k, b); }
inline m128d maskz_permutex2var_pd(mmask8 k, m128d a, m128i idx, m128d b) noexcept { return _mm_maskz_permutex2var_pd(k, a, idx, b); }
inline m128d permutex2var_pd(m128d a, m128i idx, m128d b) noexcept { return _mm_permutex2var_pd(a, idx, b); }
inline m128f mask_permutex2var_ps(m128f a, mmask8 k, m128i idx, m128f b) noexcept { return _mm_mask_permutex2var_ps(a, k, idx, b); }
inline m128f mask2_permutex2var_ps(m128f a, m128i idx, mmask8 k, m128f b) noexcept { return _mm_mask2_permutex2var_ps(a, idx, k, b); }
inline m128f maskz_permutex2var_ps(mmask8 k, m128f a, m128i idx, m128f b) noexcept { return _mm_maskz_permutex2var_ps(k, a, idx, b); }
inline m128f permutex2var_ps(m128f a, m128i idx, m128f b) noexcept { return _mm_permutex2var_ps(a, idx, b); }
inline m128i mask_permutexvar_epi16(m128i src, mmask8 k, m128i idx, m128i a) noexcept { return _mm_mask_permutexvar_epi16(src, k, idx, a); }
inline m128i maskz_permutexvar_epi16(mmask8 k, m128i idx, m128i a) noexcept { return _mm_maskz_permutexvar_epi16(k, idx, a); }
inline m128i permutexvar_epi16(m128i idx, m128i a) noexcept { return _mm_permutexvar_epi16(idx, a); }
inline m128i mask_permutexvar_epi8(m128i src, mmask16 k, m128i idx, m128i a) noexcept { return _mm_mask_permutexvar_epi8(src, k, idx, a); }
inline m128i maskz_permutexvar_epi8(mmask16 k, m128i idx, m128i a) noexcept { return _mm_maskz_permutexvar_epi8(k, idx, a); }
inline m128i permutexvar_epi8(m128i idx, m128i a) noexcept { return _mm_permutexvar_epi8(idx, a); }
inline m128i mask_popcnt_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_popcnt_epi16(src, k, a); }
inline m128i maskz_popcnt_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_popcnt_epi16(k, a); }
inline m128i popcnt_epi16(m128i a) noexcept { return _mm_popcnt_epi16(a); }
inline m128i mask_popcnt_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_popcnt_epi32(src, k, a); }
inline m128i maskz_popcnt_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_popcnt_epi32(k, a); }
inline m128i popcnt_epi32(m128i a) noexcept { return _mm_popcnt_epi32(a); }
inline m128i mask_popcnt_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_popcnt_epi64(src, k, a); }
inline m128i maskz_popcnt_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_popcnt_epi64(k, a); }
inline m128i popcnt_epi64(m128i a) noexcept { return _mm_popcnt_epi64(a); }
inline m128i mask_popcnt_epi8(m128i src, mmask16 k, m128i a) noexcept { return _mm_mask_popcnt_epi8(src, k, a); }
inline m128i maskz_popcnt_epi8(mmask16 k, m128i a) noexcept { return _mm_maskz_popcnt_epi8(k, a); }
inline m128i popcnt_epi8(m128i a) noexcept { return _mm_popcnt_epi8(a); }
template<int4 imm8> inline m128d mask_range_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_range_pd(src, k, a, b, imm8); }
template<int4 imm8> inline m128d maskz_range_pd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_range_pd(k, a, b, imm8); }
template<int4 imm8> inline m128d range_pd(m128d a, m128d b) noexcept { return _mm_range_pd(a, b, imm8); }
template<int4 imm8> inline m128f mask_range_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_range_ps(src, k, a, b, imm8); }
template<int4 imm8> inline m128f maskz_range_ps(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_range_ps(k, a, b, imm8); }
template<int4 imm8> inline m128f range_ps(m128f a, m128f b) noexcept { return _mm_range_ps(a, b, imm8); }
template<int4 imm8, int sae> inline m128d mask_range_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_range_round_sd(src, k, a, b, imm8, sae); }
template<int4 imm8, int sae> inline m128d maskz_range_round_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_range_round_sd(k, a, b, imm8, sae); }
template<int4 imm8, int sae> inline m128d range_round_sd(m128d a, m128d b) noexcept { return _mm_range_round_sd(a, b, imm8, sae); }
template<int4 imm8, int sae> inline m128f mask_range_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_range_round_ss(src, k, a, b, imm8, sae); }
template<int4 imm8, int sae> inline m128f maskz_range_round_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_range_round_ss(k, a, b, imm8, sae); }
template<int4 imm8, int sae> inline m128f range_round_ss(m128f a, m128f b) noexcept { return _mm_range_round_ss(a, b, imm8, sae); }
template<int4 imm8> inline m128d mask_range_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_range_sd(src, k, a, b, imm8); }
template<int4 imm8> inline m128d maskz_range_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_range_sd(k, a, b, imm8); }
template<int4 imm8> inline m128f mask_range_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_range_ss(src, k, a, b, imm8); }
template<int4 imm8> inline m128f maskz_range_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_range_ss(k, a, b, imm8); }
inline m128f rcp_ps(m128f a) noexcept { return _mm_rcp_ps(a); }
inline m128f rcp_ss(m128f a) noexcept { return _mm_rcp_ss(a); }
inline m128d mask_rcp14_pd(m128d src, mmask8 k, m128d a) noexcept { return _mm_mask_rcp14_pd(src, k, a); }
inline m128d maskz_rcp14_pd(mmask8 k, m128d a) noexcept { return _mm_maskz_rcp14_pd(k, a); }
inline m128d rcp14_pd(m128d a) noexcept { return _mm_rcp14_pd(a); }
inline m128f mask_rcp14_ps(m128f src, mmask8 k, m128f a) noexcept { return _mm_mask_rcp14_ps(src, k, a); }
inline m128f maskz_rcp14_ps(mmask8 k, m128f a) noexcept { return _mm_maskz_rcp14_ps(k, a); }
inline m128f rcp14_ps(m128f a) noexcept { return _mm_rcp14_ps(a); }
inline m128d mask_rcp14_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_rcp14_sd(src, k, a, b); }
inline m128d maskz_rcp14_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_rcp14_sd(k, a, b); }
inline m128d rcp14_sd(m128d a, m128d b) noexcept { return _mm_rcp14_sd(a, b); }
inline m128f mask_rcp14_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_rcp14_ss(src, k, a, b); }
inline m128f maskz_rcp14_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_rcp14_ss(k, a, b); }
inline m128f rcp14_ss(m128f a, m128f b) noexcept { return _mm_rcp14_ss(a, b); }
template<int sae> inline m128d mask_rcp28_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_rcp28_round_sd(src, k, a, b, sae); }
template<int sae> inline m128d maskz_rcp28_round_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_rcp28_round_sd(k, a, b, sae); }
template<int sae> inline m128d rcp28_round_sd(m128d a, m128d b) noexcept { return _mm_rcp28_round_sd(a, b, sae); }
template<int sae> inline m128f mask_rcp28_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_rcp28_round_ss(src, k, a, b, sae); }
template<int sae> inline m128f maskz_rcp28_round_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_rcp28_round_ss(k, a, b, sae); }
template<int sae> inline m128f rcp28_round_ss(m128f a, m128f b) noexcept { return _mm_rcp28_round_ss(a, b, sae); }
inline m128d mask_rcp28_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_rcp28_sd(src, k, a, b); }
inline m128d maskz_rcp28_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_rcp28_sd(k, a, b); }
inline m128d rcp28_sd(m128d a, m128d b) noexcept { return _mm_rcp28_sd(a, b); }
inline m128f mask_rcp28_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_rcp28_ss(src, k, a, b); }
inline m128f maskz_rcp28_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_rcp28_ss(k, a, b); }
inline m128f rcp28_ss(m128f a, m128f b) noexcept { return _mm_rcp28_ss(a, b); }
template<int4 imm8> inline m128d mask_reduce_pd(m128d src, mmask8 k, m128d a) noexcept { return _mm_mask_reduce_pd(src, k, a, imm8); }
template<int4 imm8> inline m128d maskz_reduce_pd(mmask8 k, m128d a) noexcept { return _mm_maskz_reduce_pd(k, a, imm8); }
template<int4 imm8> inline m128d reduce_pd(m128d a) noexcept { return _mm_reduce_pd(a, imm8); }
template<int4 imm8> inline m128f mask_reduce_ps(m128f src, mmask8 k, m128f a) noexcept { return _mm_mask_reduce_ps(src, k, a, imm8); }
template<int4 imm8> inline m128f maskz_reduce_ps(mmask8 k, m128f a) noexcept { return _mm_maskz_reduce_ps(k, a, imm8); }
template<int4 imm8> inline m128f reduce_ps(m128f a) noexcept { return _mm_reduce_ps(a, imm8); }
template<int4 imm8, int sae> inline m128d mask_reduce_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_reduce_round_sd(src, k, a, b, imm8, sae); }
template<int4 imm8, int sae> inline m128d maskz_reduce_round_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_reduce_round_sd(k, a, b, imm8, sae); }
template<int4 imm8, int sae> inline m128d reduce_round_sd(m128d a, m128d b) noexcept { return _mm_reduce_round_sd(a, b, imm8, sae); }
template<int4 imm8, int sae> inline m128f mask_reduce_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_reduce_round_ss(src, k, a, b, imm8, sae); }
template<int4 imm8, int sae> inline m128f maskz_reduce_round_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_reduce_round_ss(k, a, b, imm8, sae); }
template<int4 imm8, int sae> inline m128f reduce_round_ss(m128f a, m128f b) noexcept { return _mm_reduce_round_ss(a, b, imm8, sae); }
template<int4 imm8> inline m128d mask_reduce_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_reduce_sd(src, k, a, b, imm8); }
template<int4 imm8> inline m128d maskz_reduce_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_reduce_sd(k, a, b, imm8); }
template<int4 imm8> inline m128d reduce_sd(m128d a, m128d b) noexcept { return _mm_reduce_sd(a, b, imm8); }
template<int4 imm8> inline m128f mask_reduce_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_reduce_ss(src, k, a, b, imm8); }
template<int4 imm8> inline m128f maskz_reduce_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_reduce_ss(k, a, b, imm8); }
template<int4 imm8> inline m128f reduce_ss(m128f a, m128f b) noexcept { return _mm_reduce_ss(a, b, imm8); }
template<int4 imm8> inline m128i mask_rol_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_rol_epi32(src, k, a, imm8); }
template<int4 imm8> inline m128i maskz_rol_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_rol_epi32(k, a, imm8); }
template<int4 imm8> inline m128i rol_epi32(m128i a) noexcept { return _mm_rol_epi32(a, imm8); }
template<int4 imm8> inline m128i mask_rol_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_rol_epi64(src, k, a, imm8); }
template<int4 imm8> inline m128i maskz_rol_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_rol_epi64(k, a, imm8); }
template<int4 imm8> inline m128i rol_epi64(m128i a) noexcept { return _mm_rol_epi64(a, imm8); }
inline m128i mask_rolv_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_rolv_epi32(src, k, a, b); }
inline m128i maskz_rolv_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_rolv_epi32(k, a, b); }
inline m128i rolv_epi32(m128i a, m128i b) noexcept { return _mm_rolv_epi32(a, b); }
inline m128i mask_rolv_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_rolv_epi64(src, k, a, b); }
inline m128i maskz_rolv_epi64(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_rolv_epi64(k, a, b); }
inline m128i rolv_epi64(m128i a, m128i b) noexcept { return _mm_rolv_epi64(a, b); }
template<int4 imm8> inline m128i mask_ror_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_ror_epi32(src, k, a, imm8); }
template<int4 imm8> inline m128i maskz_ror_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_ror_epi32(k, a, imm8); }
template<int4 imm8> inline m128i ror_epi32(m128i a) noexcept { return _mm_ror_epi32(a, imm8); }
template<int4 imm8> inline m128i mask_ror_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_ror_epi64(src, k, a, imm8); }
template<int4 imm8> inline m128i maskz_ror_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_ror_epi64(k, a, imm8); }
template<int4 imm8> inline m128i ror_epi64(m128i a) noexcept { return _mm_ror_epi64(a, imm8); }
inline m128i mask_rorv_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_rorv_epi32(src, k, a, b); }
inline m128i maskz_rorv_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_rorv_epi32(k, a, b); }
inline m128i rorv_epi32(m128i a, m128i b) noexcept { return _mm_rorv_epi32(a, b); }
inline m128i mask_rorv_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_rorv_epi64(src, k, a, b); }
inline m128i maskz_rorv_epi64(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_rorv_epi64(k, a, b); }
inline m128i rorv_epi64(m128i a, m128i b) noexcept { return _mm_rorv_epi64(a, b); }
template<int4 rounding> inline m128d round_pd(m128d a) noexcept { return _mm_round_pd(a, rounding); }
template<int4 rounding> inline m128f round_ps(m128f a) noexcept { return _mm_round_ps(a, rounding); }
template<int4 rounding> inline m128d round_sd(m128d a, m128d b) noexcept { return _mm_round_sd(a, b, rounding); }
template<int4 rounding> inline m128f round_ss(m128f a, m128f b) noexcept { return _mm_round_ss(a, b, rounding); }
template<int4 imm8> inline m128d mask_roundscale_pd(m128d src, mmask8 k, m128d a) noexcept { return _mm_mask_roundscale_pd(src, k, a, imm8); }
template<int4 imm8> inline m128d maskz_roundscale_pd(mmask8 k, m128d a) noexcept { return _mm_maskz_roundscale_pd(k, a, imm8); }
template<int4 imm8> inline m128d roundscale_pd(m128d a) noexcept { return _mm_roundscale_pd(a, imm8); }
template<int4 imm8> inline m128f mask_roundscale_ps(m128f src, mmask8 k, m128f a) noexcept { return _mm_mask_roundscale_ps(src, k, a, imm8); }
template<int4 imm8> inline m128f maskz_roundscale_ps(mmask8 k, m128f a) noexcept { return _mm_maskz_roundscale_ps(k, a, imm8); }
template<int4 imm8> inline m128f roundscale_ps(m128f a) noexcept { return _mm_roundscale_ps(a, imm8); }
template<int4 imm8, int sae> inline m128d mask_roundscale_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_roundscale_round_sd(src, k, a, b, imm8, sae); }
template<int4 imm8, int sae> inline m128d maskz_roundscale_round_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_roundscale_round_sd(k, a, b, imm8, sae); }
template<int4 imm8, int sae> inline m128d roundscale_round_sd(m128d a, m128d b) noexcept { return _mm_roundscale_round_sd(a, b, imm8, sae); }
template<int4 imm8, int sae> inline m128f mask_roundscale_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_roundscale_round_ss(src, k, a, b, imm8, sae); }
template<int4 imm8, int sae> inline m128f maskz_roundscale_round_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_roundscale_round_ss(k, a, b, imm8, sae); }
template<int4 imm8, int sae> inline m128f roundscale_round_ss(m128f a, m128f b) noexcept { return _mm_roundscale_round_ss(a, b, imm8, sae); }
template<int4 imm8> inline m128d mask_roundscale_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_roundscale_sd(src, k, a, b, imm8); }
template<int4 imm8> inline m128d maskz_roundscale_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_roundscale_sd(k, a, b, imm8); }
template<int4 imm8> inline m128d roundscale_sd(m128d a, m128d b) noexcept { return _mm_roundscale_sd(a, b, imm8); }
template<int4 imm8> inline m128f mask_roundscale_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_roundscale_ss(src, k, a, b, imm8); }
template<int4 imm8> inline m128f maskz_roundscale_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_roundscale_ss(k, a, b, imm8); }
template<int4 imm8> inline m128f roundscale_ss(m128f a, m128f b) noexcept { return _mm_roundscale_ss(a, b, imm8); }
inline m128f rsqrt_ps(m128f a) noexcept { return _mm_rsqrt_ps(a); }
inline m128f rsqrt_ss(m128f a) noexcept { return _mm_rsqrt_ss(a); }
inline m128d mask_rsqrt14_pd(m128d src, mmask8 k, m128d a) noexcept { return _mm_mask_rsqrt14_pd(src, k, a); }
inline m128d maskz_rsqrt14_pd(mmask8 k, m128d a) noexcept { return _mm_maskz_rsqrt14_pd(k, a); }
inline m128f mask_rsqrt14_ps(m128f src, mmask8 k, m128f a) noexcept { return _mm_mask_rsqrt14_ps(src, k, a); }
inline m128f maskz_rsqrt14_ps(mmask8 k, m128f a) noexcept { return _mm_maskz_rsqrt14_ps(k, a); }
inline m128d mask_rsqrt14_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_rsqrt14_sd(src, k, a, b); }
inline m128d maskz_rsqrt14_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_rsqrt14_sd(k, a, b); }
inline m128d rsqrt14_sd(m128d a, m128d b) noexcept { return _mm_rsqrt14_sd(a, b); }
inline m128f mask_rsqrt14_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_rsqrt14_ss(src, k, a, b); }
inline m128f maskz_rsqrt14_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_rsqrt14_ss(k, a, b); }
inline m128f rsqrt14_ss(m128f a, m128f b) noexcept { return _mm_rsqrt14_ss(a, b); }
template<int sae> inline m128d mask_rsqrt28_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_rsqrt28_round_sd(src, k, a, b, sae); }
template<int sae> inline m128d maskz_rsqrt28_round_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_rsqrt28_round_sd(k, a, b, sae); }
template<int sae> inline m128d rsqrt28_round_sd(m128d a, m128d b) noexcept { return _mm_rsqrt28_round_sd(a, b, sae); }
template<int sae> inline m128f mask_rsqrt28_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_rsqrt28_round_ss(src, k, a, b, sae); }
template<int sae> inline m128f maskz_rsqrt28_round_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_rsqrt28_round_ss(k, a, b, sae); }
template<int sae> inline m128f rsqrt28_round_ss(m128f a, m128f b) noexcept { return _mm_rsqrt28_round_ss(a, b, sae); }
inline m128d mask_rsqrt28_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_rsqrt28_sd(src, k, a, b); }
inline m128d maskz_rsqrt28_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_rsqrt28_sd(k, a, b); }
inline m128d rsqrt28_sd(m128d a, m128d b) noexcept { return _mm_rsqrt28_sd(a, b); }
inline m128f mask_rsqrt28_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_rsqrt28_ss(src, k, a, b); }
inline m128f maskz_rsqrt28_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_rsqrt28_ss(k, a, b); }
inline m128f rsqrt28_ss(m128f a, m128f b) noexcept { return _mm_rsqrt28_ss(a, b); }
inline m128i sad_epu8(m128i a, m128i b) noexcept { return _mm_sad_epu8(a, b); }
inline m128d mask_scalef_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_scalef_pd(src, k, a, b); }
inline m128d maskz_scalef_pd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_scalef_pd(k, a, b); }
inline m128d scalef_pd(m128d a, m128d b) noexcept { return _mm_scalef_pd(a, b); }
inline m128f mask_scalef_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_scalef_ps(src, k, a, b); }
inline m128f maskz_scalef_ps(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_scalef_ps(k, a, b); }
inline m128f scalef_ps(m128f a, m128f b) noexcept { return _mm_scalef_ps(a, b); }
template<int4 rounding> inline m128d mask_scalef_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_scalef_round_sd(src, k, a, b, rounding); }
template<int4 rounding> inline m128d maskz_scalef_round_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_scalef_round_sd(k, a, b, rounding); }
template<int4 rounding> inline m128d scalef_round_sd(m128d a, m128d b) noexcept { return _mm_scalef_round_sd(a, b, rounding); }
template<int4 rounding> inline m128f mask_scalef_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_scalef_round_ss(src, k, a, b, rounding); }
template<int4 rounding> inline m128f maskz_scalef_round_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_scalef_round_ss(k, a, b, rounding); }
template<int4 rounding> inline m128f scalef_round_ss(m128f a, m128f b) noexcept { return _mm_scalef_round_ss(a, b, rounding); }
inline m128d mask_scalef_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_scalef_sd(src, k, a, b); }
inline m128d maskz_scalef_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_scalef_sd(k, a, b); }
inline m128d scalef_sd(m128d a, m128d b) noexcept { return _mm_scalef_sd(a, b); }
inline m128f mask_scalef_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_scalef_ss(src, k, a, b); }
inline m128f maskz_scalef_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_scalef_ss(k, a, b); }
inline m128f scalef_ss(m128f a, m128f b) noexcept { return _mm_scalef_ss(a, b); }
inline m128i set_epi16(short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0) noexcept { return _mm_set_epi16(e7, e6, e5, e4, e3, e2, e1, e0); }
inline m128i set_epi32(int e3, int e2, int e1, int e0) noexcept { return _mm_set_epi32(e3, e2, e1, e0); }
inline m128i set_epi64x(int8 e1, int8 e0) noexcept { return _mm_set_epi64x(e1, e0); }
inline m128i set_epi8(char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0) noexcept { return _mm_set_epi8(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0); }
inline m128d set_pd(double e1, double e0) noexcept { return _mm_set_pd(e1, e0); }
inline m128d set_pd1(double a) noexcept { return _mm_set_pd1(a); }
inline m128f set_ps(float e3, float e2, float e1, float e0) noexcept { return _mm_set_ps(e3, e2, e1, e0); }
inline m128f set_ps1(float a) noexcept { return _mm_set_ps1(a); }
inline m128d set_sd(double a) noexcept { return _mm_set_sd(a); }
inline m128f set_ss(float a) noexcept { return _mm_set_ss(a); }
inline m128i mask_set1_epi16(m128i src, mmask8 k, short a) noexcept { return _mm_mask_set1_epi16(src, k, a); }
inline m128i maskz_set1_epi16(mmask8 k, short a) noexcept { return _mm_maskz_set1_epi16(k, a); }
inline m128i set1_epi16(short a) noexcept { return _mm_set1_epi16(a); }
inline m128i mask_set1_epi32(m128i src, mmask8 k, int a) noexcept { return _mm_mask_set1_epi32(src, k, a); }
inline m128i maskz_set1_epi32(mmask8 k, int a) noexcept { return _mm_maskz_set1_epi32(k, a); }
inline m128i set1_epi32(int a) noexcept { return _mm_set1_epi32(a); }
inline m128i mask_set1_epi64(m128i src, mmask8 k, int8 a) noexcept { return _mm_mask_set1_epi64(src, k, a); }
inline m128i maskz_set1_epi64(mmask8 k, int8 a) noexcept { return _mm_maskz_set1_epi64(k, a); }
inline m128i set1_epi64x(int8 a) noexcept { return _mm_set1_epi64x(a); }
inline m128i mask_set1_epi8(m128i src, mmask16 k, char a) noexcept { return _mm_mask_set1_epi8(src, k, a); }
inline m128i maskz_set1_epi8(mmask16 k, char a) noexcept { return _mm_maskz_set1_epi8(k, a); }
inline m128i set1_epi8(char a) noexcept { return _mm_set1_epi8(a); }
inline m128d set1_pd(double a) noexcept { return _mm_set1_pd(a); }
inline m128f set1_ps(float a) noexcept { return _mm_set1_ps(a); }
inline m128i setr_epi16(short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0) noexcept { return _mm_setr_epi16(e7, e6, e5, e4, e3, e2, e1, e0); }
inline m128i setr_epi32(int e3, int e2, int e1, int e0) noexcept { return _mm_setr_epi32(e3, e2, e1, e0); }
inline m128i setr_epi8(char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0) noexcept { return _mm_setr_epi8(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0); }
inline m128d setr_pd(double e1, double e0) noexcept { return _mm_setr_pd(e1, e0); }
inline m128f setr_ps(float e3, float e2, float e1, float e0) noexcept { return _mm_setr_ps(e3, e2, e1, e0); }
inline m128d setzero_pd(void) noexcept { return _mm_setzero_pd(); }
inline m128f setzero_ps(void) noexcept { return _mm_setzero_ps(); }
inline m128i setzero_si128() noexcept { return _mm_setzero_si128(); }
inline m128i sha1msg1_epu32(m128i a, m128i b) noexcept { return _mm_sha1msg1_epu32(a, b); }
inline m128i sha1msg2_epu32(m128i a, m128i b) noexcept { return _mm_sha1msg2_epu32(a, b); }
inline m128i sha1nexte_epu32(m128i a, m128i b) noexcept { return _mm_sha1nexte_epu32(a, b); }
template<int func> inline m128i sha1rnds4_epu32(m128i a, m128i b) noexcept { return _mm_sha1rnds4_epu32(a, b, func); }
inline m128i sha256msg1_epu32(m128i a, m128i b) noexcept { return _mm_sha256msg1_epu32(a, b); }
inline m128i sha256msg2_epu32(m128i a, m128i b) noexcept { return _mm_sha256msg2_epu32(a, b); }
inline m128i sha256rnds2_epu32(m128i a, m128i b, m128i k) noexcept { return _mm_sha256rnds2_epu32(a, b, k); }
template<int4 imm8> inline m128i mask_shldi_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_shldi_epi16(src, k, a, b, imm8); }
template<int4 imm8> inline m128i maskz_shldi_epi16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_shldi_epi16(k, a, b, imm8); }
template<int4 imm8> inline m128i shldi_epi16(m128i a, m128i b) noexcept { return _mm_shldi_epi16(a, b, imm8); }
template<int4 imm8> inline m128i mask_shldi_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_shldi_epi32(src, k, a, b, imm8); }
template<int4 imm8> inline m128i maskz_shldi_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_shldi_epi32(k, a, b, imm8); }
template<int4 imm8> inline m128i shldi_epi32(m128i a, m128i b) noexcept { return _mm_shldi_epi32(a, b, imm8); }
template<int4 imm8> inline m128i mask_shldi_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_shldi_epi64(src, k, a, b, imm8); }
template<int4 imm8> inline m128i maskz_shldi_epi64(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_shldi_epi64(k, a, b, imm8); }
template<int4 imm8> inline m128i shldi_epi64(m128i a, m128i b) noexcept { return _mm_shldi_epi64(a, b, imm8); }
inline m128i mask_shldv_epi16(m128i a, mmask8 k, m128i b, m128i c) noexcept { return _mm_mask_shldv_epi16(a, k, b, c); }
inline m128i maskz_shldv_epi16(mmask8 k, m128i a, m128i b, m128i c) noexcept { return _mm_maskz_shldv_epi16(k, a, b, c); }
inline m128i shldv_epi16(m128i a, m128i b, m128i c) noexcept { return _mm_shldv_epi16(a, b, c); }
inline m128i mask_shldv_epi32(m128i a, mmask8 k, m128i b, m128i c) noexcept { return _mm_mask_shldv_epi32(a, k, b, c); }
inline m128i maskz_shldv_epi32(mmask8 k, m128i a, m128i b, m128i c) noexcept { return _mm_maskz_shldv_epi32(k, a, b, c); }
inline m128i shldv_epi32(m128i a, m128i b, m128i c) noexcept { return _mm_shldv_epi32(a, b, c); }
inline m128i mask_shldv_epi64(m128i a, mmask8 k, m128i b, m128i c) noexcept { return _mm_mask_shldv_epi64(a, k, b, c); }
inline m128i maskz_shldv_epi64(mmask8 k, m128i a, m128i b, m128i c) noexcept { return _mm_maskz_shldv_epi64(k, a, b, c); }
inline m128i shldv_epi64(m128i a, m128i b, m128i c) noexcept { return _mm_shldv_epi64(a, b, c); }
template<int4 imm8> inline m128i mask_shrdi_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_shrdi_epi16(src, k, a, b, imm8); }
template<int4 imm8> inline m128i maskz_shrdi_epi16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_shrdi_epi16(k, a, b, imm8); }
template<int4 imm8> inline m128i shrdi_epi16(m128i a, m128i b) noexcept { return _mm_shrdi_epi16(a, b, imm8); }
template<int4 imm8> inline m128i mask_shrdi_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_shrdi_epi32(src, k, a, b, imm8); }
template<int4 imm8> inline m128i maskz_shrdi_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_shrdi_epi32(k, a, b, imm8); }
template<int4 imm8> inline m128i shrdi_epi32(m128i a, m128i b) noexcept { return _mm_shrdi_epi32(a, b, imm8); }
template<int4 imm8> inline m128i mask_shrdi_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_shrdi_epi64(src, k, a, b, imm8); }
template<int4 imm8> inline m128i maskz_shrdi_epi64(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_shrdi_epi64(k, a, b, imm8); }
template<int4 imm8> inline m128i shrdi_epi64(m128i a, m128i b) noexcept { return _mm_shrdi_epi64(a, b, imm8); }
inline m128i mask_shrdv_epi16(m128i a, mmask8 k, m128i b, m128i c) noexcept { return _mm_mask_shrdv_epi16(a, k, b, c); }
inline m128i maskz_shrdv_epi16(mmask8 k, m128i a, m128i b, m128i c) noexcept { return _mm_maskz_shrdv_epi16(k, a, b, c); }
inline m128i shrdv_epi16(m128i a, m128i b, m128i c) noexcept { return _mm_shrdv_epi16(a, b, c); }
inline m128i mask_shrdv_epi32(m128i a, mmask8 k, m128i b, m128i c) noexcept { return _mm_mask_shrdv_epi32(a, k, b, c); }
inline m128i maskz_shrdv_epi32(mmask8 k, m128i a, m128i b, m128i c) noexcept { return _mm_maskz_shrdv_epi32(k, a, b, c); }
inline m128i shrdv_epi32(m128i a, m128i b, m128i c) noexcept { return _mm_shrdv_epi32(a, b, c); }
inline m128i mask_shrdv_epi64(m128i a, mmask8 k, m128i b, m128i c) noexcept { return _mm_mask_shrdv_epi64(a, k, b, c); }
inline m128i maskz_shrdv_epi64(mmask8 k, m128i a, m128i b, m128i c) noexcept { return _mm_maskz_shrdv_epi64(k, a, b, c); }
inline m128i shrdv_epi64(m128i a, m128i b, m128i c) noexcept { return _mm_shrdv_epi64(a, b, c); }
template<int4 imm8> inline m128i mask_shuffle_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_shuffle_epi32(src, k, a, imm8); }
template<int4 imm8> inline m128i maskz_shuffle_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_shuffle_epi32(k, a, imm8); }
template<int4 imm8> inline m128i shuffle_epi32(m128i a) noexcept { return _mm_shuffle_epi32(a, imm8); }
inline m128i mask_shuffle_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_shuffle_epi8(src, k, a, b); }
inline m128i maskz_shuffle_epi8(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_shuffle_epi8(k, a, b); }
inline m128i shuffle_epi8(m128i a, m128i b) noexcept { return _mm_shuffle_epi8(a, b); }
template<int4 imm8> inline m128d mask_shuffle_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_shuffle_pd(src, k, a, b, imm8); }
template<int4 imm8> inline m128d maskz_shuffle_pd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_shuffle_pd(k, a, b, imm8); }
template<int4 imm8> inline m128d shuffle_pd(m128d a, m128d b) noexcept { return _mm_shuffle_pd(a, b, imm8); }
template<int4 imm8> inline m128f mask_shuffle_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_shuffle_ps(src, k, a, b, imm8); }
template<int4 imm8> inline m128f maskz_shuffle_ps(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_shuffle_ps(k, a, b, imm8); }
template<nat4 imm8> inline m128f shuffle_ps(m128f a, m128f b) noexcept { return _mm_shuffle_ps(a, b, imm8); }
template<int4 imm8> inline m128i mask_shufflehi_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_shufflehi_epi16(src, k, a, imm8); }
template<int4 imm8> inline m128i maskz_shufflehi_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_shufflehi_epi16(k, a, imm8); }
template<int4 imm8> inline m128i shufflehi_epi16(m128i a) noexcept { return _mm_shufflehi_epi16(a, imm8); }
template<int4 imm8> inline m128i mask_shufflelo_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_shufflelo_epi16(src, k, a, imm8); }
template<int4 imm8> inline m128i maskz_shufflelo_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_shufflelo_epi16(k, a, imm8); }
template<int4 imm8> inline m128i shufflelo_epi16(m128i a) noexcept { return _mm_shufflelo_epi16(a, imm8); }
inline m128i sign_epi16(m128i a, m128i b) noexcept { return _mm_sign_epi16(a, b); }
inline m128i sign_epi32(m128i a, m128i b) noexcept { return _mm_sign_epi32(a, b); }
inline m128i sign_epi8(m128i a, m128i b) noexcept { return _mm_sign_epi8(a, b); }
inline m128i mask_sll_epi16(m128i src, mmask8 k, m128i a, m128i count) noexcept { return _mm_mask_sll_epi16(src, k, a, count); }
inline m128i maskz_sll_epi16(mmask8 k, m128i a, m128i count) noexcept { return _mm_maskz_sll_epi16(k, a, count); }
inline m128i sll_epi16(m128i a, m128i count) noexcept { return _mm_sll_epi16(a, count); }
inline m128i mask_sll_epi32(m128i src, mmask8 k, m128i a, m128i count) noexcept { return _mm_mask_sll_epi32(src, k, a, count); }
inline m128i maskz_sll_epi32(mmask8 k, m128i a, m128i count) noexcept { return _mm_maskz_sll_epi32(k, a, count); }
inline m128i sll_epi32(m128i a, m128i count) noexcept { return _mm_sll_epi32(a, count); }
inline m128i mask_sll_epi64(m128i src, mmask8 k, m128i a, m128i count) noexcept { return _mm_mask_sll_epi64(src, k, a, count); }
inline m128i maskz_sll_epi64(mmask8 k, m128i a, m128i count) noexcept { return _mm_maskz_sll_epi64(k, a, count); }
inline m128i sll_epi64(m128i a, m128i count) noexcept { return _mm_sll_epi64(a, count); }
template<nat4 imm8> inline m128i mask_slli_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_slli_epi16(src, k, a, imm8); }
template<nat4 imm8> inline m128i maskz_slli_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_slli_epi16(k, a, imm8); }
template<int4 imm8> inline m128i slli_epi16(m128i a) noexcept { return _mm_slli_epi16(a, imm8); }
template<nat4 imm8> inline m128i mask_slli_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_slli_epi32(src, k, a, imm8); }
template<nat4 imm8> inline m128i maskz_slli_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_slli_epi32(k, a, imm8); }
template<int4 imm8> inline m128i slli_epi32(m128i a) noexcept { return _mm_slli_epi32(a, imm8); }
template<nat4 imm8> inline m128i mask_slli_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_slli_epi64(src, k, a, imm8); }
template<nat4 imm8> inline m128i maskz_slli_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_slli_epi64(k, a, imm8); }
template<int4 imm8> inline m128i slli_epi64(m128i a) noexcept { return _mm_slli_epi64(a, imm8); }
template<int4 imm8> inline m128i slli_si128(m128i a) noexcept { return _mm_slli_si128(a, imm8); }
inline m128i mask_sllv_epi16(m128i src, mmask8 k, m128i a, m128i count) noexcept { return _mm_mask_sllv_epi16(src, k, a, count); }
inline m128i maskz_sllv_epi16(mmask8 k, m128i a, m128i count) noexcept { return _mm_maskz_sllv_epi16(k, a, count); }
inline m128i sllv_epi16(m128i a, m128i count) noexcept { return _mm_sllv_epi16(a, count); }
inline m128i mask_sllv_epi32(m128i src, mmask8 k, m128i a, m128i count) noexcept { return _mm_mask_sllv_epi32(src, k, a, count); }
inline m128i maskz_sllv_epi32(mmask8 k, m128i a, m128i count) noexcept { return _mm_maskz_sllv_epi32(k, a, count); }
inline m128i sllv_epi32(m128i a, m128i count) noexcept { return _mm_sllv_epi32(a, count); }
inline m128i mask_sllv_epi64(m128i src, mmask8 k, m128i a, m128i count) noexcept { return _mm_mask_sllv_epi64(src, k, a, count); }
inline m128i maskz_sllv_epi64(mmask8 k, m128i a, m128i count) noexcept { return _mm_maskz_sllv_epi64(k, a, count); }
inline m128i sllv_epi64(m128i a, m128i count) noexcept { return _mm_sllv_epi64(a, count); }
inline m128d mask_sqrt_pd(m128d src, mmask8 k, m128d a) noexcept { return _mm_mask_sqrt_pd(src, k, a); }
inline m128d maskz_sqrt_pd(mmask8 k, m128d a) noexcept { return _mm_maskz_sqrt_pd(k, a); }
inline m128d sqrt_pd(m128d a) noexcept { return _mm_sqrt_pd(a); }
inline m128f mask_sqrt_ps(m128f src, mmask8 k, m128f a) noexcept { return _mm_mask_sqrt_ps(src, k, a); }
inline m128f maskz_sqrt_ps(mmask8 k, m128f a) noexcept { return _mm_maskz_sqrt_ps(k, a); }
inline m128f sqrt_ps(m128f a) noexcept { return _mm_sqrt_ps(a); }
template<int4 rounding> inline m128d mask_sqrt_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_sqrt_round_sd(src, k, a, b, rounding); }
template<int4 rounding> inline m128d maskz_sqrt_round_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_sqrt_round_sd(k, a, b, rounding); }
template<int4 rounding> inline m128d sqrt_round_sd(m128d a, m128d b) noexcept { return _mm_sqrt_round_sd(a, b, rounding); }
template<int4 rounding> inline m128f mask_sqrt_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_sqrt_round_ss(src, k, a, b, rounding); }
template<int4 rounding> inline m128f maskz_sqrt_round_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_sqrt_round_ss(k, a, b, rounding); }
template<int4 rounding> inline m128f sqrt_round_ss(m128f a, m128f b) noexcept { return _mm_sqrt_round_ss(a, b, rounding); }
inline m128d mask_sqrt_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_sqrt_sd(src, k, a, b); }
inline m128d maskz_sqrt_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_sqrt_sd(k, a, b); }
inline m128d sqrt_sd(m128d a, m128d b) noexcept { return _mm_sqrt_sd(a, b); }
inline m128f mask_sqrt_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_sqrt_ss(src, k, a, b); }
inline m128f maskz_sqrt_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_sqrt_ss(k, a, b); }
inline m128f sqrt_ss(m128f a) noexcept { return _mm_sqrt_ss(a); }
inline m128i mask_sra_epi16(m128i src, mmask8 k, m128i a, m128i count) noexcept { return _mm_mask_sra_epi16(src, k, a, count); }
inline m128i maskz_sra_epi16(mmask8 k, m128i a, m128i count) noexcept { return _mm_maskz_sra_epi16(k, a, count); }
inline m128i sra_epi16(m128i a, m128i count) noexcept { return _mm_sra_epi16(a, count); }
inline m128i mask_sra_epi32(m128i src, mmask8 k, m128i a, m128i count) noexcept { return _mm_mask_sra_epi32(src, k, a, count); }
inline m128i maskz_sra_epi32(mmask8 k, m128i a, m128i count) noexcept { return _mm_maskz_sra_epi32(k, a, count); }
inline m128i sra_epi32(m128i a, m128i count) noexcept { return _mm_sra_epi32(a, count); }
inline m128i mask_sra_epi64(m128i src, mmask8 k, m128i a, m128i count) noexcept { return _mm_mask_sra_epi64(src, k, a, count); }
inline m128i maskz_sra_epi64(mmask8 k, m128i a, m128i count) noexcept { return _mm_maskz_sra_epi64(k, a, count); }
inline m128i sra_epi64(m128i a, m128i count) noexcept { return _mm_sra_epi64(a, count); }
template<nat4 imm8> inline m128i mask_srai_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_srai_epi16(src, k, a, imm8); }
template<nat4 imm8> inline m128i maskz_srai_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_srai_epi16(k, a, imm8); }
template<int4 imm8> inline m128i srai_epi16(m128i a) noexcept { return _mm_srai_epi16(a, imm8); }
template<nat4 imm8> inline m128i mask_srai_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_srai_epi32(src, k, a, imm8); }
template<nat4 imm8> inline m128i maskz_srai_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_srai_epi32(k, a, imm8); }
template<int4 imm8> inline m128i srai_epi32(m128i a) noexcept { return _mm_srai_epi32(a, imm8); }
template<nat4 imm8> inline m128i mask_srai_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_srai_epi64(src, k, a, imm8); }
template<nat4 imm8> inline m128i maskz_srai_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_srai_epi64(k, a, imm8); }
template<nat4 imm8> inline m128i srai_epi64(m128i a) noexcept { return _mm_srai_epi64(a, imm8); }
inline m128i mask_srav_epi16(m128i src, mmask8 k, m128i a, m128i count) noexcept { return _mm_mask_srav_epi16(src, k, a, count); }
inline m128i maskz_srav_epi16(mmask8 k, m128i a, m128i count) noexcept { return _mm_maskz_srav_epi16(k, a, count); }
inline m128i srav_epi16(m128i a, m128i count) noexcept { return _mm_srav_epi16(a, count); }
inline m128i mask_srav_epi32(m128i src, mmask8 k, m128i a, m128i count) noexcept { return _mm_mask_srav_epi32(src, k, a, count); }
inline m128i maskz_srav_epi32(mmask8 k, m128i a, m128i count) noexcept { return _mm_maskz_srav_epi32(k, a, count); }
inline m128i srav_epi32(m128i a, m128i count) noexcept { return _mm_srav_epi32(a, count); }
inline m128i mask_srav_epi64(m128i src, mmask8 k, m128i a, m128i count) noexcept { return _mm_mask_srav_epi64(src, k, a, count); }
inline m128i maskz_srav_epi64(mmask8 k, m128i a, m128i count) noexcept { return _mm_maskz_srav_epi64(k, a, count); }
inline m128i srav_epi64(m128i a, m128i count) noexcept { return _mm_srav_epi64(a, count); }
inline m128i mask_srl_epi16(m128i src, mmask8 k, m128i a, m128i count) noexcept { return _mm_mask_srl_epi16(src, k, a, count); }
inline m128i maskz_srl_epi16(mmask8 k, m128i a, m128i count) noexcept { return _mm_maskz_srl_epi16(k, a, count); }
inline m128i srl_epi16(m128i a, m128i count) noexcept { return _mm_srl_epi16(a, count); }
inline m128i mask_srl_epi32(m128i src, mmask8 k, m128i a, m128i count) noexcept { return _mm_mask_srl_epi32(src, k, a, count); }
inline m128i maskz_srl_epi32(mmask8 k, m128i a, m128i count) noexcept { return _mm_maskz_srl_epi32(k, a, count); }
inline m128i srl_epi32(m128i a, m128i count) noexcept { return _mm_srl_epi32(a, count); }
inline m128i mask_srl_epi64(m128i src, mmask8 k, m128i a, m128i count) noexcept { return _mm_mask_srl_epi64(src, k, a, count); }
inline m128i maskz_srl_epi64(mmask8 k, m128i a, m128i count) noexcept { return _mm_maskz_srl_epi64(k, a, count); }
inline m128i srl_epi64(m128i a, m128i count) noexcept { return _mm_srl_epi64(a, count); }
template<int4 imm8> inline m128i mask_srli_epi16(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_srli_epi16(src, k, a, imm8); }
template<int4 imm8> inline m128i maskz_srli_epi16(mmask8 k, m128i a) noexcept { return _mm_maskz_srli_epi16(k, a, imm8); }
template<int4 imm8> inline m128i srli_epi16(m128i a) noexcept { return _mm_srli_epi16(a, imm8); }
template<nat4 imm8> inline m128i mask_srli_epi32(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_srli_epi32(src, k, a, imm8); }
template<nat4 imm8> inline m128i maskz_srli_epi32(mmask8 k, m128i a) noexcept { return _mm_maskz_srli_epi32(k, a, imm8); }
template<int4 imm8> inline m128i srli_epi32(m128i a) noexcept { return _mm_srli_epi32(a, imm8); }
template<nat4 imm8> inline m128i mask_srli_epi64(m128i src, mmask8 k, m128i a) noexcept { return _mm_mask_srli_epi64(src, k, a, imm8); }
template<nat4 imm8> inline m128i maskz_srli_epi64(mmask8 k, m128i a) noexcept { return _mm_maskz_srli_epi64(k, a, imm8); }
template<int4 imm8> inline m128i srli_epi64(m128i a) noexcept { return _mm_srli_epi64(a, imm8); }
template<int4 imm8> inline m128i srli_si128(m128i a) noexcept { return _mm_srli_si128(a, imm8); }
inline m128i mask_srlv_epi16(m128i src, mmask8 k, m128i a, m128i count) noexcept { return _mm_mask_srlv_epi16(src, k, a, count); }
inline m128i maskz_srlv_epi16(mmask8 k, m128i a, m128i count) noexcept { return _mm_maskz_srlv_epi16(k, a, count); }
inline m128i srlv_epi16(m128i a, m128i count) noexcept { return _mm_srlv_epi16(a, count); }
inline m128i mask_srlv_epi32(m128i src, mmask8 k, m128i a, m128i count) noexcept { return _mm_mask_srlv_epi32(src, k, a, count); }
inline m128i maskz_srlv_epi32(mmask8 k, m128i a, m128i count) noexcept { return _mm_maskz_srlv_epi32(k, a, count); }
inline m128i srlv_epi32(m128i a, m128i count) noexcept { return _mm_srlv_epi32(a, count); }
inline m128i mask_srlv_epi64(m128i src, mmask8 k, m128i a, m128i count) noexcept { return _mm_mask_srlv_epi64(src, k, a, count); }
inline m128i maskz_srlv_epi64(mmask8 k, m128i a, m128i count) noexcept { return _mm_maskz_srlv_epi64(k, a, count); }
inline m128i srlv_epi64(m128i a, m128i count) noexcept { return _mm_srlv_epi64(a, count); }
inline void mask_store_epi32(void* mem_addr, mmask8 k, m128i a) noexcept { _mm_mask_store_epi32(mem_addr, k, a); }
inline void mask_store_epi64(void* mem_addr, mmask8 k, m128i a) noexcept { _mm_mask_store_epi64(mem_addr, k, a); }
inline void mask_store_pd(void* mem_addr, mmask8 k, m128d a) noexcept { _mm_mask_store_pd(mem_addr, k, a); }
inline void store_pd(double* mem_addr, m128d a) noexcept { _mm_store_pd(mem_addr, a); }
inline void store_pd1(double* mem_addr, m128d a) noexcept { _mm_store_pd1(mem_addr, a); }
inline void mask_store_ps(void* mem_addr, mmask8 k, m128f a) noexcept { _mm_mask_store_ps(mem_addr, k, a); }
inline void store_ps(float* mem_addr, m128f a) noexcept { _mm_store_ps(mem_addr, a); }
inline void store_ps1(float* mem_addr, m128f a) noexcept { _mm_store_ps1(mem_addr, a); }
inline void mask_store_sd(double* mem_addr, mmask8 k, m128d a) noexcept { _mm_mask_store_sd(mem_addr, k, a); }
inline void store_sd(double* mem_addr, m128d a) noexcept { _mm_store_sd(mem_addr, a); }
inline void store_si128(m128i* mem_addr, m128i a) noexcept { _mm_store_si128(mem_addr, a); }
inline void mask_store_ss(float* mem_addr, mmask8 k, m128f a) noexcept { _mm_mask_store_ss(mem_addr, k, a); }
inline void store_ss(float* mem_addr, m128f a) noexcept { _mm_store_ss(mem_addr, a); }
inline void store1_pd(double* mem_addr, m128d a) noexcept { _mm_store1_pd(mem_addr, a); }
inline void store1_ps(float* mem_addr, m128f a) noexcept { _mm_store1_ps(mem_addr, a); }
inline void storeh_pd(double* mem_addr, m128d a) noexcept { _mm_storeh_pd(mem_addr, a); }
inline void storel_epi64(m128i* mem_addr, m128i a) noexcept { _mm_storel_epi64(mem_addr, a); }
inline void storel_pd(double* mem_addr, m128d a) noexcept { _mm_storel_pd(mem_addr, a); }
inline void storer_pd(double* mem_addr, m128d a) noexcept { _mm_storer_pd(mem_addr, a); }
inline void storer_ps(float* mem_addr, m128f a) noexcept { _mm_storer_ps(mem_addr, a); }
inline void mask_storeu_epi16(void* mem_addr, mmask8 k, m128i a) noexcept { _mm_mask_storeu_epi16(mem_addr, k, a); }
inline void storeu_epi16(void* mem_addr, m128i a) noexcept { _mm_storeu_epi16(mem_addr, a); }
inline void mask_storeu_epi32(void* mem_addr, mmask8 k, m128i a) noexcept { _mm_mask_storeu_epi32(mem_addr, k, a); }
inline void storeu_epi32(void* mem_addr, m128i a) noexcept { _mm_storeu_epi32(mem_addr, a); }
inline void mask_storeu_epi64(void* mem_addr, mmask8 k, m128i a) noexcept { _mm_mask_storeu_epi64(mem_addr, k, a); }
inline void storeu_epi64(void* mem_addr, m128i a) noexcept { _mm_storeu_epi64(mem_addr, a); }
inline void mask_storeu_epi8(void* mem_addr, mmask16 k, m128i a) noexcept { _mm_mask_storeu_epi8(mem_addr, k, a); }
inline void storeu_epi8(void* mem_addr, m128i a) noexcept { _mm_storeu_epi8(mem_addr, a); }
inline void mask_storeu_pd(void* mem_addr, mmask8 k, m128d a) noexcept { _mm_mask_storeu_pd(mem_addr, k, a); }
inline void storeu_pd(double* mem_addr, m128d a) noexcept { _mm_storeu_pd(mem_addr, a); }
inline void mask_storeu_ps(void* mem_addr, mmask8 k, m128f a) noexcept { _mm_mask_storeu_ps(mem_addr, k, a); }
inline void storeu_ps(float* mem_addr, m128f a) noexcept { _mm_storeu_ps(mem_addr, a); }
inline void storeu_si128(m128i* mem_addr, m128i a) noexcept { _mm_storeu_si128(mem_addr, a); }
inline void storeu_si16(void* mem_addr, m128i a) noexcept { _mm_storeu_si16(mem_addr, a); }
inline void storeu_si32(void* mem_addr, m128i a) noexcept { _mm_storeu_si32(mem_addr, a); }
inline void storeu_si64(void* mem_addr, m128i a) noexcept { _mm_storeu_si64(mem_addr, a); }
inline m128i stream_load_si128(m128i* mem_addr) noexcept { return _mm_stream_load_si128(mem_addr); }
inline void stream_pd(double* mem_addr, m128d a) noexcept { _mm_stream_pd(mem_addr, a); }
inline void stream_ps(float* mem_addr, m128f a) noexcept { _mm_stream_ps(mem_addr, a); }
inline void stream_si128(m128i* mem_addr, m128i a) noexcept { _mm_stream_si128(mem_addr, a); }
inline m128i mask_sub_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_sub_epi16(src, k, a, b); }
inline m128i maskz_sub_epi16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_sub_epi16(k, a, b); }
inline m128i sub_epi16(m128i a, m128i b) noexcept { return _mm_sub_epi16(a, b); }
inline m128i mask_sub_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_sub_epi32(src, k, a, b); }
inline m128i maskz_sub_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_sub_epi32(k, a, b); }
inline m128i sub_epi32(m128i a, m128i b) noexcept { return _mm_sub_epi32(a, b); }
inline m128i mask_sub_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_sub_epi64(src, k, a, b); }
inline m128i maskz_sub_epi64(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_sub_epi64(k, a, b); }
inline m128i sub_epi64(m128i a, m128i b) noexcept { return _mm_sub_epi64(a, b); }
inline m128i mask_sub_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_sub_epi8(src, k, a, b); }
inline m128i maskz_sub_epi8(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_sub_epi8(k, a, b); }
inline m128i sub_epi8(m128i a, m128i b) noexcept { return _mm_sub_epi8(a, b); }
inline m128d mask_sub_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_sub_pd(src, k, a, b); }
inline m128d maskz_sub_pd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_sub_pd(k, a, b); }
inline m128d sub_pd(m128d a, m128d b) noexcept { return _mm_sub_pd(a, b); }
inline m128f mask_sub_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_sub_ps(src, k, a, b); }
inline m128f maskz_sub_ps(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_sub_ps(k, a, b); }
inline m128f sub_ps(m128f a, m128f b) noexcept { return _mm_sub_ps(a, b); }
template<int4 rounding> inline m128d mask_sub_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_sub_round_sd(src, k, a, b, rounding); }
template<int4 rounding> inline m128d maskz_sub_round_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_sub_round_sd(k, a, b, rounding); }
template<int4 rounding> inline m128d sub_round_sd(m128d a, m128d b) noexcept { return _mm_sub_round_sd(a, b, rounding); }
template<int4 rounding> inline m128f mask_sub_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_sub_round_ss(src, k, a, b, rounding); }
template<int4 rounding> inline m128f maskz_sub_round_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_sub_round_ss(k, a, b, rounding); }
template<int4 rounding> inline m128f sub_round_ss(m128f a, m128f b) noexcept { return _mm_sub_round_ss(a, b, rounding); }
inline m128d mask_sub_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_sub_sd(src, k, a, b); }
inline m128d maskz_sub_sd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_sub_sd(k, a, b); }
inline m128d sub_sd(m128d a, m128d b) noexcept { return _mm_sub_sd(a, b); }
inline m128f mask_sub_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_sub_ss(src, k, a, b); }
inline m128f maskz_sub_ss(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_sub_ss(k, a, b); }
inline m128f sub_ss(m128f a, m128f b) noexcept { return _mm_sub_ss(a, b); }
inline m128i mask_subs_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_subs_epi16(src, k, a, b); }
inline m128i maskz_subs_epi16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_subs_epi16(k, a, b); }
inline m128i subs_epi16(m128i a, m128i b) noexcept { return _mm_subs_epi16(a, b); }
inline m128i mask_subs_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_subs_epi8(src, k, a, b); }
inline m128i maskz_subs_epi8(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_subs_epi8(k, a, b); }
inline m128i subs_epi8(m128i a, m128i b) noexcept { return _mm_subs_epi8(a, b); }
inline m128i mask_subs_epu16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_subs_epu16(src, k, a, b); }
inline m128i maskz_subs_epu16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_subs_epu16(k, a, b); }
inline m128i subs_epu16(m128i a, m128i b) noexcept { return _mm_subs_epu16(a, b); }
inline m128i mask_subs_epu8(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_subs_epu8(src, k, a, b); }
inline m128i maskz_subs_epu8(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_subs_epu8(k, a, b); }
inline m128i subs_epu8(m128i a, m128i b) noexcept { return _mm_subs_epu8(a, b); }
template<int4 imm8> inline m128i mask_ternarylogic_epi32(m128i a, mmask8 k, m128i b, m128i c) noexcept { return _mm_mask_ternarylogic_epi32(a, k, b, c, imm8); }
template<int4 imm8> inline m128i maskz_ternarylogic_epi32(mmask8 k, m128i a, m128i b, m128i c) noexcept { return _mm_maskz_ternarylogic_epi32(k, a, b, c, imm8); }
template<int4 imm8> inline m128i ternarylogic_epi32(m128i a, m128i b, m128i c) noexcept { return _mm_ternarylogic_epi32(a, b, c, imm8); }
template<int4 imm8> inline m128i mask_ternarylogic_epi64(m128i a, mmask8 k, m128i b, m128i c) noexcept { return _mm_mask_ternarylogic_epi64(a, k, b, c, imm8); }
template<int4 imm8> inline m128i maskz_ternarylogic_epi64(mmask8 k, m128i a, m128i b, m128i c) noexcept { return _mm_maskz_ternarylogic_epi64(k, a, b, c, imm8); }
template<int4 imm8> inline m128i ternarylogic_epi64(m128i a, m128i b, m128i c) noexcept { return _mm_ternarylogic_epi64(a, b, c, imm8); }
inline int test_all_ones(m128i a) noexcept { return _mm_test_all_ones(a); }
inline int test_all_zeros(m128i a, m128i mask) noexcept { return _mm_test_all_zeros(a, mask); }
inline mmask8 mask_test_epi16_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_test_epi16_mask(k1, a, b); }
inline mmask8 test_epi16_mask(m128i a, m128i b) noexcept { return _mm_test_epi16_mask(a, b); }
inline mmask8 mask_test_epi32_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_test_epi32_mask(k1, a, b); }
inline mmask8 test_epi32_mask(m128i a, m128i b) noexcept { return _mm_test_epi32_mask(a, b); }
inline mmask8 mask_test_epi64_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_test_epi64_mask(k1, a, b); }
inline mmask8 test_epi64_mask(m128i a, m128i b) noexcept { return _mm_test_epi64_mask(a, b); }
inline mmask16 mask_test_epi8_mask(mmask16 k1, m128i a, m128i b) noexcept { return _mm_mask_test_epi8_mask(k1, a, b); }
inline mmask16 test_epi8_mask(m128i a, m128i b) noexcept { return _mm_test_epi8_mask(a, b); }
inline int test_mix_ones_zeros(m128i a, m128i mask) noexcept { return _mm_test_mix_ones_zeros(a, mask); }
inline int testc_pd(m128d a, m128d b) noexcept { return _mm_testc_pd(a, b); }
inline int testc_ps(m128f a, m128f b) noexcept { return _mm_testc_ps(a, b); }
inline int testc_si128(m128i a, m128i b) noexcept { return _mm_testc_si128(a, b); }
inline mmask8 mask_testn_epi16_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_testn_epi16_mask(k1, a, b); }
inline mmask8 testn_epi16_mask(m128i a, m128i b) noexcept { return _mm_testn_epi16_mask(a, b); }
inline mmask8 mask_testn_epi32_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_testn_epi32_mask(k1, a, b); }
inline mmask8 testn_epi32_mask(m128i a, m128i b) noexcept { return _mm_testn_epi32_mask(a, b); }
inline mmask8 mask_testn_epi64_mask(mmask8 k1, m128i a, m128i b) noexcept { return _mm_mask_testn_epi64_mask(k1, a, b); }
inline mmask8 testn_epi64_mask(m128i a, m128i b) noexcept { return _mm_testn_epi64_mask(a, b); }
inline mmask16 mask_testn_epi8_mask(mmask16 k1, m128i a, m128i b) noexcept { return _mm_mask_testn_epi8_mask(k1, a, b); }
inline mmask16 testn_epi8_mask(m128i a, m128i b) noexcept { return _mm_testn_epi8_mask(a, b); }
inline int testnzc_pd(m128d a, m128d b) noexcept { return _mm_testnzc_pd(a, b); }
inline int testnzc_ps(m128f a, m128f b) noexcept { return _mm_testnzc_ps(a, b); }
inline int testnzc_si128(m128i a, m128i b) noexcept { return _mm_testnzc_si128(a, b); }
inline int testz_pd(m128d a, m128d b) noexcept { return _mm_testz_pd(a, b); }
inline int testz_ps(m128f a, m128f b) noexcept { return _mm_testz_ps(a, b); }
inline int testz_si128(m128i a, m128i b) noexcept { return _mm_testz_si128(a, b); }
inline void TRANSPOSE4_PS(m128f row0, m128f row1, m128f row2, m128f row3) noexcept { _MM_TRANSPOSE4_PS(row0, row1, row2, row3); }
inline int ucomieq_sd(m128d a, m128d b) noexcept { return _mm_ucomieq_sd(a, b); }
inline int ucomieq_ss(m128f a, m128f b) noexcept { return _mm_ucomieq_ss(a, b); }
inline int ucomige_sd(m128d a, m128d b) noexcept { return _mm_ucomige_sd(a, b); }
inline int ucomige_ss(m128f a, m128f b) noexcept { return _mm_ucomige_ss(a, b); }
inline int ucomigt_sd(m128d a, m128d b) noexcept { return _mm_ucomigt_sd(a, b); }
inline int ucomigt_ss(m128f a, m128f b) noexcept { return _mm_ucomigt_ss(a, b); }
inline int ucomile_sd(m128d a, m128d b) noexcept { return _mm_ucomile_sd(a, b); }
inline int ucomile_ss(m128f a, m128f b) noexcept { return _mm_ucomile_ss(a, b); }
inline int ucomilt_sd(m128d a, m128d b) noexcept { return _mm_ucomilt_sd(a, b); }
inline int ucomilt_ss(m128f a, m128f b) noexcept { return _mm_ucomilt_ss(a, b); }
inline int ucomineq_sd(m128d a, m128d b) noexcept { return _mm_ucomineq_sd(a, b); }
inline int ucomineq_ss(m128f a, m128f b) noexcept { return _mm_ucomineq_ss(a, b); }
inline m128d undefined_pd(void) noexcept { return _mm_undefined_pd(); }
inline m128f undefined_ps(void) noexcept { return _mm_undefined_ps(); }
inline m128i undefined_si128(void) noexcept { return _mm_undefined_si128(); }
inline m128i mask_unpackhi_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_unpackhi_epi16(src, k, a, b); }
inline m128i maskz_unpackhi_epi16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_unpackhi_epi16(k, a, b); }
inline m128i unpackhi_epi16(m128i a, m128i b) noexcept { return _mm_unpackhi_epi16(a, b); }
inline m128i mask_unpackhi_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_unpackhi_epi32(src, k, a, b); }
inline m128i maskz_unpackhi_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_unpackhi_epi32(k, a, b); }
inline m128i unpackhi_epi32(m128i a, m128i b) noexcept { return _mm_unpackhi_epi32(a, b); }
inline m128i mask_unpackhi_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_unpackhi_epi64(src, k, a, b); }
inline m128i maskz_unpackhi_epi64(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_unpackhi_epi64(k, a, b); }
inline m128i unpackhi_epi64(m128i a, m128i b) noexcept { return _mm_unpackhi_epi64(a, b); }
inline m128i mask_unpackhi_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_unpackhi_epi8(src, k, a, b); }
inline m128i maskz_unpackhi_epi8(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_unpackhi_epi8(k, a, b); }
inline m128i unpackhi_epi8(m128i a, m128i b) noexcept { return _mm_unpackhi_epi8(a, b); }
inline m128d mask_unpackhi_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_unpackhi_pd(src, k, a, b); }
inline m128d maskz_unpackhi_pd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_unpackhi_pd(k, a, b); }
inline m128d unpackhi_pd(m128d a, m128d b) noexcept { return _mm_unpackhi_pd(a, b); }
inline m128f mask_unpackhi_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_unpackhi_ps(src, k, a, b); }
inline m128f maskz_unpackhi_ps(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_unpackhi_ps(k, a, b); }
inline m128f unpackhi_ps(m128f a, m128f b) noexcept { return _mm_unpackhi_ps(a, b); }
inline m128i mask_unpacklo_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_unpacklo_epi16(src, k, a, b); }
inline m128i maskz_unpacklo_epi16(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_unpacklo_epi16(k, a, b); }
inline m128i unpacklo_epi16(m128i a, m128i b) noexcept { return _mm_unpacklo_epi16(a, b); }
inline m128i mask_unpacklo_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_unpacklo_epi32(src, k, a, b); }
inline m128i maskz_unpacklo_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_unpacklo_epi32(k, a, b); }
inline m128i unpacklo_epi32(m128i a, m128i b) noexcept { return _mm_unpacklo_epi32(a, b); }
inline m128i mask_unpacklo_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_unpacklo_epi64(src, k, a, b); }
inline m128i maskz_unpacklo_epi64(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_unpacklo_epi64(k, a, b); }
inline m128i unpacklo_epi64(m128i a, m128i b) noexcept { return _mm_unpacklo_epi64(a, b); }
inline m128i mask_unpacklo_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept { return _mm_mask_unpacklo_epi8(src, k, a, b); }
inline m128i maskz_unpacklo_epi8(mmask16 k, m128i a, m128i b) noexcept { return _mm_maskz_unpacklo_epi8(k, a, b); }
inline m128i unpacklo_epi8(m128i a, m128i b) noexcept { return _mm_unpacklo_epi8(a, b); }
inline m128d mask_unpacklo_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_unpacklo_pd(src, k, a, b); }
inline m128d maskz_unpacklo_pd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_unpacklo_pd(k, a, b); }
inline m128d unpacklo_pd(m128d a, m128d b) noexcept { return _mm_unpacklo_pd(a, b); }
inline m128f mask_unpacklo_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_unpacklo_ps(src, k, a, b); }
inline m128f maskz_unpacklo_ps(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_unpacklo_ps(k, a, b); }
inline m128f unpacklo_ps(m128f a, m128f b) noexcept { return _mm_unpacklo_ps(a, b); }
inline m128i mask_xor_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_xor_epi32(src, k, a, b); }
inline m128i maskz_xor_epi32(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_xor_epi32(k, a, b); }
inline m128i mask_xor_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept { return _mm_mask_xor_epi64(src, k, a, b); }
inline m128i maskz_xor_epi64(mmask8 k, m128i a, m128i b) noexcept { return _mm_maskz_xor_epi64(k, a, b); }
inline m128d mask_xor_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept { return _mm_mask_xor_pd(src, k, a, b); }
inline m128d maskz_xor_pd(mmask8 k, m128d a, m128d b) noexcept { return _mm_maskz_xor_pd(k, a, b); }
inline m128d xor_pd(m128d a, m128d b) noexcept { return _mm_xor_pd(a, b); }
inline m128f mask_xor_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept { return _mm_mask_xor_ps(src, k, a, b); }
inline m128f maskz_xor_ps(mmask8 k, m128f a, m128f b) noexcept { return _mm_maskz_xor_ps(k, a, b); }
inline m128f xor_ps(m128f a, m128f b) noexcept { return _mm_xor_ps(a, b); }
inline m128i xor_si128(m128i a, m128i b) noexcept { return _mm_xor_si128(a, b); }
} // namespace yw::intrin::inline m128
#else
namespace yw::intrin::inline m128 {
union alignas(16) m128d {
  fat8 _[2];
};
union alignas(16) m128f {
  fat4 _[4];
};
union alignas(16) m128i {
  int8 _[2];
};
using mmask8 = nat1;
using mmask16 = nat2;
enum MANTISSA_NORM_ENUM {
  MANT_NORM_1_2, // interval [1, 2)
  MANT_NORM_p5_2, // interval [1.5, 2)
  MANT_NORM_p5_1, // interval [1.5, 1)
  MANT_NORM_p75_1p5 // interval [0.75, 1.5)
};
extern m128i abs_epi16(m128i a) noexcept;
extern m128i mask_abs_epi16(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_abs_epi16(mmask8 k, m128i a) noexcept;
extern m128i abs_epi32(m128i a) noexcept;
extern m128i mask_abs_epi32(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_abs_epi32(mmask8 k, m128i a) noexcept;
extern m128i abs_epi64(m128i a) noexcept;
extern m128i mask_abs_epi64(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_abs_epi64(mmask8 k, m128i a) noexcept;
extern m128i abs_epi8(m128i a) noexcept;
extern m128i mask_abs_epi8(m128i src, mmask16 k, m128i a) noexcept;
extern m128i maskz_abs_epi8(mmask16 k, m128i a) noexcept;
extern m128i add_epi16(m128i a, m128i b) noexcept;
extern m128i mask_add_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_add_epi16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i add_epi32(m128i a, m128i b) noexcept;
extern m128i mask_add_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_add_epi32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i add_epi64(m128i a, m128i b) noexcept;
extern m128i mask_add_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_add_epi64(mmask8 k, m128i a, m128i b) noexcept;
extern m128i add_epi8(m128i a, m128i b) noexcept;
extern m128i mask_add_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept;
extern m128i maskz_add_epi8(mmask16 k, m128i a, m128i b) noexcept;
extern m128d add_pd(m128d a, m128d b) noexcept;
extern m128d mask_add_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_add_pd(mmask8 k, m128d a, m128d b) noexcept;
extern m128f add_ps(m128f a, m128f b) noexcept;
extern m128f mask_add_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_add_ps(mmask8 k, m128f a, m128f b) noexcept;
template<int4 rounding> extern m128d add_round_sd(m128d a, m128d b) noexcept;
template<int4 rounding> extern m128d mask_add_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int4 rounding> extern m128d maskz_add_round_sd(mmask8 k, m128d a, m128d b) noexcept;
template<int4 rounding> extern m128f add_round_ss(m128f a, m128f b) noexcept;
template<int4 rounding> extern m128f mask_add_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int4 rounding> extern m128f maskz_add_round_ss(mmask8 k, m128f a, m128f b) noexcept;
extern m128d add_sd(m128d a, m128d b) noexcept;
extern m128d mask_add_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_add_sd(mmask8 k, m128d a, m128d b) noexcept;
extern m128f add_ss(m128f a, m128f b) noexcept;
extern m128f mask_add_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_add_ss(mmask8 k, m128f a, m128f b) noexcept;
extern m128i adds_epi16(m128i a, m128i b) noexcept;
extern m128i mask_adds_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_adds_epi16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i adds_epi8(m128i a, m128i b) noexcept;
extern m128i mask_adds_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept;
extern m128i maskz_adds_epi8(mmask16 k, m128i a, m128i b) noexcept;
extern m128i adds_epu16(m128i a, m128i b) noexcept;
extern m128i mask_adds_epu16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_adds_epu16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i adds_epu8(m128i a, m128i b) noexcept;
extern m128i mask_adds_epu8(m128i src, mmask16 k, m128i a, m128i b) noexcept;
extern m128i maskz_adds_epu8(mmask16 k, m128i a, m128i b) noexcept;
extern m128d addsub_pd(m128d a, m128d b) noexcept;
extern m128f addsub_ps(m128f a, m128f b) noexcept;
extern m128i aesdec_si128(m128i a, m128i RoundKey) noexcept;
extern nat1 aesdec128kl_u8(m128i* __odata, m128i __idata, const void* __h) noexcept;
extern nat1 aesdec256kl_u8(m128i* __odata, m128i __idata, const void* __h) noexcept;
extern m128i aesdeclast_si128(m128i a, m128i RoundKey) noexcept;
extern nat1 aesdecwide128kl_u8(m128i* __odata, const m128i* __idata, const void* __h) noexcept;
extern nat1 aesdecwide256kl_u8(m128i* __odata, const m128i* __idata, const void* __h) noexcept;
extern m128i aesenc_si128(m128i a, m128i RoundKey) noexcept;
extern nat1 aesenc128kl_u8(m128i* __odata, m128i __idata, const void* __h) noexcept;
extern nat1 aesenc256kl_u8(m128i* __odata, m128i __idata, const void* __h) noexcept;
extern m128i aesenclast_si128(m128i a, m128i RoundKey) noexcept;
extern nat1 aesencwide128kl_u8(m128i* __odata, const m128i* __idata, const void* __h) noexcept;
extern nat1 aesencwide256kl_u8(m128i* __odata, const m128i* __idata, const void* __h) noexcept;
extern m128i aesimc_si128(m128i a) noexcept;
template<int4 imm8> extern m128i aeskeygenassist_si128(m128i a) noexcept;
template<int4 imm8> extern m128i alignr_epi32(m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i mask_alignr_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i maskz_alignr_epi32(mmask8 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i alignr_epi64(m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i mask_alignr_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i maskz_alignr_epi64(mmask8 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i alignr_epi8(m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i mask_alignr_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i maskz_alignr_epi8(mmask16 k, m128i a, m128i b) noexcept;
extern m128i mask_and_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_and_epi32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i mask_and_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_and_epi64(mmask8 k, m128i a, m128i b) noexcept;
extern m128d and_pd(m128d a, m128d b) noexcept;
extern m128d mask_and_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_and_pd(mmask8 k, m128d a, m128d b) noexcept;
extern m128f and_ps(m128f a, m128f b) noexcept;
extern m128f mask_and_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_and_ps(mmask8 k, m128f a, m128f b) noexcept;
extern m128i and_si128(m128i a, m128i b) noexcept;
extern m128i mask_andnot_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_andnot_epi32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i mask_andnot_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_andnot_epi64(mmask8 k, m128i a, m128i b) noexcept;
extern m128d andnot_pd(m128d a, m128d b) noexcept;
extern m128d mask_andnot_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_andnot_pd(mmask8 k, m128d a, m128d b) noexcept;
extern m128f andnot_ps(m128f a, m128f b) noexcept;
extern m128f mask_andnot_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_andnot_ps(mmask8 k, m128f a, m128f b) noexcept;
extern m128i andnot_si128(m128i a, m128i b) noexcept;
extern m128i avg_epu16(m128i a, m128i b) noexcept;
extern m128i mask_avg_epu16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_avg_epu16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i avg_epu8(m128i a, m128i b) noexcept;
extern m128i mask_avg_epu8(m128i src, mmask16 k, m128i a, m128i b) noexcept;
extern m128i maskz_avg_epu8(mmask16 k, m128i a, m128i b) noexcept;
extern mmask16 bitshuffle_epi64_mask(m128i b, m128i c) noexcept;
extern mmask16 mask_bitshuffle_epi64_mask(mmask16 k, m128i b, m128i c) noexcept;
template<int4 imm8> extern m128i blend_epi16(m128i a, m128i b) noexcept;
extern m128i mask_blend_epi16(mmask8 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i blend_epi32(m128i a, m128i b) noexcept;
extern m128i mask_blend_epi32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i mask_blend_epi64(mmask8 k, m128i a, m128i b) noexcept;
extern m128i mask_blend_epi8(mmask16 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128d blend_pd(m128d a, m128d b) noexcept;
extern m128d mask_blend_pd(mmask8 k, m128d a, m128d b) noexcept;
template<int4 imm8> extern m128f blend_ps(m128f a, m128f b) noexcept;
extern m128f mask_blend_ps(mmask8 k, m128f a, m128f b) noexcept;
extern m128i blendv_epi8(m128i a, m128i b, m128i mask) noexcept;
extern m128d blendv_pd(m128d a, m128d b, m128d mask) noexcept;
extern m128f blendv_ps(m128f a, m128f b, m128f mask) noexcept;
extern m128i broadcast_i32x2(m128i a) noexcept;
extern m128i mask_broadcast_i32x2(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_broadcast_i32x2(mmask8 k, m128i a) noexcept;
extern m128f broadcast_ss(float const* mem_addr) noexcept;
extern m128i broadcastb_epi8(m128i a) noexcept;
extern m128i mask_broadcastb_epi8(m128i src, mmask16 k, m128i a) noexcept;
extern m128i maskz_broadcastb_epi8(mmask16 k, m128i a) noexcept;
extern m128i broadcastd_epi32(m128i a) noexcept;
extern m128i mask_broadcastd_epi32(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_broadcastd_epi32(mmask8 k, m128i a) noexcept;
extern m128i broadcastmb_epi64(mmask8 k) noexcept;
extern m128i broadcastmw_epi32(mmask16 k) noexcept;
extern m128i broadcastq_epi64(m128i a) noexcept;
extern m128i mask_broadcastq_epi64(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_broadcastq_epi64(mmask8 k, m128i a) noexcept;
extern m128d broadcastsd_pd(m128d a) noexcept;
extern m128f broadcastss_ps(m128f a) noexcept;
extern m128f mask_broadcastss_ps(m128f src, mmask8 k, m128f a) noexcept;
extern m128f maskz_broadcastss_ps(mmask8 k, m128f a) noexcept;
extern m128i broadcastw_epi16(m128i a) noexcept;
extern m128i mask_broadcastw_epi16(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_broadcastw_epi16(mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i bslli_si128(m128i a) noexcept;
template<int4 imm8> extern m128i bsrli_si128(m128i a) noexcept;
extern m128f castpd_ps(m128d a) noexcept;
extern m128i castpd_si128(m128d a) noexcept;
extern m128d castps_pd(m128f a) noexcept;
extern m128i castps_si128(m128f a) noexcept;
extern m128d castsi128_pd(m128i a) noexcept;
extern m128f castsi128_ps(m128i a) noexcept;
extern m128d ceil_pd(m128d a) noexcept;
extern m128f ceil_ps(m128f a) noexcept;
extern m128d ceil_sd(m128d a, m128d b) noexcept;
extern m128f ceil_ss(m128f a, m128f b) noexcept;
template<int4 imm8> extern m128i clmulepi64_si128(m128i a, m128i b) noexcept;
template<int4 imm8> extern mmask8 cmp_epi16_mask(m128i a, m128i b) noexcept;
template<int4 imm8> extern mmask8 mask_cmp_epi16_mask(mmask8 k1, m128i a, m128i b) noexcept;
template<int4 imm8> extern mmask8 cmp_epi32_mask(m128i a, m128i b) noexcept;
template<int4 imm8> extern mmask8 mask_cmp_epi32_mask(mmask8 k1, m128i a, m128i b) noexcept;
template<int4 imm8> extern mmask8 cmp_epi64_mask(m128i a, m128i b) noexcept;
template<int4 imm8> extern mmask8 mask_cmp_epi64_mask(mmask8 k1, m128i a, m128i b) noexcept;
template<int4 imm8> extern mmask16 cmp_epi8_mask(m128i a, m128i b) noexcept;
template<int4 imm8> extern mmask16 mask_cmp_epi8_mask(mmask16 k1, m128i a, m128i b) noexcept;
template<int4 imm8> extern mmask8 cmp_epu16_mask(m128i a, m128i b) noexcept;
template<int4 imm8> extern mmask8 mask_cmp_epu16_mask(mmask8 k1, m128i a, m128i b) noexcept;
template<int4 imm8> extern mmask8 cmp_epu32_mask(m128i a, m128i b) noexcept;
template<int4 imm8> extern mmask8 mask_cmp_epu32_mask(mmask8 k1, m128i a, m128i b) noexcept;
template<int4 imm8> extern mmask8 cmp_epu64_mask(m128i a, m128i b) noexcept;
template<int4 imm8> extern mmask8 mask_cmp_epu64_mask(mmask8 k1, m128i a, m128i b) noexcept;
template<int4 imm8> extern mmask16 cmp_epu8_mask(m128i a, m128i b) noexcept;
template<int4 imm8> extern mmask16 mask_cmp_epu8_mask(mmask16 k1, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128d cmp_pd(m128d a, m128d b) noexcept;
template<int4 imm8> extern mmask8 cmp_pd_mask(m128d a, m128d b) noexcept;
template<int4 imm8> extern mmask8 mask_cmp_pd_mask(mmask8 k1, m128d a, m128d b) noexcept;
template<int4 imm8> extern m128f cmp_ps(m128f a, m128f b) noexcept;
template<int4 imm8> extern mmask8 cmp_ps_mask(m128f a, m128f b) noexcept;
template<int4 imm8> extern mmask8 mask_cmp_ps_mask(mmask8 k1, m128f a, m128f b) noexcept;
template<int4 imm8, int sae> extern mmask8 cmp_round_sd_mask(m128d a, m128d b) noexcept;
template<int4 imm8, int sae> extern mmask8 mask_cmp_round_sd_mask(mmask8 k1, m128d a, m128d b) noexcept;
template<int4 imm8, int sae> extern mmask8 cmp_round_ss_mask(m128f a, m128f b) noexcept;
template<int4 imm8, int sae> extern mmask8 mask_cmp_round_ss_mask(mmask8 k1, m128f a, m128f b) noexcept;
template<int4 imm8> extern m128d cmp_sd(m128d a, m128d b) noexcept;
template<int4 imm8> extern mmask8 cmp_sd_mask(m128d a, m128d b) noexcept;
template<int4 imm8> extern mmask8 mask_cmp_sd_mask(mmask8 k1, m128d a, m128d b) noexcept;
template<int4 imm8> extern m128f cmp_ss(m128f a, m128f b) noexcept;
template<int4 imm8> extern mmask8 cmp_ss_mask(m128f a, m128f b) noexcept;
template<int4 imm8> extern mmask8 mask_cmp_ss_mask(mmask8 k1, m128f a, m128f b) noexcept;
extern m128i cmpeq_epi16(m128i a, m128i b) noexcept;
extern mmask8 cmpeq_epi16_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpeq_epi16_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern m128i cmpeq_epi32(m128i a, m128i b) noexcept;
extern mmask8 cmpeq_epi32_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpeq_epi32_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern m128i cmpeq_epi64(m128i a, m128i b) noexcept;
extern mmask8 cmpeq_epi64_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpeq_epi64_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern m128i cmpeq_epi8(m128i a, m128i b) noexcept;
extern mmask16 cmpeq_epi8_mask(m128i a, m128i b) noexcept;
extern mmask16 mask_cmpeq_epi8_mask(mmask16 k1, m128i a, m128i b) noexcept;
extern mmask8 cmpeq_epu16_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpeq_epu16_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmpeq_epu32_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpeq_epu32_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmpeq_epu64_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpeq_epu64_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask16 cmpeq_epu8_mask(m128i a, m128i b) noexcept;
extern mmask16 mask_cmpeq_epu8_mask(mmask16 k1, m128i a, m128i b) noexcept;
extern m128d cmpeq_pd(m128d a, m128d b) noexcept;
extern m128f cmpeq_ps(m128f a, m128f b) noexcept;
extern m128d cmpeq_sd(m128d a, m128d b) noexcept;
extern m128f cmpeq_ss(m128f a, m128f b) noexcept;
template<int4 imm8> extern int cmpestra(m128i a, int la, m128i b, int lb) noexcept;
template<int4 imm8> extern int cmpestrc(m128i a, int la, m128i b, int lb) noexcept;
template<int4 imm8> extern int cmpestri(m128i a, int la, m128i b, int lb) noexcept;
template<int4 imm8> extern m128i cmpestrm(m128i a, int la, m128i b, int lb) noexcept;
template<int4 imm8> extern int cmpestro(m128i a, int la, m128i b, int lb) noexcept;
template<int4 imm8> extern int cmpestrs(m128i a, int la, m128i b, int lb) noexcept;
template<int4 imm8> extern int cmpestrz(m128i a, int la, m128i b, int lb) noexcept;
extern mmask8 cmpge_epi16_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpge_epi16_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmpge_epi32_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpge_epi32_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmpge_epi64_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpge_epi64_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask16 cmpge_epi8_mask(m128i a, m128i b) noexcept;
extern mmask16 mask_cmpge_epi8_mask(mmask16 k1, m128i a, m128i b) noexcept;
extern mmask8 cmpge_epu16_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpge_epu16_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmpge_epu32_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpge_epu32_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmpge_epu64_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpge_epu64_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask16 cmpge_epu8_mask(m128i a, m128i b) noexcept;
extern mmask16 mask_cmpge_epu8_mask(mmask16 k1, m128i a, m128i b) noexcept;
extern m128d cmpge_pd(m128d a, m128d b) noexcept;
extern m128f cmpge_ps(m128f a, m128f b) noexcept;
extern m128d cmpge_sd(m128d a, m128d b) noexcept;
extern m128f cmpge_ss(m128f a, m128f b) noexcept;
extern m128i cmpgt_epi16(m128i a, m128i b) noexcept;
extern mmask8 cmpgt_epi16_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpgt_epi16_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern m128i cmpgt_epi32(m128i a, m128i b) noexcept;
extern mmask8 cmpgt_epi32_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpgt_epi32_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern m128i cmpgt_epi64(m128i a, m128i b) noexcept;
extern mmask8 cmpgt_epi64_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpgt_epi64_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern m128i cmpgt_epi8(m128i a, m128i b) noexcept;
extern mmask16 cmpgt_epi8_mask(m128i a, m128i b) noexcept;
extern mmask16 mask_cmpgt_epi8_mask(mmask16 k1, m128i a, m128i b) noexcept;
extern mmask8 cmpgt_epu16_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpgt_epu16_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmpgt_epu32_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpgt_epu32_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmpgt_epu64_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpgt_epu64_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask16 cmpgt_epu8_mask(m128i a, m128i b) noexcept;
extern mmask16 mask_cmpgt_epu8_mask(mmask16 k1, m128i a, m128i b) noexcept;
extern m128d cmpgt_pd(m128d a, m128d b) noexcept;
extern m128f cmpgt_ps(m128f a, m128f b) noexcept;
extern m128d cmpgt_sd(m128d a, m128d b) noexcept;
extern m128f cmpgt_ss(m128f a, m128f b) noexcept;
template<int4 imm8> extern int cmpistra(m128i a, m128i b) noexcept;
template<int4 imm8> extern int cmpistrc(m128i a, m128i b) noexcept;
template<int4 imm8> extern int cmpistri(m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i cmpistrm(m128i a, m128i b) noexcept;
template<int4 imm8> extern int cmpistro(m128i a, m128i b) noexcept;
template<int4 imm8> extern int cmpistrs(m128i a, m128i b) noexcept;
template<int4 imm8> extern int cmpistrz(m128i a, m128i b) noexcept;
extern mmask8 cmple_epi16_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmple_epi16_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmple_epi32_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmple_epi32_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmple_epi64_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmple_epi64_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask16 cmple_epi8_mask(m128i a, m128i b) noexcept;
extern mmask16 mask_cmple_epi8_mask(mmask16 k1, m128i a, m128i b) noexcept;
extern mmask8 cmple_epu16_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmple_epu16_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmple_epu32_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmple_epu32_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmple_epu64_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmple_epu64_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask16 cmple_epu8_mask(m128i a, m128i b) noexcept;
extern mmask16 mask_cmple_epu8_mask(mmask16 k1, m128i a, m128i b) noexcept;
extern m128d cmple_pd(m128d a, m128d b) noexcept;
extern m128f cmple_ps(m128f a, m128f b) noexcept;
extern m128d cmple_sd(m128d a, m128d b) noexcept;
extern m128f cmple_ss(m128f a, m128f b) noexcept;
extern m128i cmplt_epi16(m128i a, m128i b) noexcept;
extern mmask8 cmplt_epi16_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmplt_epi16_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern m128i cmplt_epi32(m128i a, m128i b) noexcept;
extern mmask8 cmplt_epi32_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmplt_epi32_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmplt_epi64_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmplt_epi64_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern m128i cmplt_epi8(m128i a, m128i b) noexcept;
extern mmask16 cmplt_epi8_mask(m128i a, m128i b) noexcept;
extern mmask16 mask_cmplt_epi8_mask(mmask16 k1, m128i a, m128i b) noexcept;
extern mmask8 cmplt_epu16_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmplt_epu16_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmplt_epu32_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmplt_epu32_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmplt_epu64_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmplt_epu64_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask16 cmplt_epu8_mask(m128i a, m128i b) noexcept;
extern mmask16 mask_cmplt_epu8_mask(mmask16 k1, m128i a, m128i b) noexcept;
extern m128d cmplt_pd(m128d a, m128d b) noexcept;
extern m128f cmplt_ps(m128f a, m128f b) noexcept;
extern m128d cmplt_sd(m128d a, m128d b) noexcept;
extern m128f cmplt_ss(m128f a, m128f b) noexcept;
extern mmask8 cmpneq_epi16_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpneq_epi16_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmpneq_epi32_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpneq_epi32_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmpneq_epi64_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpneq_epi64_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask16 cmpneq_epi8_mask(m128i a, m128i b) noexcept;
extern mmask16 mask_cmpneq_epi8_mask(mmask16 k1, m128i a, m128i b) noexcept;
extern mmask8 cmpneq_epu16_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpneq_epu16_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmpneq_epu32_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpneq_epu32_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 cmpneq_epu64_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_cmpneq_epu64_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask16 cmpneq_epu8_mask(m128i a, m128i b) noexcept;
extern mmask16 mask_cmpneq_epu8_mask(mmask16 k1, m128i a, m128i b) noexcept;
extern m128d cmpneq_pd(m128d a, m128d b) noexcept;
extern m128f cmpneq_ps(m128f a, m128f b) noexcept;
extern m128d cmpneq_sd(m128d a, m128d b) noexcept;
extern m128f cmpneq_ss(m128f a, m128f b) noexcept;
extern m128d cmpnge_pd(m128d a, m128d b) noexcept;
extern m128f cmpnge_ps(m128f a, m128f b) noexcept;
extern m128d cmpnge_sd(m128d a, m128d b) noexcept;
extern m128f cmpnge_ss(m128f a, m128f b) noexcept;
extern m128d cmpngt_pd(m128d a, m128d b) noexcept;
extern m128f cmpngt_ps(m128f a, m128f b) noexcept;
extern m128d cmpngt_sd(m128d a, m128d b) noexcept;
extern m128f cmpngt_ss(m128f a, m128f b) noexcept;
extern m128d cmpnle_pd(m128d a, m128d b) noexcept;
extern m128f cmpnle_ps(m128f a, m128f b) noexcept;
extern m128d cmpnle_sd(m128d a, m128d b) noexcept;
extern m128f cmpnle_ss(m128f a, m128f b) noexcept;
extern m128d cmpnlt_pd(m128d a, m128d b) noexcept;
extern m128f cmpnlt_ps(m128f a, m128f b) noexcept;
extern m128d cmpnlt_sd(m128d a, m128d b) noexcept;
extern m128f cmpnlt_ss(m128f a, m128f b) noexcept;
extern m128d cmpord_pd(m128d a, m128d b) noexcept;
extern m128f cmpord_ps(m128f a, m128f b) noexcept;
extern m128d cmpord_sd(m128d a, m128d b) noexcept;
extern m128f cmpord_ss(m128f a, m128f b) noexcept;
extern m128d cmpunord_pd(m128d a, m128d b) noexcept;
extern m128f cmpunord_ps(m128f a, m128f b) noexcept;
extern m128d cmpunord_sd(m128d a, m128d b) noexcept;
extern m128f cmpunord_ss(m128f a, m128f b) noexcept;
template<int4 imm8, int sae> extern int comi_round_sd(m128d a, m128d b) noexcept;
template<int4 imm8, int sae> extern int comi_round_ss(m128f a, m128f b) noexcept;
extern int comieq_sd(m128d a, m128d b) noexcept;
extern int comieq_ss(m128f a, m128f b) noexcept;
extern int comige_sd(m128d a, m128d b) noexcept;
extern int comige_ss(m128f a, m128f b) noexcept;
extern int comigt_sd(m128d a, m128d b) noexcept;
extern int comigt_ss(m128f a, m128f b) noexcept;
extern int comile_sd(m128d a, m128d b) noexcept;
extern int comile_ss(m128f a, m128f b) noexcept;
extern int comilt_sd(m128d a, m128d b) noexcept;
extern int comilt_ss(m128f a, m128f b) noexcept;
extern int comineq_sd(m128d a, m128d b) noexcept;
extern int comineq_ss(m128f a, m128f b) noexcept;
extern m128i mask_compress_epi16(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_compress_epi16(mmask8 k, m128i a) noexcept;
extern m128i mask_compress_epi32(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_compress_epi32(mmask8 k, m128i a) noexcept;
extern m128i mask_compress_epi64(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_compress_epi64(mmask8 k, m128i a) noexcept;
extern m128i mask_compress_epi8(m128i src, mmask16 k, m128i a) noexcept;
extern m128i maskz_compress_epi8(mmask16 k, m128i a) noexcept;
extern m128d mask_compress_pd(m128d src, mmask8 k, m128d a) noexcept;
extern m128d maskz_compress_pd(mmask8 k, m128d a) noexcept;
extern m128f mask_compress_ps(m128f src, mmask8 k, m128f a) noexcept;
extern m128f maskz_compress_ps(mmask8 k, m128f a) noexcept;
extern void mask_compressstoreu_epi16(void* base_addr, mmask8 k, m128i a) noexcept;
extern void mask_compressstoreu_epi32(void* base_addr, mmask8 k, m128i a) noexcept;
extern void mask_compressstoreu_epi64(void* base_addr, mmask8 k, m128i a) noexcept;
extern void mask_compressstoreu_epi8(void* base_addr, mmask16 k, m128i a) noexcept;
extern void mask_compressstoreu_pd(void* base_addr, mmask8 k, m128d a) noexcept;
extern void mask_compressstoreu_ps(void* base_addr, mmask8 k, m128f a) noexcept;
extern m128i conflict_epi32(m128i a) noexcept;
extern m128i mask_conflict_epi32(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_conflict_epi32(mmask8 k, m128i a) noexcept;
extern m128i conflict_epi64(m128i a) noexcept;
extern m128i mask_conflict_epi64(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_conflict_epi64(mmask8 k, m128i a) noexcept;
template<int4 rounding> extern m128f cvt_roundi32_ss(m128f a, int b) noexcept;
template<int4 rounding> extern m128d cvt_roundi64_sd(m128d a, int8 b) noexcept;
template<int4 rounding> extern m128f cvt_roundi64_ss(m128f a, int8 b) noexcept;
template<int4 rounding> extern int cvt_roundsd_i32(m128d a) noexcept;
template<int4 rounding> extern int8 cvt_roundsd_i64(m128d a) noexcept;
template<int4 rounding> extern int cvt_roundsd_si32(m128d a) noexcept;
template<int4 rounding> extern int8 cvt_roundsd_si64(m128d a) noexcept;
template<int4 rounding> extern m128f cvt_roundsd_ss(m128f a, m128d b) noexcept;
template<int4 rounding> extern m128f mask_cvt_roundsd_ss(m128f src, mmask8 k, m128f a, m128d b) noexcept;
template<int4 rounding> extern m128f maskz_cvt_roundsd_ss(mmask8 k, m128f a, m128d b) noexcept;
template<int4 rounding> extern nat4 cvt_roundsd_u32(m128d a) noexcept;
template<int4 rounding> extern nat8 cvt_roundsd_u64(m128d a) noexcept;
template<int4 rounding> extern m128f cvt_roundsi32_ss(m128f a, int b) noexcept;
template<int4 rounding> extern m128d cvt_roundsi64_sd(m128d a, int8 b) noexcept;
template<int4 rounding> extern m128f cvt_roundsi64_ss(m128f a, int8 b) noexcept;
template<int4 rounding> extern int cvt_roundss_i32(m128f a) noexcept;
template<int4 rounding> extern int8 cvt_roundss_i64(m128f a) noexcept;
template<int sae> extern m128d cvt_roundss_sd(m128d a, m128f b) noexcept;
template<int sae> extern m128d mask_cvt_roundss_sd(m128d src, mmask8 k, m128d a, m128f b) noexcept;
template<int sae> extern m128d maskz_cvt_roundss_sd(mmask8 k, m128d a, m128f b) noexcept;
template<int4 rounding> extern int cvt_roundss_si32(m128f a) noexcept;
template<int4 rounding> extern int8 cvt_roundss_si64(m128f a) noexcept;
template<int4 rounding> extern nat4 cvt_roundss_u32(m128f a) noexcept;
template<int4 rounding> extern nat8 cvt_roundss_u64(m128f a) noexcept;
template<int4 rounding> extern m128f cvt_roundu32_ss(m128f a, nat4 b) noexcept;
template<int4 rounding> extern m128d cvt_roundu64_sd(m128d a, nat8 b) noexcept;
template<int4 rounding> extern m128f cvt_roundu64_ss(m128f a, nat8 b) noexcept;
extern m128f cvt_si2ss(m128f a, int b) noexcept;
extern int cvt_ss2si(m128f a) noexcept;
extern m128i cvtepi16_epi32(m128i a) noexcept;
extern m128i mask_cvtepi16_epi32(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtepi16_epi32(mmask8 k, m128i a) noexcept;
extern m128i cvtepi16_epi64(m128i a) noexcept;
extern m128i mask_cvtepi16_epi64(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtepi16_epi64(mmask8 k, m128i a) noexcept;
extern m128i cvtepi16_epi8(m128i a) noexcept;
extern m128i mask_cvtepi16_epi8(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtepi16_epi8(mmask8 k, m128i a) noexcept;
extern void mask_cvtepi16_storeu_epi8(void* base_addr, mmask8 k, m128i a) noexcept;
extern m128i cvtepi32_epi16(m128i a) noexcept;
extern m128i mask_cvtepi32_epi16(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtepi32_epi16(mmask8 k, m128i a) noexcept;
extern m128i cvtepi32_epi64(m128i a) noexcept;
extern m128i mask_cvtepi32_epi64(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtepi32_epi64(mmask8 k, m128i a) noexcept;
extern m128i cvtepi32_epi8(m128i a) noexcept;
extern m128i mask_cvtepi32_epi8(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtepi32_epi8(mmask8 k, m128i a) noexcept;
extern m128d cvtepi32_pd(m128i a) noexcept;
extern m128d mask_cvtepi32_pd(m128d src, mmask8 k, m128i a) noexcept;
extern m128d maskz_cvtepi32_pd(mmask8 k, m128i a) noexcept;
extern m128f cvtepi32_ps(m128i a) noexcept;
extern m128f mask_cvtepi32_ps(m128f src, mmask8 k, m128i a) noexcept;
extern m128f maskz_cvtepi32_ps(mmask8 k, m128i a) noexcept;
extern void mask_cvtepi32_storeu_epi16(void* base_addr, mmask8 k, m128i a) noexcept;
extern void mask_cvtepi32_storeu_epi8(void* base_addr, mmask8 k, m128i a) noexcept;
extern m128i cvtepi64_epi16(m128i a) noexcept;
extern m128i mask_cvtepi64_epi16(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtepi64_epi16(mmask8 k, m128i a) noexcept;
extern m128i cvtepi64_epi32(m128i a) noexcept;
extern m128i mask_cvtepi64_epi32(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtepi64_epi32(mmask8 k, m128i a) noexcept;
extern m128i cvtepi64_epi8(m128i a) noexcept;
extern m128i mask_cvtepi64_epi8(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtepi64_epi8(mmask8 k, m128i a) noexcept;
extern m128d cvtepi64_pd(m128i a) noexcept;
extern m128d mask_cvtepi64_pd(m128d src, mmask8 k, m128i a) noexcept;
extern m128d maskz_cvtepi64_pd(mmask8 k, m128i a) noexcept;
extern m128f cvtepi64_ps(m128i a) noexcept;
extern m128f mask_cvtepi64_ps(m128f src, mmask8 k, m128i a) noexcept;
extern m128f maskz_cvtepi64_ps(mmask8 k, m128i a) noexcept;
extern void mask_cvtepi64_storeu_epi16(void* base_addr, mmask8 k, m128i a) noexcept;
extern void mask_cvtepi64_storeu_epi32(void* base_addr, mmask8 k, m128i a) noexcept;
extern void mask_cvtepi64_storeu_epi8(void* base_addr, mmask8 k, m128i a) noexcept;
extern m128i cvtepi8_epi16(m128i a) noexcept;
extern m128i mask_cvtepi8_epi16(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtepi8_epi16(mmask8 k, m128i a) noexcept;
extern m128i cvtepi8_epi32(m128i a) noexcept;
extern m128i mask_cvtepi8_epi32(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtepi8_epi32(mmask8 k, m128i a) noexcept;
extern m128i cvtepi8_epi64(m128i a) noexcept;
extern m128i mask_cvtepi8_epi64(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtepi8_epi64(mmask8 k, m128i a) noexcept;
extern m128i cvtepu16_epi32(m128i a) noexcept;
extern m128i mask_cvtepu16_epi32(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtepu16_epi32(mmask8 k, m128i a) noexcept;
extern m128i cvtepu16_epi64(m128i a) noexcept;
extern m128i mask_cvtepu16_epi64(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtepu16_epi64(mmask8 k, m128i a) noexcept;
extern m128i cvtepu32_epi64(m128i a) noexcept;
extern m128i mask_cvtepu32_epi64(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtepu32_epi64(mmask8 k, m128i a) noexcept;
extern m128d cvtepu32_pd(m128i a) noexcept;
extern m128d mask_cvtepu32_pd(m128d src, mmask8 k, m128i a) noexcept;
extern m128d maskz_cvtepu32_pd(mmask8 k, m128i a) noexcept;
extern m128d cvtepu64_pd(m128i a) noexcept;
extern m128d mask_cvtepu64_pd(m128d src, mmask8 k, m128i a) noexcept;
extern m128d maskz_cvtepu64_pd(mmask8 k, m128i a) noexcept;
extern m128f cvtepu64_ps(m128i a) noexcept;
extern m128f mask_cvtepu64_ps(m128f src, mmask8 k, m128i a) noexcept;
extern m128f maskz_cvtepu64_ps(mmask8 k, m128i a) noexcept;
extern m128i cvtepu8_epi16(m128i a) noexcept;
extern m128i mask_cvtepu8_epi16(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtepu8_epi16(mmask8 k, m128i a) noexcept;
extern m128i cvtepu8_epi32(m128i a) noexcept;
extern m128i mask_cvtepu8_epi32(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtepu8_epi32(mmask8 k, m128i a) noexcept;
extern m128i cvtepu8_epi64(m128i a) noexcept;
extern m128i mask_cvtepu8_epi64(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtepu8_epi64(mmask8 k, m128i a) noexcept;
extern m128d cvti32_sd(m128d a, int b) noexcept;
extern m128f cvti32_ss(m128f a, int b) noexcept;
extern m128d cvti64_sd(m128d a, int8 b) noexcept;
extern m128f cvti64_ss(m128f a, int8 b) noexcept;
extern m128i cvtpd_epi32(m128d a) noexcept;
extern m128i mask_cvtpd_epi32(m128i src, mmask8 k, m128d a) noexcept;
extern m128i maskz_cvtpd_epi32(mmask8 k, m128d a) noexcept;
extern m128i cvtpd_epi64(m128d a) noexcept;
extern m128i mask_cvtpd_epi64(m128i src, mmask8 k, m128d a) noexcept;
extern m128i maskz_cvtpd_epi64(mmask8 k, m128d a) noexcept;
extern m128i cvtpd_epu32(m128d a) noexcept;
extern m128i mask_cvtpd_epu32(m128i src, mmask8 k, m128d a) noexcept;
extern m128i maskz_cvtpd_epu32(mmask8 k, m128d a) noexcept;
extern m128i cvtpd_epu64(m128d a) noexcept;
extern m128i mask_cvtpd_epu64(m128i src, mmask8 k, m128d a) noexcept;
extern m128i maskz_cvtpd_epu64(mmask8 k, m128d a) noexcept;
extern m128f cvtpd_ps(m128d a) noexcept;
extern m128f mask_cvtpd_ps(m128f src, mmask8 k, m128d a) noexcept;
extern m128f maskz_cvtpd_ps(mmask8 k, m128d a) noexcept;
extern m128f cvtph_ps(m128i a) noexcept;
extern m128f mask_cvtph_ps(m128f src, mmask8 k, m128i a) noexcept;
extern m128f maskz_cvtph_ps(mmask8 k, m128i a) noexcept;
extern m128i cvtps_epi32(m128f a) noexcept;
extern m128i mask_cvtps_epi32(m128i src, mmask8 k, m128f a) noexcept;
extern m128i maskz_cvtps_epi32(mmask8 k, m128f a) noexcept;
extern m128i cvtps_epi64(m128f a) noexcept;
extern m128i mask_cvtps_epi64(m128i src, mmask8 k, m128f a) noexcept;
extern m128i maskz_cvtps_epi64(mmask8 k, m128f a) noexcept;
extern m128i cvtps_epu32(m128f a) noexcept;
extern m128i mask_cvtps_epu32(m128i src, mmask8 k, m128f a) noexcept;
extern m128i maskz_cvtps_epu32(mmask8 k, m128f a) noexcept;
extern m128i cvtps_epu64(m128f a) noexcept;
extern m128i mask_cvtps_epu64(m128i src, mmask8 k, m128f a) noexcept;
extern m128i maskz_cvtps_epu64(mmask8 k, m128f a) noexcept;
extern m128d cvtps_pd(m128f a) noexcept;
extern double cvtsd_f64(m128d a) noexcept;
extern int cvtsd_i32(m128d a) noexcept;
extern int8 cvtsd_i64(m128d a) noexcept;
extern int cvtsd_si32(m128d a) noexcept;
extern int8 cvtsd_si64(m128d a) noexcept;
extern int8 cvtsd_si64x(m128d a) noexcept;
extern m128f cvtsd_ss(m128f a, m128d b) noexcept;
extern m128f mask_cvtsd_ss(m128f src, mmask8 k, m128f a, m128d b) noexcept;
extern m128f maskz_cvtsd_ss(mmask8 k, m128f a, m128d b) noexcept;
extern nat4 cvtsd_u32(m128d a) noexcept;
extern nat8 cvtsd_u64(m128d a) noexcept;
extern m128i cvtsepi16_epi8(m128i a) noexcept;
extern m128i mask_cvtsepi16_epi8(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtsepi16_epi8(mmask8 k, m128i a) noexcept;
extern void mask_cvtsepi16_storeu_epi8(void* base_addr, mmask8 k, m128i a) noexcept;
extern m128i cvtsepi32_epi16(m128i a) noexcept;
extern m128i mask_cvtsepi32_epi16(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtsepi32_epi16(mmask8 k, m128i a) noexcept;
extern m128i cvtsepi32_epi8(m128i a) noexcept;
extern m128i mask_cvtsepi32_epi8(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtsepi32_epi8(mmask8 k, m128i a) noexcept;
extern void mask_cvtsepi32_storeu_epi16(void* base_addr, mmask8 k, m128i a) noexcept;
extern void mask_cvtsepi32_storeu_epi8(void* base_addr, mmask8 k, m128i a) noexcept;
extern m128i cvtsepi64_epi16(m128i a) noexcept;
extern m128i mask_cvtsepi64_epi16(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtsepi64_epi16(mmask8 k, m128i a) noexcept;
extern m128i cvtsepi64_epi32(m128i a) noexcept;
extern m128i mask_cvtsepi64_epi32(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtsepi64_epi32(mmask8 k, m128i a) noexcept;
extern m128i cvtsepi64_epi8(m128i a) noexcept;
extern m128i mask_cvtsepi64_epi8(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtsepi64_epi8(mmask8 k, m128i a) noexcept;
extern void mask_cvtsepi64_storeu_epi16(void* base_addr, mmask8 k, m128i a) noexcept;
extern void mask_cvtsepi64_storeu_epi32(void* base_addr, mmask8 k, m128i a) noexcept;
extern void mask_cvtsepi64_storeu_epi8(void* base_addr, mmask8 k, m128i a) noexcept;
extern int cvtsi128_si32(m128i a) noexcept;
extern int8 cvtsi128_si64(m128i a) noexcept;
extern int8 cvtsi128_si64x(m128i a) noexcept;
extern m128d cvtsi32_sd(m128d a, int b) noexcept;
extern m128i cvtsi32_si128(int a) noexcept;
extern m128f cvtsi32_ss(m128f a, int b) noexcept;
extern m128d cvtsi64_sd(m128d a, int8 b) noexcept;
extern m128i cvtsi64_si128(int8 a) noexcept;
extern m128f cvtsi64_ss(m128f a, int8 b) noexcept;
extern m128d cvtsi64x_sd(m128d a, int8 b) noexcept;
extern m128i cvtsi64x_si128(int8 a) noexcept;
extern float cvtss_f32(m128f a) noexcept;
extern int cvtss_i32(m128f a) noexcept;
extern int8 cvtss_i64(m128f a) noexcept;
extern m128d cvtss_sd(m128d a, m128f b) noexcept;
extern m128d mask_cvtss_sd(m128d src, mmask8 k, m128d a, m128f b) noexcept;
extern m128d maskz_cvtss_sd(mmask8 k, m128d a, m128f b) noexcept;
extern int cvtss_si32(m128f a) noexcept;
extern int8 cvtss_si64(m128f a) noexcept;
extern nat4 cvtss_u32(m128f a) noexcept;
extern nat8 cvtss_u64(m128f a) noexcept;
template<int sae> extern int cvtt_roundsd_i32(m128d a) noexcept;
template<int sae> extern int8 cvtt_roundsd_i64(m128d a) noexcept;
template<int sae> extern int cvtt_roundsd_si32(m128d a) noexcept;
template<int sae> extern int8 cvtt_roundsd_si64(m128d a) noexcept;
template<int sae> extern nat4 cvtt_roundsd_u32(m128d a) noexcept;
template<int sae> extern nat8 cvtt_roundsd_u64(m128d a) noexcept;
template<int sae> extern int cvtt_roundss_i32(m128f a) noexcept;
template<int sae> extern int8 cvtt_roundss_i64(m128f a) noexcept;
template<int sae> extern int cvtt_roundss_si32(m128f a) noexcept;
template<int sae> extern int8 cvtt_roundss_si64(m128f a) noexcept;
template<int sae> extern nat4 cvtt_roundss_u32(m128f a) noexcept;
template<int sae> extern nat8 cvtt_roundss_u64(m128f a) noexcept;
extern int cvtt_ss2si(m128f a) noexcept;
extern m128i cvttpd_epi32(m128d a) noexcept;
extern m128i mask_cvttpd_epi32(m128i src, mmask8 k, m128d a) noexcept;
extern m128i maskz_cvttpd_epi32(mmask8 k, m128d a) noexcept;
extern m128i cvttpd_epi64(m128d a) noexcept;
extern m128i mask_cvttpd_epi64(m128i src, mmask8 k, m128d a) noexcept;
extern m128i maskz_cvttpd_epi64(mmask8 k, m128d a) noexcept;
extern m128i cvttpd_epu32(m128d a) noexcept;
extern m128i mask_cvttpd_epu32(m128i src, mmask8 k, m128d a) noexcept;
extern m128i maskz_cvttpd_epu32(mmask8 k, m128d a) noexcept;
extern m128i cvttpd_epu64(m128d a) noexcept;
extern m128i mask_cvttpd_epu64(m128i src, mmask8 k, m128d a) noexcept;
extern m128i maskz_cvttpd_epu64(mmask8 k, m128d a) noexcept;
extern m128i cvttps_epi32(m128f a) noexcept;
extern m128i mask_cvttps_epi32(m128i src, mmask8 k, m128f a) noexcept;
extern m128i maskz_cvttps_epi32(mmask8 k, m128f a) noexcept;
extern m128i cvttps_epi64(m128f a) noexcept;
extern m128i mask_cvttps_epi64(m128i src, mmask8 k, m128f a) noexcept;
extern m128i maskz_cvttps_epi64(mmask8 k, m128f a) noexcept;
extern m128i cvttps_epu32(m128f a) noexcept;
extern m128i mask_cvttps_epu32(m128i src, mmask8 k, m128f a) noexcept;
extern m128i maskz_cvttps_epu32(mmask8 k, m128f a) noexcept;
extern m128i cvttps_epu64(m128f a) noexcept;
extern m128i mask_cvttps_epu64(m128i src, mmask8 k, m128f a) noexcept;
extern m128i maskz_cvttps_epu64(mmask8 k, m128f a) noexcept;
extern int cvttsd_i32(m128d a) noexcept;
extern int8 cvttsd_i64(m128d a) noexcept;
extern int cvttsd_si32(m128d a) noexcept;
extern int8 cvttsd_si64(m128d a) noexcept;
extern int8 cvttsd_si64x(m128d a) noexcept;
extern nat4 cvttsd_u32(m128d a) noexcept;
extern nat8 cvttsd_u64(m128d a) noexcept;
extern int cvttss_i32(m128f a) noexcept;
extern int8 cvttss_i64(m128f a) noexcept;
extern int cvttss_si32(m128f a) noexcept;
extern int8 cvttss_si64(m128f a) noexcept;
extern nat4 cvttss_u32(m128f a) noexcept;
extern nat8 cvttss_u64(m128f a) noexcept;
extern m128d cvtu32_sd(m128d a, nat4 b) noexcept;
extern m128f cvtu32_ss(m128f a, nat4 b) noexcept;
extern m128d cvtu64_sd(m128d a, nat8 b) noexcept;
extern m128f cvtu64_ss(m128f a, nat8 b) noexcept;
extern m128i cvtusepi16_epi8(m128i a) noexcept;
extern m128i mask_cvtusepi16_epi8(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtusepi16_epi8(mmask8 k, m128i a) noexcept;
extern void mask_cvtusepi16_storeu_epi8(void* base_addr, mmask8 k, m128i a) noexcept;
extern m128i cvtusepi32_epi16(m128i a) noexcept;
extern m128i mask_cvtusepi32_epi16(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtusepi32_epi16(mmask8 k, m128i a) noexcept;
extern m128i cvtusepi32_epi8(m128i a) noexcept;
extern m128i mask_cvtusepi32_epi8(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtusepi32_epi8(mmask8 k, m128i a) noexcept;
extern void mask_cvtusepi32_storeu_epi16(void* base_addr, mmask8 k, m128i a) noexcept;
extern void mask_cvtusepi32_storeu_epi8(void* base_addr, mmask8 k, m128i a) noexcept;
extern m128i cvtusepi64_epi16(m128i a) noexcept;
extern m128i mask_cvtusepi64_epi16(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtusepi64_epi16(mmask8 k, m128i a) noexcept;
extern m128i cvtusepi64_epi32(m128i a) noexcept;
extern m128i mask_cvtusepi64_epi32(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtusepi64_epi32(mmask8 k, m128i a) noexcept;
extern m128i cvtusepi64_epi8(m128i a) noexcept;
extern m128i mask_cvtusepi64_epi8(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_cvtusepi64_epi8(mmask8 k, m128i a) noexcept;
extern void mask_cvtusepi64_storeu_epi16(void* base_addr, mmask8 k, m128i a) noexcept;
extern void mask_cvtusepi64_storeu_epi32(void* base_addr, mmask8 k, m128i a) noexcept;
extern void mask_cvtusepi64_storeu_epi8(void* base_addr, mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i dbsad_epu8(m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i mask_dbsad_epu8(m128i src, mmask8 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i maskz_dbsad_epu8(mmask8 k, m128i a, m128i b) noexcept;
extern m128i div_epi16(m128i a, m128i b) noexcept;
extern m128i div_epi32(m128i a, m128i b) noexcept;
extern m128i div_epi64(m128i a, m128i b) noexcept;
extern m128i div_epi8(m128i a, m128i b) noexcept;
extern m128i div_epu16(m128i a, m128i b) noexcept;
extern m128i div_epu32(m128i a, m128i b) noexcept;
extern m128i div_epu64(m128i a, m128i b) noexcept;
extern m128i div_epu8(m128i a, m128i b) noexcept;
extern m128d div_pd(m128d a, m128d b) noexcept;
extern m128d mask_div_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_div_pd(mmask8 k, m128d a, m128d b) noexcept;
extern m128f div_ps(m128f a, m128f b) noexcept;
extern m128f mask_div_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_div_ps(mmask8 k, m128f a, m128f b) noexcept;
template<int4 rounding> extern m128d div_round_sd(m128d a, m128d b) noexcept;
template<int4 rounding> extern m128d mask_div_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int4 rounding> extern m128d maskz_div_round_sd(mmask8 k, m128d a, m128d b) noexcept;
template<int4 rounding> extern m128f div_round_ss(m128f a, m128f b) noexcept;
template<int4 rounding> extern m128f mask_div_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int4 rounding> extern m128f maskz_div_round_ss(mmask8 k, m128f a, m128f b) noexcept;
extern m128d div_sd(m128d a, m128d b) noexcept;
extern m128d mask_div_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_div_sd(mmask8 k, m128d a, m128d b) noexcept;
extern m128f div_ss(m128f a, m128f b) noexcept;
extern m128f mask_div_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_div_ss(mmask8 k, m128f a, m128f b) noexcept;
template<int4 imm8> extern m128d dp_pd(m128d a, m128d b) noexcept;
template<int4 imm8> extern m128f dp_ps(m128f a, m128f b) noexcept;
extern m128i dpbusd_avx_epi32(m128i src, m128i a, m128i b) noexcept;
extern m128i dpbusd_epi32(m128i src, m128i a, m128i b) noexcept;
extern m128i dpbusd_epi32(m128i src, m128i a, m128i b) noexcept;
extern m128i mask_dpbusd_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_dpbusd_epi32(mmask8 k, m128i src, m128i a, m128i b) noexcept;
extern m128i dpbusds_avx_epi32(m128i src, m128i a, m128i b) noexcept;
extern m128i dpbusds_epi32(m128i src, m128i a, m128i b) noexcept;
extern m128i dpbusds_epi32(m128i src, m128i a, m128i b) noexcept;
extern m128i mask_dpbusds_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_dpbusds_epi32(mmask8 k, m128i src, m128i a, m128i b) noexcept;
extern m128i dpwssd_avx_epi32(m128i src, m128i a, m128i b) noexcept;
extern m128i dpwssd_epi32(m128i src, m128i a, m128i b) noexcept;
extern m128i dpwssd_epi32(m128i src, m128i a, m128i b) noexcept;
extern m128i mask_dpwssd_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_dpwssd_epi32(mmask8 k, m128i src, m128i a, m128i b) noexcept;
extern m128i dpwssds_avx_epi32(m128i src, m128i a, m128i b) noexcept;
extern m128i dpwssds_epi32(m128i src, m128i a, m128i b) noexcept;
extern m128i dpwssds_epi32(m128i src, m128i a, m128i b) noexcept;
extern m128i mask_dpwssds_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_dpwssds_epi32(mmask8 k, m128i src, m128i a, m128i b) noexcept;
extern nat4 encodekey128_u32(nat4 __htype, m128i __key, void* __h) noexcept;
extern nat4 encodekey256_u32(nat4 __htype, m128i __key_lo, m128i __key_hi, void* __h) noexcept;
extern m128d erf_pd(m128d a) noexcept;
extern m128f erf_ps(m128f a) noexcept;
extern m128d erfc_pd(m128d a) noexcept;
extern m128f erfc_ps(m128f a) noexcept;
extern m128d erfcinv_pd(m128d a) noexcept;
extern m128f erfcinv_ps(m128f a) noexcept;
extern m128d erfinv_pd(m128d a) noexcept;
extern m128f erfinv_ps(m128f a) noexcept;
extern m128i mask_expand_epi16(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_expand_epi16(mmask8 k, m128i a) noexcept;
extern m128i mask_expand_epi32(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_expand_epi32(mmask8 k, m128i a) noexcept;
extern m128i mask_expand_epi64(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_expand_epi64(mmask8 k, m128i a) noexcept;
extern m128i mask_expand_epi8(m128i src, mmask16 k, m128i a) noexcept;
extern m128i maskz_expand_epi8(mmask16 k, m128i a) noexcept;
extern m128d mask_expand_pd(m128d src, mmask8 k, m128d a) noexcept;
extern m128d maskz_expand_pd(mmask8 k, m128d a) noexcept;
extern m128f mask_expand_ps(m128f src, mmask8 k, m128f a) noexcept;
extern m128f maskz_expand_ps(mmask8 k, m128f a) noexcept;
extern m128i mask_expandloadu_epi16(m128i src, mmask8 k, const void* mem_addr) noexcept;
extern m128i maskz_expandloadu_epi16(mmask8 k, const void* mem_addr) noexcept;
extern m128i mask_expandloadu_epi32(m128i src, mmask8 k, void const* mem_addr) noexcept;
extern m128i maskz_expandloadu_epi32(mmask8 k, void const* mem_addr) noexcept;
extern m128i mask_expandloadu_epi64(m128i src, mmask8 k, void const* mem_addr) noexcept;
extern m128i maskz_expandloadu_epi64(mmask8 k, void const* mem_addr) noexcept;
extern m128i mask_expandloadu_epi8(m128i src, mmask16 k, const void* mem_addr) noexcept;
extern m128i maskz_expandloadu_epi8(mmask16 k, const void* mem_addr) noexcept;
extern m128d mask_expandloadu_pd(m128d src, mmask8 k, void const* mem_addr) noexcept;
extern m128d maskz_expandloadu_pd(mmask8 k, void const* mem_addr) noexcept;
extern m128f mask_expandloadu_ps(m128f src, mmask8 k, void const* mem_addr) noexcept;
extern m128f maskz_expandloadu_ps(mmask8 k, void const* mem_addr) noexcept;
template<int4 imm8> extern int extract_epi16(m128i a) noexcept;
template<int4 imm8> extern int extract_epi32(m128i a) noexcept;
template<int4 imm8> extern int8 extract_epi64(m128i a) noexcept;
template<int4 imm8> extern int extract_epi8(m128i a) noexcept;
template<int4 imm8> extern int extract_ps(m128f a) noexcept;
template<int4 imm8> extern m128d fixupimm_pd(m128d a, m128d b, m128i c) noexcept;
template<int4 imm8> extern m128d mask_fixupimm_pd(m128d a, mmask8 k, m128d b, m128i c) noexcept;
template<int4 imm8> extern m128d maskz_fixupimm_pd(mmask8 k, m128d a, m128d b, m128i c) noexcept;
template<int4 imm8> extern m128f fixupimm_ps(m128f a, m128f b, m128i c) noexcept;
template<int4 imm8> extern m128f mask_fixupimm_ps(m128f a, mmask8 k, m128f b, m128i c) noexcept;
template<int4 imm8> extern m128f maskz_fixupimm_ps(mmask8 k, m128f a, m128f b, m128i c) noexcept;
template<int4 imm8, int sae> extern m128d fixupimm_round_sd(m128d a, m128d b, m128i c) noexcept;
template<int4 imm8, int sae> extern m128d mask_fixupimm_round_sd(m128d a, mmask8 k, m128d b, m128i c) noexcept;
template<int4 imm8, int sae> extern m128d maskz_fixupimm_round_sd(mmask8 k, m128d a, m128d b, m128i c) noexcept;
template<int4 imm8, int sae> extern m128f fixupimm_round_ss(m128f a, m128f b, m128i c) noexcept;
template<int4 imm8, int sae> extern m128f mask_fixupimm_round_ss(m128f a, mmask8 k, m128f b, m128i c) noexcept;
template<int4 imm8, int sae> extern m128f maskz_fixupimm_round_ss(mmask8 k, m128f a, m128f b, m128i c) noexcept;
template<int4 imm8> extern m128d fixupimm_sd(m128d a, m128d b, m128i c) noexcept;
template<int4 imm8> extern m128d mask_fixupimm_sd(m128d a, mmask8 k, m128d b, m128i c) noexcept;
template<int4 imm8> extern m128d maskz_fixupimm_sd(mmask8 k, m128d a, m128d b, m128i c) noexcept;
template<int4 imm8> extern m128f fixupimm_ss(m128f a, m128f b, m128i c) noexcept;
template<int4 imm8> extern m128f mask_fixupimm_ss(m128f a, mmask8 k, m128f b, m128i c) noexcept;
template<int4 imm8> extern m128f maskz_fixupimm_ss(mmask8 k, m128f a, m128f b, m128i c) noexcept;
extern m128d floor_pd(m128d a) noexcept;
extern m128f floor_ps(m128f a) noexcept;
extern m128d floor_sd(m128d a, m128d b) noexcept;
extern m128f floor_ss(m128f a, m128f b) noexcept;
extern m128d fmadd_pd(m128d a, m128d b, m128d c) noexcept;
extern m128d mask_fmadd_pd(m128d a, mmask8 k, m128d b, m128d c) noexcept;
extern m128d mask3_fmadd_pd(m128d a, m128d b, m128d c, mmask8 k) noexcept;
extern m128d maskz_fmadd_pd(mmask8 k, m128d a, m128d b, m128d c) noexcept;
extern m128f fmadd_ps(m128f a, m128f b, m128f c) noexcept;
extern m128f mask_fmadd_ps(m128f a, mmask8 k, m128f b, m128f c) noexcept;
extern m128f mask3_fmadd_ps(m128f a, m128f b, m128f c, mmask8 k) noexcept;
extern m128f maskz_fmadd_ps(mmask8 k, m128f a, m128f b, m128f c) noexcept;
template<int4 rounding> extern m128d fmadd_round_sd(m128d a, m128d b, m128d c) noexcept;
template<int4 rounding> extern m128d mask_fmadd_round_sd(m128d a, mmask8 k, m128d b, m128d c) noexcept;
template<int4 rounding> extern m128d mask3_fmadd_round_sd(m128d a, m128d b, m128d c, mmask8 k) noexcept;
template<int4 rounding> extern m128d maskz_fmadd_round_sd(mmask8 k, m128d a, m128d b, m128d c) noexcept;
template<int4 rounding> extern m128f fmadd_round_ss(m128f a, m128f b, m128f c) noexcept;
template<int4 rounding> extern m128f mask_fmadd_round_ss(m128f a, mmask8 k, m128f b, m128f c) noexcept;
template<int4 rounding> extern m128f mask3_fmadd_round_ss(m128f a, m128f b, m128f c, mmask8 k) noexcept;
template<int4 rounding> extern m128f maskz_fmadd_round_ss(mmask8 k, m128f a, m128f b, m128f c) noexcept;
extern m128d fmadd_sd(m128d a, m128d b, m128d c) noexcept;
extern m128d mask_fmadd_sd(m128d a, mmask8 k, m128d b, m128d c) noexcept;
extern m128d mask3_fmadd_sd(m128d a, m128d b, m128d c, mmask8 k) noexcept;
extern m128d maskz_fmadd_sd(mmask8 k, m128d a, m128d b, m128d c) noexcept;
extern m128f fmadd_ss(m128f a, m128f b, m128f c) noexcept;
extern m128f mask_fmadd_ss(m128f a, mmask8 k, m128f b, m128f c) noexcept;
extern m128f mask3_fmadd_ss(m128f a, m128f b, m128f c, mmask8 k) noexcept;
extern m128f maskz_fmadd_ss(mmask8 k, m128f a, m128f b, m128f c) noexcept;
extern m128d fmaddsub_pd(m128d a, m128d b, m128d c) noexcept;
extern m128d mask_fmaddsub_pd(m128d a, mmask8 k, m128d b, m128d c) noexcept;
extern m128d mask3_fmaddsub_pd(m128d a, m128d b, m128d c, mmask8 k) noexcept;
extern m128d maskz_fmaddsub_pd(mmask8 k, m128d a, m128d b, m128d c) noexcept;
extern m128f fmaddsub_ps(m128f a, m128f b, m128f c) noexcept;
extern m128f mask_fmaddsub_ps(m128f a, mmask8 k, m128f b, m128f c) noexcept;
extern m128f mask3_fmaddsub_ps(m128f a, m128f b, m128f c, mmask8 k) noexcept;
extern m128f maskz_fmaddsub_ps(mmask8 k, m128f a, m128f b, m128f c) noexcept;
extern m128d fmsub_pd(m128d a, m128d b, m128d c) noexcept;
extern m128d mask_fmsub_pd(m128d a, mmask8 k, m128d b, m128d c) noexcept;
extern m128d mask3_fmsub_pd(m128d a, m128d b, m128d c, mmask8 k) noexcept;
extern m128d maskz_fmsub_pd(mmask8 k, m128d a, m128d b, m128d c) noexcept;
extern m128f fmsub_ps(m128f a, m128f b, m128f c) noexcept;
extern m128f mask_fmsub_ps(m128f a, mmask8 k, m128f b, m128f c) noexcept;
extern m128f mask3_fmsub_ps(m128f a, m128f b, m128f c, mmask8 k) noexcept;
extern m128f maskz_fmsub_ps(mmask8 k, m128f a, m128f b, m128f c) noexcept;
template<int4 rounding> extern m128d fmsub_round_sd(m128d a, m128d b, m128d c) noexcept;
template<int4 rounding> extern m128d mask_fmsub_round_sd(m128d a, mmask8 k, m128d b, m128d c) noexcept;
template<int4 rounding> extern m128d mask3_fmsub_round_sd(m128d a, m128d b, m128d c, mmask8 k) noexcept;
template<int4 rounding> extern m128d maskz_fmsub_round_sd(mmask8 k, m128d a, m128d b, m128d c) noexcept;
template<int4 rounding> extern m128f fmsub_round_ss(m128f a, m128f b, m128f c) noexcept;
template<int4 rounding> extern m128f mask_fmsub_round_ss(m128f a, mmask8 k, m128f b, m128f c) noexcept;
template<int4 rounding> extern m128f mask3_fmsub_round_ss(m128f a, m128f b, m128f c, mmask8 k) noexcept;
template<int4 rounding> extern m128f maskz_fmsub_round_ss(mmask8 k, m128f a, m128f b, m128f c) noexcept;
extern m128d fmsub_sd(m128d a, m128d b, m128d c) noexcept;
extern m128d mask_fmsub_sd(m128d a, mmask8 k, m128d b, m128d c) noexcept;
extern m128d mask3_fmsub_sd(m128d a, m128d b, m128d c, mmask8 k) noexcept;
extern m128d maskz_fmsub_sd(mmask8 k, m128d a, m128d b, m128d c) noexcept;
extern m128f fmsub_ss(m128f a, m128f b, m128f c) noexcept;
extern m128f mask_fmsub_ss(m128f a, mmask8 k, m128f b, m128f c) noexcept;
extern m128f mask3_fmsub_ss(m128f a, m128f b, m128f c, mmask8 k) noexcept;
extern m128f maskz_fmsub_ss(mmask8 k, m128f a, m128f b, m128f c) noexcept;
extern m128d fmsubadd_pd(m128d a, m128d b, m128d c) noexcept;
extern m128d mask_fmsubadd_pd(m128d a, mmask8 k, m128d b, m128d c) noexcept;
extern m128d mask3_fmsubadd_pd(m128d a, m128d b, m128d c, mmask8 k) noexcept;
extern m128d maskz_fmsubadd_pd(mmask8 k, m128d a, m128d b, m128d c) noexcept;
extern m128f fmsubadd_ps(m128f a, m128f b, m128f c) noexcept;
extern m128f mask_fmsubadd_ps(m128f a, mmask8 k, m128f b, m128f c) noexcept;
extern m128f mask3_fmsubadd_ps(m128f a, m128f b, m128f c, mmask8 k) noexcept;
extern m128f maskz_fmsubadd_ps(mmask8 k, m128f a, m128f b, m128f c) noexcept;
extern m128d fnmadd_pd(m128d a, m128d b, m128d c) noexcept;
extern m128d mask_fnmadd_pd(m128d a, mmask8 k, m128d b, m128d c) noexcept;
extern m128d mask3_fnmadd_pd(m128d a, m128d b, m128d c, mmask8 k) noexcept;
extern m128d maskz_fnmadd_pd(mmask8 k, m128d a, m128d b, m128d c) noexcept;
extern m128f fnmadd_ps(m128f a, m128f b, m128f c) noexcept;
extern m128f mask_fnmadd_ps(m128f a, mmask8 k, m128f b, m128f c) noexcept;
extern m128f mask3_fnmadd_ps(m128f a, m128f b, m128f c, mmask8 k) noexcept;
extern m128f maskz_fnmadd_ps(mmask8 k, m128f a, m128f b, m128f c) noexcept;
template<int4 rounding> extern m128d fnmadd_round_sd(m128d a, m128d b, m128d c) noexcept;
template<int4 rounding> extern m128d mask_fnmadd_round_sd(m128d a, mmask8 k, m128d b, m128d c) noexcept;
template<int4 rounding> extern m128d mask3_fnmadd_round_sd(m128d a, m128d b, m128d c, mmask8 k) noexcept;
template<int4 rounding> extern m128d maskz_fnmadd_round_sd(mmask8 k, m128d a, m128d b, m128d c) noexcept;
template<int4 rounding> extern m128f fnmadd_round_ss(m128f a, m128f b, m128f c) noexcept;
template<int4 rounding> extern m128f mask_fnmadd_round_ss(m128f a, mmask8 k, m128f b, m128f c) noexcept;
template<int4 rounding> extern m128f mask3_fnmadd_round_ss(m128f a, m128f b, m128f c, mmask8 k) noexcept;
template<int4 rounding> extern m128f maskz_fnmadd_round_ss(mmask8 k, m128f a, m128f b, m128f c) noexcept;
extern m128d fnmadd_sd(m128d a, m128d b, m128d c) noexcept;
extern m128d mask_fnmadd_sd(m128d a, mmask8 k, m128d b, m128d c) noexcept;
extern m128d mask3_fnmadd_sd(m128d a, m128d b, m128d c, mmask8 k) noexcept;
extern m128d maskz_fnmadd_sd(mmask8 k, m128d a, m128d b, m128d c) noexcept;
extern m128f fnmadd_ss(m128f a, m128f b, m128f c) noexcept;
extern m128f mask_fnmadd_ss(m128f a, mmask8 k, m128f b, m128f c) noexcept;
extern m128f mask3_fnmadd_ss(m128f a, m128f b, m128f c, mmask8 k) noexcept;
extern m128f maskz_fnmadd_ss(mmask8 k, m128f a, m128f b, m128f c) noexcept;
extern m128d fnmsub_pd(m128d a, m128d b, m128d c) noexcept;
extern m128d mask_fnmsub_pd(m128d a, mmask8 k, m128d b, m128d c) noexcept;
extern m128d mask3_fnmsub_pd(m128d a, m128d b, m128d c, mmask8 k) noexcept;
extern m128d maskz_fnmsub_pd(mmask8 k, m128d a, m128d b, m128d c) noexcept;
extern m128f fnmsub_ps(m128f a, m128f b, m128f c) noexcept;
extern m128f mask_fnmsub_ps(m128f a, mmask8 k, m128f b, m128f c) noexcept;
extern m128f mask3_fnmsub_ps(m128f a, m128f b, m128f c, mmask8 k) noexcept;
extern m128f maskz_fnmsub_ps(mmask8 k, m128f a, m128f b, m128f c) noexcept;
template<int4 rounding> extern m128d fnmsub_round_sd(m128d a, m128d b, m128d c) noexcept;
template<int4 rounding> extern m128d mask_fnmsub_round_sd(m128d a, mmask8 k, m128d b, m128d c) noexcept;
template<int4 rounding> extern m128d mask3_fnmsub_round_sd(m128d a, m128d b, m128d c, mmask8 k) noexcept;
template<int4 rounding> extern m128d maskz_fnmsub_round_sd(mmask8 k, m128d a, m128d b, m128d c) noexcept;
template<int4 rounding> extern m128f fnmsub_round_ss(m128f a, m128f b, m128f c) noexcept;
template<int4 rounding> extern m128f mask_fnmsub_round_ss(m128f a, mmask8 k, m128f b, m128f c) noexcept;
template<int4 rounding> extern m128f mask3_fnmsub_round_ss(m128f a, m128f b, m128f c, mmask8 k) noexcept;
template<int4 rounding> extern m128f maskz_fnmsub_round_ss(mmask8 k, m128f a, m128f b, m128f c) noexcept;
extern m128d fnmsub_sd(m128d a, m128d b, m128d c) noexcept;
extern m128d mask_fnmsub_sd(m128d a, mmask8 k, m128d b, m128d c) noexcept;
extern m128d mask3_fnmsub_sd(m128d a, m128d b, m128d c, mmask8 k) noexcept;
extern m128d maskz_fnmsub_sd(mmask8 k, m128d a, m128d b, m128d c) noexcept;
extern m128f fnmsub_ss(m128f a, m128f b, m128f c) noexcept;
extern m128f mask_fnmsub_ss(m128f a, mmask8 k, m128f b, m128f c) noexcept;
extern m128f mask3_fnmsub_ss(m128f a, m128f b, m128f c, mmask8 k) noexcept;
extern m128f maskz_fnmsub_ss(mmask8 k, m128f a, m128f b, m128f c) noexcept;
template<int4 imm8> extern mmask8 fpclass_pd_mask(m128d a) noexcept;
template<int4 imm8> extern mmask8 mask_fpclass_pd_mask(mmask8 k1, m128d a) noexcept;
template<int4 imm8> extern mmask8 fpclass_ps_mask(m128f a) noexcept;
template<int4 imm8> extern mmask8 mask_fpclass_ps_mask(mmask8 k1, m128f a) noexcept;
template<int4 imm8> extern mmask8 fpclass_sd_mask(m128d a) noexcept;
template<int4 imm8> extern mmask8 mask_fpclass_sd_mask(mmask8 k1, m128d a) noexcept;
template<int4 imm8> extern mmask8 fpclass_ss_mask(m128f a) noexcept;
template<int4 imm8> extern mmask8 mask_fpclass_ss_mask(mmask8 k1, m128f a) noexcept;
extern m128d getexp_pd(m128d a) noexcept;
extern m128d mask_getexp_pd(m128d src, mmask8 k, m128d a) noexcept;
extern m128d maskz_getexp_pd(mmask8 k, m128d a) noexcept;
extern m128f getexp_ps(m128f a) noexcept;
extern m128f mask_getexp_ps(m128f src, mmask8 k, m128f a) noexcept;
extern m128f maskz_getexp_ps(mmask8 k, m128f a) noexcept;
template<int sae> extern m128d getexp_round_sd(m128d a, m128d b) noexcept;
template<int sae> extern m128d mask_getexp_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int sae> extern m128d maskz_getexp_round_sd(mmask8 k, m128d a, m128d b) noexcept;
template<int sae> extern m128f getexp_round_ss(m128f a, m128f b) noexcept;
template<int sae> extern m128f mask_getexp_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int sae> extern m128f maskz_getexp_round_ss(mmask8 k, m128f a, m128f b) noexcept;
extern m128d getexp_sd(m128d a, m128d b) noexcept;
extern m128d mask_getexp_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_getexp_sd(mmask8 k, m128d a, m128d b) noexcept;
extern m128f getexp_ss(m128f a, m128f b) noexcept;
extern m128f mask_getexp_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_getexp_ss(mmask8 k, m128f a, m128f b) noexcept;
template<int sc> extern m128d getmant_pd(m128d a, MANTISSA_NORM_ENUM interv) noexcept;
template<int sc> extern m128d mask_getmant_pd(m128d src, mmask8 k, m128d a, MANTISSA_NORM_ENUM interv) noexcept;
template<int sc> extern m128d maskz_getmant_pd(mmask8 k, m128d a, MANTISSA_NORM_ENUM interv) noexcept;
template<int sc> extern m128f getmant_ps(m128f a, MANTISSA_NORM_ENUM interv) noexcept;
template<int sc> extern m128f mask_getmant_ps(m128f src, mmask8 k, m128f a, MANTISSA_NORM_ENUM interv) noexcept;
template<int sc> extern m128f maskz_getmant_ps(mmask8 k, m128f a, MANTISSA_NORM_ENUM interv) noexcept;
template<int sae, int sc> extern m128d getmant_round_sd(m128d a, m128d b, MANTISSA_NORM_ENUM interv) noexcept;
template<int sae, int sc> extern m128d mask_getmant_round_sd(m128d src, mmask8 k, m128d a, m128d b, MANTISSA_NORM_ENUM interv) noexcept;
template<int sae, int sc> extern m128d maskz_getmant_round_sd(mmask8 k, m128d a, m128d b, MANTISSA_NORM_ENUM interv) noexcept;
template<int sae, int sc> extern m128f getmant_round_ss(m128f a, m128f b, MANTISSA_NORM_ENUM interv) noexcept;
template<int sae, int sc> extern m128f mask_getmant_round_ss(m128f src, mmask8 k, m128f a, m128f b, MANTISSA_NORM_ENUM interv) noexcept;
template<int sae, int sc> extern m128f maskz_getmant_round_ss(mmask8 k, m128f a, m128f b, MANTISSA_NORM_ENUM interv) noexcept;
template<int sc> extern m128d getmant_sd(m128d a, m128d b, MANTISSA_NORM_ENUM interv) noexcept;
template<int sc> extern m128d mask_getmant_sd(m128d src, mmask8 k, m128d a, m128d b, MANTISSA_NORM_ENUM interv) noexcept;
template<int sc> extern m128d maskz_getmant_sd(mmask8 k, m128d a, m128d b, MANTISSA_NORM_ENUM interv) noexcept;
template<int sc> extern m128f getmant_ss(m128f a, m128f b, MANTISSA_NORM_ENUM interv) noexcept;
template<int sc> extern m128f mask_getmant_ss(m128f src, mmask8 k, m128f a, m128f b, MANTISSA_NORM_ENUM interv) noexcept;
template<int sc> extern m128f maskz_getmant_ss(mmask8 k, m128f a, m128f b, MANTISSA_NORM_ENUM interv) noexcept;
template<int b> extern m128i gf2p8affine_epi64_epi8(m128i x, m128i A) noexcept;
template<int b> extern m128i mask_gf2p8affine_epi64_epi8(m128i src, mmask16 k, m128i x, m128i A) noexcept;
template<int b> extern m128i maskz_gf2p8affine_epi64_epi8(mmask16 k, m128i x, m128i A) noexcept;
template<int b> extern m128i gf2p8affineinv_epi64_epi8(m128i x, m128i A) noexcept;
template<int b> extern m128i mask_gf2p8affineinv_epi64_epi8(m128i src, mmask16 k, m128i x, m128i A) noexcept;
template<int b> extern m128i maskz_gf2p8affineinv_epi64_epi8(mmask16 k, m128i x, m128i A) noexcept;
extern m128i gf2p8mul_epi8(m128i a, m128i b) noexcept;
extern m128i mask_gf2p8mul_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept;
extern m128i maskz_gf2p8mul_epi8(mmask16 k, m128i a, m128i b) noexcept;
extern m128i hadd_epi16(m128i a, m128i b) noexcept;
extern m128i hadd_epi32(m128i a, m128i b) noexcept;
extern m128d hadd_pd(m128d a, m128d b) noexcept;
extern m128f hadd_ps(m128f a, m128f b) noexcept;
extern m128i hadds_epi16(m128i a, m128i b) noexcept;
extern m128i hsub_epi16(m128i a, m128i b) noexcept;
extern m128i hsub_epi32(m128i a, m128i b) noexcept;
extern m128d hsub_pd(m128d a, m128d b) noexcept;
extern m128f hsub_ps(m128f a, m128f b) noexcept;
extern m128i hsubs_epi16(m128i a, m128i b) noexcept;
template<int scale> extern m128i i32gather_epi32(int const* base_addr, m128i vindex) noexcept;
template<int scale> extern m128i mask_i32gather_epi32(m128i src, int const* base_addr, m128i vindex, m128i mask) noexcept;
template<int scale> extern m128i mmask_i32gather_epi32(m128i src, mmask8 k, m128i vindex, void const* base_addr) noexcept;
template<int scale> extern m128i i32gather_epi64(int8 const* base_addr, m128i vindex) noexcept;
template<int scale> extern m128i mask_i32gather_epi64(m128i src, int8 const* base_addr, m128i vindex, m128i mask) noexcept;
template<int scale> extern m128i mmask_i32gather_epi64(m128i src, mmask8 k, m128i vindex, void const* base_addr) noexcept;
template<int scale> extern m128d i32gather_pd(double const* base_addr, m128i vindex) noexcept;
template<int scale> extern m128d mask_i32gather_pd(m128d src, double const* base_addr, m128i vindex, m128d mask) noexcept;
template<int scale> extern m128d mmask_i32gather_pd(m128d src, mmask8 k, m128i vindex, void const* base_addr) noexcept;
template<int scale> extern m128f i32gather_ps(float const* base_addr, m128i vindex) noexcept;
template<int scale> extern m128f mask_i32gather_ps(m128f src, float const* base_addr, m128i vindex, m128f mask) noexcept;
template<int scale> extern m128f mmask_i32gather_ps(m128f src, mmask8 k, m128i vindex, void const* base_addr) noexcept;
template<int scale> extern void i32scatter_epi32(void* base_addr, m128i vindex, m128i a) noexcept;
template<int scale> extern void mask_i32scatter_epi32(void* base_addr, mmask8 k, m128i vindex, m128i a) noexcept;
template<int scale> extern void i32scatter_epi64(void* base_addr, m128i vindex, m128i a) noexcept;
template<int scale> extern void mask_i32scatter_epi64(void* base_addr, mmask8 k, m128i vindex, m128i a) noexcept;
template<int scale> extern void i32scatter_pd(void* base_addr, m128i vindex, m128d a) noexcept;
template<int scale> extern void mask_i32scatter_pd(void* base_addr, mmask8 k, m128i vindex, m128d a) noexcept;
template<int scale> extern void i32scatter_ps(void* base_addr, m128i vindex, m128f a) noexcept;
template<int scale> extern void mask_i32scatter_ps(void* base_addr, mmask8 k, m128i vindex, m128f a) noexcept;
template<int scale> extern m128i i64gather_epi32(int const* base_addr, m128i vindex) noexcept;
template<int scale> extern m128i mask_i64gather_epi32(m128i src, int const* base_addr, m128i vindex, m128i mask) noexcept;
template<int scale> extern m128i mmask_i64gather_epi32(m128i src, mmask8 k, m128i vindex, void const* base_addr) noexcept;
template<int scale> extern m128i i64gather_epi64(int8 const* base_addr, m128i vindex) noexcept;
template<int scale> extern m128i mask_i64gather_epi64(m128i src, int8 const* base_addr, m128i vindex, m128i mask) noexcept;
template<int scale> extern m128i mmask_i64gather_epi64(m128i src, mmask8 k, m128i vindex, void const* base_addr) noexcept;
template<int scale> extern m128d i64gather_pd(double const* base_addr, m128i vindex) noexcept;
template<int scale> extern m128d mask_i64gather_pd(m128d src, double const* base_addr, m128i vindex, m128d mask) noexcept;
template<int scale> extern m128d mmask_i64gather_pd(m128d src, mmask8 k, m128i vindex, void const* base_addr) noexcept;
template<int scale> extern m128f i64gather_ps(float const* base_addr, m128i vindex) noexcept;
template<int scale> extern m128f mask_i64gather_ps(m128f src, float const* base_addr, m128i vindex, m128f mask) noexcept;
template<int scale> extern m128f mmask_i64gather_ps(m128f src, mmask8 k, m128i vindex, void const* base_addr) noexcept;
template<int scale> extern void i64scatter_epi32(void* base_addr, m128i vindex, m128i a) noexcept;
template<int scale> extern void mask_i64scatter_epi32(void* base_addr, mmask8 k, m128i vindex, m128i a) noexcept;
template<int scale> extern void i64scatter_epi64(void* base_addr, m128i vindex, m128i a) noexcept;
template<int scale> extern void mask_i64scatter_epi64(void* base_addr, mmask8 k, m128i vindex, m128i a) noexcept;
template<int scale> extern void i64scatter_pd(void* base_addr, m128i vindex, m128d a) noexcept;
template<int scale> extern void mask_i64scatter_pd(void* base_addr, mmask8 k, m128i vindex, m128d a) noexcept;
template<int scale> extern void i64scatter_ps(void* base_addr, m128i vindex, m128f a) noexcept;
template<int scale> extern void mask_i64scatter_ps(void* base_addr, mmask8 k, m128i vindex, m128f a) noexcept;
template<int4 imm8> extern m128i insert_epi16(m128i a, int i) noexcept;
template<int4 imm8> extern m128i insert_epi32(m128i a, int i) noexcept;
template<int4 imm8> extern m128i insert_epi64(m128i a, int8 i) noexcept;
template<int4 imm8> extern m128i insert_epi8(m128i a, int i) noexcept;
template<int4 imm8> extern m128f insert_ps(m128f a, m128f b) noexcept;
extern m128i lddqu_si128(m128i const* mem_addr) noexcept;
extern m128i mask_load_epi32(m128i src, mmask8 k, void const* mem_addr) noexcept;
extern m128i maskz_load_epi32(mmask8 k, void const* mem_addr) noexcept;
extern m128i mask_load_epi64(m128i src, mmask8 k, void const* mem_addr) noexcept;
extern m128i maskz_load_epi64(mmask8 k, void const* mem_addr) noexcept;
extern m128d load_pd(double const* mem_addr) noexcept;
extern m128d mask_load_pd(m128d src, mmask8 k, void const* mem_addr) noexcept;
extern m128d maskz_load_pd(mmask8 k, void const* mem_addr) noexcept;
extern m128d load_pd1(double const* mem_addr) noexcept;
extern m128f load_ps(float const* mem_addr) noexcept;
extern m128f mask_load_ps(m128f src, mmask8 k, void const* mem_addr) noexcept;
extern m128f maskz_load_ps(mmask8 k, void const* mem_addr) noexcept;
extern m128f load_ps1(float const* mem_addr) noexcept;
extern m128d load_sd(double const* mem_addr) noexcept;
extern m128d mask_load_sd(m128d src, mmask8 k, const double* mem_addr) noexcept;
extern m128d maskz_load_sd(mmask8 k, const double* mem_addr) noexcept;
extern m128i load_si128(m128i const* mem_addr) noexcept;
extern m128f load_ss(float const* mem_addr) noexcept;
extern m128f mask_load_ss(m128f src, mmask8 k, const float* mem_addr) noexcept;
extern m128f maskz_load_ss(mmask8 k, const float* mem_addr) noexcept;
extern m128d load1_pd(double const* mem_addr) noexcept;
extern m128f load1_ps(float const* mem_addr) noexcept;
extern m128d loaddup_pd(double const* mem_addr) noexcept;
extern m128d loadh_pd(m128d a, double const* mem_addr) noexcept;
extern void loadiwkey(nat4 __ctl, m128i __intkey, m128i __enkey_lo, m128i __enkey_hi) noexcept;
extern m128i loadl_epi64(m128i const* mem_addr) noexcept;
extern m128d loadl_pd(m128d a, double const* mem_addr) noexcept;
extern m128d loadr_pd(double const* mem_addr) noexcept;
extern m128f loadr_ps(float const* mem_addr) noexcept;
extern m128i loadu_epi16(void const* mem_addr) noexcept;
extern m128i mask_loadu_epi16(m128i src, mmask8 k, void const* mem_addr) noexcept;
extern m128i maskz_loadu_epi16(mmask8 k, void const* mem_addr) noexcept;
extern m128i loadu_epi32(void const* mem_addr) noexcept;
extern m128i mask_loadu_epi32(m128i src, mmask8 k, void const* mem_addr) noexcept;
extern m128i maskz_loadu_epi32(mmask8 k, void const* mem_addr) noexcept;
extern m128i loadu_epi64(void const* mem_addr) noexcept;
extern m128i mask_loadu_epi64(m128i src, mmask8 k, void const* mem_addr) noexcept;
extern m128i maskz_loadu_epi64(mmask8 k, void const* mem_addr) noexcept;
extern m128i loadu_epi8(void const* mem_addr) noexcept;
extern m128i mask_loadu_epi8(m128i src, mmask16 k, void const* mem_addr) noexcept;
extern m128i maskz_loadu_epi8(mmask16 k, void const* mem_addr) noexcept;
extern m128d loadu_pd(double const* mem_addr) noexcept;
extern m128d mask_loadu_pd(m128d src, mmask8 k, void const* mem_addr) noexcept;
extern m128d maskz_loadu_pd(mmask8 k, void const* mem_addr) noexcept;
extern m128f loadu_ps(float const* mem_addr) noexcept;
extern m128f mask_loadu_ps(m128f src, mmask8 k, void const* mem_addr) noexcept;
extern m128f maskz_loadu_ps(mmask8 k, void const* mem_addr) noexcept;
extern m128i loadu_si128(m128i const* mem_addr) noexcept;
extern m128i loadu_si16(void const* mem_addr) noexcept;
extern m128i loadu_si32(void const* mem_addr) noexcept;
extern m128i loadu_si64(void const* mem_addr) noexcept;
extern m128i lzcnt_epi32(m128i a) noexcept;
extern m128i mask_lzcnt_epi32(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_lzcnt_epi32(mmask8 k, m128i a) noexcept;
extern m128i lzcnt_epi64(m128i a) noexcept;
extern m128i mask_lzcnt_epi64(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_lzcnt_epi64(mmask8 k, m128i a) noexcept;
extern m128i madd_epi16(m128i a, m128i b) noexcept;
extern m128i mask_madd_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_madd_epi16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i madd52hi_epu64(m128i a, m128i b, m128i c) noexcept;
extern m128i mask_madd52hi_epu64(m128i a, mmask8 k, m128i b, m128i c) noexcept;
extern m128i maskz_madd52hi_epu64(mmask8 k, m128i a, m128i b, m128i c) noexcept;
extern m128i madd52lo_epu64(m128i a, m128i b, m128i c) noexcept;
extern m128i mask_madd52lo_epu64(m128i a, mmask8 k, m128i b, m128i c) noexcept;
extern m128i maskz_madd52lo_epu64(mmask8 k, m128i a, m128i b, m128i c) noexcept;
extern m128i maddubs_epi16(m128i a, m128i b) noexcept;
extern m128i mask_maddubs_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_maddubs_epi16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskload_epi32(int const* mem_addr, m128i mask) noexcept;
extern m128i maskload_epi64(int8 const* mem_addr, m128i mask) noexcept;
extern m128d maskload_pd(double const* mem_addr, m128i mask) noexcept;
extern m128f maskload_ps(float const* mem_addr, m128i mask) noexcept;
extern void maskmoveu_si128(m128i a, m128i mask, char* mem_addr) noexcept;
extern void maskstore_epi32(int* mem_addr, m128i mask, m128i a) noexcept;
extern void maskstore_epi64(int8* mem_addr, m128i mask, m128i a) noexcept;
extern void maskstore_pd(double* mem_addr, m128i mask, m128d a) noexcept;
extern void maskstore_ps(float* mem_addr, m128i mask, m128f a) noexcept;
extern m128i mask_max_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_max_epi16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i max_epi16(m128i a, m128i b) noexcept;
extern m128i mask_max_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_max_epi32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i max_epi32(m128i a, m128i b) noexcept;
extern m128i mask_max_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_max_epi64(mmask8 k, m128i a, m128i b) noexcept;
extern m128i max_epi64(m128i a, m128i b) noexcept;
extern m128i mask_max_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept;
extern m128i maskz_max_epi8(mmask16 k, m128i a, m128i b) noexcept;
extern m128i max_epi8(m128i a, m128i b) noexcept;
extern m128i mask_max_epu16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_max_epu16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i max_epu16(m128i a, m128i b) noexcept;
extern m128i mask_max_epu32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_max_epu32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i max_epu32(m128i a, m128i b) noexcept;
extern m128i mask_max_epu64(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_max_epu64(mmask8 k, m128i a, m128i b) noexcept;
extern m128i max_epu64(m128i a, m128i b) noexcept;
extern m128i mask_max_epu8(m128i src, mmask16 k, m128i a, m128i b) noexcept;
extern m128i maskz_max_epu8(mmask16 k, m128i a, m128i b) noexcept;
extern m128i max_epu8(m128i a, m128i b) noexcept;
extern m128d mask_max_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_max_pd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d max_pd(m128d a, m128d b) noexcept;
extern m128f mask_max_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_max_ps(mmask8 k, m128f a, m128f b) noexcept;
extern m128f max_ps(m128f a, m128f b) noexcept;
template<int sae> extern m128d mask_max_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int sae> extern m128d maskz_max_round_sd(mmask8 k, m128d a, m128d b) noexcept;
template<int sae> extern m128d max_round_sd(m128d a, m128d b) noexcept;
template<int sae> extern m128f mask_max_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int sae> extern m128f maskz_max_round_ss(mmask8 k, m128f a, m128f b) noexcept;
template<int sae> extern m128f max_round_ss(m128f a, m128f b) noexcept;
extern m128d mask_max_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_max_sd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d max_sd(m128d a, m128d b) noexcept;
extern m128f mask_max_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_max_ss(mmask8 k, m128f a, m128f b) noexcept;
extern m128f max_ss(m128f a, m128f b) noexcept;
extern m128i mask_min_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_min_epi16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i min_epi16(m128i a, m128i b) noexcept;
extern m128i mask_min_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_min_epi32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i min_epi32(m128i a, m128i b) noexcept;
extern m128i mask_min_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_min_epi64(mmask8 k, m128i a, m128i b) noexcept;
extern m128i min_epi64(m128i a, m128i b) noexcept;
extern m128i mask_min_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept;
extern m128i maskz_min_epi8(mmask16 k, m128i a, m128i b) noexcept;
extern m128i min_epi8(m128i a, m128i b) noexcept;
extern m128i mask_min_epu16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_min_epu16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i min_epu16(m128i a, m128i b) noexcept;
extern m128i mask_min_epu32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_min_epu32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i min_epu32(m128i a, m128i b) noexcept;
extern m128i mask_min_epu64(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_min_epu64(mmask8 k, m128i a, m128i b) noexcept;
extern m128i min_epu64(m128i a, m128i b) noexcept;
extern m128i mask_min_epu8(m128i src, mmask16 k, m128i a, m128i b) noexcept;
extern m128i maskz_min_epu8(mmask16 k, m128i a, m128i b) noexcept;
extern m128i min_epu8(m128i a, m128i b) noexcept;
extern m128d mask_min_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_min_pd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d min_pd(m128d a, m128d b) noexcept;
extern m128f mask_min_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_min_ps(mmask8 k, m128f a, m128f b) noexcept;
extern m128f min_ps(m128f a, m128f b) noexcept;
template<int sae> extern m128d mask_min_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int sae> extern m128d maskz_min_round_sd(mmask8 k, m128d a, m128d b) noexcept;
template<int sae> extern m128d min_round_sd(m128d a, m128d b) noexcept;
template<int sae> extern m128f mask_min_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int sae> extern m128f maskz_min_round_ss(mmask8 k, m128f a, m128f b) noexcept;
template<int sae> extern m128f min_round_ss(m128f a, m128f b) noexcept;
extern m128d mask_min_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_min_sd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d min_sd(m128d a, m128d b) noexcept;
extern m128f mask_min_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_min_ss(mmask8 k, m128f a, m128f b) noexcept;
extern m128f min_ss(m128f a, m128f b) noexcept;
extern m128i minpos_epu16(m128i a) noexcept;
extern m128i mask_mov_epi16(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_mov_epi16(mmask8 k, m128i a) noexcept;
extern m128i mask_mov_epi32(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_mov_epi32(mmask8 k, m128i a) noexcept;
extern m128i mask_mov_epi64(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_mov_epi64(mmask8 k, m128i a) noexcept;
extern m128i mask_mov_epi8(m128i src, mmask16 k, m128i a) noexcept;
extern m128i maskz_mov_epi8(mmask16 k, m128i a) noexcept;
extern m128d mask_mov_pd(m128d src, mmask8 k, m128d a) noexcept;
extern m128d maskz_mov_pd(mmask8 k, m128d a) noexcept;
extern m128f mask_mov_ps(m128f src, mmask8 k, m128f a) noexcept;
extern m128f maskz_mov_ps(mmask8 k, m128f a) noexcept;
extern m128i move_epi64(m128i a) noexcept;
extern m128d mask_move_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_move_sd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d move_sd(m128d a, m128d b) noexcept;
extern m128f mask_move_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_move_ss(mmask8 k, m128f a, m128f b) noexcept;
extern m128f move_ss(m128f a, m128f b) noexcept;
extern m128d mask_movedup_pd(m128d src, mmask8 k, m128d a) noexcept;
extern m128d maskz_movedup_pd(mmask8 k, m128d a) noexcept;
extern m128d movedup_pd(m128d a) noexcept;
extern m128f mask_movehdup_ps(m128f src, mmask8 k, m128f a) noexcept;
extern m128f maskz_movehdup_ps(mmask8 k, m128f a) noexcept;
extern m128f movehdup_ps(m128f a) noexcept;
extern m128f movehl_ps(m128f a, m128f b) noexcept;
extern m128f mask_moveldup_ps(m128f src, mmask8 k, m128f a) noexcept;
extern m128f maskz_moveldup_ps(mmask8 k, m128f a) noexcept;
extern m128f moveldup_ps(m128f a) noexcept;
extern m128f movelh_ps(m128f a, m128f b) noexcept;
extern int movemask_epi8(m128i a) noexcept;
extern int movemask_pd(m128d a) noexcept;
extern int movemask_ps(m128f a) noexcept;
extern mmask8 movepi16_mask(m128i a) noexcept;
extern mmask8 movepi32_mask(m128i a) noexcept;
extern mmask8 movepi64_mask(m128i a) noexcept;
extern mmask16 movepi8_mask(m128i a) noexcept;
extern m128i movm_epi16(mmask8 k) noexcept;
extern m128i movm_epi32(mmask8 k) noexcept;
extern m128i movm_epi64(mmask8 k) noexcept;
extern m128i movm_epi8(mmask16 k) noexcept;
template<int4 imm8> extern m128i mpsadbw_epu8(m128i a, m128i b) noexcept;
extern m128i mask_mul_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_mul_epi32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i mul_epi32(m128i a, m128i b) noexcept;
extern m128i mask_mul_epu32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_mul_epu32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i mul_epu32(m128i a, m128i b) noexcept;
extern m128d mask_mul_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_mul_pd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d mul_pd(m128d a, m128d b) noexcept;
extern m128f mask_mul_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_mul_ps(mmask8 k, m128f a, m128f b) noexcept;
extern m128f mul_ps(m128f a, m128f b) noexcept;
template<int4 rounding> extern m128d mask_mul_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int4 rounding> extern m128d maskz_mul_round_sd(mmask8 k, m128d a, m128d b) noexcept;
template<int4 rounding> extern m128d mul_round_sd(m128d a, m128d b) noexcept;
template<int4 rounding> extern m128f mask_mul_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int4 rounding> extern m128f maskz_mul_round_ss(mmask8 k, m128f a, m128f b) noexcept;
template<int4 rounding> extern m128f mul_round_ss(m128f a, m128f b) noexcept;
extern m128d mask_mul_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_mul_sd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d mul_sd(m128d a, m128d b) noexcept;
extern m128f mask_mul_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_mul_ss(mmask8 k, m128f a, m128f b) noexcept;
extern m128f mul_ss(m128f a, m128f b) noexcept;
extern m128i mask_mulhi_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_mulhi_epi16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i mulhi_epi16(m128i a, m128i b) noexcept;
extern m128i mask_mulhi_epu16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_mulhi_epu16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i mulhi_epu16(m128i a, m128i b) noexcept;
extern m128i mask_mulhrs_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_mulhrs_epi16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i mulhrs_epi16(m128i a, m128i b) noexcept;
extern m128i mask_mullo_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_mullo_epi16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i mullo_epi16(m128i a, m128i b) noexcept;
extern m128i mask_mullo_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_mullo_epi32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i mullo_epi32(m128i a, m128i b) noexcept;
extern m128i mask_mullo_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_mullo_epi64(mmask8 k, m128i a, m128i b) noexcept;
extern m128i mullo_epi64(m128i a, m128i b) noexcept;
extern m128i mask_multishift_epi64_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept;
extern m128i maskz_multishift_epi64_epi8(mmask16 k, m128i a, m128i b) noexcept;
extern m128i multishift_epi64_epi8(m128i a, m128i b) noexcept;
extern m128i mask_or_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_or_epi32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i mask_or_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_or_epi64(mmask8 k, m128i a, m128i b) noexcept;
extern m128d mask_or_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_or_pd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d or_pd(m128d a, m128d b) noexcept;
extern m128f mask_or_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_or_ps(mmask8 k, m128f a, m128f b) noexcept;
extern m128f or_ps(m128f a, m128f b) noexcept;
extern m128i or_si128(m128i a, m128i b) noexcept;
extern m128i mask_packs_epi16(m128i src, mmask16 k, m128i a, m128i b) noexcept;
extern m128i maskz_packs_epi16(mmask16 k, m128i a, m128i b) noexcept;
extern m128i packs_epi16(m128i a, m128i b) noexcept;
extern m128i mask_packs_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_packs_epi32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i packs_epi32(m128i a, m128i b) noexcept;
extern m128i mask_packus_epi16(m128i src, mmask16 k, m128i a, m128i b) noexcept;
extern m128i maskz_packus_epi16(mmask16 k, m128i a, m128i b) noexcept;
extern m128i packus_epi16(m128i a, m128i b) noexcept;
extern m128i mask_packus_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_packus_epi32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i packus_epi32(m128i a, m128i b) noexcept;
template<int4 imm8> extern m128d mask_permute_pd(m128d src, mmask8 k, m128d a) noexcept;
template<int4 imm8> extern m128d maskz_permute_pd(mmask8 k, m128d a) noexcept;
template<int4 imm8> extern m128d permute_pd(m128d a) noexcept;
template<int4 imm8> extern m128f mask_permute_ps(m128f src, mmask8 k, m128f a) noexcept;
template<int4 imm8> extern m128f maskz_permute_ps(mmask8 k, m128f a) noexcept;
template<int4 imm8> extern m128f permute_ps(m128f a) noexcept;
extern m128d mask_permutevar_pd(m128d src, mmask8 k, m128d a, m128i b) noexcept;
extern m128d maskz_permutevar_pd(mmask8 k, m128d a, m128i b) noexcept;
extern m128d permutevar_pd(m128d a, m128i b) noexcept;
extern m128f mask_permutevar_ps(m128f src, mmask8 k, m128f a, m128i b) noexcept;
extern m128f maskz_permutevar_ps(mmask8 k, m128f a, m128i b) noexcept;
extern m128f permutevar_ps(m128f a, m128i b) noexcept;
extern m128i mask_permutex2var_epi16(m128i a, mmask8 k, m128i idx, m128i b) noexcept;
extern m128i mask2_permutex2var_epi16(m128i a, m128i idx, mmask8 k, m128i b) noexcept;
extern m128i maskz_permutex2var_epi16(mmask8 k, m128i a, m128i idx, m128i b) noexcept;
extern m128i permutex2var_epi16(m128i a, m128i idx, m128i b) noexcept;
extern m128i mask_permutex2var_epi32(m128i a, mmask8 k, m128i idx, m128i b) noexcept;
extern m128i mask2_permutex2var_epi32(m128i a, m128i idx, mmask8 k, m128i b) noexcept;
extern m128i maskz_permutex2var_epi32(mmask8 k, m128i a, m128i idx, m128i b) noexcept;
extern m128i permutex2var_epi32(m128i a, m128i idx, m128i b) noexcept;
extern m128i mask_permutex2var_epi64(m128i a, mmask8 k, m128i idx, m128i b) noexcept;
extern m128i mask2_permutex2var_epi64(m128i a, m128i idx, mmask8 k, m128i b) noexcept;
extern m128i maskz_permutex2var_epi64(mmask8 k, m128i a, m128i idx, m128i b) noexcept;
extern m128i permutex2var_epi64(m128i a, m128i idx, m128i b) noexcept;
extern m128i mask_permutex2var_epi8(m128i a, mmask16 k, m128i idx, m128i b) noexcept;
extern m128i mask2_permutex2var_epi8(m128i a, m128i idx, mmask16 k, m128i b) noexcept;
extern m128i maskz_permutex2var_epi8(mmask16 k, m128i a, m128i idx, m128i b) noexcept;
extern m128i permutex2var_epi8(m128i a, m128i idx, m128i b) noexcept;
extern m128d mask_permutex2var_pd(m128d a, mmask8 k, m128i idx, m128d b) noexcept;
extern m128d mask2_permutex2var_pd(m128d a, m128i idx, mmask8 k, m128d b) noexcept;
extern m128d maskz_permutex2var_pd(mmask8 k, m128d a, m128i idx, m128d b) noexcept;
extern m128d permutex2var_pd(m128d a, m128i idx, m128d b) noexcept;
extern m128f mask_permutex2var_ps(m128f a, mmask8 k, m128i idx, m128f b) noexcept;
extern m128f mask2_permutex2var_ps(m128f a, m128i idx, mmask8 k, m128f b) noexcept;
extern m128f maskz_permutex2var_ps(mmask8 k, m128f a, m128i idx, m128f b) noexcept;
extern m128f permutex2var_ps(m128f a, m128i idx, m128f b) noexcept;
extern m128i mask_permutexvar_epi16(m128i src, mmask8 k, m128i idx, m128i a) noexcept;
extern m128i maskz_permutexvar_epi16(mmask8 k, m128i idx, m128i a) noexcept;
extern m128i permutexvar_epi16(m128i idx, m128i a) noexcept;
extern m128i mask_permutexvar_epi8(m128i src, mmask16 k, m128i idx, m128i a) noexcept;
extern m128i maskz_permutexvar_epi8(mmask16 k, m128i idx, m128i a) noexcept;
extern m128i permutexvar_epi8(m128i idx, m128i a) noexcept;
extern m128i mask_popcnt_epi16(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_popcnt_epi16(mmask8 k, m128i a) noexcept;
extern m128i popcnt_epi16(m128i a) noexcept;
extern m128i mask_popcnt_epi32(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_popcnt_epi32(mmask8 k, m128i a) noexcept;
extern m128i popcnt_epi32(m128i a) noexcept;
extern m128i mask_popcnt_epi64(m128i src, mmask8 k, m128i a) noexcept;
extern m128i maskz_popcnt_epi64(mmask8 k, m128i a) noexcept;
extern m128i popcnt_epi64(m128i a) noexcept;
extern m128i mask_popcnt_epi8(m128i src, mmask16 k, m128i a) noexcept;
extern m128i maskz_popcnt_epi8(mmask16 k, m128i a) noexcept;
extern m128i popcnt_epi8(m128i a) noexcept;
template<int4 imm8> extern m128d mask_range_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int4 imm8> extern m128d maskz_range_pd(mmask8 k, m128d a, m128d b) noexcept;
template<int4 imm8> extern m128d range_pd(m128d a, m128d b) noexcept;
template<int4 imm8> extern m128f mask_range_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int4 imm8> extern m128f maskz_range_ps(mmask8 k, m128f a, m128f b) noexcept;
template<int4 imm8> extern m128f range_ps(m128f a, m128f b) noexcept;
template<int4 imm8, int sae> extern m128d mask_range_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int4 imm8, int sae> extern m128d maskz_range_round_sd(mmask8 k, m128d a, m128d b) noexcept;
template<int4 imm8, int sae> extern m128d range_round_sd(m128d a, m128d b) noexcept;
template<int4 imm8, int sae> extern m128f mask_range_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int4 imm8, int sae> extern m128f maskz_range_round_ss(mmask8 k, m128f a, m128f b) noexcept;
template<int4 imm8, int sae> extern m128f range_round_ss(m128f a, m128f b) noexcept;
template<int4 imm8> extern m128d mask_range_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int4 imm8> extern m128d maskz_range_sd(mmask8 k, m128d a, m128d b) noexcept;
template<int4 imm8> extern m128f mask_range_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int4 imm8> extern m128f maskz_range_ss(mmask8 k, m128f a, m128f b) noexcept;
extern m128f rcp_ps(m128f a) noexcept;
extern m128f rcp_ss(m128f a) noexcept;
extern m128d mask_rcp14_pd(m128d src, mmask8 k, m128d a) noexcept;
extern m128d maskz_rcp14_pd(mmask8 k, m128d a) noexcept;
extern m128d rcp14_pd(m128d a) noexcept;
extern m128f mask_rcp14_ps(m128f src, mmask8 k, m128f a) noexcept;
extern m128f maskz_rcp14_ps(mmask8 k, m128f a) noexcept;
extern m128f rcp14_ps(m128f a) noexcept;
extern m128d mask_rcp14_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_rcp14_sd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d rcp14_sd(m128d a, m128d b) noexcept;
extern m128f mask_rcp14_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_rcp14_ss(mmask8 k, m128f a, m128f b) noexcept;
extern m128f rcp14_ss(m128f a, m128f b) noexcept;
template<int sae> extern m128d mask_rcp28_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int sae> extern m128d maskz_rcp28_round_sd(mmask8 k, m128d a, m128d b) noexcept;
template<int sae> extern m128d rcp28_round_sd(m128d a, m128d b) noexcept;
template<int sae> extern m128f mask_rcp28_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int sae> extern m128f maskz_rcp28_round_ss(mmask8 k, m128f a, m128f b) noexcept;
template<int sae> extern m128f rcp28_round_ss(m128f a, m128f b) noexcept;
extern m128d mask_rcp28_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_rcp28_sd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d rcp28_sd(m128d a, m128d b) noexcept;
extern m128f mask_rcp28_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_rcp28_ss(mmask8 k, m128f a, m128f b) noexcept;
extern m128f rcp28_ss(m128f a, m128f b) noexcept;
template<int4 imm8> extern m128d mask_reduce_pd(m128d src, mmask8 k, m128d a) noexcept;
template<int4 imm8> extern m128d maskz_reduce_pd(mmask8 k, m128d a) noexcept;
template<int4 imm8> extern m128d reduce_pd(m128d a) noexcept;
template<int4 imm8> extern m128f mask_reduce_ps(m128f src, mmask8 k, m128f a) noexcept;
template<int4 imm8> extern m128f maskz_reduce_ps(mmask8 k, m128f a) noexcept;
template<int4 imm8> extern m128f reduce_ps(m128f a) noexcept;
template<int4 imm8, int sae> extern m128d mask_reduce_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int4 imm8, int sae> extern m128d maskz_reduce_round_sd(mmask8 k, m128d a, m128d b) noexcept;
template<int4 imm8, int sae> extern m128d reduce_round_sd(m128d a, m128d b) noexcept;
template<int4 imm8, int sae> extern m128f mask_reduce_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int4 imm8, int sae> extern m128f maskz_reduce_round_ss(mmask8 k, m128f a, m128f b) noexcept;
template<int4 imm8, int sae> extern m128f reduce_round_ss(m128f a, m128f b) noexcept;
template<int4 imm8> extern m128d mask_reduce_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int4 imm8> extern m128d maskz_reduce_sd(mmask8 k, m128d a, m128d b) noexcept;
template<int4 imm8> extern m128d reduce_sd(m128d a, m128d b) noexcept;
template<int4 imm8> extern m128f mask_reduce_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int4 imm8> extern m128f maskz_reduce_ss(mmask8 k, m128f a, m128f b) noexcept;
template<int4 imm8> extern m128f reduce_ss(m128f a, m128f b) noexcept;
template<int4 imm8> extern m128i mask_rol_epi32(m128i src, mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i maskz_rol_epi32(mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i rol_epi32(m128i a) noexcept;
template<int4 imm8> extern m128i mask_rol_epi64(m128i src, mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i maskz_rol_epi64(mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i rol_epi64(m128i a) noexcept;
extern m128i mask_rolv_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_rolv_epi32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i rolv_epi32(m128i a, m128i b) noexcept;
extern m128i mask_rolv_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_rolv_epi64(mmask8 k, m128i a, m128i b) noexcept;
extern m128i rolv_epi64(m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i mask_ror_epi32(m128i src, mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i maskz_ror_epi32(mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i ror_epi32(m128i a) noexcept;
template<int4 imm8> extern m128i mask_ror_epi64(m128i src, mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i maskz_ror_epi64(mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i ror_epi64(m128i a) noexcept;
extern m128i mask_rorv_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_rorv_epi32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i rorv_epi32(m128i a, m128i b) noexcept;
extern m128i mask_rorv_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_rorv_epi64(mmask8 k, m128i a, m128i b) noexcept;
extern m128i rorv_epi64(m128i a, m128i b) noexcept;
template<int4 rounding> extern m128d round_pd(m128d a) noexcept;
template<int4 rounding> extern m128f round_ps(m128f a) noexcept;
template<int4 rounding> extern m128d round_sd(m128d a, m128d b) noexcept;
template<int4 rounding> extern m128f round_ss(m128f a, m128f b) noexcept;
template<int4 imm8> extern m128d mask_roundscale_pd(m128d src, mmask8 k, m128d a) noexcept;
template<int4 imm8> extern m128d maskz_roundscale_pd(mmask8 k, m128d a) noexcept;
template<int4 imm8> extern m128d roundscale_pd(m128d a) noexcept;
template<int4 imm8> extern m128f mask_roundscale_ps(m128f src, mmask8 k, m128f a) noexcept;
template<int4 imm8> extern m128f maskz_roundscale_ps(mmask8 k, m128f a) noexcept;
template<int4 imm8> extern m128f roundscale_ps(m128f a) noexcept;
template<int4 imm8, int sae> extern m128d mask_roundscale_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int4 imm8, int sae> extern m128d maskz_roundscale_round_sd(mmask8 k, m128d a, m128d b) noexcept;
template<int4 imm8, int sae> extern m128d roundscale_round_sd(m128d a, m128d b) noexcept;
template<int4 imm8, int sae> extern m128f mask_roundscale_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int4 imm8, int sae> extern m128f maskz_roundscale_round_ss(mmask8 k, m128f a, m128f b) noexcept;
template<int4 imm8, int sae> extern m128f roundscale_round_ss(m128f a, m128f b) noexcept;
template<int4 imm8> extern m128d mask_roundscale_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int4 imm8> extern m128d maskz_roundscale_sd(mmask8 k, m128d a, m128d b) noexcept;
template<int4 imm8> extern m128d roundscale_sd(m128d a, m128d b) noexcept;
template<int4 imm8> extern m128f mask_roundscale_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int4 imm8> extern m128f maskz_roundscale_ss(mmask8 k, m128f a, m128f b) noexcept;
template<int4 imm8> extern m128f roundscale_ss(m128f a, m128f b) noexcept;
extern m128f rsqrt_ps(m128f a) noexcept;
extern m128f rsqrt_ss(m128f a) noexcept;
extern m128d mask_rsqrt14_pd(m128d src, mmask8 k, m128d a) noexcept;
extern m128d maskz_rsqrt14_pd(mmask8 k, m128d a) noexcept;
extern m128f mask_rsqrt14_ps(m128f src, mmask8 k, m128f a) noexcept;
extern m128f maskz_rsqrt14_ps(mmask8 k, m128f a) noexcept;
extern m128d mask_rsqrt14_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_rsqrt14_sd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d rsqrt14_sd(m128d a, m128d b) noexcept;
extern m128f mask_rsqrt14_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_rsqrt14_ss(mmask8 k, m128f a, m128f b) noexcept;
extern m128f rsqrt14_ss(m128f a, m128f b) noexcept;
template<int sae> extern m128d mask_rsqrt28_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int sae> extern m128d maskz_rsqrt28_round_sd(mmask8 k, m128d a, m128d b) noexcept;
template<int sae> extern m128d rsqrt28_round_sd(m128d a, m128d b) noexcept;
template<int sae> extern m128f mask_rsqrt28_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int sae> extern m128f maskz_rsqrt28_round_ss(mmask8 k, m128f a, m128f b) noexcept;
template<int sae> extern m128f rsqrt28_round_ss(m128f a, m128f b) noexcept;
extern m128d mask_rsqrt28_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_rsqrt28_sd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d rsqrt28_sd(m128d a, m128d b) noexcept;
extern m128f mask_rsqrt28_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_rsqrt28_ss(mmask8 k, m128f a, m128f b) noexcept;
extern m128f rsqrt28_ss(m128f a, m128f b) noexcept;
extern m128i sad_epu8(m128i a, m128i b) noexcept;
extern m128d mask_scalef_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_scalef_pd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d scalef_pd(m128d a, m128d b) noexcept;
extern m128f mask_scalef_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_scalef_ps(mmask8 k, m128f a, m128f b) noexcept;
extern m128f scalef_ps(m128f a, m128f b) noexcept;
template<int4 rounding> extern m128d mask_scalef_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int4 rounding> extern m128d maskz_scalef_round_sd(mmask8 k, m128d a, m128d b) noexcept;
template<int4 rounding> extern m128d scalef_round_sd(m128d a, m128d b) noexcept;
template<int4 rounding> extern m128f mask_scalef_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int4 rounding> extern m128f maskz_scalef_round_ss(mmask8 k, m128f a, m128f b) noexcept;
template<int4 rounding> extern m128f scalef_round_ss(m128f a, m128f b) noexcept;
extern m128d mask_scalef_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_scalef_sd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d scalef_sd(m128d a, m128d b) noexcept;
extern m128f mask_scalef_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_scalef_ss(mmask8 k, m128f a, m128f b) noexcept;
extern m128f scalef_ss(m128f a, m128f b) noexcept;
extern m128i set_epi16(short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0) noexcept;
extern m128i set_epi32(int e3, int e2, int e1, int e0) noexcept;
extern m128i set_epi64x(int8 e1, int8 e0) noexcept;
extern m128i set_epi8(char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0) noexcept;
extern m128d set_pd(double e1, double e0) noexcept;
extern m128d set_pd1(double a) noexcept;
extern m128f set_ps(float e3, float e2, float e1, float e0) noexcept;
extern m128f set_ps1(float a) noexcept;
extern m128d set_sd(double a) noexcept;
extern m128f set_ss(float a) noexcept;
extern m128i mask_set1_epi16(m128i src, mmask8 k, short a) noexcept;
extern m128i maskz_set1_epi16(mmask8 k, short a) noexcept;
extern m128i set1_epi16(short a) noexcept;
extern m128i mask_set1_epi32(m128i src, mmask8 k, int a) noexcept;
extern m128i maskz_set1_epi32(mmask8 k, int a) noexcept;
extern m128i set1_epi32(int a) noexcept;
extern m128i mask_set1_epi64(m128i src, mmask8 k, int8 a) noexcept;
extern m128i maskz_set1_epi64(mmask8 k, int8 a) noexcept;
extern m128i set1_epi64x(int8 a) noexcept;
extern m128i mask_set1_epi8(m128i src, mmask16 k, char a) noexcept;
extern m128i maskz_set1_epi8(mmask16 k, char a) noexcept;
extern m128i set1_epi8(char a) noexcept;
extern m128d set1_pd(double a) noexcept;
extern m128f set1_ps(float a) noexcept;
extern m128i setr_epi16(short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0) noexcept;
extern m128i setr_epi32(int e3, int e2, int e1, int e0) noexcept;
extern m128i setr_epi8(char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0) noexcept;
extern m128d setr_pd(double e1, double e0) noexcept;
extern m128f setr_ps(float e3, float e2, float e1, float e0) noexcept;
extern m128d setzero_pd(void) noexcept;
extern m128f setzero_ps(void) noexcept;
extern m128i setzero_si128() noexcept;
extern m128i sha1msg1_epu32(m128i a, m128i b) noexcept;
extern m128i sha1msg2_epu32(m128i a, m128i b) noexcept;
extern m128i sha1nexte_epu32(m128i a, m128i b) noexcept;
template<int func> extern m128i sha1rnds4_epu32(m128i a, m128i b) noexcept;
extern m128i sha256msg1_epu32(m128i a, m128i b) noexcept;
extern m128i sha256msg2_epu32(m128i a, m128i b) noexcept;
extern m128i sha256rnds2_epu32(m128i a, m128i b, m128i k) noexcept;
template<int4 imm8> extern m128i mask_shldi_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i maskz_shldi_epi16(mmask8 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i shldi_epi16(m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i mask_shldi_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i maskz_shldi_epi32(mmask8 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i shldi_epi32(m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i mask_shldi_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i maskz_shldi_epi64(mmask8 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i shldi_epi64(m128i a, m128i b) noexcept;
extern m128i mask_shldv_epi16(m128i a, mmask8 k, m128i b, m128i c) noexcept;
extern m128i maskz_shldv_epi16(mmask8 k, m128i a, m128i b, m128i c) noexcept;
extern m128i shldv_epi16(m128i a, m128i b, m128i c) noexcept;
extern m128i mask_shldv_epi32(m128i a, mmask8 k, m128i b, m128i c) noexcept;
extern m128i maskz_shldv_epi32(mmask8 k, m128i a, m128i b, m128i c) noexcept;
extern m128i shldv_epi32(m128i a, m128i b, m128i c) noexcept;
extern m128i mask_shldv_epi64(m128i a, mmask8 k, m128i b, m128i c) noexcept;
extern m128i maskz_shldv_epi64(mmask8 k, m128i a, m128i b, m128i c) noexcept;
extern m128i shldv_epi64(m128i a, m128i b, m128i c) noexcept;
template<int4 imm8> extern m128i mask_shrdi_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i maskz_shrdi_epi16(mmask8 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i shrdi_epi16(m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i mask_shrdi_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i maskz_shrdi_epi32(mmask8 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i shrdi_epi32(m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i mask_shrdi_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i maskz_shrdi_epi64(mmask8 k, m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i shrdi_epi64(m128i a, m128i b) noexcept;
extern m128i mask_shrdv_epi16(m128i a, mmask8 k, m128i b, m128i c) noexcept;
extern m128i maskz_shrdv_epi16(mmask8 k, m128i a, m128i b, m128i c) noexcept;
extern m128i shrdv_epi16(m128i a, m128i b, m128i c) noexcept;
extern m128i mask_shrdv_epi32(m128i a, mmask8 k, m128i b, m128i c) noexcept;
extern m128i maskz_shrdv_epi32(mmask8 k, m128i a, m128i b, m128i c) noexcept;
extern m128i shrdv_epi32(m128i a, m128i b, m128i c) noexcept;
extern m128i mask_shrdv_epi64(m128i a, mmask8 k, m128i b, m128i c) noexcept;
extern m128i maskz_shrdv_epi64(mmask8 k, m128i a, m128i b, m128i c) noexcept;
extern m128i shrdv_epi64(m128i a, m128i b, m128i c) noexcept;
template<int4 imm8> extern m128i mask_shuffle_epi32(m128i src, mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i maskz_shuffle_epi32(mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i shuffle_epi32(m128i a) noexcept;
extern m128i mask_shuffle_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept;
extern m128i maskz_shuffle_epi8(mmask16 k, m128i a, m128i b) noexcept;
extern m128i shuffle_epi8(m128i a, m128i b) noexcept;
template<int4 imm8> extern m128d mask_shuffle_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int4 imm8> extern m128d maskz_shuffle_pd(mmask8 k, m128d a, m128d b) noexcept;
template<int4 imm8> extern m128d shuffle_pd(m128d a, m128d b) noexcept;
template<int4 imm8> extern m128f mask_shuffle_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int4 imm8> extern m128f maskz_shuffle_ps(mmask8 k, m128f a, m128f b) noexcept;
template<nat4 imm8> extern m128f shuffle_ps(m128f a, m128f b) noexcept;
template<int4 imm8> extern m128i mask_shufflehi_epi16(m128i src, mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i maskz_shufflehi_epi16(mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i shufflehi_epi16(m128i a) noexcept;
template<int4 imm8> extern m128i mask_shufflelo_epi16(m128i src, mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i maskz_shufflelo_epi16(mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i shufflelo_epi16(m128i a) noexcept;
extern m128i sign_epi16(m128i a, m128i b) noexcept;
extern m128i sign_epi32(m128i a, m128i b) noexcept;
extern m128i sign_epi8(m128i a, m128i b) noexcept;
extern m128i mask_sll_epi16(m128i src, mmask8 k, m128i a, m128i count) noexcept;
extern m128i maskz_sll_epi16(mmask8 k, m128i a, m128i count) noexcept;
extern m128i sll_epi16(m128i a, m128i count) noexcept;
extern m128i mask_sll_epi32(m128i src, mmask8 k, m128i a, m128i count) noexcept;
extern m128i maskz_sll_epi32(mmask8 k, m128i a, m128i count) noexcept;
extern m128i sll_epi32(m128i a, m128i count) noexcept;
extern m128i mask_sll_epi64(m128i src, mmask8 k, m128i a, m128i count) noexcept;
extern m128i maskz_sll_epi64(mmask8 k, m128i a, m128i count) noexcept;
extern m128i sll_epi64(m128i a, m128i count) noexcept;
template<nat4 imm8> extern m128i mask_slli_epi16(m128i src, mmask8 k, m128i a) noexcept;
template<nat4 imm8> extern m128i maskz_slli_epi16(mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i slli_epi16(m128i a) noexcept;
template<nat4 imm8> extern m128i mask_slli_epi32(m128i src, mmask8 k, m128i a) noexcept;
template<nat4 imm8> extern m128i maskz_slli_epi32(mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i slli_epi32(m128i a) noexcept;
template<nat4 imm8> extern m128i mask_slli_epi64(m128i src, mmask8 k, m128i a) noexcept;
template<nat4 imm8> extern m128i maskz_slli_epi64(mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i slli_epi64(m128i a) noexcept;
template<int4 imm8> extern m128i slli_si128(m128i a) noexcept;
extern m128i mask_sllv_epi16(m128i src, mmask8 k, m128i a, m128i count) noexcept;
extern m128i maskz_sllv_epi16(mmask8 k, m128i a, m128i count) noexcept;
extern m128i sllv_epi16(m128i a, m128i count) noexcept;
extern m128i mask_sllv_epi32(m128i src, mmask8 k, m128i a, m128i count) noexcept;
extern m128i maskz_sllv_epi32(mmask8 k, m128i a, m128i count) noexcept;
extern m128i sllv_epi32(m128i a, m128i count) noexcept;
extern m128i mask_sllv_epi64(m128i src, mmask8 k, m128i a, m128i count) noexcept;
extern m128i maskz_sllv_epi64(mmask8 k, m128i a, m128i count) noexcept;
extern m128i sllv_epi64(m128i a, m128i count) noexcept;
extern m128d mask_sqrt_pd(m128d src, mmask8 k, m128d a) noexcept;
extern m128d maskz_sqrt_pd(mmask8 k, m128d a) noexcept;
extern m128d sqrt_pd(m128d a) noexcept;
extern m128f mask_sqrt_ps(m128f src, mmask8 k, m128f a) noexcept;
extern m128f maskz_sqrt_ps(mmask8 k, m128f a) noexcept;
extern m128f sqrt_ps(m128f a) noexcept;
template<int4 rounding> extern m128d mask_sqrt_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int4 rounding> extern m128d maskz_sqrt_round_sd(mmask8 k, m128d a, m128d b) noexcept;
template<int4 rounding> extern m128d sqrt_round_sd(m128d a, m128d b) noexcept;
template<int4 rounding> extern m128f mask_sqrt_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int4 rounding> extern m128f maskz_sqrt_round_ss(mmask8 k, m128f a, m128f b) noexcept;
template<int4 rounding> extern m128f sqrt_round_ss(m128f a, m128f b) noexcept;
extern m128d mask_sqrt_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_sqrt_sd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d sqrt_sd(m128d a, m128d b) noexcept;
extern m128f mask_sqrt_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_sqrt_ss(mmask8 k, m128f a, m128f b) noexcept;
extern m128f sqrt_ss(m128f a) noexcept;
extern m128i mask_sra_epi16(m128i src, mmask8 k, m128i a, m128i count) noexcept;
extern m128i maskz_sra_epi16(mmask8 k, m128i a, m128i count) noexcept;
extern m128i sra_epi16(m128i a, m128i count) noexcept;
extern m128i mask_sra_epi32(m128i src, mmask8 k, m128i a, m128i count) noexcept;
extern m128i maskz_sra_epi32(mmask8 k, m128i a, m128i count) noexcept;
extern m128i sra_epi32(m128i a, m128i count) noexcept;
extern m128i mask_sra_epi64(m128i src, mmask8 k, m128i a, m128i count) noexcept;
extern m128i maskz_sra_epi64(mmask8 k, m128i a, m128i count) noexcept;
extern m128i sra_epi64(m128i a, m128i count) noexcept;
template<nat4 imm8> extern m128i mask_srai_epi16(m128i src, mmask8 k, m128i a) noexcept;
template<nat4 imm8> extern m128i maskz_srai_epi16(mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i srai_epi16(m128i a) noexcept;
template<nat4 imm8> extern m128i mask_srai_epi32(m128i src, mmask8 k, m128i a) noexcept;
template<nat4 imm8> extern m128i maskz_srai_epi32(mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i srai_epi32(m128i a) noexcept;
template<nat4 imm8> extern m128i mask_srai_epi64(m128i src, mmask8 k, m128i a) noexcept;
template<nat4 imm8> extern m128i maskz_srai_epi64(mmask8 k, m128i a) noexcept;
template<nat4 imm8> extern m128i srai_epi64(m128i a) noexcept;
extern m128i mask_srav_epi16(m128i src, mmask8 k, m128i a, m128i count) noexcept;
extern m128i maskz_srav_epi16(mmask8 k, m128i a, m128i count) noexcept;
extern m128i srav_epi16(m128i a, m128i count) noexcept;
extern m128i mask_srav_epi32(m128i src, mmask8 k, m128i a, m128i count) noexcept;
extern m128i maskz_srav_epi32(mmask8 k, m128i a, m128i count) noexcept;
extern m128i srav_epi32(m128i a, m128i count) noexcept;
extern m128i mask_srav_epi64(m128i src, mmask8 k, m128i a, m128i count) noexcept;
extern m128i maskz_srav_epi64(mmask8 k, m128i a, m128i count) noexcept;
extern m128i srav_epi64(m128i a, m128i count) noexcept;
extern m128i mask_srl_epi16(m128i src, mmask8 k, m128i a, m128i count) noexcept;
extern m128i maskz_srl_epi16(mmask8 k, m128i a, m128i count) noexcept;
extern m128i srl_epi16(m128i a, m128i count) noexcept;
extern m128i mask_srl_epi32(m128i src, mmask8 k, m128i a, m128i count) noexcept;
extern m128i maskz_srl_epi32(mmask8 k, m128i a, m128i count) noexcept;
extern m128i srl_epi32(m128i a, m128i count) noexcept;
extern m128i mask_srl_epi64(m128i src, mmask8 k, m128i a, m128i count) noexcept;
extern m128i maskz_srl_epi64(mmask8 k, m128i a, m128i count) noexcept;
extern m128i srl_epi64(m128i a, m128i count) noexcept;
template<int4 imm8> extern m128i mask_srli_epi16(m128i src, mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i maskz_srli_epi16(mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i srli_epi16(m128i a) noexcept;
template<nat4 imm8> extern m128i mask_srli_epi32(m128i src, mmask8 k, m128i a) noexcept;
template<nat4 imm8> extern m128i maskz_srli_epi32(mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i srli_epi32(m128i a) noexcept;
template<nat4 imm8> extern m128i mask_srli_epi64(m128i src, mmask8 k, m128i a) noexcept;
template<nat4 imm8> extern m128i maskz_srli_epi64(mmask8 k, m128i a) noexcept;
template<int4 imm8> extern m128i srli_epi64(m128i a) noexcept;
template<int4 imm8> extern m128i srli_si128(m128i a) noexcept;
extern m128i mask_srlv_epi16(m128i src, mmask8 k, m128i a, m128i count) noexcept;
extern m128i maskz_srlv_epi16(mmask8 k, m128i a, m128i count) noexcept;
extern m128i srlv_epi16(m128i a, m128i count) noexcept;
extern m128i mask_srlv_epi32(m128i src, mmask8 k, m128i a, m128i count) noexcept;
extern m128i maskz_srlv_epi32(mmask8 k, m128i a, m128i count) noexcept;
extern m128i srlv_epi32(m128i a, m128i count) noexcept;
extern m128i mask_srlv_epi64(m128i src, mmask8 k, m128i a, m128i count) noexcept;
extern m128i maskz_srlv_epi64(mmask8 k, m128i a, m128i count) noexcept;
extern m128i srlv_epi64(m128i a, m128i count) noexcept;
extern void mask_store_epi32(void* mem_addr, mmask8 k, m128i a) noexcept;
extern void mask_store_epi64(void* mem_addr, mmask8 k, m128i a) noexcept;
extern void mask_store_pd(void* mem_addr, mmask8 k, m128d a) noexcept;
extern void store_pd(double* mem_addr, m128d a) noexcept;
extern void store_pd1(double* mem_addr, m128d a) noexcept;
extern void mask_store_ps(void* mem_addr, mmask8 k, m128f a) noexcept;
extern void store_ps(float* mem_addr, m128f a) noexcept;
extern void store_ps1(float* mem_addr, m128f a) noexcept;
extern void mask_store_sd(double* mem_addr, mmask8 k, m128d a) noexcept;
extern void store_sd(double* mem_addr, m128d a) noexcept;
extern void store_si128(m128i* mem_addr, m128i a) noexcept;
extern void mask_store_ss(float* mem_addr, mmask8 k, m128f a) noexcept;
extern void store_ss(float* mem_addr, m128f a) noexcept;
extern void store1_pd(double* mem_addr, m128d a) noexcept;
extern void store1_ps(float* mem_addr, m128f a) noexcept;
extern void storeh_pd(double* mem_addr, m128d a) noexcept;
extern void storel_epi64(m128i* mem_addr, m128i a) noexcept;
extern void storel_pd(double* mem_addr, m128d a) noexcept;
extern void storer_pd(double* mem_addr, m128d a) noexcept;
extern void storer_ps(float* mem_addr, m128f a) noexcept;
extern void mask_storeu_epi16(void* mem_addr, mmask8 k, m128i a) noexcept;
extern void storeu_epi16(void* mem_addr, m128i a) noexcept;
extern void mask_storeu_epi32(void* mem_addr, mmask8 k, m128i a) noexcept;
extern void storeu_epi32(void* mem_addr, m128i a) noexcept;
extern void mask_storeu_epi64(void* mem_addr, mmask8 k, m128i a) noexcept;
extern void storeu_epi64(void* mem_addr, m128i a) noexcept;
extern void mask_storeu_epi8(void* mem_addr, mmask16 k, m128i a) noexcept;
extern void storeu_epi8(void* mem_addr, m128i a) noexcept;
extern void mask_storeu_pd(void* mem_addr, mmask8 k, m128d a) noexcept;
extern void storeu_pd(double* mem_addr, m128d a) noexcept;
extern void mask_storeu_ps(void* mem_addr, mmask8 k, m128f a) noexcept;
extern void storeu_ps(float* mem_addr, m128f a) noexcept;
extern void storeu_si128(m128i* mem_addr, m128i a) noexcept;
extern void storeu_si16(void* mem_addr, m128i a) noexcept;
extern void storeu_si32(void* mem_addr, m128i a) noexcept;
extern void storeu_si64(void* mem_addr, m128i a) noexcept;
extern m128i stream_load_si128(m128i* mem_addr) noexcept;
extern void stream_pd(double* mem_addr, m128d a) noexcept;
extern void stream_ps(float* mem_addr, m128f a) noexcept;
extern void stream_si128(m128i* mem_addr, m128i a) noexcept;
extern m128i mask_sub_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_sub_epi16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i sub_epi16(m128i a, m128i b) noexcept;
extern m128i mask_sub_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_sub_epi32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i sub_epi32(m128i a, m128i b) noexcept;
extern m128i mask_sub_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_sub_epi64(mmask8 k, m128i a, m128i b) noexcept;
extern m128i sub_epi64(m128i a, m128i b) noexcept;
extern m128i mask_sub_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept;
extern m128i maskz_sub_epi8(mmask16 k, m128i a, m128i b) noexcept;
extern m128i sub_epi8(m128i a, m128i b) noexcept;
extern m128d mask_sub_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_sub_pd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d sub_pd(m128d a, m128d b) noexcept;
extern m128f mask_sub_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_sub_ps(mmask8 k, m128f a, m128f b) noexcept;
extern m128f sub_ps(m128f a, m128f b) noexcept;
template<int4 rounding> extern m128d mask_sub_round_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
template<int4 rounding> extern m128d maskz_sub_round_sd(mmask8 k, m128d a, m128d b) noexcept;
template<int4 rounding> extern m128d sub_round_sd(m128d a, m128d b) noexcept;
template<int4 rounding> extern m128f mask_sub_round_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
template<int4 rounding> extern m128f maskz_sub_round_ss(mmask8 k, m128f a, m128f b) noexcept;
template<int4 rounding> extern m128f sub_round_ss(m128f a, m128f b) noexcept;
extern m128d mask_sub_sd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_sub_sd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d sub_sd(m128d a, m128d b) noexcept;
extern m128f mask_sub_ss(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_sub_ss(mmask8 k, m128f a, m128f b) noexcept;
extern m128f sub_ss(m128f a, m128f b) noexcept;
extern m128i mask_subs_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_subs_epi16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i subs_epi16(m128i a, m128i b) noexcept;
extern m128i mask_subs_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept;
extern m128i maskz_subs_epi8(mmask16 k, m128i a, m128i b) noexcept;
extern m128i subs_epi8(m128i a, m128i b) noexcept;
extern m128i mask_subs_epu16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_subs_epu16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i subs_epu16(m128i a, m128i b) noexcept;
extern m128i mask_subs_epu8(m128i src, mmask16 k, m128i a, m128i b) noexcept;
extern m128i maskz_subs_epu8(mmask16 k, m128i a, m128i b) noexcept;
extern m128i subs_epu8(m128i a, m128i b) noexcept;
template<int4 imm8> extern m128i mask_ternarylogic_epi32(m128i a, mmask8 k, m128i b, m128i c) noexcept;
template<int4 imm8> extern m128i maskz_ternarylogic_epi32(mmask8 k, m128i a, m128i b, m128i c) noexcept;
template<int4 imm8> extern m128i ternarylogic_epi32(m128i a, m128i b, m128i c) noexcept;
template<int4 imm8> extern m128i mask_ternarylogic_epi64(m128i a, mmask8 k, m128i b, m128i c) noexcept;
template<int4 imm8> extern m128i maskz_ternarylogic_epi64(mmask8 k, m128i a, m128i b, m128i c) noexcept;
template<int4 imm8> extern m128i ternarylogic_epi64(m128i a, m128i b, m128i c) noexcept;
extern int test_all_ones(m128i a) noexcept;
extern int test_all_zeros(m128i a, m128i mask) noexcept;
extern mmask8 mask_test_epi16_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 test_epi16_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_test_epi32_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 test_epi32_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_test_epi64_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 test_epi64_mask(m128i a, m128i b) noexcept;
extern mmask16 mask_test_epi8_mask(mmask16 k1, m128i a, m128i b) noexcept;
extern mmask16 test_epi8_mask(m128i a, m128i b) noexcept;
extern int test_mix_ones_zeros(m128i a, m128i mask) noexcept;
extern int testc_pd(m128d a, m128d b) noexcept;
extern int testc_ps(m128f a, m128f b) noexcept;
extern int testc_si128(m128i a, m128i b) noexcept;
extern mmask8 mask_testn_epi16_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 testn_epi16_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_testn_epi32_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 testn_epi32_mask(m128i a, m128i b) noexcept;
extern mmask8 mask_testn_epi64_mask(mmask8 k1, m128i a, m128i b) noexcept;
extern mmask8 testn_epi64_mask(m128i a, m128i b) noexcept;
extern mmask16 mask_testn_epi8_mask(mmask16 k1, m128i a, m128i b) noexcept;
extern mmask16 testn_epi8_mask(m128i a, m128i b) noexcept;
extern int testnzc_pd(m128d a, m128d b) noexcept;
extern int testnzc_ps(m128f a, m128f b) noexcept;
extern int testnzc_si128(m128i a, m128i b) noexcept;
extern int testz_pd(m128d a, m128d b) noexcept;
extern int testz_ps(m128f a, m128f b) noexcept;
extern int testz_si128(m128i a, m128i b) noexcept;
extern void TRANSPOSE4_PS(m128f row0, m128f row1, m128f row2, m128f row3) noexcept;
extern int ucomieq_sd(m128d a, m128d b) noexcept;
extern int ucomieq_ss(m128f a, m128f b) noexcept;
extern int ucomige_sd(m128d a, m128d b) noexcept;
extern int ucomige_ss(m128f a, m128f b) noexcept;
extern int ucomigt_sd(m128d a, m128d b) noexcept;
extern int ucomigt_ss(m128f a, m128f b) noexcept;
extern int ucomile_sd(m128d a, m128d b) noexcept;
extern int ucomile_ss(m128f a, m128f b) noexcept;
extern int ucomilt_sd(m128d a, m128d b) noexcept;
extern int ucomilt_ss(m128f a, m128f b) noexcept;
extern int ucomineq_sd(m128d a, m128d b) noexcept;
extern int ucomineq_ss(m128f a, m128f b) noexcept;
extern m128d undefined_pd(void) noexcept;
extern m128f undefined_ps(void) noexcept;
extern m128i undefined_si128(void) noexcept;
extern m128i mask_unpackhi_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_unpackhi_epi16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i unpackhi_epi16(m128i a, m128i b) noexcept;
extern m128i mask_unpackhi_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_unpackhi_epi32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i unpackhi_epi32(m128i a, m128i b) noexcept;
extern m128i mask_unpackhi_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_unpackhi_epi64(mmask8 k, m128i a, m128i b) noexcept;
extern m128i unpackhi_epi64(m128i a, m128i b) noexcept;
extern m128i mask_unpackhi_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept;
extern m128i maskz_unpackhi_epi8(mmask16 k, m128i a, m128i b) noexcept;
extern m128i unpackhi_epi8(m128i a, m128i b) noexcept;
extern m128d mask_unpackhi_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_unpackhi_pd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d unpackhi_pd(m128d a, m128d b) noexcept;
extern m128f mask_unpackhi_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_unpackhi_ps(mmask8 k, m128f a, m128f b) noexcept;
extern m128f unpackhi_ps(m128f a, m128f b) noexcept;
extern m128i mask_unpacklo_epi16(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_unpacklo_epi16(mmask8 k, m128i a, m128i b) noexcept;
extern m128i unpacklo_epi16(m128i a, m128i b) noexcept;
extern m128i mask_unpacklo_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_unpacklo_epi32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i unpacklo_epi32(m128i a, m128i b) noexcept;
extern m128i mask_unpacklo_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_unpacklo_epi64(mmask8 k, m128i a, m128i b) noexcept;
extern m128i unpacklo_epi64(m128i a, m128i b) noexcept;
extern m128i mask_unpacklo_epi8(m128i src, mmask16 k, m128i a, m128i b) noexcept;
extern m128i maskz_unpacklo_epi8(mmask16 k, m128i a, m128i b) noexcept;
extern m128i unpacklo_epi8(m128i a, m128i b) noexcept;
extern m128d mask_unpacklo_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_unpacklo_pd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d unpacklo_pd(m128d a, m128d b) noexcept;
extern m128f mask_unpacklo_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_unpacklo_ps(mmask8 k, m128f a, m128f b) noexcept;
extern m128f unpacklo_ps(m128f a, m128f b) noexcept;
extern m128i mask_xor_epi32(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_xor_epi32(mmask8 k, m128i a, m128i b) noexcept;
extern m128i mask_xor_epi64(m128i src, mmask8 k, m128i a, m128i b) noexcept;
extern m128i maskz_xor_epi64(mmask8 k, m128i a, m128i b) noexcept;
extern m128d mask_xor_pd(m128d src, mmask8 k, m128d a, m128d b) noexcept;
extern m128d maskz_xor_pd(mmask8 k, m128d a, m128d b) noexcept;
extern m128d xor_pd(m128d a, m128d b) noexcept;
extern m128f mask_xor_ps(m128f src, mmask8 k, m128f a, m128f b) noexcept;
extern m128f maskz_xor_ps(mmask8 k, m128f a, m128f b) noexcept;
extern m128f xor_ps(m128f a, m128f b) noexcept;
extern m128i xor_si128(m128i a, m128i b) noexcept;
} // namespace yw::intrin::inline m128
#endif
