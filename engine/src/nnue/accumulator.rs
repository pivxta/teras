use crate::utils::simd::SimdAlign;
use dama::{ByColor, Color, Piece, Position, Square};

use super::{Transformer, feature::feature, model};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Accumulator {
    accumulator: ByColor<SimdAlign<[i16; model::FT_OUT]>>,
}

impl Accumulator {
    #[inline]
    pub fn from_position(transformer: &Transformer, position: &Position) -> Self {
        let mut accum = Accumulator {
            accumulator: ByColor::from_fn(|_| [0; model::FT_OUT].into()),
        };
        accum.refresh(transformer, position, Color::White);
        accum.refresh(transformer, position, Color::Black);
        accum
    }

    #[inline]
    pub fn refresh(&mut self, transformer: &Transformer, position: &Position, persp: Color) {
        transformer.reset(&mut self.accumulator[persp]);
        for color in Color::all() {
            for piece in Piece::all() {
                for square in position.colored(color) & position.pieces(piece) {
                    let feature = feature(persp, color, piece, square);
                    transformer.add(feature, &mut self.accumulator[persp]);
                }
            }
        }
    }

    #[inline]
    pub fn add(&mut self, transformer: &Transformer, color: Color, piece: Piece, square: Square) {
        for persp in Color::all() {
            let feature = feature(persp, color, piece, square);
            transformer.add(feature, &mut self.accumulator[persp]);
        }
    }

    #[inline]
    pub fn sub(&mut self, transformer: &Transformer, color: Color, piece: Piece, square: Square) {
        for persp in Color::all() {
            let feature = feature(persp, color, piece, square);
            transformer.sub(feature, &mut self.accumulator[persp]);
        }
    }

    #[inline]
    pub fn get(&self, color: Color) -> &[i16; model::FT_OUT] {
        &self.accumulator[color]
    }
}
