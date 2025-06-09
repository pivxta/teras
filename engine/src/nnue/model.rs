use dama::Position;
use std::{
    io::{self, Read},
    sync::Arc,
};

use super::{
    Accumulator,
    layers::{self, Linear, LinearOut, cat, clipped_relu, clipped_relu16},
};
use crate::{eval::Eval, utils::simd::SimdAlign};

pub const FT_SIZE: usize = 768;
pub const FT_OUT: usize = 256;
pub const HIDDEN1_SIZE: usize = 16;
pub const HIDDEN2_SIZE: usize = 32;

#[derive(Debug, Clone)]
pub struct Transformer(Arc<layers::Transformer<FT_SIZE, FT_OUT>>);

impl Transformer {
    fn load_from(reader: &mut impl Read) -> io::Result<Transformer> {
        Ok(Self(Arc::new(layers::Transformer::load_from(reader)?)))
    }

    #[inline]
    pub fn reset(&self, dst: &mut [i16; FT_OUT]) {
        self.0.reset(dst)
    }

    #[inline]
    pub fn add(&self, feature: usize, dst: &mut SimdAlign<[i16; FT_OUT]>) {
        self.0.add(feature, dst);
    }

    #[inline]
    pub fn sub(&self, feature: usize, dst: &mut SimdAlign<[i16; FT_OUT]>) {
        self.0.sub(feature, dst);
    }
}

#[derive(Debug)]
struct LayerStack {
    hidden1: Linear<{ FT_OUT * 2 }, HIDDEN1_SIZE>,
    hidden2: Linear<HIDDEN1_SIZE, HIDDEN2_SIZE>,
    out: LinearOut<HIDDEN2_SIZE>,
}

impl LayerStack {
    #[inline]
    pub fn load_from(mut reader: impl Read) -> io::Result<LayerStack> {
        Ok(LayerStack {
            hidden1: Linear::load_from(&mut reader)?,
            hidden2: Linear::load_from(&mut reader)?,
            out: LinearOut::load_from(&mut reader)?,
        })
    }

    fn propagate(&self, stm: &[i16; FT_OUT], non_stm: &[i16; FT_OUT]) -> i32 {
        let mut ft = SimdAlign([0i16; FT_OUT * 2]);
        cat(stm, non_stm, &mut ft);

        let mut hidden1_in = SimdAlign([0i8; FT_OUT * 2]);
        clipped_relu16(&ft, &mut hidden1_in);

        let mut hidden1_out = SimdAlign([0i32; HIDDEN1_SIZE]);
        self.hidden1.propagate(&hidden1_in, &mut hidden1_out);

        let mut hidden2_in = SimdAlign([0i8; HIDDEN1_SIZE]);
        clipped_relu(&hidden1_out, &mut hidden2_in);

        let mut hidden2_out = SimdAlign([0i32; HIDDEN2_SIZE]);
        self.hidden2.propagate(&hidden2_in, &mut hidden2_out);

        let mut output_in = SimdAlign([0i8; HIDDEN2_SIZE]);
        clipped_relu(&hidden2_out, &mut output_in);

        self.out.propagate(&output_in)
    }
}

#[derive(Debug, Clone)]
pub struct Model(Arc<LayerStack>);

impl Model {
    #[inline]
    pub fn load_default() -> (Model, Transformer) {
        let default_model = include_bytes!("default.nnue");
        Model::load_from(&default_model[..]).expect("failed to load default model")
    }

    #[inline]
    pub fn load_from(mut reader: impl Read) -> io::Result<(Model, Transformer)> {
        let transformer = Transformer::load_from(&mut reader)?;
        let model = Model(Arc::new(LayerStack::load_from(&mut reader)?));
        Ok((model, transformer))
    }

    pub fn evaluate(&self, position: &Position, accumulator: &Accumulator) -> Eval {
        let value = self.0.propagate(
            accumulator.get(position.side_to_move()),
            accumulator.get(!position.side_to_move()),
        );
        Eval::centipawns(value)
    }
}
