use crate::eval::Eval;
use dama::Position;
use std::ops;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Phase(i32);

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct PhaseEval {
    pub midgame_centipawns: i32,
    pub endgame_centipawns: i32,
}

impl Phase {
    const KNIGHT_PHASE: i32 = 1;
    const BISHOP_PHASE: i32 = 1;
    const ROOK_PHASE: i32 = 2;
    const QUEEN_PHASE: i32 = 4;
    const MAX_PHASE: i32 = 4 * Self::KNIGHT_PHASE
        + 4 * Self::BISHOP_PHASE
        + 4 * Self::ROOK_PHASE
        + 2 * Self::QUEEN_PHASE;

    #[inline]
    pub fn new(knights: u32, bishops: u32, rooks: u32, queens: u32) -> Phase {
        let phase = Self::KNIGHT_PHASE * knights as i32
            + Self::BISHOP_PHASE * bishops as i32
            + Self::ROOK_PHASE * rooks as i32
            + Self::QUEEN_PHASE * queens as i32;

        Phase(phase.min(Self::MAX_PHASE))
    }

    #[inline]
    pub fn from_position(position: &Position) -> Phase {
        Phase::new(
            position.knights().count(),
            position.bishops().count(),
            position.rooks().count(),
            position.queens().count(),
        )
    }
}

impl PhaseEval {
    #[inline]
    pub const fn new(midgame_centipawns: i32, endgame_centipawns: i32) -> Self {
        Self {
            midgame_centipawns,
            endgame_centipawns,
        }
    }

    #[inline]
    pub fn interpolate(self, phase: Phase) -> Eval {
        Eval::centipawns(
            (self.midgame_centipawns * phase.0
                + self.endgame_centipawns * (Phase::MAX_PHASE - phase.0))
                / Phase::MAX_PHASE,
        )
    }
}

impl ops::AddAssign for PhaseEval {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.midgame_centipawns += other.midgame_centipawns;
        self.endgame_centipawns += other.endgame_centipawns;
    }
}

impl ops::SubAssign for PhaseEval {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.midgame_centipawns -= other.midgame_centipawns;
        self.endgame_centipawns -= other.endgame_centipawns;
    }
}

impl ops::Neg for PhaseEval {
    type Output = PhaseEval;

    #[inline]
    fn neg(self) -> PhaseEval {
        PhaseEval {
            midgame_centipawns: -self.midgame_centipawns,
            endgame_centipawns: -self.endgame_centipawns,
        }
    }
}

impl ops::Add for PhaseEval {
    type Output = PhaseEval;

    #[inline]
    fn add(self, other: PhaseEval) -> PhaseEval {
        Self {
            midgame_centipawns: self.midgame_centipawns + other.midgame_centipawns,
            endgame_centipawns: self.endgame_centipawns + other.endgame_centipawns,
        }
    }
}

impl ops::Sub for PhaseEval {
    type Output = PhaseEval;

    #[inline]
    fn sub(self, other: PhaseEval) -> PhaseEval {
        Self {
            midgame_centipawns: self.midgame_centipawns - other.midgame_centipawns,
            endgame_centipawns: self.endgame_centipawns - other.endgame_centipawns,
        }
    }
}
