mod taper;

use crate::search::MAX_PLY;
use core::fmt;
use std::ops;
pub use taper::*;

#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub enum Bound {
    #[default]
    Exact = 0,
    Lower,
    Upper,
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Eval(pub i32);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EvalKind {
    Centipawns(i32),
    MateIn(u32),
    MatedIn(u32),
}

impl Eval {
    const MATE_VALUE: i32 = 16777216;
    const MAX_CENTIPAWNS: i32 = Self::MATE_VALUE - MAX_PLY as i32 - 1;
    const MIN_CENTIPAWNS: i32 = -Self::MATE_VALUE + MAX_PLY as i32 + 1;

    pub const ZERO: Eval = Eval(0);
    pub const DRAW: Eval = Eval(0);
    pub const MATE: Eval = Eval(Self::MATE_VALUE);
    pub const MATED: Eval = Eval(-Self::MATE_VALUE);
    pub const MAX: Eval = Eval(i32::MAX);
    pub const MIN: Eval = Eval(-i32::MAX);

    #[inline]
    pub const fn centipawns(cp: i32) -> Eval {
        Eval(if cp < Self::MIN_CENTIPAWNS {
            Self::MIN_CENTIPAWNS
        } else if cp > Self::MAX_CENTIPAWNS {
            Self::MAX_CENTIPAWNS
        } else {
            cp
        })
    }

    #[inline]
    pub const fn mate_in(ply: u32) -> Eval {
        debug_assert!(ply <= MAX_PLY);
        Eval(Self::MATE_VALUE - ply as i32)
    }

    #[inline]
    pub const fn mated_in(ply: u32) -> Eval {
        debug_assert!(ply <= MAX_PLY);
        Eval(-Self::MATE_VALUE + ply as i32)
    }

    #[inline]
    pub const fn is_mate(self) -> bool {
        self.0 < Self::MIN_CENTIPAWNS || self.0 > Self::MAX_CENTIPAWNS
    }

    #[inline]
    pub const fn is_mated(self) -> bool {
        self.0 < Self::MIN_CENTIPAWNS
    }

    #[inline]
    pub const fn plies_from_mate(self) -> Option<u32> {
        let abs = self.0.abs();
        if abs > Self::MAX_CENTIPAWNS {
            Some((Self::MATE_VALUE - abs) as u32)
        } else {
            None
        }
    }

    #[inline]
    pub const fn as_centipawns(self) -> Option<i32> {
        if self.is_mate() {
            return None;
        }
        Some(self.0)
    }

    #[inline]
    pub const fn kind(self) -> EvalKind {
        if let Some(plies_from_mate) = self.plies_from_mate() {
            if self.0 > 0 {
                return EvalKind::MateIn(plies_from_mate);
            } else {
                return EvalKind::MatedIn(plies_from_mate);
            }
        }
        EvalKind::Centipawns(self.0)
    }

    #[inline]
    pub const fn average(self, other: Eval) -> Eval {
        Self((self.0 + other.0) / 2)
    }
}

impl ops::Neg for Eval {
    type Output = Eval;

    #[inline]
    fn neg(self) -> Eval {
        Eval(-self.0)
    }
}

impl ops::AddAssign<i32> for Eval {
    #[inline]
    fn add_assign(&mut self, other: i32) {
        self.0 += other;
    }
}

impl ops::SubAssign<i32> for Eval {
    #[inline]
    fn sub_assign(&mut self, other: i32) {
        self.0 -= other;
    }
}

impl ops::MulAssign<i32> for Eval {
    #[inline]
    fn mul_assign(&mut self, other: i32) {
        self.0 *= other;
    }
}

impl ops::DivAssign<i32> for Eval {
    #[inline]
    fn div_assign(&mut self, other: i32) {
        self.0 /= other;
    }
}

impl ops::Add<Eval> for i32 {
    type Output = Eval;

    #[inline]
    fn add(self, other: Eval) -> Eval {
        Eval(self + other.0)
    }
}

impl ops::Add<i32> for Eval {
    type Output = Eval;

    #[inline]
    fn add(self, other: i32) -> Eval {
        Eval(self.0 + other)
    }
}

impl ops::Sub for Eval {
    type Output = i32;

    #[inline]
    fn sub(self, other: Eval) -> i32 {
        self.0.saturating_sub(other.0)
    }
}

impl ops::Sub<i32> for Eval {
    type Output = Eval;

    #[inline]
    fn sub(self, other: i32) -> Eval {
        Eval(self.0 - other)
    }
}

impl ops::Mul<Eval> for i32 {
    type Output = Eval;

    #[inline]
    fn mul(self, other: Eval) -> Eval {
        Eval(self * other.0)
    }
}

impl ops::Mul<i32> for Eval {
    type Output = Eval;

    #[inline]
    fn mul(self, other: i32) -> Eval {
        Eval(self.0 * other)
    }
}

impl ops::Div<i32> for Eval {
    type Output = Eval;

    #[inline]
    fn div(self, other: i32) -> Eval {
        Eval(self.0 / other)
    }
}

impl fmt::Display for Eval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind() {
            EvalKind::MateIn(plies) => write!(f, "+M{}", plies.div_ceil(2)),
            EvalKind::MatedIn(plies) => write!(f, "-M{}", plies.div_ceil(2)),
            EvalKind::Centipawns(cp) => write!(f, "{:+.02}", cp as f64 / 100.0),
        }
    }
}
